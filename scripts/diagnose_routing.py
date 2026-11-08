"""Does the model know what it does not know? Routing-signal diagnostic (no training).

Phase 4 asks the model to choose <CALL> exactly when its own answer would be wrong. If that fails, there are two very
different explanations:
  (a) no signal: nothing inside the model separates facts it can answer from facts it cannot, or
  (b) a signal exists but Phase 4 does not learn to read it (for example, too few unknown training facts).
This script measures the signal directly on a frozen model, for every dev prompt and a sample of training prompts:

  output confidence  after <ANSWER>: probability of the top next token, entropy, mean log-prob of the greedy answer
  linear probes      logistic regression on the final hidden state, trained on delegation_train prompts and scored on
                     dev, at the decision position (after <ASSISTANT>) and at the answer position (after <ANSWER>)

The target is "the direct answer is wrong" (greedy exact match), i.e. the cases where calling could help. AUROC 0.5 =
no signal. A clearly higher AUROC (for example >= 0.75) at the decision position means the information is there and
readable where the router decides, so the Phase 4 recipe, not the model, is the bottleneck.

    python scripts/diagnose_routing.py --model runs/preflight_40m --data data_v2 --out routing_diagnostic.json
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import formats  # noqa: E402
from src.evaluation import metrics as M  # noqa: E402
from src.evaluation.generate import greedy, sequence_logprob  # noqa: E402
from src.evaluation.predict import load_prompts  # noqa: E402
from src.model.load import load_model  # noqa: E402


@torch.no_grad()
def last_position(model, prompts, batch_size=256):
    """Final-layer hidden state (after the last norm) and next-token distribution at the last position of each prompt."""
    model.eval()
    dev = next(model.parameters()).device
    store = {}
    h = model.norm_f.register_forward_hook(lambda m, i, o: store.__setitem__("h", o))
    hid, probs = [None] * len(prompts), [None] * len(prompts)
    groups = defaultdict(list)
    for i, p in enumerate(prompts):
        groups[len(p)].append(i)
    try:
        for _, idx in groups.items():
            for k in range(0, len(idx), batch_size):
                chunk = idx[k : k + batch_size]
                x = torch.tensor([prompts[i] for i in chunk], device=dev)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    logits, _ = model(x)
                pr = torch.softmax(logits[:, -1].float(), -1)
                hs = store["h"][:, -1].float()
                for row, i in enumerate(chunk):
                    hid[i] = hs[row].cpu().numpy()
                    probs[i] = pr[row]
    finally:
        h.remove()
    maxp = np.array([p.max().item() for p in probs])
    ent = np.array([-(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum().item() for p in probs])
    return np.stack(hid), maxp, ent


def features(model, tok, prompts, fmt):
    dec = [formats.decision_ids(tok, fmt, p["question"], p["relation"], p["entity"]) for p in prompts]
    ans = dec if fmt == "answer_slot" else [d + [formats.ANSWER] for d in dec]
    h_dec, _, _ = last_position(model, dec)
    h_ans, maxp, ent = last_position(model, ans)
    gens = greedy(model, ans, max_new=10, stop_ids=[formats.EOS, formats.BOS],
                  ban_ids=[formats.CALL, formats.EV_OPEN, formats.EV_CLOSE, formats.ANSWER])
    texts = [tok.decode(g).strip() for g in gens]
    wrong = np.array([0 if formats.exact_match(t, p["aliases"]) else 1 for t, p in zip(texts, prompts)])
    safe = [g if g else [formats.EOS] for g in gens]
    mlp = np.array(sequence_logprob(model, ans, safe))
    tiers = np.array([p["exposure_count"] for p in prompts])
    return {"h_dec": h_dec, "h_ans": h_ans, "maxp": maxp, "entropy": ent, "mean_logprob": mlp, "wrong": wrong, "tier": tiers}


def probe_auroc(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return float("nan")
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return float(M.auroc(clf.predict_proba(sc.transform(Xte))[:, 1], yte.astype(bool)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model run dir or .pt (base or delegation checkpoint)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--format", default=None, help="decision format (default: the checkpoint's, else 'question')")
    ap.add_argument("--train_prompts", type=int, default=4000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    data = Path(a.data)
    tok = Tokenizer.from_file(str(data / "tokenizer" / "tokenizer.json"))
    model, cfg, info = load_model(a.model, a.device)
    fmt = a.format or (formats.format_of(cfg) if "delegation" in (cfg or {}) else "question")
    train = load_prompts(data / "facts", "delegation_train")
    random.Random(0).shuffle(train)
    train = train[: a.train_prompts]
    dev = load_prompts(data / "facts", "dev")
    tr, dv = features(model, tok, train, fmt), features(model, tok, dev, fmt)

    y = dv["wrong"]
    rep = {"model": str(a.model), "step": info["step"], "format": fmt, "n_train": len(train), "n_dev": len(dev),
           "dev_answer_wrong_rate": float(y.mean()),
           "auroc_dev": {
               "low_top_prob_after_ANSWER": float(M.auroc(-dv["maxp"], y.astype(bool))),
               "high_entropy_after_ANSWER": float(M.auroc(dv["entropy"], y.astype(bool))),
               "low_mean_logprob_of_answer": float(M.auroc(-dv["mean_logprob"], y.astype(bool))),
               "probe_decision_position": probe_auroc(tr["h_dec"], tr["wrong"], dv["h_dec"], y),
               "probe_answer_position": probe_auroc(tr["h_ans"], tr["wrong"], dv["h_ans"], y),
           }}
    by_tier = pd.DataFrame({"tier": dv["tier"], "wrong": y, "top_prob": dv["maxp"], "entropy": dv["entropy"]}) \
        .groupby("tier").mean().round(3)
    rep["dev_by_tier"] = by_tier.reset_index().to_dict(orient="records")
    best_conf = max(rep["auroc_dev"][k] for k in ("low_top_prob_after_ANSWER", "high_entropy_after_ANSWER", "low_mean_logprob_of_answer"))
    dec = rep["auroc_dev"]["probe_decision_position"]
    if dec >= 0.75:
        verdict = "SIGNAL AT THE DECISION POSITION: the model can tell; the Phase 4 recipe is the bottleneck"
    elif best_conf >= 0.75:
        verdict = "SIGNAL ONLY IN THE ANSWER DISTRIBUTION: readable after <ANSWER>, weak where the router decides"
    else:
        verdict = "WEAK SIGNAL: the model does not reliably know what it does not know at this scale"
    rep["verdict"] = verdict
    json.dump(rep, open(a.out, "w"), indent=1, default=float)

    print(f"model step {info['step']} | format {fmt} | train prompts {len(train):,} | dev prompts {len(dev):,} | "
          f"dev direct answers wrong: {y.mean():.3f}")
    print("\nAUROC on dev for 'the direct answer is wrong' (0.5 = no signal):")
    for k, v in rep["auroc_dev"].items():
        print(f"  {k:30s} {v:.3f}")
    print("\ndev by exposure tier (answer-wrong rate, top next-token probability, entropy after <ANSWER>):")
    print(by_tier.to_string())
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
