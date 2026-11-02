"""Knowledge-based delegation labels (handbook section 17 fallback, declared deviation).

data_v2: the label is question-format recall (the ANSWER path itself); cloze recall is recorded alongside.

For every fact outside the test split, ask the frozen base model a cloze question in the pretraining format
("Varnek Holloway was born in the city of ___"). knows = 1 if its greedy completion matches an accepted alias.
Delegation targets are then ANSWER if knows else CALL, so ANSWER means "the model actually knows this".

    python scripts/label_knowledge.py --base /kaggle/input/.../base_v1.pt --data /kaggle/input/.../data_v1 \
        --out /kaggle/working/facts_knowledge.parquet
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import formats  # noqa: E402
from src.evaluation.generate import greedy, sequence_logprob  # noqa: E402
from src.model.load import load_model  # noqa: E402

# the knowledge probe uses exactly the ANSWER-path recitation, so an ANSWER label means "the ANSWER path works"
CLOZE = formats.RECITATION


def qa_probe(model, tok, ledger, prompts_path, batch_size=256):
    """Question-format recall: <BOS><USER> question<ASSISTANT><ANSWER> -> greedy answer, i.e. exactly the ANSWER path.
    One main-family question per fact (the lowest template id among the fact's prompts)."""
    rows = [json.loads(l) for l in open(prompts_path)]
    q = {}
    for r in sorted((r for r in rows if r["template_family"] == "q_main"), key=lambda r: int(r["template_id"].split("_q")[1])):
        q.setdefault(r["fact_id"], r["question"])
    prefixes = [tok.encode(formats.delegation_prompt(q[f])).ids + [formats.ANSWER] for f in ledger.fact_id]
    gens = greedy(model, prefixes, max_new=10, stop_ids=[formats.EOS, formats.BOS], batch_size=batch_size)
    preds = [tok.decode(g).strip() for g in gens]
    return preds, [formats.exact_match(p, json.loads(a)) for p, a in zip(preds, ledger.accepted_aliases)]


def label(model, tok, ledger, batch_size=256, prompts_path=None, probe="cloze"):
    prefixes = [[formats.BOS] + tok.encode(CLOZE[r.relation].format(e=r.pseudonym)).ids for r in ledger.itertuples()]
    gens = greedy(model, prefixes, max_new=10, stop_ids=[formats.EOS, formats.BOS], batch_size=batch_size)
    golds = [tok.encode(" " + v).ids for v in ledger.value_text]
    lps = sequence_logprob(model, prefixes, golds, batch_size)
    out = ledger[["fact_id", "entity_id", "split", "relation", "exposure_status", "exposure_count"]].copy()
    preds = [re.split(r"\.(?:\s|$)", tok.decode(g).strip(), maxsplit=1)[0] for g in gens]
    out["cloze_pred"] = preds
    out["knows_cloze"] = [formats.exact_match(p, json.loads(a)) for p, a in zip(preds, ledger.accepted_aliases)]
    out["gold_logprob"] = lps
    if prompts_path is not None:
        out["qa_pred"], out["knows_qa"] = qa_probe(model, tok, ledger, prompts_path, batch_size)
    out["knows"] = out["knows_qa"] if probe == "qa" else out["knows_cloze"]
    out["label_probe"] = probe
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="delegation_train,replay_construction,dev")
    ap.add_argument("--probe", default="auto", choices=["auto", "cloze", "qa"],
                    help="which recall defines the ANSWER label; auto = qa for data_v2 (Q&A seen in pretraining), else cloze")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    t0 = time.time()
    data = Path(a.data)
    tok = Tokenizer.from_file(str(data / "tokenizer" / "tokenizer.json"))
    model, _, info = load_model(a.base, a.device)
    led = pd.read_parquet(data / "facts" / "exposure_ledger.parquet")
    splits = a.splits.split(",")
    assert "test" not in splits, "test facts are never labelled here"
    led = led[led.split.isin(splits)].reset_index(drop=True)
    v2 = "statement_count" in led.columns
    probe = a.probe if a.probe != "auto" else ("qa" if v2 else "cloze")
    out = label(model, tok, led, prompts_path=data / "facts" / "prompts.jsonl", probe=probe)
    out["base_step"] = info["step"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    tab = out.pivot_table(index="split", columns="exposure_count", values="knows", aggfunc="mean").round(3)
    print(f"labelled {len(out):,} facts in {time.time() - t0:.0f}s with base step {info['step']} | label = {probe} recall")
    both = out.groupby("exposure_count")[["knows_cloze", "knows_qa"]].mean().round(3)
    print("\nrecall by exposure tier, sentence (cloze) vs question (qa) format:")
    print(both.to_string())
    print("\nshare the base model knows (1 = ANSWER label), by split x exposure tier:")
    print(tab.to_string())
    print("\nlabel balance:", out.groupby("split").knows.agg(["mean", "size"]).round(3).to_dict(orient="index"))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
