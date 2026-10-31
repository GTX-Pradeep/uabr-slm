"""Phase 5: does plain SFT damage the delegation policy in a way that recalibration cannot fix?

Scores delegation_v1 (0%) and every pilot checkpoint (25%, 50%, 100% per seed) on the dev split, then applies the
decision rule fixed in docs/pilot_decision.md before any result was seen:

  GO                  in every seed, the 100% checkpoint has a significant drop (paired-bootstrap 95% CI below 0)
                      in routing AUROC or in accuracy at a re-picked 10% call budget. Neither can be repaired by
                      moving the threshold, because both are threshold-free.
  RECALIBRATION_ONLY  not GO, but in every seed something dropped, and at least the frozen threshold (chosen on
                      delegation_v1) loses accuracy: the ranking survived and only calibration moved.
  NO_CHANGE           otherwise.

    python scripts/pilot_eval.py --data data_v1 --delegation delegation_v1.pt \
        --runs runs/sft_standard_s11,runs/sft_standard_s29 --out runs/pilot
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import formats  # noqa: E402
from src.evaluation import metrics as M  # noqa: E402
from src.evaluation.predict import load_prompts, predict  # noqa: E402
from src.model.load import load_model  # noqa: E402
from src.retrieval.evidence import EvidenceIndex  # noqa: E402
from src.train.sft import collate  # noqa: E402
from src.train.sft_batches import load_sft  # noqa: E402

BUDGETS = (0.05, 0.10, 0.20)


@torch.no_grad()
def sft_val_loss(model, examples, batch_size=32):
    """Mean per-token loss on the held-out instruction responses (is SFT actually learning?)."""
    model.eval()
    dev = next(model.parameters()).device
    tot, n = 0.0, 0
    for k in range(0, len(examples), batch_size):
        x, y = collate(examples[k : k + batch_size], dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            _, loss = model(x, y)
        m = int((y != -100).sum())
        tot += loss.item() * m; n += m
    return tot / max(n, 1)


def checkpoints(run_dir):
    """[(fraction, path)] for step_N.pt and final.pt of one SFT run."""
    run = Path(run_dir)
    total = json.load(open(run / "budget.json"))["updates"]
    out = [(int(re.findall(r"\d+", p.stem)[0]) / total, p) for p in run.glob("step_*.pt")]
    out.append((1.0, run / "final.pt"))
    return sorted(out)


def score(path, tok, prompts, evidence, sft_val, run_id, label, device, fmt):
    model, config, info = load_model(path, device)
    assert formats.format_of(config) == fmt, f"{path} uses format {formats.format_of(config)}, expected {fmt}"
    df = predict(model, tok, prompts, evidence, fmt, run_id=run_id, checkpoint=f"{label}@{info['step']}")
    vl = sft_val_loss(model, sft_val)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return df, vl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--delegation", required=True, help="delegation_v1.pt (the 0% point)")
    ap.add_argument("--runs", required=True, help="comma-separated SFT run folders, one per seed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--quick", action="store_true", help="400 dev prompts, 200 bootstrap resamples")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    t0 = time.time()
    data, out = Path(a.data), Path(a.out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(data / "tokenizer" / "tokenizer.json"))
    evidence = EvidenceIndex(data / "facts")
    prompts = load_prompts(data / "facts", "dev", limit=400 if a.quick else None, seed=0)
    sft_val = load_sft(data, "val")
    n_boot = 200 if a.quick else a.n_boot
    print(f"dev prompts {len(prompts):,} | instruction val examples {len(sft_val)} | bootstrap {n_boot}")

    # 0%: delegation_v1. Its dev thresholds for each call rate are frozen and reused for every later checkpoint.
    fmt = formats.format_of(load_model(a.delegation, "cpu")[1])
    print(f"delegation format: {fmt}")
    base_df, base_vl = score(a.delegation, tok, prompts, evidence, sft_val, "delegation_v1", "delegation_v1", a.device, fmt)
    base_df.to_parquet(out / "predictions" / "delegation_v1.parquet", index=False)
    thr = {b: M.threshold_for_rate(base_df.call_score, b) for b in BUDGETS}

    def row(df, vl, seed, frac):
        s = M.summarize(df, BUDGETS, dev_thresholds=thr)
        return {"seed": seed, "frac": frac, "instruction_val_loss": vl, "auroc": s["auroc"],
                "average_precision": s["average_precision"], "default_call_rate": s["default_call_rate"],
                **{f"acc_budget_{int(b * 100)}": s[f"budget_{int(b * 100)}"]["accuracy"] for b in BUDGETS},
                **{f"acc_frozen_thr_{int(b * 100)}": s[f"frozen_threshold_{int(b * 100)}"]["accuracy"] for b in BUDGETS},
                "frozen_thr_10_call_rate": s["frozen_threshold_10"]["call_rate"],
                "harmful_answer_rate_10": s["budget_10"]["harmful_answer_rate"],
                "mutual_information": s["decisiveness"]["mutual_information"],
                "call_rate_tier_0": s["call_rate_by_tier"].get(0), "call_rate_tier_80": s["call_rate_by_tier"].get(80),
                "ece": s["ece"], "brier": s["brier"]}

    rows = [row(base_df, base_vl, "all", 0.0)]
    finals = {}
    for run in a.runs.split(","):
        seed = json.load(open(Path(run) / "budget.json"))["seed"]
        for frac, path in checkpoints(run):
            label = f"sft_s{seed}_{int(round(frac * 100))}pct"
            df, vl = score(path, tok, prompts, evidence, sft_val, label, label, a.device, fmt)
            df.to_parquet(out / "predictions" / f"{label}.parquet", index=False)
            rows.append(row(df, vl, seed, frac))
            print(f"  {label}: AUROC {rows[-1]['auroc']:.3f} | acc@10% {rows[-1]['acc_budget_10']:.3f} | "
                  f"frozen-thr acc@10% {rows[-1]['acc_frozen_thr_10']:.3f} | instr val loss {vl:.3f}", flush=True)
            if frac == 1.0:
                finals[seed] = df

    # paired bootstrap of 100% minus 0% on the same dev prompts
    metric_fns = {
        "auroc": lambda d: M.auroc(d.call_score, d.benefit > 0),
        "acc_budget_10": lambda d: M.outcome(d, M.route_top_fraction(d.call_score, 0.10))["accuracy"],
        "acc_frozen_thr_10": lambda d: M.outcome(d, d.call_score.to_numpy() >= thr[0.10])["accuracy"],
    }
    per_seed = {}
    for seed, df in finals.items():
        diffs = {k: dict(zip(("diff", "ci_lo", "ci_hi"), M.paired_bootstrap(df, base_df, fn, n_boot=n_boot)))
                 for k, fn in metric_fns.items()}
        sig = {k: v["ci_hi"] < 0 for k, v in diffs.items()}
        go = sig["auroc"] or sig["acc_budget_10"]
        per_seed[str(seed)] = {"diff_100_minus_0": diffs, "significant_drop": sig, "go": go,
                               "recalibration_only": (not go) and sig["acc_frozen_thr_10"]}
    seeds = list(per_seed.values())
    if seeds and all(s["go"] for s in seeds):
        decision = "GO"
    elif seeds and all(s["go"] or s["recalibration_only"] for s in seeds):
        decision = "RECALIBRATION_ONLY"
    else:
        decision = "NO_CHANGE"

    table = pd.DataFrame(rows)
    table.to_csv(out / "pilot_table.csv", index=False)
    sft_learned = bool(all(table[(table.seed != "all") & (table.frac == 1.0)].instruction_val_loss < base_vl))
    report = {"decision": decision, "per_seed": per_seed, "sft_learned": sft_learned,
              "frozen_thresholds_from_delegation_v1": thr, "delegation_format": fmt, "n_dev_prompts": len(prompts), "n_boot": n_boot,
              "quick": a.quick, "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(report, open(out / "pilot_report.json", "w"), indent=1, default=float)

    pd.set_option("display.width", 220)
    cols = ["seed", "frac", "instruction_val_loss", "auroc", "acc_budget_10", "acc_frozen_thr_10", "frozen_thr_10_call_rate",
            "default_call_rate", "harmful_answer_rate_10", "mutual_information", "call_rate_tier_0", "call_rate_tier_80"]
    print("\n=== Interference pilot (dev) ===")
    print(table[cols].round(3).to_string(index=False))
    for seed, r in per_seed.items():
        d = r["diff_100_minus_0"]
        print(f"seed {seed}: 100% minus 0% | " + " | ".join(
            f"{k} {v['diff']:+.3f} [{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]" for k, v in d.items()))
    print(f"SFT learned (instruction val loss fell in every seed): {sft_learned}")
    print(f"DECISION: {decision}  (rule: docs/pilot_decision.md)")


if __name__ == "__main__":
    main()
