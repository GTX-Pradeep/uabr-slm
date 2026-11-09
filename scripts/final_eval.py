"""Phase 7: the one evaluation on the test split.

Scores delegation_v1 and every SFT branch (Standard, Random, BPR, UABR; one checkpoint per seed) on all test prompts,
then compares methods with paired bootstrap 95% CIs on the same prompts. Primary endpoint: accuracy at a 10% call
budget (top 10% of prompts by call_score call, the rest answer directly). This is the only script that reads test labels.

    python scripts/final_eval.py --data data_v2 --delegation delegation_v1.pt \
        --branch standard=11=runs/sft_standard_s11/final.pt --branch uabr=11=runs/sft_uabr_s11/final.pt ... \
        --out runs/final
"""
import argparse
import json
import sys
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

BUDGETS = (0.05, 0.10, 0.20)
COMPARE = [("uabr", "standard"), ("uabr", "bpr"), ("uabr", "random"), ("bpr", "standard"), ("random", "standard")]
METRICS = {
    "acc_budget_10": lambda d: M.outcome(d, M.route_top_fraction(d.call_score, 0.10))["accuracy"],
    "auroc": lambda d: M.auroc(d.call_score, d.benefit > 0),
}


def row(name, seed, df):
    s = M.summarize(df, BUDGETS)
    tiers = s["call_rate_by_tier"]
    b10 = M.outcome(df, M.route_top_fraction(df.call_score, 0.10))
    return {"method": name, "seed": seed, "auroc": s["auroc"], "average_precision": s["average_precision"],
            **{f"acc_budget_{int(b * 100)}": s[f"budget_{int(b * 100)}"]["accuracy"] for b in BUDGETS},
            "harmful_answer_rate_10": s["budget_10"]["harmful_answer_rate"],
            "wasteful_call_rate_10": float(((M.route_top_fraction(df.call_score, 0.10)) & (df.reward_answer == 1)).mean()),
            "both_fail_rate": s["both_fail_rate"], "acc_always_answer": s["acc_always_answer"],
            "default_call_rate": s["default_call_rate"], "mutual_information": s["decisiveness"]["mutual_information"],
            "call_rate_tier_0": tiers.get(0), "call_rate_tier_top": tiers.get(max(tiers)) if tiers else None,
            "ece": s["ece"], "acc_10_check": b10["accuracy"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--delegation", required=True)
    ap.add_argument("--branch", action="append", required=True, help="method=seed=path_to_final.pt (repeat)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--quick", action="store_true", help="400 test prompts, 200 resamples (pipeline check only)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(data / "tokenizer" / "tokenizer.json"))
    evidence = EvidenceIndex(data / "facts")
    prompts = load_prompts(data / "facts", "test", limit=400 if a.quick else None, seed=0)
    n_boot = 200 if a.quick else a.n_boot
    fmt = formats.format_of(load_model(a.delegation, "cpu")[1])
    print(f"test prompts {len(prompts):,} | format {fmt} | bootstrap {n_boot}")

    preds, rows = {}, []
    for spec in [f"delegation_v1=0={a.delegation}"] + a.branch:
        name, seed, path = spec.split("=", 2)
        model, cfg, info = load_model(path, a.device)
        assert formats.format_of(cfg) == fmt, f"{path}: format {formats.format_of(cfg)} != {fmt}"
        df = predict(model, tok, prompts, evidence, fmt, run_id=f"{name}_s{seed}", checkpoint=f"{name}@{info['step']}")
        df.to_parquet(out / "predictions" / f"{name}_s{seed}.parquet", index=False)
        preds[(name, int(seed))] = df
        rows.append(row(name, int(seed), df))
        print(f"  {name:13s} seed {seed:>2}: acc@10% {rows[-1]['acc_budget_10']:.3f} | AUROC {rows[-1]['auroc']:.3f} | "
              f"harmful@10% {rows[-1]['harmful_answer_rate_10']:.3f} | MI {rows[-1]['mutual_information']:.3f}", flush=True)
        del model
        if a.device == "cuda":
            torch.cuda.empty_cache()

    table = pd.DataFrame(rows)
    table.to_csv(out / "results_table.csv", index=False)
    comps = []
    seeds = sorted({s for (_, s) in preds if s != 0})
    for x, y in COMPARE:
        for s in seeds:
            if (x, s) in preds and (y, s) in preds:
                for mname, fn in METRICS.items():
                    d, lo, hi = M.paired_bootstrap(preds[(x, s)], preds[(y, s)], fn, n_boot=n_boot)
                    comps.append({"comparison": f"{x} - {y}", "seed": s, "metric": mname, "diff": d, "ci_lo": lo, "ci_hi": hi,
                                  "significant": bool(lo > 0 or hi < 0)})
    comps = pd.DataFrame(comps)
    comps.to_csv(out / "comparisons.csv", index=False)
    mean = table[table.method != "delegation_v1"].groupby("method").mean(numeric_only=True).drop(columns="seed")
    json.dump({"n_test_prompts": len(prompts), "format": fmt, "n_boot": n_boot, "quick": a.quick,
               "by_method_mean_over_seeds": mean.round(4).to_dict(orient="index"),
               "comparisons": comps.to_dict(orient="records")}, open(out / "results.json", "w"), indent=1, default=float)

    try:
        import matplotlib.pyplot as plt
        order = [m for m in ["standard", "random", "bpr", "uabr"] if m in set(table.method)]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ref = table[table.method == "delegation_v1"].iloc[0]
        for k, (col, title) in enumerate([("acc_budget_10", "Accuracy at 10% call budget (test)"), ("auroc", "Routing AUROC (test)")]):
            means = [table[table.method == m][col].mean() for m in order]
            ax[k].bar(order, means, color="#8fb3d9")
            for i, m in enumerate(order):
                ax[k].scatter([i] * (table.method == m).sum(), table[table.method == m][col], color="black", zorder=3, s=14)
            ax[k].axhline(ref[col], ls="--", color="gray", label="delegation_v1 (before SFT)")
            ax[k].set_title(title); ax[k].legend(); ax[k].grid(axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig(out / "final_results.png", dpi=140)
    except Exception as e:  # plotting must never block the numbers
        print("plot skipped:", e)

    pd.set_option("display.width", 220)
    cols = ["method", "seed", "acc_budget_10", "acc_budget_5", "acc_budget_20", "auroc", "harmful_answer_rate_10",
            "wasteful_call_rate_10", "mutual_information", "call_rate_tier_0", "call_rate_tier_top"]
    print("\n=== Test results (per seed) ===")
    print(table[cols].round(3).to_string(index=False))
    print("\n=== Mean over seeds ===")
    print(mean[[c for c in cols if c in mean.columns]].round(3).to_string())
    print("\n=== Paired bootstrap, same test prompts (95% CI) ===")
    for r in comps.itertuples():
        print(f"  {r.comparison:18s} seed {r.seed}: {r.metric:13s} {r.diff:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
              f"{'  *' if r.significant else ''}")


if __name__ == "__main__":
    main()
