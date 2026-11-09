"""Phase 6, step 2: write the replay files for every method and seed (configs/replay.yaml).

    python scripts/select_replay.py --table runs/replay/replay_table.parquet --instruction_tokens 6000009 \
        --out runs/replay
Writes {method}_s{seed}.jsonl (rows for scripts/run_sft.py --replay) and selection_report.json.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.replay.selectors import jaccard, label_target, select  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--instruction_tokens", type=int, required=True, help="instruction tokens of one SFT run (budget.json)")
    ap.add_argument("--config", default="configs/replay.yaml")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    t = pd.read_parquet(a.table)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tl = label_target(t)
    pool = int(sum(r.tokens_call if g == "CALL" else r.tokens_answer for r, g in zip(t.itertuples(), tl) if g))
    select_tokens = int(cfg["select_frac"] * pool)
    replay_tokens = int(cfg["replay_frac"] * a.instruction_tokens)
    rep = {"pool_tokens": pool, "select_tokens": select_tokens, "replay_tokens": replay_tokens, "methods": {}}
    picks = {}
    for seed in cfg["seeds"]:
        for m in cfg["methods"]:
            rows, distinct = select(t, m, select_tokens, replay_tokens, seed=seed, tau=cfg["uabr"]["tau"],
                                    call_cost=cfg["uabr"]["call_cost"])
            with open(out / f"{m}_s{seed}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            picks[(m, seed)] = distinct
            rep["methods"][f"{m}_s{seed}"] = {
                "distinct": len(distinct), "rows": len(rows),
                "targets": distinct.target.value_counts().to_dict(),
                "mean_margin": float(distinct.call_score.abs().mean()),
                "benefit_counts": {int(k): int(v) for k, v in distinct.benefit.value_counts().sort_index().items()},
                "share_unknown_facts": float((distinct.knows == 0).mean()),
            }
    s0 = cfg["seeds"][0]
    rep["overlap_jaccard"] = {
        "uabr_vs_bpr": jaccard(picks[("uabr", s0)].prompt_id, picks[("bpr", s0)].prompt_id),
        "uabr_vs_random": jaccard(picks[("uabr", s0)].prompt_id, picks[("random", s0)].prompt_id),
        "bpr_vs_random": jaccard(picks[("bpr", s0)].prompt_id, picks[("random", s0)].prompt_id),
    }
    json.dump(rep, open(out / "selection_report.json", "w"), indent=1, default=str)
    print(f"pool {pool:,} tokens | each method selects {select_tokens:,} tokens, replays {replay_tokens:,} tokens")
    for k, v in rep["methods"].items():
        print(f"  {k:10s} distinct {v['distinct']:4d} | rows {v['rows']:5d} | targets {v['targets']} | "
              f"mean margin {v['mean_margin']:.2f} | benefit {v['benefit_counts']} | unknown facts {v['share_unknown_facts']:.2f}")
    print("overlap (Jaccard of selected prompts):", {k: round(v, 3) for k, v in rep["overlap_jaccard"].items()})


if __name__ == "__main__":
    main()
