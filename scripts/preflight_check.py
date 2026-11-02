"""data_v2 preflight decision: is it worth spending ~5 GPU-hours on base_v2?

Reads the knowledge labels and the Phase 4 gate report of the 1/10-scale preflight model and prints GO or STOP.
The criteria were fixed before any data_v2 result existed (docs/results.md, "Version 2"):

  1. question_recall    at the highest exposure tier, question-format recall >= 0.5, and >= 0.7 x sentence (cloze)
                         recall. base_v1 had 0.19 vs 0.75 on its top tier (ratio 0.25), the root cause of Phase 4's failures.
  2. exposure_gradient  question recall at the top tier >= the lowest taught tier + 0.1, and <= 0.15 for never-shown facts.
  3. real_decision      the Phase 4 gate's decisive_policy (mutual information >= 0.1) and tier_sensitive checks both pass.

    python scripts/preflight_check.py --knowledge facts_knowledge.parquet --gate runs/delegation_question/gate_report.json \
        --out preflight_report.json
"""
import argparse
import json

import pandas as pd


def decide(kn, gate):
    by_tier = kn.groupby("exposure_count")[["knows_cloze", "knows_qa"]].mean()
    taught = sorted(t for t in by_tier.index if t > 0)
    top, low = taught[-1], taught[0]
    qa_top, cloze_top = float(by_tier.loc[top, "knows_qa"]), float(by_tier.loc[top, "knows_cloze"])
    qa_low = float(by_tier.loc[low, "knows_qa"])
    qa_never = float(by_tier.loc[0, "knows_qa"]) if 0 in by_tier.index else float("nan")
    checks = {
        "question_recall": qa_top >= 0.5 and qa_top >= 0.7 * cloze_top,
        "exposure_gradient": qa_top >= qa_low + 0.1 and (qa_never != qa_never or qa_never <= 0.15),
        "real_decision": bool(gate["checks"]["decisive_policy"] and gate["checks"]["tier_sensitive"]),
    }
    s = gate["summary"]
    return {
        "decision": "GO" if all(checks.values()) else "STOP",
        "checks": checks,
        "recall_by_tier": {int(t): {"cloze": round(float(r.knows_cloze), 3), "question": round(float(r.knows_qa), 3)}
                           for t, r in by_tier.iterrows()},
        "question_over_cloze_top_tier": round(qa_top / cloze_top, 3) if cloze_top else None,
        "gate": {"all_pass": gate["all_pass"], "checks": gate["checks"], "auroc": gate["auroc"],
                 "auroc_95ci": gate["auroc_95ci"], "mutual_information": s["decisiveness"]["mutual_information"],
                 "call_rate_by_tier": s["call_rate_by_tier"], "acc_budget_10": s["budget_10"]["accuracy"],
                 "call_path_accuracy_record_retrieved": gate.get("call_path_accuracy_record_retrieved")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knowledge", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = decide(pd.read_parquet(a.knowledge), json.load(open(a.gate)))
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print("recall by exposure tier (sentence vs question format):")
    for t, r in rep["recall_by_tier"].items():
        print(f"  {t:>4}x  cloze {r['cloze']:.3f} | question {r['question']:.3f}")
    print(f"question / cloze at the top tier: {rep['question_over_cloze_top_tier']}  (base_v1: 0.25)")
    g = rep["gate"]
    print(f"Phase 4 gate: AUROC {g['auroc']:.3f} | mutual information {g['mutual_information']:.3f} | "
          f"CALL rate by tier {g['call_rate_by_tier']} | failed {[k for k, v in g['checks'].items() if not v] or 'none'}")
    for k, v in rep["checks"].items():
        print(("PASS " if v else "FAIL ") + k)
    print(f"PREFLIGHT: {rep['decision']}")


if __name__ == "__main__":
    main()
