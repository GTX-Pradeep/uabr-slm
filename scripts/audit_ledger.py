"""The data audits, as one pass/fail report: six for data_v1, eight for data_v2 (+ retrieval, Q&A hold-out).

    python scripts/audit_ledger.py --data data/processed
Writes data/processed/audit_report.json and audit_alias_sample.csv (100 rows to check by hand).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.formats import normalize_answer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--min_class", type=int, default=100,
                    help="min facts per split and class (data_v2: 50, because only 10% of facts are withheld)")
    a = ap.parse_args()
    data = Path(a.data)
    led = pd.read_parquet(data / "facts" / "exposure_ledger.parquet")
    prompts = pd.read_json(data / "facts" / "prompts.jsonl", lines=True)
    ev = pd.read_json(data / "facts" / "evidence.jsonl", lines=True)
    rep, ok = {}, {}

    # 1. duplicate facts across splits (and entities split cleanly)
    fact_splits = led.groupby("fact_id").split.nunique()
    ent_splits = led.groupby("entity_id").split.nunique()
    rep["1_duplicate_facts"] = {"facts_in_multiple_splits": int((fact_splits > 1).sum()),
                                "entities_in_multiple_splits": int((ent_splits > 1).sum()),
                                "duplicate_fact_ids": int(led.fact_id.duplicated().sum())}
    ok["1_duplicate_facts"] = all(v == 0 for v in rep["1_duplicate_facts"].values())

    # 2. normalized prompt overlap across splits
    prompts["norm"] = prompts.question.map(normalize_answer)
    cross = prompts.groupby("norm").split.nunique()
    rep["2_prompt_overlap"] = {"normalized_prompts_crossing_splits": int((cross > 1).sum()),
                               "prompt_split_matches_fact_split": bool(
                                   (prompts.merge(led[["fact_id", "split"]], on="fact_id", suffixes=("", "_f"))
                                    .eval("split == split_f")).all())}
    ok["2_prompt_overlap"] = rep["2_prompt_overlap"]["normalized_prompts_crossing_splits"] == 0 and \
        rep["2_prompt_overlap"]["prompt_split_matches_fact_split"]

    # 3. class balance (report counts; fail only if a split lacks either class)
    bal = led.groupby(["split", "exposure_status"]).size().unstack(fill_value=0)
    tiers = led[led.exposure_count > 0].groupby(["split", "exposure_count"]).size().unstack(fill_value=0)
    rep["3_class_balance"] = {"facts": bal.to_dict(orient="index"), "exposed_tiers": tiers.to_dict(orient="index"),
                              "relations": led.relation.value_counts().to_dict()}
    ok["3_class_balance"] = bool((bal.min(axis=1) >= a.min_class).all())

    # 4. evidence completeness: exactly one evidence record per fact
    ev_per_fact = ev.groupby("fact_id").size()
    missing = set(led.fact_id) - set(ev.fact_id)
    rep["4_evidence"] = {"facts_missing_evidence": len(missing), "facts_with_multiple_evidence": int((ev_per_fact > 1).sum()),
                         "evidence_contains_answer": float(
                             ev.merge(led[["fact_id", "value_text"]], on="fact_id")
                             .apply(lambda r: r.value_text in r.text, axis=1).mean())}
    ok["4_evidence"] = rep["4_evidence"]["facts_missing_evidence"] == 0 and \
        rep["4_evidence"]["facts_with_multiple_evidence"] == 0 and rep["4_evidence"]["evidence_contains_answer"] == 1.0

    # 5. alias audit: automatic checks + 100-row sample for a manual look
    aliases = led.accepted_aliases.map(json.loads)
    short = aliases.map(lambda al: any(len(normalize_answer(x)) <= 1 for x in al))
    shared = Counter(normalize_answer(x) for al in led.drop_duplicates("value_id").accepted_aliases.map(json.loads) for x in al)
    ambiguous = sorted(k for k, c in shared.items() if c > 1)
    led.sample(100, random_state=0)[["fact_id", "relation", "value_text", "accepted_aliases"]].to_csv(
        data / "audit_alias_sample.csv", index=False, encoding="utf-8-sig")  # BOM so Excel shows accents
    rep["5_aliases"] = {"facts_with_1char_alias": int(short.sum()), "aliases_shared_by_different_values": ambiguous[:20],
                        "manual_sample": "audit_alias_sample.csv - check 100 rows by hand and record the result"}
    ok["5_aliases"] = int(short.sum()) == 0

    # 6. corpus leakage: pseudonyms found in the Wikipedia sample (from the tokenizer audit)
    tr_path = data / "tokenizer" / "tokenizer_report.json"
    hits = json.load(open(tr_path)).get("pseudonym_hits_in_wiki_sample", None) if tr_path.exists() else None
    rep["6_leakage"] = {"pseudonym_hits_in_wiki_sample": hits,
                        "shard_ids_recorded_for_exposed": float((led[led.exposure_count > 0].pretrain_shard_ids != "").mean())
                        if "pretrain_shard_ids" in led else None,
                        "withheld_with_shard_ids": int((led[led.exposure_count == 0].pretrain_shard_ids != "").sum())}
    ok["6_leakage"] = hits == [] and rep["6_leakage"]["withheld_with_shard_ids"] == 0

    # data_v2 only: 7. retrieval integrity, 8. the held-out question family never appears in pretraining Q&A
    if "retrieved" in prompts.columns:
        indexed = dict(zip(ev.evidence_doc_id, ev.indexed))
        own_indexed = prompts.evidence_doc_id.map(indexed)
        rep["7_retrieval"] = {
            "retrieved_unindexed_records": int(prompts.retrieved.map(lambda ds: sum(not indexed[d] for d in ds)).sum()),
            "prompts_with_fewer_than_k": int((prompts.retrieved.map(len) < prompts.retrieved.map(len).max()).sum()),
            "record_retrieved_share": float(prompts.record_retrieved.mean()),
            "record_retrieved_matches_indexed": bool((prompts.record_retrieved == own_indexed).all()),
            "record_retrieved_by_split": prompts.groupby("split").record_retrieved.mean().round(4).to_dict(),
        }
        ok["7_retrieval"] = rep["7_retrieval"]["retrieved_unindexed_records"] == 0 and \
            rep["7_retrieval"]["prompts_with_fewer_than_k"] == 0 and rep["7_retrieval"]["record_retrieved_matches_indexed"]
        qa_texts = {json.loads(l)["text"] for l in open(data / "facts" / "statements.jsonl") if '"kind": "qa"' in l}
        qa_questions = {t.split("<USER> ", 1)[1].split("<ASSISTANT>", 1)[0] for t in qa_texts}
        alt = prompts[prompts.template_family == "q_alt"]
        rep["8_qa_holdout"] = {"qa_exposures_distinct": len(qa_texts),
                               "held_out_questions_seen_in_pretraining": int(alt.question.isin(qa_questions).sum()),
                               "withheld_facts_with_qa": int(led[(led.exposure_count == 0) & (led.qa_count > 0)].shape[0])}
        ok["8_qa_holdout"] = rep["8_qa_holdout"]["held_out_questions_seen_in_pretraining"] == 0 and \
            rep["8_qa_holdout"]["withheld_facts_with_qa"] == 0

    rep["pass"] = ok
    rep["all_pass"] = all(ok.values())
    json.dump(rep, open(data / "audit_report.json", "w"), indent=1, default=str)
    for k, v in ok.items():
        print(("PASS " if v else "FAIL ") + k)
    print("ALL PASS" if rep["all_pass"] else "SOME AUDITS FAILED - see audit_report.json")
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
