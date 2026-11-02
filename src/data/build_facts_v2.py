"""Controlled fact layer, version 2 (data_v2).

Why version 2 (see docs/results.md, Phase 4 attempts 1-3): base_v1 stored facts only as statements and never saw the
chat/action tokens, so it could not retrieve facts in question form (0.75 cloze vs 0.19 question recall on 80x facts) and
the router had no internal signal of what it knew. Version 2 changes the fact layer so that this can be learned in
pretraining, and makes the CALL path fallible so that calling is a real decision:

  * fewer facts, more repetition: 2,400 entities x 3 relations = 7,200 facts; 90% taught, 10% never shown;
    exposure tiers 40 / 120 / 300 (small models memorise by repetition, not coverage)
  * question-answer exposures: ~30% of each taught fact's exposures are chat Q&A in exactly the delegation format,
        <BOS><USER> Where was Kael Dorn born?<ASSISTANT><ANSWER> Oslo<EOS>
    so <USER>, <ASSISTANT> and <ANSWER> have trained embeddings and question-format recall exists before Phase 4.
    Only the main question family is used; the held-out family (q_alt) never appears in pretraining.
  * fallible CALL: evidence comes from a top-3 keyword search (src/retrieval/search.py); 20% of records are missing
    from the index, and first names are shared, so look-alike records are returned as distractors.

Writes the same files as version 1 plus retrieval fields:
  facts/exposure_ledger.parquet   one row per fact; + statement_count, qa_count, record_indexed
  facts/statements.jsonl          statements and Q&A exposures (field "kind"), interleaved into pretraining by pack.py
  facts/prompts.jsonl             question prompts; + retrieved (top-3 doc ids, rank-free order), record_retrieved
  facts/evidence.jsonl            the full registry, one record per fact; + indexed (False = missing from the search)

    python -m src.data.build_facts_v2 --vocab data/raw/value_vocab.json --out data/processed/facts
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

from src.data.build_facts import (ALT_FAMILY, CODAS, EVIDENCE, MAIN_FAMILY, ONSETS, QUESTIONS, SPLITS, STATEMENTS,
                                  VOWELS, _fmt)
from src.data.vocab import RELATIONS
from src.retrieval.search import KeywordIndex, stable_shuffle

PRESET = {"n_entities": 2400, "rels_per_entity": 3, "p_exposed": 0.9, "tiers": [40, 120, 300], "qa_frac": 0.3,
          "missing_record_frac": 0.2, "retrieval_k": 3, "n_first_names": 300, "seed": 2026}

# extra words each record is searchable by, so questions about a relation can match its records
REL_KEYWORDS = {
    "born_in": "born birthplace birth city",
    "citizen_of": "citizen citizenship country nation national passport",
    "occupation": "occupation profession job work works",
    "language": "language speaks speak spoken fluent",
    "birth_year": "born birth year",
}


def qa_text(question, answer):
    """A Q&A exposure; pack.py adds <BOS> ... <EOS>, giving exactly the delegation ANSWER format."""
    return f"<USER> {question}<ASSISTANT><ANSWER> {answer}"


def _word(rng, syll):
    return "".join(rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS) for _ in range(syll)).capitalize()


def make_names(n, n_first, rng):
    """Unique surnames; first names drawn from a pool of n_first, so each first name is shared by ~n/n_first people."""
    firsts = set()
    while len(firsts) < n_first:
        w = _word(rng, 2)
        if len(w) <= 11:
            firsts.add(w)
    firsts = sorted(firsts)
    out, lasts = [], set()
    while len(out) < n:
        last = _word(rng, rng.choice([2, 3, 3]))
        name = f"{rng.choice(firsts)} {last}"
        if last in lasts or last in firsts or len(name) > 24:
            continue
        lasts.add(last)
        out.append(name)
    return out


def build(vocab_path, out_dir, n_entities, rels_per_entity, p_exposed, tiers, qa_frac, missing_record_frac,
          retrieval_k, n_first_names, seed):
    rng = random.Random(seed)
    vocab = json.load(open(vocab_path))
    from src.data.vocab import clean_vocab
    values = clean_vocab(vocab["relations"])
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    names = make_names(n_entities, n_first_names, rng)
    order = list(range(n_entities)); rng.shuffle(order)
    split_of, start = {}, 0
    for i, (sname, frac) in enumerate(SPLITS):
        end = n_entities if i == len(SPLITS) - 1 else start + round(frac * n_entities)
        for k in order[start:end]:
            split_of[k] = sname
        start = end

    ledger, statements, prompts, evidence = [], [], [], []
    rel_keys = list(RELATIONS)
    for k, name in enumerate(names):
        ent_id = f"E{k:05d}"
        for rel in rng.sample(rel_keys, rels_per_entity):
            cand = values[rel]
            w = [1.0] * len(cand) if rel == "birth_year" else [1.0 / (r + 1) ** 0.8 for r in range(len(cand))]
            val = rng.choices(cand, weights=w, k=1)[0]
            fact_id = f"{ent_id}-{RELATIONS[rel]}-{val['value_id']}"
            exposed = rng.random() < p_exposed
            count = rng.choice(tiers) if exposed else 0
            n_qa = round(count * qa_frac)
            n_st = count - n_qa
            ev_id = f"ev_{ent_id}_{RELATIONS[rel]}"
            indexed = rng.random() >= missing_record_frac
            split = split_of[k]
            ledger.append({
                "fact_id": fact_id, "entity_id": ent_id, "pseudonym": name, "relation": rel,
                "relation_id": RELATIONS[rel], "value_id": val["value_id"], "value_text": val["label"],
                "accepted_aliases": json.dumps(val["aliases"], ensure_ascii=False),
                "exposure_status": "exposed" if exposed else "withheld", "exposure_count": count,
                "statement_count": n_st, "qa_count": n_qa, "record_indexed": indexed,
                "pretrain_shard_ids": "", "evidence_doc_id": ev_id, "split": split,
                "source": f"value: {vocab['source']} ({vocab['license']}); entity: pseudonym",
            })
            evidence.append({"evidence_doc_id": ev_id, "fact_id": fact_id, "title": "Controlled registry record",
                             "text": _fmt(EVIDENCE[rel], name, val["label"]), "relation": rel, "indexed": indexed,
                             "source": "derived from value vocab, pseudonymous entity"})
            st_order = rng.sample(range(len(STATEMENTS[rel])), len(STATEMENTS[rel]))
            for j in range(n_st):
                statements.append({"fact_id": fact_id, "kind": "statement",
                                   "text": _fmt(STATEMENTS[rel][st_order[j % len(st_order)]], name, val["label"])})
            q_order = rng.sample(MAIN_FAMILY, len(MAIN_FAMILY))       # held-out family never enters pretraining
            for j in range(n_qa):
                q = QUESTIONS[rel][q_order[j % len(q_order)]].format(e=name)
                statements.append({"fact_id": fact_id, "kind": "qa", "text": qa_text(q, val["label"])})
            if split == "delegation_train":
                tids = MAIN_FAMILY
            elif split == "replay_construction":
                tids = rng.sample(MAIN_FAMILY, 2)
            else:
                tids = [rng.choice(MAIN_FAMILY), rng.choice(ALT_FAMILY)]
            for t in tids:
                q = QUESTIONS[rel][t].format(e=name)
                pid = hashlib.sha1(f"{fact_id}|{t}".encode()).hexdigest()[:12]
                prompts.append({"prompt_id": pid, "fact_id": fact_id, "split": split, "relation": rel,
                                "template_id": f"{rel}_q{t}", "template_family": "q_alt" if t in ALT_FAMILY else "q_main",
                                "question": q, "answer": val["label"], "aliases": val["aliases"],
                                "exposure_status": "exposed" if exposed else "withheld", "exposure_count": count,
                                "evidence_doc_id": ev_id})

    # fallible CALL: top-k keyword search over the indexed records only, precomputed and frozen per prompt
    index = KeywordIndex((e["evidence_doc_id"], e["text"] + " " + REL_KEYWORDS[e["relation"]]) for e in evidence if e["indexed"])
    for p in prompts:
        hits = index.search(p["question"], retrieval_k)
        p["retrieved"] = stable_shuffle(hits, p["prompt_id"])
        p["record_retrieved"] = p["evidence_doc_id"] in hits

    rng.shuffle(statements)
    df = pd.DataFrame(ledger)
    df.to_parquet(out / "exposure_ledger.parquet", index=False)
    for fname, rows in (("statements.jsonl", statements), ("prompts.jsonl", prompts), ("evidence.jsonl", evidence)):
        with open(out / fname, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pdf = pd.DataFrame(prompts)
    summary = {
        "data_design": "v2", "preset": {"n_entities": n_entities, "rels_per_entity": rels_per_entity, "p_exposed": p_exposed,
                                        "tiers": tiers, "qa_frac": qa_frac, "missing_record_frac": missing_record_frac,
                                        "retrieval_k": retrieval_k, "n_first_names": n_first_names, "seed": seed},
        "n_entities": n_entities, "n_facts": len(df), "n_prompts": len(prompts),
        "n_statements": int(sum(r["kind"] == "statement" for r in statements)),
        "n_qa_exposures": int(sum(r["kind"] == "qa" for r in statements)),
        "vocab_source": vocab["source"],
        "facts_by_split_and_status": df.groupby(["split", "exposure_status"]).size().unstack().to_dict(),
        "exposed_by_tier": df[df.exposure_count > 0].exposure_count.value_counts().sort_index().to_dict(),
        "records_indexed": float(df.record_indexed.mean()),
        "record_retrieved_by_split": pdf.groupby("split").record_retrieved.mean().round(4).to_dict(),
        "record_retrieved_if_indexed": float(pdf[pdf.evidence_doc_id.map(dict(zip(df.evidence_doc_id, df.record_indexed)))]
                                             .record_retrieved.mean()),
    }
    json.dump(summary, open(out / "facts_summary.json", "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="data/raw/value_vocab.json")
    ap.add_argument("--out", default="data/processed/facts")
    for k, v in PRESET.items():
        if isinstance(v, list):
            ap.add_argument(f"--{k}", type=int, nargs="+", default=v)
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v)
    a = vars(ap.parse_args())
    build(a.pop("vocab"), a.pop("out"), **a)
