"""Phase 6, step 1: score every replay_construction prompt once with delegation_v1 (the replay table).

    python scripts/build_replay_table.py --delegation delegation_v1.pt --data data_v2 \
        --knowledge facts_knowledge.parquet --out runs/replay/replay_table.parquet
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import formats  # noqa: E402
from src.evaluation.predict import load_prompts, predict  # noqa: E402
from src.model.load import load_model  # noqa: E402
from src.retrieval.evidence import EvidenceIndex  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delegation", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--knowledge", required=True, help="facts_knowledge.parquet from Phase 4 (the training labels)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(data / "tokenizer" / "tokenizer.json"))
    evidence = EvidenceIndex(data / "facts")
    model, cfg, info = load_model(a.delegation, a.device)
    fmt = formats.format_of(cfg)
    prompts = load_prompts(data / "facts", "replay_construction")
    df = predict(model, tok, prompts, evidence, fmt, run_id="replay_table", checkpoint=f"delegation_v1@{info['step']}")
    kn = pd.read_parquet(a.knowledge)
    df["knows"] = df.fact_id.map(dict(zip(kn.fact_id, kn.knows))).astype(int)
    tok_a, tok_c = [], []
    for p in prompts:
        prefix = formats.decision_ids(tok, fmt, p["question"], p["relation"], p["entity"])
        tok_a.append(len(formats.build_example_ids(tok, prefix, formats.answer_target_for(fmt, p["answer"]))[0]))
        tok_c.append(len(formats.build_example_ids(tok, prefix, formats.call_target(evidence.for_prompt(p), p["answer"]))[0]))
    df["tokens_answer"], df["tokens_call"] = tok_a, tok_c
    df.to_parquet(out, index=False)
    summ = {"prompts": len(df), "format": fmt, "known_share": float(df.knows.mean()),
            "record_retrieved_share": float(df.record_retrieved.mean()),
            "benefit_counts": {int(k): int(v) for k, v in df.benefit.value_counts().sort_index().items()},
            "median_margin": float(df.call_score.abs().median()),
            "pool_tokens_label_target": int(sum(c if (k == 0 and r) else aa for aa, c, k, r in
                                                zip(df.tokens_answer, df.tokens_call, df.knows, df.record_retrieved)))}
    json.dump(summ, open(out.with_suffix(".json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
