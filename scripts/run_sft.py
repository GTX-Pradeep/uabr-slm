"""One SFT branch from delegation_v1: Standard SFT (Phase 5 pilot) or SFT + replay (Phase 6).

    # Standard SFT, seed 11, with 25% / 50% checkpoints (the interference pilot)
    python scripts/run_sft.py --init delegation_v1.pt --data data_v1 --seed 11 --out runs/sft_standard_s11

    # SFT + replay: the selector's output lists replay_construction prompt ids and the target action for each
    python scripts/run_sft.py --init delegation_v1.pt --data data_v1 --seed 11 --out runs/sft_uabr_s11 \
        --replay runs/replay/uabr.jsonl

Every branch with the same seed uses identical instruction batches; replay examples are added inside them, so all
branches make the same number of optimizer updates. Writes final.pt (100%), step_N.pt at --save_fracs, budget.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import formats  # noqa: E402
from src.evaluation.predict import load_prompts  # noqa: E402
from src.model.load import load_model  # noqa: E402
from src.retrieval.evidence import EvidenceIndex  # noqa: E402
from src.train.sft import train  # noqa: E402
from src.train.sft_batches import add_replay, count_tokens, instruction_batches, load_sft  # noqa: E402


def replay_examples(tok, data, evidence, rows, fmt):
    """rows: [{"prompt_id", "target": "ANSWER" | "CALL"}] -> (ids, labels) delegation examples in format fmt."""
    prompts = {p["prompt_id"]: p for p in load_prompts(Path(data) / "facts", "replay_construction")}
    out = []
    for r in rows:
        p = prompts[r["prompt_id"]]  # KeyError = a selector picked a prompt outside replay_construction
        prefix = formats.decision_ids(tok, fmt, p["question"], p["relation"], p["entity"])
        if r["target"] == "ANSWER":
            target = formats.answer_target(p["answer"])
        elif r["target"] == "CALL":
            target = formats.call_target(evidence.lookup(p["evidence_doc_id"]), p["answer"])
        else:
            raise ValueError(f"unknown target {r['target']!r}")
        out.append(formats.build_example_ids(tok, prefix, target))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="delegation_v1.pt")
    ap.add_argument("--data", required=True, help="data_v1 folder")
    ap.add_argument("--config", default="configs/sft_methods/standard.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replay", default=None, help="jsonl of {prompt_id, target}; omit for Standard SFT")
    ap.add_argument("--save_fracs", default=None, help="e.g. 0.25,0.5 (default: the config's pilot.save_fracs)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_minutes", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    t0 = time.time()
    cfg = yaml.safe_load(open(a.config))
    ft = dict(cfg["finetune"]); ft["seed"] = a.seed
    assert ft.get("prebatched"), "SFT branches must use shared prebatched instruction batches"
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer.from_file(str(data / "tokenizer" / "tokenizer.json"))
    instr = load_sft(data, "train")
    batches = instruction_batches(instr, ft["batch_size"], a.seed)
    if a.quick:
        batches = batches[: cfg["quick"]["max_batches"]]
    instr_used = [e for b in batches for e in b]

    model, base_config, info = load_model(a.init, a.device)
    fmt = formats.format_of(base_config)
    rep_rows, rep = [], []
    if a.replay:
        rep_rows = [json.loads(l) for l in open(a.replay)]
        evidence = EvidenceIndex(data / "facts")
        rep = replay_examples(tok, data, evidence, rep_rows, fmt)
        batches = add_replay(batches, rep, a.seed)

    total = len(batches)
    fracs = [float(x) for x in a.save_fracs.split(",")] if a.save_fracs else cfg["pilot"]["save_fracs"]
    save_at = sorted({max(1, round(f * total)) for f in fracs if 0 < f < 1})
    print(f"init {a.init} (step {info['step']}, format {fmt}) | seed {a.seed} | {total} updates | "
          f"{len(instr_used):,} instruction + {len(rep):,} replay examples | checkpoints at {save_at} + final")

    budget = {
        "seed": a.seed, "init": str(a.init), "delegation_format": fmt, "replay_file": a.replay, "updates": total, "save_at_steps": save_at,
        "instruction": count_tokens(instr_used), "replay": count_tokens(rep),
        "replay_targets": {t: sum(r["target"] == t for r in rep_rows) for t in ("ANSWER", "CALL")},
        "quick": a.quick, "config": ft,
    }
    json.dump(budget, open(out / "budget.json", "w"), indent=1)
    summ = train(model, [batches], ft, base_config, out, save_at_steps=save_at, resume=a.resume, max_minutes=a.max_minutes)
    budget["train"] = summ
    budget["minutes"] = round((time.time() - t0) / 60, 2)
    json.dump(budget, open(out / "budget.json", "w"), indent=1)
    if summ["status"] != "completed":
        print(f"stopped at {summ['steps_done']}/{summ['total_steps']}: rerun the same command with --resume")
        return
    print(f"done: {total} updates | instruction tokens {budget['instruction']['tokens']:,} | "
          f"replay tokens {budget['replay']['tokens']:,} | {budget['minutes']} min | {out / 'final.pt'}")


if __name__ == "__main__":
    main()
