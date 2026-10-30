"""Instruction batches shared by every SFT branch (Phases 5-6).

Every branch with the same seed sees exactly the same instruction batches in the same order. Replay branches add
their replay examples *inside* those batches, spread evenly through training, so all branches make the same number of
optimizer updates and differ only in which replay examples they add.
"""
import json
import random
from pathlib import Path


def sft_labels(ids, prompt_len):
    """Shifted labels (labels[t] = ids[t+1]) supervised only on the response, i.e. positions t+1 >= prompt_len."""
    return [ids[t + 1] if t + 1 >= prompt_len else -100 for t in range(len(ids) - 1)] + [-100]


def load_sft(data_dir, split="train"):
    """(ids, labels) for data_v1/sft/sft_{split}.jsonl."""
    rows = [json.loads(l) for l in open(Path(data_dir) / "sft" / f"sft_{split}.jsonl")]
    return [(r["ids"], sft_labels(r["ids"], r["prompt_len"])) for r in rows]


def instruction_batches(examples, batch_size, seed):
    """Seeded shuffle, then consecutive batches (the last one may be smaller)."""
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    ex = [examples[i] for i in order]
    return [ex[k : k + batch_size] for k in range(0, len(ex), batch_size)]


def add_replay(batches, replay_examples, seed):
    """Insert replay examples evenly through the run: example j joins batch floor(j * n_batches / n_replay).

    The replay order is shuffled with `seed`, so which batch an example lands in does not depend on how the selector
    ranked it. Returns new batches; the input batches are not modified.
    """
    out = [list(b) for b in batches]
    rep = list(replay_examples)
    random.Random(seed + 7919).shuffle(rep)
    n_b, n_r = len(out), len(rep)
    for j, e in enumerate(rep):
        out[min(n_b - 1, j * n_b // n_r)].append(e)
    return out


def count_tokens(examples):
    """Total and supervised (loss-bearing) tokens of a list of (ids, labels)."""
    return {"examples": len(examples), "tokens": sum(len(i) for i, _ in examples),
            "supervised_tokens": sum(sum(1 for t in l if t != -100) for _, l in examples)}
