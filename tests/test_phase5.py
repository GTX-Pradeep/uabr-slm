"""Phase 5: shared instruction batches, replay mixing, SFT branches and the pilot evaluation (tiny, CPU)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train.sft_batches import add_replay, count_tokens, instruction_batches, sft_labels  # noqa: E402


def test_sft_labels_supervise_only_the_response():
    ids = [1, 8, 50, 51, 9, 60, 61, 2]          # <BOS><USER> a b <ASSISTANT> x y <EOS>, prompt_len = 5
    lab = sft_labels(ids, 5)
    assert lab == [-100, -100, -100, -100, 60, 61, 2, -100]   # labels[t] = ids[t+1] only where t+1 >= prompt_len


def test_instruction_batches_are_seeded_and_complete():
    ex = [([i], [i]) for i in range(70)]
    a, b, c = instruction_batches(ex, 32, 11), instruction_batches(ex, 32, 11), instruction_batches(ex, 32, 29)
    assert a == b and a != c
    assert [len(x) for x in a] == [32, 32, 6]
    assert sorted(e[0][0] for bt in a for e in bt) == list(range(70))


def test_add_replay_keeps_instruction_batches_and_spreads_replay():
    instr = instruction_batches([([i], [i]) for i in range(100)], 10, 11)
    rep = [([1000 + j], [1000 + j]) for j in range(20)]
    mixed = add_replay(instr, rep, 11)
    assert len(mixed) == len(instr)                                   # same number of updates
    for m, b in zip(mixed, instr):
        assert m[: len(b)] == b                                       # instruction part unchanged and in order
    added = [len(m) - len(b) for m, b in zip(mixed, instr)]
    assert sum(added) == 20 and max(added) - min(added) <= 1          # spread evenly through the run
    assert instr[0] == instruction_batches([([i], [i]) for i in range(100)], 10, 11)[0]  # input not modified
    assert count_tokens(rep) == {"examples": 20, "tokens": 20, "supervised_tokens": 20}


def run(*args):
    subprocess.check_call([sys.executable, *args], cwd=ROOT)


@pytest.fixture(scope="module")
def pilot(built, tmp_path_factory):
    d = tmp_path_factory.mktemp("p5")
    cfg = yaml.safe_load(open(ROOT / "configs/tiny_cpu.yaml"))
    cfg["model"]["max_seq_len"] = 512                                 # SFT examples are up to 512 tokens
    cfg["training"].update({"micro_batch_size": 2, "grad_accum_steps": 1})
    yaml.safe_dump(cfg, open(d / "tiny512.yaml", "w"))
    run("src/train/pretrain.py", "--config", str(d / "tiny512.yaml"), "--data_dir", str(built / "pretrain"),
        "--output_dir", str(d / "base"), "--max_steps", "5", "--device", "cpu")
    sft_cfg = yaml.safe_load(open(ROOT / "configs/sft_methods/standard.yaml"))
    sft_cfg["finetune"].update({"batch_size": 8, "warmup_steps": 1, "log_every_steps": 1000, "checkpoint_every_steps": 0})
    sft_cfg["quick"]["max_batches"] = 8
    yaml.safe_dump(sft_cfg, open(d / "sft.yaml", "w"))
    common = ["--init", str(d / "base"), "--data", str(built), "--config", str(d / "sft.yaml"), "--quick", "--device", "cpu"]
    run("scripts/run_sft.py", *common, "--seed", "11", "--out", str(d / "std"))
    prompts = [json.loads(l) for l in open(built / "facts" / "prompts.jsonl")]
    rc = [p["prompt_id"] for p in prompts if p["split"] == "replay_construction"][:6]
    with open(d / "replay.jsonl", "w") as f:
        for i, pid in enumerate(rc):
            f.write(json.dumps({"prompt_id": pid, "target": "CALL" if i % 2 else "ANSWER"}) + "\n")
    run("scripts/run_sft.py", *common, "--seed", "11", "--out", str(d / "rep"), "--replay", str(d / "replay.jsonl"))
    run("scripts/pilot_eval.py", "--data", str(built), "--delegation", str(d / "base"), "--runs", str(d / "std"),
        "--out", str(d / "pilot"), "--quick", "--n_boot", "20", "--device", "cpu")
    return d


def test_sft_branch_outputs(pilot):
    b = json.load(open(pilot / "std" / "budget.json"))
    assert b["updates"] == 8 and b["save_at_steps"] == [2, 4] and b["train"]["status"] == "completed"
    for f in ["step_2.pt", "step_4.pt", "final.pt"]:
        assert (pilot / "std" / f).exists(), f


def test_replay_branch_has_same_updates_and_instruction_tokens(pilot):
    s = json.load(open(pilot / "std" / "budget.json"))
    r = json.load(open(pilot / "rep" / "budget.json"))
    assert r["updates"] == s["updates"]
    assert r["instruction"] == s["instruction"]
    assert r["replay"]["examples"] == 6 and r["replay"]["tokens"] > 0 and s["replay"]["examples"] == 0
    assert r["replay_targets"] == {"ANSWER": 3, "CALL": 3}


def test_pilot_report(pilot):
    rep = json.load(open(pilot / "pilot" / "pilot_report.json"))
    assert rep["decision"] in {"GO", "RECALIBRATION_ONLY", "NO_CHANGE"}
    assert set(rep["per_seed"]) == {"11"}
    assert set(rep["per_seed"]["11"]["diff_100_minus_0"]) == {"auroc", "acc_budget_10", "acc_frozen_thr_10"}
    import pandas as pd
    t = pd.read_csv(pilot / "pilot" / "pilot_table.csv")
    assert sorted(t.frac.round(2)) == [0.0, 0.25, 0.5, 1.0]
