"""Phase 6/7: replay selection (pure pandas) and the build-table -> select -> SFT -> final-eval pipeline (tiny, CPU)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.replay.selectors import label_target, rank, select, uabr_scores  # noqa: E402


def table(n=200, seed=0):
    rng = np.random.default_rng(seed)
    ra = rng.integers(0, 2, n)
    rc = rng.integers(0, 2, n)
    return pd.DataFrame({
        "prompt_id": [f"p{i:04d}" for i in range(n)],
        "call_score": rng.normal(0, 2, n),
        "reward_answer": ra, "reward_call": rc, "benefit": rc - ra,
        "knows": rng.integers(0, 2, n), "record_retrieved": rng.random(n) < 0.8,
        "tokens_answer": rng.integers(20, 40, n), "tokens_call": rng.integers(60, 120, n),
    })


def test_label_target_never_calls_without_a_record():
    t = table()
    tl = label_target(t)
    assert (tl[t.knows == 1] == "ANSWER").all()
    assert (tl[(t.knows == 0) & t.record_retrieved] == "CALL").all()
    assert (tl[(t.knows == 0) & ~t.record_retrieved] == "").all()


def test_uabr_target_is_the_action_that_works():
    t = table()
    _, target, du = uabr_scores(t, tau=1.0, call_cost=0.1)
    assert (target[~t.record_retrieved] == "ANSWER").all()                     # no record: CALL cannot work
    ok = t.record_retrieved
    assert (target[ok & (t.reward_call == 1) & (t.reward_answer == 0)] == "CALL").all()
    assert (target[ok & (t.reward_answer == 1)] == "ANSWER").all()             # call_cost breaks the tie
    assert np.allclose(du[ok], (t.reward_call - t.reward_answer - 0.1)[ok])


def test_uabr_prefers_close_and_consequential_prompts():
    t = table()
    r = rank(t, "uabr", tau="median", call_cost=0.1)
    tau = t.call_score.abs().median()
    exp = np.exp(-t.call_score.abs() / tau) * uabr_scores(t, tau, 0.1)[2].abs()
    assert list(r.prompt_id[:20]) == list(t.assign(e=exp).sort_values(["e", "prompt_id"], ascending=[False, True]).prompt_id[:20])


def test_bpr_takes_smallest_margin_first_and_random_depends_on_seed():
    t = table()
    b = rank(t, "bpr")
    assert b.call_score.abs().is_monotonic_increasing
    assert list(rank(t, "random", seed=11).prompt_id) != list(rank(t, "random", seed=29).prompt_id)
    assert list(rank(t, "random", seed=11).prompt_id) == list(rank(t, "random", seed=11).prompt_id)


@pytest.mark.parametrize("method", ["random", "bpr", "uabr"])
def test_select_matches_token_budgets(method):
    t = table()
    rows, distinct = select(t, method, select_tokens=2000, replay_tokens=9000, seed=11, tau="median")
    assert distinct.tokens.sum() <= 2000 and distinct.tokens.sum() > 2000 - 130
    tok = dict(zip(distinct.prompt_id, distinct.tokens))
    total = sum(tok[r["prompt_id"]] for r in rows)
    assert 9000 <= total < 9000 + 130                                           # same replay tokens for every method
    assert set(r["prompt_id"] for r in rows) == set(distinct.prompt_id)          # every pick is replayed
    rr = t.set_index("prompt_id").record_retrieved
    assert all(rr[r["prompt_id"]] for r in rows if r["target"] == "CALL")


def run(*args):
    subprocess.check_call([sys.executable, *args], cwd=ROOT)


@pytest.fixture(scope="module")
def phase67(built, tmp_path_factory):
    d = tmp_path_factory.mktemp("p67")
    cfg = yaml.safe_load(open(ROOT / "configs/tiny_cpu.yaml"))
    cfg["model"]["max_seq_len"] = 512
    cfg["training"].update({"micro_batch_size": 2, "grad_accum_steps": 1})
    yaml.safe_dump(cfg, open(d / "tiny512.yaml", "w"))
    run("src/train/pretrain.py", "--config", str(d / "tiny512.yaml"), "--data_dir", str(built / "pretrain"),
        "--output_dir", str(d / "base"), "--max_steps", "5", "--device", "cpu")
    prompts = [json.loads(l) for l in open(built / "facts" / "prompts.jsonl")]
    pd.DataFrame({"fact_id": sorted({p["fact_id"] for p in prompts})}).assign(knows=lambda x: np.arange(len(x)) % 2) \
        .to_parquet(d / "knowledge.parquet", index=False)
    run("scripts/build_replay_table.py", "--delegation", str(d / "base"), "--data", str(built),
        "--knowledge", str(d / "knowledge.parquet"), "--out", str(d / "replay" / "replay_table.parquet"), "--device", "cpu")
    run("scripts/select_replay.py", "--table", str(d / "replay" / "replay_table.parquet"), "--instruction_tokens", "40000",
        "--out", str(d / "replay"))
    sft_cfg = yaml.safe_load(open(ROOT / "configs/sft_methods/standard.yaml"))
    sft_cfg["finetune"].update({"batch_size": 8, "warmup_steps": 1, "log_every_steps": 1000, "checkpoint_every_steps": 0})
    sft_cfg["quick"]["max_batches"] = 4
    yaml.safe_dump(sft_cfg, open(d / "sft.yaml", "w"))
    common = ["--init", str(d / "base"), "--data", str(built), "--config", str(d / "sft.yaml"), "--quick", "--device", "cpu",
              "--seed", "11"]
    run("scripts/run_sft.py", *common, "--out", str(d / "standard"))
    branches = ["--branch", f"standard=11={d / 'standard' / 'final.pt'}"]
    for m in ["random", "bpr", "uabr"]:
        run("scripts/run_sft.py", *common, "--out", str(d / m), "--replay", str(d / "replay" / f"{m}_s11.jsonl"))
        branches += ["--branch", f"{m}=11={d / m / 'final.pt'}"]
    run("scripts/final_eval.py", "--data", str(built), "--delegation", str(d / "base"), *branches,
        "--out", str(d / "final"), "--quick", "--n_boot", "20", "--device", "cpu")
    return d


def test_replay_files_and_report(phase67):
    rep = json.load(open(phase67 / "replay" / "selection_report.json"))
    assert set(rep["methods"]) == {f"{m}_s{s}" for m in ["random", "bpr", "uabr"] for s in [11, 29]}
    assert rep["replay_tokens"] == 2000                                          # 5% of 40,000
    for m in ["random", "bpr", "uabr"]:
        assert (phase67 / "replay" / f"{m}_s11.jsonl").exists()


def test_replay_branches_are_matched(phase67):
    std = json.load(open(phase67 / "standard" / "budget.json"))
    toks = []
    for m in ["random", "bpr", "uabr"]:
        b = json.load(open(phase67 / m / "budget.json"))
        assert b["updates"] == std["updates"] and b["instruction"] == std["instruction"]
        toks.append(b["replay"]["tokens"])
    assert min(toks) >= 2000 and max(toks) - min(toks) < 600                    # equal replay budget (one example of slack)


def test_final_eval_outputs(phase67):
    t = pd.read_csv(phase67 / "final" / "results_table.csv")
    assert set(t.method) == {"delegation_v1", "standard", "random", "bpr", "uabr"}
    c = pd.read_csv(phase67 / "final" / "comparisons.csv")
    assert {"uabr - standard", "uabr - bpr", "uabr - random"} <= set(c.comparison)
    c = c.dropna(subset=["diff"])                       # AUROC is undefined when an untrained tiny model never benefits from CALL
    assert len(c) and ((c.ci_lo <= c["diff"] + 1e-9) & (c["diff"] <= c.ci_hi + 1e-9)).all()
    r = json.load(open(phase67 / "final" / "results.json"))
    assert r["quick"] and r["n_test_prompts"] > 0
