"""data_v2: Q&A exposures, keyword-search retrieval, audits, the preflight rule and Phase 4 on a tiny model (CPU)."""
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import formats  # noqa: E402
from src.retrieval.search import KeywordIndex, stable_shuffle  # noqa: E402

WORDS = ("the of and in to was is for on as with by he at from his an were are which this be its first new "
         "city river school war year state national century music film team university known also after").split()


def run(*args):
    subprocess.check_call([sys.executable, *args], cwd=ROOT)


@pytest.fixture(scope="session")
def built_v2(tmp_path_factory):
    d = tmp_path_factory.mktemp("v2")
    rng = random.Random(1)
    with open(d / "wiki.jsonl", "w") as f:
        for _ in range(600):
            text = "\n".join(" ".join(rng.choice(WORDS) for _ in range(rng.randint(60, 140))).capitalize() + "."
                             for _ in range(4))
            f.write(json.dumps({"text": text}) + "\n")
    with open(d / "sft_local.jsonl", "w") as f:
        for i in range(300):
            f.write(json.dumps({"id": f"x{i}", "source": "toy", "instruction": f"Explain topic {i} briefly.",
                                "context": "", "response": " ".join(rng.choice(WORDS) for _ in range(30))}) + "\n")
    data = d / "data_v2"
    run("-m", "src.data.vocab", "--offline", "--out", str(d / "vocab.json"))
    run("-m", "src.data.build_facts_v2", "--vocab", str(d / "vocab.json"), "--out", str(data / "facts"),
        "--n_entities", "300", "--tiers", "4", "8", "12", "--n_first_names", "40")
    run("-m", "src.data.sft_data", "--local", str(d / "sft_local.jsonl"), "--out", str(d / "sft_raw.jsonl"))
    run("-m", "src.data.tokenizer", "--facts", str(data / "facts"), "--sft", str(d / "sft_raw.jsonl"),
        "--out", str(data / "tokenizer"), "--vocab_size", "2000", "--sample_chars", "300000",
        "--wiki_source", "local", "--wiki_local", str(d / "wiki.jsonl"))
    run("-m", "src.data.pack", "--data", str(data), "--sft_raw", str(d / "sft_raw.jsonl"), "--target_tokens", "1500000",
        "--preflight_tokens", "600000", "--val_tokens", "20000", "--sft_val", "50", "--wiki_source", "local",
        "--wiki_local", str(d / "wiki.jsonl"), "--version", "data_v2")
    run("scripts/audit_ledger.py", "--data", str(data), "--min_class", "2")  # tiny set: ~4 withheld dev facts
    return data


def test_search_ranks_by_idf_and_is_deterministic():
    idx = KeywordIndex([("a", "Kael Dorn place of birth Oslo"), ("b", "Kael Brin place of birth Rome"),
                        ("c", "Mira Dorn language Tamil"), ("d", "Tovi Sek occupation baker")])
    assert idx.search("Where was Kael Dorn born?", 2)[0] == "a"           # both name words beat one
    assert set(idx.search("Where was Kael Dorn born?", 3)) == {"a", "b", "c"}
    assert idx.search("Where was Kael Dorn born?", 3) == idx.search("Where was Kael Dorn born?", 3)
    assert stable_shuffle(["a", "b", "c"], "p1") == stable_shuffle(["c", "a", "b"], "p1")


def test_v2_audits_and_manifest(built_v2):
    rep = json.load(open(built_v2 / "audit_report.json"))
    assert rep["all_pass"], rep["pass"]
    assert set(rep["pass"]) >= {"7_retrieval", "8_qa_holdout"}
    m = json.load(open(built_v2 / "manifest.json"))
    assert m["data_version"] == "data_v2" and m["pretrain_preflight"]["train_tokens"] > 0
    assert (built_v2 / "pretrain_preflight" / "train.bin").exists()
    assert set(m["pretrain"]["exposures_by_kind"]) == {"statement", "qa"}


def test_v2_ledger_design(built_v2):
    led = pd.read_parquet(built_v2 / "facts" / "exposure_ledger.parquet")
    assert len(led) == 900 and set(led.exposure_count) <= {0, 4, 8, 12}
    taught = led[led.exposure_count > 0]
    assert 0.8 < len(taught) / len(led) < 0.97                              # ~90% taught
    assert (taught.statement_count + taught.qa_count == taught.exposure_count).all()
    assert (led[led.exposure_count == 0].qa_count == 0).all()
    assert 0.7 < led.record_indexed.mean() < 0.9                             # ~20% of records missing
    assert led.pseudonym.str.split().str[0].value_counts().max() > 5        # first names are shared


def test_qa_exposure_is_exactly_the_delegation_answer_format(built_v2):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(built_v2 / "tokenizer" / "tokenizer.json"))
    qa = [json.loads(l) for l in open(built_v2 / "facts" / "statements.jsonl") if '"kind": "qa"' in l][:20]
    for r in qa:
        q = r["text"].split("<USER> ", 1)[1].split("<ASSISTANT>", 1)[0]
        a = r["text"].split("<ANSWER> ", 1)[1]
        packed = [formats.BOS] + tok.encode(r["text"]).ids + [formats.EOS]
        assert packed == tok.encode(formats.delegation_prompt(q) + formats.answer_target(a)).ids


def test_call_evidence_is_the_frozen_search_result(built_v2):
    from src.evaluation.predict import load_prompts
    from src.retrieval.evidence import EvidenceIndex
    ev = EvidenceIndex(built_v2 / "facts")
    ps = load_prompts(built_v2 / "facts", "dev")
    assert ps and all(len(p["retrieved"]) == 3 for p in ps)
    for p in ps:
        text = ev.for_prompt(p)
        assert text.count(" | ") == 2
        assert (ev.lookup(p["evidence_doc_id"]) in text) == p["record_retrieved"]


def test_preflight_rule():
    from scripts.preflight_check import decide
    rng = np.random.default_rng(0)

    def kn(qa, cloze):
        return pd.DataFrame([{"exposure_count": t, "knows_qa": int(rng.random() < qa[t]), "knows_cloze": int(rng.random() < cloze[t])}
                             for t in qa for _ in range(400)])

    def gate(decisive, tiered):
        return {"checks": {"decisive_policy": decisive, "tier_sensitive": tiered}, "all_pass": decisive and tiered,
                "auroc": 0.7, "auroc_95ci": [0.65, 0.75],
                "summary": {"decisiveness": {"mutual_information": 0.3}, "call_rate_by_tier": {0: 0.9, 300: 0.2},
                            "budget_10": {"accuracy": 0.8}}}
    v1_like = decide(kn({0: .05, 5: .07, 20: .18, 80: .19}, {0: .05, 5: .07, 20: .18, 80: .75}), gate(True, False))
    healthy = decide(kn({0: .04, 40: .5, 120: .8, 300: .9}, {0: .05, 40: .55, 120: .85, 300: .93}), gate(True, True))
    assert v1_like["decision"] == "STOP" and not v1_like["checks"]["question_recall"]
    assert healthy["decision"] == "GO"


@pytest.fixture(scope="module")
def delegation_v2(built_v2, tmp_path_factory):
    d = tmp_path_factory.mktemp("p4v2")
    tiny = yaml.safe_load(open(ROOT / "configs/tiny_cpu.yaml"))
    tiny["model"]["max_seq_len"] = 160               # three evidence records make CALL examples longer
    yaml.safe_dump(tiny, open(d / "tiny.yaml", "w"))
    run("src/train/pretrain.py", "--config", str(d / "tiny.yaml"), "--data_dir", str(built_v2 / "pretrain_preflight"),
        "--output_dir", str(d / "base"), "--max_steps", "10", "--device", "cpu")
    run("scripts/label_knowledge.py", "--base", str(d / "base"), "--data", str(built_v2), "--out", str(d / "k.parquet"),
        "--device", "cpu")
    k = pd.read_parquet(d / "k.parquet")
    assert set(k.label_probe) == {"qa"} and {"knows_qa", "knows_cloze"} <= set(k.columns)
    k["knows"] = (k.exposure_count == 12).astype(int)   # force both labels for the tiny model
    k.to_parquet(d / "k.parquet")
    cfg = yaml.safe_load(open(ROOT / "configs/delegation.yaml"))
    cfg["quick"].update({"max_train_examples": 120, "epochs": 1, "dev_eval_prompts": 40})
    cfg["finetune"]["eval_every_steps"] = 0
    yaml.safe_dump(cfg, open(d / "cfg.yaml", "w"))
    run("scripts/train_delegation.py", "--config", str(d / "cfg.yaml"), "--base", str(d / "base"), "--data", str(built_v2),
        "--knowledge", str(d / "k.parquet"), "--out", str(d / "deleg"), "--format", "question", "--quick", "--device", "cpu")
    return d


def test_copy_example_teaches_reading_not_routing(built_v2):
    from tokenizers import Tokenizer
    from scripts.train_delegation import copy_example
    tok = Tokenizer.from_file(str(built_v2 / "tokenizer" / "tokenizer.json"))
    prefix = tok.encode(formats.delegation_prompt("Where was Kael Dorn born?")).ids
    ids, lab = copy_example(tok, prefix, "Registry record for Kael Dorn: place of birth: Oslo.", "Oslo")
    assert ids[len(prefix)] == formats.CALL and ids[len(prefix) + 1] == formats.EV_OPEN
    assert [t for t in lab if t != -100] == tok.encode(" Oslo").ids + [formats.EOS]   # loss on the answer only


def test_v2_delegation_gate(delegation_v2):
    rep = json.load(open(delegation_v2 / "deleg" / "gate_report.json"))
    assert rep["format"] == "question"
    assert 0.6 < rep["record_retrieved_share"] < 0.95
    assert "call_path_accuracy_record_retrieved" in rep
    assert set(rep["checks"]) == {"both_actions_learned", "useful_ranking", "direct_path_works", "call_path_works",
                                  "decisive_policy", "tier_sensitive"}
    df = pd.read_parquet(delegation_v2 / "deleg" / "dev_predictions.parquet")
    assert "record_retrieved" in df and set(df.benefit) <= {-1, 0, 1}
