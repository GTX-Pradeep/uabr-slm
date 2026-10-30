# UABR-SLM

**Preserving selective delegation during instruction tuning of a scratch-trained small language model.**

AFML course project · Team Straw Hats

---

## The idea in one paragraph

A 40M-parameter language model cannot memorise every fact, so it learns a habit: for each question it either
**answers directly** (`<ANSWER>`) or **calls a frozen evidence source** (`<CALL>`). Instruction tuning (SFT) can
quietly break that habit, because SFT demonstrations almost always reward answering directly. This project asks:

1. **Does ordinary SFT damage the answer-or-call policy?** (a pilot experiment)
2. **If it does, which replay method protects it best?** We compare replaying old delegation examples chosen at
   random, by boundary margin (BPR), and by **UABR (Utility-Anchored Boundary Replay)**, which picks examples that are
   both close to the decision boundary *and* where the choice actually changes the outcome:

$$ s_i = \exp(-m_i/\tau)\cdot|b_i|, \qquad b_i = R_i(\text{call}) - R_i(\text{answer}) $$

All methods get identical batches, update counts and replay-token budgets. The primary metric is held-out
exact-match accuracy at a fixed 10% call budget.

## Pipeline

```
Wikipedia + pseudonymous facts ─► 40M base model ─► <ANSWER>/<CALL> policy ─► SFT (+ replay variants) ─► one test evaluation
        Phase 2                     Phase 3              Phase 4               Phases 5–6                  Phase 7
```

| Phase | What it does | Where it runs | Gate to pass | Status |
|---|---|---|---|---|
| **1** | Model, trainer, tests, GPU smoke run, resume check | Kaggle / Colab T4, ~20 min | 7 tests pass, loss falls, resume continuous, tokens/s recorded | ✅ passed |
| 2 | Data: Wikipedia corpus, controlled facts, exposure ledger, tokenizer | Kaggle CPU | 6 ledger audits pass, tokenizer frozen | ✅ code in repo · built `data_v1` reused |
| 3 | Pretrain the 40M base model (1B tokens) | Kaggle T4, ~6 h | Val loss falls, facts recalled by exposure tier | ✅ code in repo · trained `base_v1` reused |
| 4 | Frozen evidence index + delegation training | Kaggle / Colab, < 1 h | Both actions learned, AUROC > 0.5 | 🟡 code in repo · run next |
| 5 | No-replay interference pilot | Kaggle / Colab, ~2 h | Go / no-go note written | 🟡 code in repo · run after Phase 4 |
| 6 | Utility labels + SFT with each replay method × 2 seeds | Kaggle, spread over accounts | Equal-budget checkpoints | ⏳ |
| 7 | Test evaluation, paired bootstrap, error analysis | Colab, < 1 h | Frozen prediction tables + figures | ⏳ |

Every phase runs a miniature version on the same code path first, then the full version.

## Model

Decoder-only transformer, trained from random initialisation:

| | |
|---|---|
| Layers / width / heads | 8 / 512 / 8 |
| Feed-forward | SwiGLU, 2048 |
| Positions | RoPE (θ = 10,000) |
| Norm | RMSNorm (pre-norm) |
| Embeddings | Tied input/output, vocab 12,000 |
| Parameters | 39,707,136 |
| Precision | FP16 mixed precision + GradScaler (T4/P100 have no fast BF16) |

## Phase 1 quick start

Open `notebooks/phase1_setup_and_smoke.ipynb` on Kaggle (Accelerator: **GPU T4 x2**, Internet: **On**) or on Colab
(Runtime: **T4 GPU**) and Run All. Or by hand:

```bash
git clone https://github.com/GTX-Pradeep/uabr-slm.git && cd uabr-slm
pip install -q -r requirements.txt          # torch is preinstalled on Kaggle/Colab
python scripts/check_gpu.py
python -m pytest tests -q                   # 7 tests, ~1 min
python scripts/make_synthetic_data.py --out_dir data/processed/synthetic
python src/train/pretrain.py --config configs/smoke_40m.yaml --max_steps 300 --mirror_dir persist/smoke_pretrain
python src/train/pretrain.py --config configs/smoke_40m.yaml --resume persist/smoke_pretrain --max_steps 350
```

Synthetic data is a Markov chain with ideal loss ln 4 ≈ 1.39, so a working model drops from ~9.4 towards ~1.4.

Phase 1 results are in [`docs/results.md`](docs/results.md).

## Phases 2–3: data and pretraining

These were already built once by the team, so the outputs are reused rather than rebuilt:

- **`data_v1`** (Phase 2 output): `pretrain/train.bin` + `val.bin`, `facts/` (exposure ledger, prompts,
  evidence), `sft/` (instruction data), `tokenizer/` and the audit report.
- **`base_v1`** (Phase 3 output): the pretrained 40M checkpoint.

On Kaggle, attach them with **+ Add Input → Notebook Output Files**. Later notebooks find `data_v1` automatically
under `/kaggle/input/`.

**How the data is designed.** 12,000 invented people (for example "Varnek Holloway") each get 3 real-looking facts,
36,000 in total. Because the names are invented, Wikipedia cannot leak the answers. About half the facts appear in the
pretraining text 5, 20 or 80 times; the rest never appear and exist only in the evidence source. A 40M model learns
the 80× facts but not the 5× ones, so real examples land on both sides of the answer-or-call boundary. Facts are split
by person: delegation_train 50%, replay_construction 20%, dev 10%, test 20%.

To rebuild from scratch: run `notebooks/phase2_build_data.ipynb` (CPU, Internet on, about 3–5 h with `QUICK = False`),
then `notebooks/phase3_pretrain.ipynb` (GPU, about 6 h).

## Phase 4: answer or call

The base model learns to open every factual answer with one action token. Loss covers only the assistant tokens;
evidence text is copied by the tool, so it is masked.

```
<BOS><USER> Where was Varnek Holloway born?<ASSISTANT><ANSWER> Lyon<EOS>
<BOS><USER> What language did Orsa Kellin speak?<ASSISTANT><CALL><EVIDENCE> Orsa Kellin spoke Portuguese. </EVIDENCE> Portuguese<EOS>
```

Labels come from what `base_v1` actually knows: each non-test fact is asked as a cloze, and a correct greedy answer
means ANSWER, otherwise CALL. This is the plan's documented fallback to exposure-based labels and is reported as a
declared deviation. The routing score is `call_score = log p(<CALL>) − log p(<ANSWER>)` at the first assistant token.
The kept checkpoint is the epoch with the best dev routing AUROC, saved as `delegation_v1`.

The gate has six checks: both actions used (each ≥ 5%, recall ≥ 0.5 per label), useful ranking (AUROC 95% CI lower
bound > 0.55), the direct path works on known facts, the CALL path works (≥ 90%), and two anti-cheating checks:

- **Decisive policy:** mutual information between prompt and action ≥ 0.1 bits. This is overall action entropy minus
  mean per-prompt entropy. "Always CALL" (low overall entropy) and "50/50 for every prompt" (high per-prompt entropy)
  both give ≈ 0, while a real policy, mixed overall but confident on each prompt, gives a clearly positive value.
- **Tier sensitive:** the CALL rate for never-shown facts is at least 0.2 above the rate for 80× facts. A flat rate
  across tiers means a global habit rather than a per-fact decision.

Run `notebooks/phase4_delegation.ipynb` with `data_v1` and `base_v1` attached as inputs.

## Phase 5: interference pilot

Plain instruction tuning (Dolly + Alpaca, ~6M tokens, 1 epoch, LR 5e-5 constant, batch 32) from `delegation_v1`, with
seeds 11 and 29 and checkpoints at 25%, 50% and 100%. Every checkpoint is scored on all dev prompts, and the
preregistered rule in [`docs/pilot_decision.md`](docs/pilot_decision.md) prints **GO**, **RECALIBRATION_ONLY** or
**NO_CHANGE**. The comparison is 100% minus 0% on the same prompts, with a paired bootstrap:

- **Real damage:** routing AUROC or accuracy at a *re-picked* 10% call budget falls. Both are threshold-free, so moving
  the threshold cannot fix them.
- **Calibration only:** only accuracy at the *frozen* threshold (chosen on `delegation_v1`) falls.

All SFT branches share one runner (`scripts/run_sft.py`). Instruction batches are built once per seed
(`src/train/sft_batches.py`); replay branches in Phase 6 add their examples *inside* those batches, so every branch
makes the same number of updates on the same instruction data. The 100% checkpoints double as the Standard SFT branch.

Run `notebooks/phase5_pilot.ipynb` with `data_v1` and the Phase 4 output attached.

## Training flags for free GPUs

| Flag | Purpose |
|---|---|
| `--max_minutes N` | Stop cleanly and checkpoint after N minutes (Kaggle ≈ 690, Colab ≈ 150) |
| `--mirror_dir PATH` | Copy every checkpoint, manifest and metrics file to persistent storage |
| `--resume PATH` | Continue from a checkpoint or run directory; the LR schedule and data order continue exactly |
| `--compile` | `torch.compile`; about 1.4× faster on a T4 |

## Repository layout

```
configs/    pretrain.yaml (full 40M, 1B tokens) · smoke_40m.yaml · tiny_cpu.yaml
src/model/  model.py (transformer) · load.py (checkpoint loading)
src/train/  pretrain.py (FP16 AMP, grad accumulation, cosine LR, full-state checkpoints)
src/data/   formats.py (special tokens, sequence formats, answer matching) · vocab.py (Wikidata answer lists)
            build_facts.py (pseudonymous facts, exposure tiers, splits, prompts, evidence, ledger)
            wiki.py (Wikipedia stream) · sft_data.py (Dolly + Alpaca) · tokenizer.py (BPE 12k) · pack.py (.bin files)
src/retrieval/  evidence.py (frozen evidence lookup, hashed as the index version)
src/evaluation/ generate.py (batched greedy decoding, next-token scores) · predict.py (one prediction table per checkpoint)
                metrics.py (AUROC, AP, call-budget accuracy, calibration, risk-coverage, bootstrap)
src/train/  sft.py (shared fine-tuning loop for delegation, SFT and every replay branch)
            sft_batches.py (instruction batches shared by all SFT branches; replay mixed in evenly)
scripts/    check_gpu.py · make_synthetic_data.py · audit_ledger.py · probe_facts.py · sample_text.py
            label_knowledge.py · train_delegation.py · run_sft.py · pilot_eval.py
configs/sft_methods/  standard.yaml (the SFT setup every Phase 5-6 branch shares)
tests/      test_model.py · test_data.py · test_phase4.py · test_phase5.py · conftest.py
notebooks/  phase1_setup_and_smoke.ipynb · phase2_build_data.ipynb · phase3_pretrain.ipynb · phase4_delegation.ipynb · phase5_pilot.ipynb
docs/       results.md (gate results per phase) · pilot_decision.md (preregistered pilot rule)
```

Later phases add `src/replay` and one notebook per phase.

## Ground rules

1. Miniature run first, on the same code path, then the full run.
2. No gate passed, no next phase.
3. Test labels are read only by the final evaluation script.
4. Every run folder holds its resolved config, manifest, metrics and checkpoint pointer.
5. Nothing large goes into Git: data and checkpoints live in Kaggle Datasets, Google Drive or the Hugging Face Hub.

## Scope

Claims are limited to this ~40M model, controlled factual data, supervised instruction tuning and a frozen evidence
source.
