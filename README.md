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
| **1** | Model, trainer, tests, GPU smoke run, resume check | Kaggle / Colab T4, ~20 min | 7 tests pass, loss falls, resume continuous, tokens/s recorded | ✅ in repo |
| 2 | Data: Wikipedia corpus, controlled facts, exposure ledger, tokenizer | Kaggle CPU | 6 ledger audits pass, tokenizer frozen | ⏳ |
| 3 | Pretrain the 40M base model (1B tokens) | Kaggle T4, ~6 h | Val loss falls, facts recalled by exposure tier | ⏳ |
| 4 | Frozen evidence index + delegation training | Kaggle / Colab, < 1 h | Both actions learned, AUROC > 0.5 | ⏳ |
| 5 | No-replay interference pilot | Kaggle / Colab, ~2 h | Go / no-go note written | ⏳ |
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
scripts/    check_gpu.py · make_synthetic_data.py
tests/      test_model.py
notebooks/  phase1_setup_and_smoke.ipynb
```

Later phases add `src/data`, `src/retrieval`, `src/replay`, `src/evaluation` and one notebook per phase.

## Ground rules

1. Miniature run first, on the same code path, then the full run.
2. No gate passed, no next phase.
3. Test labels are read only by the final evaluation script.
4. Every run folder holds its resolved config, manifest, metrics and checkpoint pointer.
5. Nothing large goes into Git: data and checkpoints live in Kaggle Datasets, Google Drive or the Hugging Face Hub.

## Scope

Claims are limited to this ~40M model, controlled factual data, supervised instruction tuning and a frozen evidence
source.
