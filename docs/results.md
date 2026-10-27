# Results log

One entry per phase gate. Numbers are copied from the notebook output of the run named in each entry.

## Phase 1: model code, GPU smoke run, resume

**Run:** `notebooks/phase1_setup_and_smoke.ipynb` on Kaggle, GPU T4 (1 of 2 used), FP16 AMP. 27 Oct 2026.
**Gate: PASSED**

| Check | Result |
|---|---|
| Parameters | 39,707,136 |
| Training loss, step 20 → 300 (synthetic data, ideal ln 4 ≈ 1.39) | 7.82 → 1.51 |
| Validation loss at steps 100 / 200 / 300 | 1.755 / 1.569 / 1.532 |
| Non-finite loss | none |
| Resume continuity (restart from persistent copy) | step 300: 1.515 → step 310: 1.488 |
| Throughput, eager | 38,376 tokens/s |
| Throughput, `--compile` | 54,468 tokens/s (1.42×), identical loss at step 200 (1.5663) |
| Peak GPU memory | 8.02 GB |
| 1B-token pretraining estimate | 5.9 h with `--compile` (one Kaggle session) |

Decision: use `--compile` for every pretraining and resume command.

## Phase 2: data_v1

Reused from an existing team build of `notebooks/phase2_build_data.ipynb` (Kaggle notebook output `data_v1`).
Manifest and audit numbers to be recorded here.

## Phase 3: base_v1

Reused from the existing team pretraining run of `notebooks/phase3_pretrain.ipynb`.
Validation loss, fact probe by exposure tier and checkpoint details to be recorded here.
