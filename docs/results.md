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

**Run:** `notebooks/phase2_build_data.ipynb` on Kaggle CPU, `QUICK = False` (Kaggle notebook `uabr-p2-data`, version 2). 30 Oct 2026.
**Gate: PASSED**

| Item | Result |
|---|---|
| Answer vocabulary source | Wikidata (live SPARQL, not the fallback lists) |
| Facts | 12,000 pseudonymous entities × 3 relations = 36,000 facts; ~50% exposed (5 / 20 / 80 statements), ~50% withheld |
| Fact statements | 624,565 statements, ~9.4M tokens (< 1% of the corpus) |
| Pretraining tokens | 1,005,000,040 (reached target) |
| Instruction SFT data | Dolly-15k + Alpaca-cleaned, 6.0M tokens, ~34.8k examples; 500 held out for validation |
| Tokenizer | Byte-level BPE, 12,000 tokens, special tokens at IDs 0–9; ~3.6 characters per token on Wikipedia |
| Ledger audits | 6 / 6 PASS (duplicate facts, prompt overlap, class balance, evidence, aliases, leakage) |

## Phase 3: base_v1

**Run:** `notebooks/phase3_pretrain.ipynb` on Kaggle GPU T4, `QUICK = False` (Kaggle notebook `uabr-p3-pretrain`). 30–31 Oct 2026, one session, 5.2 h.
**Gate: PASSED**

| Item | Result |
|---|---|
| Steps / tokens | 15,260 / 1,000,079,360 (one pass over data_v1) |
| Throughput | ~54,000 tokens/s with `--compile`, no NaN, peak memory 5.2 GB |
| Validation loss | 4.458 (step 500) → 3.349 (2k) → 3.080 (5k) → 2.896 (10k) → **2.814** (15k), perplexity 16.7 |
| Final training loss | 2.738 (train and validation curves stay together: no overfitting) |
| Checkpoint | `base_v1.pt`, 159 MB |

**Fact probe** (cloze prompt, greedy exact match, 300 facts per tier):

| Exposure | 0 (withheld) | 5× | 20× | 80× |
|---|---|---|---|---|
| Exact match | 0.050 | 0.073 | 0.183 | **0.750** |
| Mean gold log-prob | −5.37 | −4.73 | −3.30 | −1.24 |

By relation at 80×: occupation 0.95, citizen_of 0.89, language 0.88, born_in 0.57, birth_year 0.44. Withheld facts are at
the guessing floor (~5%), 80× facts are mostly stored, and the 20× tier is partly known, so real examples fall on both
sides of the answer-or-call boundary, as the design intended.

## Phase 4: delegation_v1

### Attempt 1: bare answer format (gate NOT passed)

**Run:** `notebooks/phase4_delegation.ipynb`, `QUICK = False` (Kaggle notebook `uabr-p4-delegation`). 31 Oct 2026. 29 tests passed.

Knowledge labels from `base_v1` (cloze probe), share labelled ANSWER:

| Split | 0 | 5× | 20× | 80× |
|---|---|---|---|---|
| delegation_train | 0.036 | 0.070 | 0.216 | 0.747 |
| dev | 0.040 | 0.046 | 0.218 | 0.759 |
| replay_construction | 0.034 | 0.077 | 0.225 | 0.728 |

Training: 3 epochs × 40,776 examples (50% ANSWER), 1,914 updates. Dev routing AUROC by epoch: 0.586 → 0.584 → 0.491;
training loss reached 0.0003 (the model memorised the training entities). Best checkpoint (epoch 1) on all 7,200 dev prompts:

| Measure | Value |
|---|---|
| Routing AUROC | 0.598 (95% CI 0.574–0.620) |
| Direct-answer accuracy, facts labelled known / unknown | 0.271 / 0.027 |
| CALL-path accuracy | 1.000 |
| Accuracy at 10% call budget | 0.172 |
| Mutual information | 0.094 bits; 37.8% of prompts within 0.1 of a coin flip |
| CALL rate by tier (0 / 5 / 20 / 80) | 0.54 / 0.57 / 0.53 / 0.48 |
| Checks | both_actions, useful_ranking, direct_path, call_path PASS; **decisive_policy, tier_sensitive FAIL** |

**Diagnosis.** The base model stores facts in statement form but cannot retrieve them as answers to questions: direct
accuracy on 80× facts was 0.19 in question-answer format against 0.75 in the cloze probe. So about 73% of the dev facts
labelled ANSWER were facts the ANSWER path then got wrong, and the router was trained on labels that disagree with
the outcome it is scored on. The quick run (47 updates) also showed the value of the anti-cheating checks: three of
the four original checks passed on a model that was 50/50 on every prompt (mutual information 0.001).

**Change for attempt 2** (made before seeing any attempt-2 result; gate thresholds unchanged):
1. Recitation answer path: after `<ANSWER>` the system writes the cloze sentence opening (for example "Kael Dorn was
   born in the city of") and the model completes it. It is the same text as the knowledge probe, so an ANSWER label now
   means exactly "the ANSWER path will be right". Like the evidence block on the CALL path, the scaffold is masked in
   training and fed at evaluation; the routing decision comes before it.
2. Dev check every 200 updates, keeping the best routing AUROC, to stop before memorisation.
