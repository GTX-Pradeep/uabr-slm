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

### Attempt 2: recitation after `<ANSWER>` (gate NOT passed)

**Run:** same notebook, `QUICK = False`. 31 Oct 2026. 30 tests passed. Same knowledge labels as attempt 1.

Format: `<USER> question<ASSISTANT>` → decision → `<ANSWER> Kael Dorn was born in the city of| Oslo` (recitation
scaffold fed after the action token). Dev every 200 updates; best routing AUROC 0.645 at update 1,000 (2,000-prompt
subset). Best checkpoint on all 7,200 dev prompts:

| Measure | Attempt 1 | Attempt 2 |
|---|---|---|
| Routing AUROC (95% CI) | 0.598 (0.574–0.620) | 0.621 (0.601–0.640) |
| Mutual information | 0.094 | **0.235** |
| Near-coin-flip prompts | 37.8% | 7.0% |
| Direct-answer accuracy, facts labelled known / unknown | 0.271 / 0.027 | 0.270 / 0.038 |
| CALL rate by tier (0 / 5 / 20 / 80) | 0.54 / 0.57 / 0.53 / 0.48 | 0.88 / 0.90 / 0.87 / 0.86 |
| Accuracy at 10% call budget | 0.172 | 0.182 |
| Failed checks | decisive_policy, tier_sensitive | both_actions_learned, tier_sensitive |

**Diagnosis.** The recitation did not restore recall inside the chat context. Before any training, the base model with
the scaffold scored 0.053 direct accuracy on dev, against ~0.19 for the same sentence given on its own (the knowledge
probe). The chat tokens `<USER>`, `<ASSISTANT>` and `<ANSWER>` never occur in pretraining, and placing them in front of
the recitation disrupts retrieval; delegation training on ~3.4k known facts did not repair it for unseen entities
(direct accuracy on 80× dev facts 0.186). The router still cannot tell known from unknown facts in that context: the
CALL rate is flat across tiers.

**Change for attempt 3** (made before seeing any attempt-3 result; gate thresholds unchanged): move the decision to the
position where the answer word would start, after the recitation, so the action is chosen exactly where the model's
knowledge of the next word lives. Two formats are trained on the same labels, data and settings:

- `recite`: `<BOS><USER> question<ASSISTANT> Kael Dorn was born in the city of` → `<ANSWER> Oslo` / `<CALL><EVIDENCE> … </EVIDENCE> Oslo`
- `cloze`: `<BOS>Kael Dorn was born in the city of` → same two actions (exactly the knowledge-probe prefix)

**Selection rule (fixed now):** keep `recite` if its gate passes, because it keeps the chat context that instruction
tuning acts on; otherwise keep `cloze` if its gate passes; otherwise keep the format with the higher full-dev routing
AUROC and report Phase 4 as not passing. The choice is written to `delegation_choice.json`.

### Attempt 3: decision at the answer position, `recite` and `cloze` (gate NOT passed)

**Run:** same notebook, `QUICK = False`, 31 Oct 2026. 30 tests passed. Same labels and settings; dev every 200 updates.

| Measure | `recite` | `cloze` |
|---|---|---|
| Best dev-subset AUROC (update) | 0.628 (1,200) | 0.609 (400) |
| Routing AUROC, full dev (95% CI) | 0.634 (0.612–0.657) | 0.634 (0.611–0.656) |
| Mutual information | 0.202 | 0.145 |
| Default call rate | 0.955 | 0.760 |
| Direct-answer accuracy, facts labelled known / unknown | 0.248 / 0.033 | 0.306 / 0.037 |
| Direct-answer accuracy on 80× facts | — | 0.236 |
| CALL rate by tier (0 / 5 / 20 / 80) | 0.96 / 0.96 / 0.94 / 0.96 | 0.77 / 0.78 / 0.77 / 0.71 |
| Accuracy at 10% call budget | 0.173 | 0.187 |
| Failed checks | both_actions_learned, tier_sensitive | both_actions_learned, tier_sensitive |

**Preregistered choice:** neither passes, so `cloze` (higher full-dev AUROC; tie at three decimals broken by the rule's
`max`) is kept as `delegation_v1` and Phase 4 is reported as **not passing**.

**Diagnosis across all three attempts.** Routing AUROC stayed at 0.60–0.63 in every format, and the CALL rate never
separated 80× facts from never-shown facts by more than 0.07. The cleanest evidence is the `cloze` format: its prefix is
*exactly* the knowledge-probe prompt, on which `base_v1` recalls 75% of 80× facts, yet before any delegation training
its direct-answer accuracy with `<ANSWER>` inserted before the answer was **0.000**, and after training it reached only
0.236 on 80× facts. So any special token that never occurred in pretraining (`<USER>`, `<ASSISTANT>`, `<ANSWER>`, `<CALL>`)
breaks retrieval at the position where it appears, and fine-tuning on ~3.4k known facts memorises the training
entities (loss → 0.0002) without restoring recall for unseen ones. With recall broken, the model has no internal signal
of what it knows, and no decision format can fix that from inside Phase 4.

**Consequence.** The fix has to be upstream, in pretraining: the delegation tokens and the question format must be seen
during pretraining, so they carry trained embeddings and question-format recall exists before Phase 4 starts. This is
version 2 of the data and base model (the team's plan): question-answer examples in pretraining, fewer facts with more
repetition, a fallible CALL path, and a 1/10-scale preflight gate before the full pretraining run.
