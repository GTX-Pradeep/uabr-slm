# Phase 5: interference pilot, decision note

**Status:** rule fixed before any pilot result was seen. The results section is filled in after the run and signed off
by all four members.

## Question

Does ordinary instruction tuning (SFT) with no replay damage the `<ANSWER>`/`<CALL>` policy of `delegation_v1`, in a way
that re-choosing the call threshold cannot repair?

## Setup (fixed)

| Item | Value |
|---|---|
| Start | `delegation_v1` (Phase 4) |
| SFT data | `data_v1/sft/sft_train.jsonl`: Dolly-15k + Alpaca-cleaned, ~6M tokens, 1 epoch |
| Optimiser | AdamW, LR 5e-5 constant after 30 warmup steps, batch 32 sequences, FP16 (`configs/sft_methods/standard.yaml`) |
| Seeds | 11 and 29 (the seed fixes the instruction batch order) |
| Durations | 25%, 50% and 100% of the planned updates, saved from one run per seed. With a constant LR these equal separate shorter runs |
| Evaluation | All dev prompts; `delegation_v1` is the 0% point |
| Re-use | The 100% checkpoints are the Standard SFT branch of Phase 6 |

## Measures, at 0%, 25%, 50% and 100% of SFT

- Instruction validation loss (sanity check: SFT must actually learn)
- Routing AUROC and average precision (threshold-free ranking quality)
- Accuracy at a 5 / 10 / 20% call budget, top-k routing (the threshold is re-picked at every checkpoint)
- Accuracy at the frozen thresholds chosen on `delegation_v1` for 5 / 10 / 20% (what a deployed system would see)
- Default-decision call rate, harmful-answer rate, mutual information, CALL rate for never-shown and 80× facts

## Decision rule

The comparison is the 100% checkpoint minus `delegation_v1` on the same dev prompts, with a paired bootstrap
(1,000 resamples, 95% CI). A drop is significant when the upper CI bound is below 0.

- **GO:** in **both** seeds, routing AUROC **or** accuracy at the re-picked 10% budget drops significantly. Both
  measures are threshold-free, so recalibration cannot repair the damage. → Phase 6 is the main claim.
- **RECALIBRATION ONLY:** not GO, but in both seeds there is damage, and at least the frozen-threshold accuracy drops.
  → Report threshold recalibration as the simpler fix; Phase 6 results become secondary.
- **NO CHANGE:** otherwise. Check that SFT learned (instruction val loss fell) and that `delegation_v1` is not collapsed.
  Optionally run the preregistered 200% duration once. If still flat, report the robustness boundary, which is still a
  valid course result.

Phase 6 may start before this note is final. A non-GO result changes what the write-up claims, not which runs exist.

## Results

*(to be filled from `runs/pilot/pilot_report.json` and `pilot_table.csv`)*

| Seed | Δ AUROC [95% CI] | Δ acc@10% re-picked [95% CI] | Δ acc@10% frozen [95% CI] | Instruction loss fell |
|---|---|---|---|---|
| 11 | | | | |
| 29 | | | | |

**Decision:**

**Signed off:** ☐ ☐ ☐ ☐
