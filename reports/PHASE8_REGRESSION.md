# Phase 8 Base-Capability Regression Analysis (Sec. 22)

This is a mandatory gate, not an optional check: Phase 8 must explicitly
determine whether SFT damaged the Base's general capability, using
identical evaluation conditions on the Base and every candidate. This
report documents a real regression this session found, how it was found,
and the corrective action taken.

## What was measured

Two distinct "validation loss" metrics exist in this repository and they
are **not interchangeable**:

1. **SFT-masked validation loss** (`sft_pipeline.compute_validation_metrics`,
   what every Phase 8 milestone reports): teacher-forced loss over the
   `MaskedSftDataset`, where most positions (prompt, `<tool_result>`) are
   `label=-100` and excluded from the loss. This measures "does the model
   predict the assistant's supervised tokens correctly," not general
   language-modeling ability.
2. **Unmasked full-corpus validation loss** (what Phase 5-7 report as
   "validation loss," e.g. the Base's documented 0.600296): teacher-forced
   loss over the frozen `juniper-math-dataset-v1` validation split rendered
   and tokenized the ordinary way, with **every** token loss-bearing. This
   is the actual Sec. 22 "general validation loss" Base-regression metric.

**The original Sec. 19 preflight (`reports/PHASE8_PREFLIGHT.md`) only
measured metric (1) and the tool-interaction metrics — it did not measure
metric (2).** That is a real gap in this session's own methodology, found
only after the first full training run completed and its milestone
evaluations (which also only report metric (1)) looked encouraging.

## Finding: LR=8e-4 causes severe catastrophic forgetting; LR=2e-4 does not

Computed post hoc, identically for every checkpoint (2,000-example
fixed-stride sample of the frozen validation split, `micro_batch_size=16`,
same tokenizer/architecture, `smoke_data.TokenizedSmokeDataset` — the exact
methodology Phase 5-7 use for this metric):

| Checkpoint | Steps | LR | Unmasked full-corpus validation loss |
| --- | --- | --- | --- |
| Phase 7 Base (`step_007483_final.pt`) | — | — | **0.606183** (documented Base value: 0.600296, over the full split rather than a 2,000-example sample — consistent) |
| Preflight Candidate A | 300 | 8e-4 | 1.344935 (+122%) |
| Preflight Candidate B | 300 | 2e-4 | 0.628191 (+3.6%) |
| **Rejected v1 full run**, milestone step 900/20% | 900 | 8e-4 | 3.664990 (+505%) |
| **Rejected v1 full run**, milestone step 1800/40% | 1800 | 8e-4 | 3.998799 (+560%) |
| **Rejected v1 full run**, milestone step 2700/60% | 2700 | 8e-4 | 3.975701 (+556%) |
| **Rejected v1 full run**, milestone step 3600/80% | 3600 | 8e-4 | 3.838812 (+533%) |
| **Rejected v1 full run**, final step 4500/100% | 4500 | 8e-4 | 3.617611 (+497%) |

Already at 300 steps, 8e-4 more than doubles the unmasked validation loss;
by 900 steps it is roughly 6x the Base value, and it never recovers to
anywhere near the Base's own value even after the full 4,500-step run (the
partial late-training decline from 4.0 to 3.6 nats is real but nowhere
near sufficient — it is still >5x the Base's loss). At the same step count,
2e-4 leaves this metric essentially unchanged (+3.6% at 300 steps).

## Why this didn't show up as a tool-metric red flag

The Sec. 19 preflight and the milestone tool-interaction metrics *did*
show real, apparently-positive signal at 8e-4 (valid-call rate, tool-name
match, routing accuracy, and especially fabricated-tool-result elimination
all improved). The frozen v2 suite accuracies (math/adversarial/
calibration) also did not collapse to near-zero — they stayed within a
similarly noisy low range across milestones. This is consistent with a
narrow interpretation: at 8e-4, the model rapidly specializes toward the
SFT distribution's short, structured completions (tool-call JSON,
`<final>`/behavior tags) and gets measurably better at *that* narrow task,
while its ability to continue arbitrary longer-form corpus text the way
the Base could is severely damaged. The frozen v2 suites and the
tool-interaction suite both score short, structured completions — they
were not sensitive enough on their own to catch general-capability
collapse. **This is exactly why Sec. 22 exists as its own explicit,
separate gate rather than being inferred from downstream task accuracy.**

## Corrective action taken

Per Sec. 34 ("if Phase 8 genuinely fails to produce an acceptable
checkpoint, preserve the evidence, identify the failure, correct
reasonable Phase 8 implementation/data/training problems, and rerun
bounded experiments as necessary" — not "change the architecture"):

1. The rejected v1 full run's checkpoints/logs/manifest were preserved,
   not deleted: `checkpoints/phase8-sft-v1-lr8e-4-rejected/`,
   `experiments/phase8-sft-v1-lr8e-4-rejected/`,
   `data/processed/phase8-sft-v1-lr8e-4-rejected/`.
2. `config/training_phase8_sft.yaml` was revised: `run_id` ->
   `phase8-sft-v2`, `learning_rate` -> `2.0e-4` (Candidate B's rate from
   the original preflight — the one that did not show this regression at
   300 steps).
3. The full 4,500-step run was repeated from the identical verified Base,
   identical SFT dataset selection (only the learning rate differs),
   identical seed.
4. This report's methodology (the unmasked full-corpus validation loss
   check) is now run on the winning candidate's milestones too — see the
   comparison in `reports/PHASE8_RESULTS.md`, so the same blind spot is not
   repeated for the corrected run's own checkpoint-selection decision.

## Corrected run (v2, LR=2e-4): regression is modest but non-zero

Same methodology, applied to every milestone of the corrected full run:

| Checkpoint | Steps | Unmasked full-corpus validation loss | vs. Base |
| --- | --- | --- | --- |
| Phase 7 Base | — | 0.606183 | — |
| v2 step 900 | 900 | 0.792412 | +30.7% |
| v2 step 1800 | 1800 | 0.743898 | +22.7% |
| v2 step 2700 | 2700 | 0.743720 | +22.7% |
| v2 step 3600 | 3600 | 0.688130 | +13.5% |
| v2 step 4500 (final) | 4500 | 0.681840 | +12.5% |

This is a real, non-negligible regression against the Sec. 8 pre-committed
tolerance of ≤0.05 absolute nats (every candidate exceeds it, by 0.076-0.186
nats) — but two orders of magnitude smaller than the rejected 8e-4 run's
regression, and it *shrinks* monotonically over the run rather than growing.
See `reports/PHASE8_RESULTS.md` for the full checkpoint-selection table
this feeds, including the honest finding that a larger (n=200) evaluation
sample shows the tool-metric gains at this LR are smaller and more mixed
than the noisier small-sample (n=40-100) milestone/preflight numbers
suggested — some frozen/tool metrics (`tool_use_format.valid_rate`,
`argument_execution_successful`) are measurably *worse* than the Base at
every corrected-run candidate.

## Lesson for future phases

Sec. 22's Base-regression check must be run as an integral part of *every*
milestone evaluation during training, not as a post-hoc audit after a run
already looks encouraging on narrower metrics. `sft_pipeline.run_milestone`
was not updated to add this check retroactively to the rejected v1 run (its
checkpoints are frozen historical evidence, per the same "don't rewrite
history" principle Phase 7's own invalidated-candidate handling used), but
this is recorded as a known process gap for Terra's review and for any
future phase that fine-tunes further.
