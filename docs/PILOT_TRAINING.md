# Phase 6: Pilot Pretraining

Canonical config: [`config/training_phase6_pilot.yaml`](../config/training_phase6_pilot.yaml).
This document describes the design; see
[`reports/PHASE6_PLAN.md`](../reports/PHASE6_PLAN.md) for the plan and
[`reports/PHASE6_RESULTS.md`](../reports/PHASE6_RESULTS.md) for the
measured results of the canonical run.

## Purpose

Phase 5 (`docs/TRAINING.md`) proved the training pipeline mechanics work
end to end on a tiny, deliberately inexpensive smoke subset — it made no
claim about model behavior. Phase 6 is the first phase where model
behavior is meant to matter: it trains on a deterministic,
category-stratified subset of the frozen train split large enough to
observe real learning dynamics (loss trajectory, validation behavior,
category-level differences, generation change) while remaining
meaningfully smaller than a full Phase 7 base-pretraining run. Its output
is not a capable model — it is evidence for what Phase 7's configuration
should be.

## Pilot subset selection

[`src/juniper_math/pilot_data.py`](../src/juniper_math/pilot_data.py)
extends Phase 5's own approved fixed-stride selection primitive
(`compute_stride_selection`, `juniper_math.smoke_data`) to run once per
category instead of once per split:

1. **Pass 1** (`count_categories`): a single sequential scan of the split's
   shards, counting exact per-category record and token totals.
2. **Target computation** (`compute_category_targets`): each category's
   target example count is proportional to its share of the split's
   tokens against the configured `target_train_tokens` budget, floored at
   `min(min_category_examples, available)` — so a rare category (e.g.
   `tool_error`, `missing_information`) is never proportionally rounded
   away.
3. **Pass 2** (`select_pilot_examples`): using each category's own
   `(stride, offset)` (the exact same `compute_stride_selection` Phase 5's
   review already covers), a second sequential scan takes every
   `index`-th record satisfying the per-category stride test.

The validation pilot subset is selected the same way, from the
`validation` split only, sized from `pilot_subset.validation_examples`,
and frozen for the entire run — every milestone evaluates the identical
subset (Sec. 16).

A manifest (`pilot_manifest.json`) and a full selection audit
(`pilot_selection_audit.json`) are written under
`pilot_subset.pilot_dataset_dir` (`data/processed/phase6-pilot/` by
default, not committed — reproducible from the frozen dataset + this
config + seed) recording category counts, token counts, difficulty
counts, family counts, and per-category stride/offset for audit.

## Packing

The frozen corpus's examples are short relative to the architecture's
1024-token context (median 27 tokens, p99 194 —
`data/processed/juniper-math-dataset-v1/stats.json`). Padding each example
individually out to a fixed sequence length — Phase 5's approach, fine at
smoke scale — would waste most of every pilot training step's compute.
`pack_sequences()` (in `pilot_data.py`) performs simple, deterministic,
single-pass first-fit packing: each example is independently tokenized,
BOS/EOS-wrapped, and length-clipped to fit `max_sequence_length` (never
split across two packed sequences), then examples are appended to the
current packed sequence in selection order until the next one would
overflow it. Causal attention already only lets each position see earlier
positions, so packing introduces no non-causal leakage; loss is computed
over every real (non-pad) token, including the boundary between two
packed examples — an EOS token's target is simply the next example's BOS,
standard practice for packed causal-LM corpora at this scale. See
`reports/PHASE6_RESULTS.md` §Sequence length and packing for the measured
padding-fraction result.

Validation deliberately does **not** use packing — a packed sequence can
interleave examples from different categories, which would make a
per-category validation loss meaningless. Validation instead reuses Phase
5's unpacked `TokenizedSmokeDataset` at a separate, shorter
`VALIDATION_MAX_SEQUENCE_LENGTH` (256, matching Phase 5's smoke length —
comfortably covers the p99 example length): small enough that the padding
cost is immaterial, and it keeps every category's loss cleanly
attributable via `torch.utils.data.Subset`.

## Training loop

Reuses `juniper_math.trainer` unchanged in mechanics — no second training
loop was written for Phase 6. The loop's `TrainingConfig`/
`TokenizedSmokeDataset` type hints were broadened to a structural
`TrainingConfigLike` `Protocol` and `torch.utils.data.Dataset[Any]`, so
Phase 5's `TrainingConfig`+`TokenizedSmokeDataset` and Phase 6's
`PilotTrainingConfig`+`PackedPilotDataset` both satisfy the same function
signatures. This is a pure static-typing change — the loop's runtime
behavior, gradient-accumulation semantics, finite-value checks, and
checkpoint cadence are exactly Phase 5's, now shared rather than
duplicated.

## Milestone evaluation

At each of `milestone_fractions` (default `[0.0, 0.25, 0.5, 0.75, 1.0]` of
`schedule.total_steps`), `pilot_pipeline.run_milestone` runs:

- Overall + per-category validation loss (`compute_validation_metrics`).
- All four frozen v2 evaluation suites: math, calibration, and adversarial
  via the new `juniper_math.pilot_eval.run_capability_evaluation` (Phase 5
  left these three unscored by any model-facing CLI command); tool_use via
  Phase 5's existing, unmodified `tool_format_eval.run_tool_format_evaluation`
  (tool-call *syntax* validity — a distinct question from numeric
  correctness).
- The fixed-seed qualitative generation set
  (`fixed_generation_prompts`, greedy decoding, frozen `max_new_tokens`).

`pilot_eval.run_capability_evaluation` scores a case "correct" only
against the exact same terminal-tag/value ground truth
`juniper_math.dataset.shard.render_training_text` already uses
(`expected_completion`) — never a second, hand-invented notion of
correctness. A case with no recognizable tag, or an unparseable `<final>`
value, is scored incorrect and still counted in the denominator (Sec. 19,
Sec. 30's "no false PASS on silently skipped input" rule).

## Resume verification

`train pilot-resume-test` mirrors Phase 5's `train resume-test`
methodology (Sec. 22) exactly, at pilot scale: one shared init checkpoint,
an uninterrupted Run A, and an interrupted-then-resumed Run B (fresh
model/optimizer/scheduler objects at both the interrupt and the resume
point, simulating separate processes). Equivalence is a tolerance check
(`max_loss_diff < 1e-2` and `max_param_diff < 1e-2`), not an assumption of
bitwise identity — Phase 5's bitwise-exact result on this GPU was a good
outcome for that run length, not a hardware guarantee that automatically
extends to a longer pilot-scale run (documented directly in the
comparison's own output notes).

## CLI

```bash
python -m juniper_math train pilot-run [--config PATH] [--max-steps N] [--eval-sample-size N] [--no-milestone-eval]
python -m juniper_math train pilot-resume-test [--config PATH]
python -m juniper_math pilot-evaluate --checkpoint PATH [--config PATH] [--sample-size N]
python -m juniper_math pilot-infer --checkpoint PATH --prompt TEXT [--max-new-tokens N]
```

## Scope boundary

Phase 6 does not: modify the frozen architecture, tokenizer, dataset, or
tool protocol; train on validation/test data or the eval-suite-seed-
isolated frozen evaluation cases; claim mathematical capability; or begin
Phase 7. See `reports/PHASE6_PLAN.md` and `reports/PHASE6_RESULTS.md` for
the full evidence chain and the Phase 7 recommendation.
