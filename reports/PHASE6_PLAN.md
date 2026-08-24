# Phase 6 Plan — Pilot Pretraining

## Research question this phase must answer

> What training configuration should Juniper Math 1 use for serious Phase 7
> base pretraining on the available hardware?

Phase 5 proved the training *pipeline* works end to end (data → tokenize →
batch → forward → backward → optimize → validate → checkpoint → resume →
generate → evaluate) on a tiny, deliberately inexpensive smoke subset.
Phase 6 must establish whether the model learns *stably* from a meaningful,
representative slice of the frozen corpus, and produce an evidence-backed
recommendation for Phase 7's configuration. It is not attempting to
produce a capable model — that is explicitly out of scope (Sec. 41).

## Starting state

- Baseline: `phase-5-smoke` (approved, `reports/PHASE5_FINAL_APPROVAL.md`).
- `config/project.yaml`: `current_phase: 5`, `next_phase: {number: 6, name:
  "Pilot Pretraining", status: "AUTHORIZED — NOT STARTED"}`.
- Frozen and unchanged by this phase: architecture (`config/architecture.yaml`,
  5,004,032 params), tokenizer (`juniper-math-tokenizer-v1`), dataset
  (`juniper-math-dataset-v1`), tool protocol (`juniper-tool-protocol-v1`
  v1.0.0), the four frozen v2 evaluation suites.

## What Phase 6 adds (new code, no changes to frozen artifacts)

| File | Purpose |
|---|---|
| `config/training_phase6_pilot.yaml` | Dedicated pilot config — does not touch `config/training.yaml` |
| `src/juniper_math/pilot_training_config.py` | Loader/validator for the pilot config; reuses `training_config`'s `DataConfig`/`OptimizerConfig`/`SchedulerConfig`/`ScheduleConfig`/`ResumeTestConfig` dataclasses directly rather than redefining them |
| `src/juniper_math/pilot_data.py` | Category-stratified deterministic pilot-subset selection + packing |
| `src/juniper_math/pilot_eval.py` | Model-scoring evaluator for the three frozen v2 suites Phase 5 left unscored (math/calibration/adversarial) |
| `src/juniper_math/pilot_pipeline.py` | Orchestration: pilot train/resume-test/evaluate/infer, milestone evaluation |
| `src/juniper_math/trainer.py` (extended, not rewritten) | Loop's `TrainingConfig`/`TokenizedSmokeDataset` type hints broadened to structural `Protocol`/`Dataset[Any]` so Phase 5's and Phase 6's configs/datasets share one training loop |
| `src/juniper_math/dataset/shard.py` (additive) | Exports `BEHAVIOR_TAG` and a new `expected_completion()` helper — the same ground-truth mapping `render_training_text` already used, now reusable by the eval scorer instead of a second copy |
| `experiments/phase6-pilot/`, `checkpoints/phase6-pilot/`, `data/processed/phase6-pilot/` | Phase 6's own namespace, mirroring `docs/EXPERIMENT_NAMING.md` |

Phase 5's `config/training.yaml`, `training_config.py`, `smoke_data.py`,
and `train pilot-run`-adjacent `train run`/`evaluate`/`infer` commands are
unmodified and re-verified working (`tests/test_train_cli.py`,
`tests/test_pilot_training_config.py::test_pilot_does_not_mutate_phase5_smoke_config`).

## Pilot dataset design

Extends Phase 5's own reviewed, approved selection primitive
(`compute_stride_selection`, `juniper_math.smoke_data`) rather than
inventing a new one: the same fixed-stride rule is applied once **per
category** instead of once per split, so all 24 frozen categories —
including the rare truthfulness/error ones (`tool_error`,
`missing_information`, `incorrect_tool_call`, `undefined_operation`,
`ambiguity`, `unsupported_capability`) — are guaranteed at least
`min_category_examples` regardless of proportional rounding. See
`src/juniper_math/pilot_data.py`'s module docstring for the exact
two-pass algorithm and `reports/PHASE6_RESULTS.md` for the measured
category/token counts.

Packing (`pack_sequences`, simple deterministic first-fit, single pass, no
example ever split across two packed sequences) exists because the frozen
corpus's examples are short (median 27 tokens, p99 194 —
`data/processed/juniper-math-dataset-v1/stats.json`) relative to the
1024-token architectural context; padding each one individually the way
Phase 5's smoke pipeline does would waste the large majority of every
pilot training step's compute. Validation deliberately does **not** reuse
packing — see `pilot_pipeline.py`'s module docstring for why per-category
validation loss requires the unpacked path.

## Token budget and sequence length

- Target: 5,000,000 loss-bearing training tokens — the midpoint of the
  authorized 3–10M envelope (`pilot_training_config.py` hard-rejects
  configs outside it).
- Sequence length: 1024 (the full frozen architectural context), justified
  by `scripts/benchmark_phase1.py` on the actual RTX 2060 (already existed
  from Phase 1, reused rather than rewritten): 757.6 MiB peak VRAM at
  seq_len 1024/batch 4 vs. a 6,144 MiB budget, with packing keeping it
  data-efficient rather than padding-dominated. See
  `reports/PHASE6_RESULTS.md` §Sequence length for the full 128/512/1024
  comparison.

## Controlled experiments performed before the canonical run

A short (60-optimizer-step) learning-rate screen — the one variable Sec.
14 asks to check before accepting the Phase 5 baseline unchanged — found
6.0e-4 (2x the Phase 5 baseline of 3.0e-4) converges clearly faster with
no non-finite loss and a bounded pre-clip gradient norm. Adopted as the
pilot's peak learning rate; see `config/training_phase6_pilot.yaml`'s
`optimizer` section comment and `reports/PHASE6_RESULTS.md` §Controlled
experiments for the full screen (three learning rates, plus a follow-up
point at 1e-3 that was deliberately not adopted).

## What Phase 6 explicitly does not do

- Does not modify the frozen architecture, tokenizer, dataset, or tool
  protocol.
- Does not train on validation or test splits, or on the eval-suite-seed-
  isolated frozen evaluation cases.
- Does not claim capability — the pilot model is not expected to solve
  math reliably; see Sec. 41 and `reports/PHASE6_RESULTS.md`'s
  interpretation notes.
- Does not authorize or begin Phase 7.
- Does not create the final `phase-6-pilot` approval tag or mark
  `terra_independent_review`/`terra_final_approval` — that is GPT-5.6
  Terra's independent-review boundary (Sec. 3, 38-39).
