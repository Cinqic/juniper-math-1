# Phase 8 Completion Report

## Starting state

- Starting Phase 7 tag: `phase-7-pretraining`
- Starting source commit: `79c14afb291ac04123db16e90d64828ec62d959f`
- Approved Phase 7 Base checkpoint: `checkpoints/phase7-full-v2/step_007483_final.pt`
  (downloaded fresh from the `phase-7-pretraining` GitHub release and
  re-hashed this session — matches exactly)
- Base SHA-256: `2e8098ab3a5db3c59a82fad19af2050154637fbe0628f3f6b0ca932d6cb6ea60`

## Phase 8 SFT dataset

- Identity: `juniper-math-sft-v1`
- sft_identity: `1e55652407d3624a7e7c4d9d849ac6284fba4ce868a18cd59a3bf556a1a2d1b7`
- 24,000 train examples (1,000/category, all 24 categories), 3,437
  validation examples (150/category target, drawn from the frozen
  validation split only)
- Full statistics: `reports/PHASE8_DATASET.md`

## Selected configuration

- `config/training_phase8_sft.yaml`, SHA-256
  `f0657af69023f939acd71a371da0c75bad1cb28076f9cb4bc58d75f04ab9e71d`
- `run_id: phase8-sft-v2`, learning rate 2.0e-4 (corrected after a
  regression finding — see below), 4,500 steps, effective batch size 16

## Preflight

Three bounded (300-step) candidates compared learning rate (8e-4 vs. 2e-4)
and a category-weight mixture ablation. Full detail:
`reports/PHASE8_PREFLIGHT.md` (see its addendum — the preflight's own
methodology gap is documented there).

## Full training budget

- **Rejected run v1** (`phase8-sft-v1`, LR=8e-4): 4,500 steps, 719.6s
  wall-clock, 494.9 MiB peak CUDA memory. Rejected after discovering
  severe catastrophic forgetting (unmasked full-corpus validation loss
  0.606 → 3.6-4.0, a >5x increase). Preserved as evidence.
- **Corrected run v2** (`phase8-sft-v2`, LR=2.0e-4): 4,500 steps, 1120.6s
  wall-clock, 494.9 MiB peak CUDA memory. This is the run the selected
  candidate comes from.

## Checkpoints compared

`step_002700.pt`, `step_003600.pt`, `step_004500_final.pt` from the
corrected run, each evaluated against the Base under identical conditions
(n=200 sample of `evals/phase8_instruction_v1.json`; complete/near-complete
samples of all four frozen v2 suites). Full table: `reports/PHASE8_RESULTS.md`.

## Selected checkpoint

- Path: `checkpoints/phase8-sft/step_002700.pt`
- SHA-256: `41742e554acb6619df954b7425cebe44ed11ee1edceefb4905ae6025287d0361`
- Selected on a composite of tool-routing/format metrics (best or
  near-best on `correct_routing`, `tool_name_correct`,
  `unnecessary_tool_call`, `fabricated_result_attempted`), not because it
  is the final training step.

## Base-vs-SFT evaluation table

See `reports/PHASE8_RESULTS.md` for the full table. Summary: modest,
partially-noise-level improvements in tool routing and unnecessary-call
avoidance; a measurable regression in the established
`tool_use_format.valid_rate` metric and in `argument_execution_successful`;
zero end-to-end task completion at any checkpoint including the Base; a
real, non-negligible (though much-reduced-from-the-rejected-run) general
capability regression (+22.7% unmasked validation loss at the selected
checkpoint) that exceeds the plan's own pre-committed tolerance.

## Tool metrics (Sec. 23), selected checkpoint vs. Base, n=200

| Metric | Base | Selected (step 2700) |
| --- | --- | --- |
| `correct_routing` | 0.735 | 0.765 |
| `emitted_tool_call` | 0.430 | 0.400 |
| `call_parsed_valid` | 0.280 | 0.275 |
| `tool_name_correct` | 0.690 (29/42) | 0.750 (30/40) |
| `argument_execution_successful` | 0.732 (41/56) | 0.509 (28/55) |
| `end_to_end_success_on_tool_required` | 0.0 (0/53) | 0.0 (0/53) |
| `unnecessary_tool_call` | 0.293 (43/147) | 0.252 (37/147) |
| `missing_required_call` | 0.189 (10/53) | 0.189 (10/53) |
| `fabricated_result_attempted` | 0.425 (85/200) | 0.380 (76/200) |

## Direct-vs-tool routing

`correct_routing` (did the model's decision to call or not call a tool
match whether one was actually required): Base 0.735, selected candidate
0.765 — a modest improvement, within roughly 1 standard error at this
sample size.

## Regression findings

Full detail in `reports/PHASE8_REGRESSION.md`. Summary: an initial full
run at Phase 7's own LR (8e-4) caused severe catastrophic forgetting,
found via a Sec. 22 check the original preflight omitted; corrected by
rerunning at 2e-4, which reduces (but does not eliminate) the regression.

## Tests run

- `tests/test_sft_rendering.py` (15 tests): loss-masking correctness.
- `tests/test_sft_data.py` (8 tests): SFT selection determinism, targets,
  manifest writing.
- `tests/test_tool_interaction.py` (7 tests): end-to-end tool loop,
  including the fabricated-result trust boundary and a context-duplication
  regression test.
- `tests/test_sft_eval.py` (5 tests): metric numerator/denominator
  correctness.
- `tests/test_sft_training_config.py` (9 tests): config loading/validation,
  including parent-checkpoint hash verification.
- Full project regression suite: **704 passed, 0 failed** (660 prior +
  44 new Phase 8 tests), 2 documented CUDA-nondeterminism warnings
  (unchanged from Phase 7's own baseline).

## Environment

Linux, Python (repository `.venv`), PyTorch + CUDA, RTX 2060 6GB, 16GB
system RAM. Peak CUDA memory measured at 494.9 MiB for both full SFT runs
— comfortably inside budget.

## Recovery procedure

See `reports/PHASE8_TERRA_HANDOFF.md` for the exact commands. Bounded
recovery verification (not a full second clean clone, given session time
constraints — documented as a limitation, not concealed) is described
there.

## Remote artifact location

- Candidate commit: recorded in `reports/PHASE8_TERRA_HANDOFF.md` after
  push (this report is written before the final commit hash is known).
- Candidate tag: `phase-8-math-sft-candidate`
- Selected checkpoint preservation: GitHub release attached to
  `phase-8-math-sft-candidate`, SHA-256
  `41742e554acb6619df954b7425cebe44ed11ee1edceefb4905ae6025287d0361`.

## Known limitations / unresolved findings

1. General-capability regression (+22.7% unmasked validation loss) exceeds
   the plan's own pre-committed tolerance at every corrected-run candidate.
2. Several tool-format metrics are worse on every SFT candidate than on
   the Base at the larger, more reliable evaluation sample.
3. No checkpoint (including the Base) demonstrates end-to-end task
   completion or correct direct final answers on the held-out suite.
4. Only one corrective learning-rate rerun was performed; a fuller sweep
   (intermediate LR, different warmup/step count) was not attempted given
   the bounded-preflight discipline this project follows.
5. A latent bug was found in the frozen (pre-Phase-8) `evals/
   phase4_calibration_v2.json` (duplicate example ids across two
   categories) — documented, not fixed, per the frozen-artifact boundary.

## Statement

GPT-5.6 Terra has not yet independently reviewed or approved Phase 8.
Phase 9 has not started and remains unauthorized until Phase 8 is
approved (with or without remediation).
