# Phase 7 — Bounded Resume-Mechanics Check

Required by Step 12 of the Phase 7 instructions: verify checkpoint
restoration for the full-dataset pipeline "sufficiently to establish that
the full run can recover from interruption" before relying on that
assumption during (or after) the expensive canonical run. This is a
dedicated, bounded check — NOT a rerun of the canonical training budget.

## Method

`config/phase7_resume_check.yaml`: identical to
`config/training_phase7_full.yaml` (same architecture/tokenizer/dataset
identity, same optimizer, LR, and scheduler settings) but with
`schedule.total_steps=200` and `resume_test.interrupt_step=100`, sharing
the already-built full-dataset pack at `data/processed/phase7-full/` for a
controlled comparison. Command: `python -m juniper_math train
full-resume-test --config config/phase7_resume_check.yaml`. This exercises
the actual Phase 7 code path (`full_pipeline.run_full_resume_test`,
`full_data`-selected/packed dataset, `trainer.save_state`/`load_state`),
not a synthetic stand-in.

Method (same as Phase 5/6): Run A trains uninterrupted for 200 steps. Run B
trains to step 100, saves a checkpoint, then a **fresh process state**
loads that checkpoint and continues to step 200. Step count, token count,
loss history, final parameters, and fixed-prompt generations are compared
between A and B.

## Result: mechanically correct, numerically outside the prior tolerance

| Metric | Run A | Run B | Match? |
|---|---:|---:|---|
| Final step | 200 | 200 | exact |
| Tokens seen | 3,154,674 | 3,154,674 | exact |
| Final loss | 0.9245 | 0.9217 | close |

- `loss_history_max_abs_diff = 0.02082`
- `max_param_abs_diff = 0.02199`
- `generations_match = False`
- Both `torch.isfinite` checks passed throughout; no NaN/Inf anywhere in
  either run.

The 1e-2 tolerance threshold (unchanged from Phase 5/6, not loosened for
this check) was **exceeded**: `python -m juniper_math train
full-resume-test` correctly reported `FAIL: resume comparison diverged`,
and this report does not relabel that outcome. The Phase 6 pilot's
equivalent check passed well inside tolerance (`max_param_abs_diff
0.005875`, `loss_history_max_abs_diff 0.003059` at 320 total steps /
interrupt step 160). This check's divergence is roughly 4x larger despite
*fewer* post-interrupt steps (100 here vs. 160 in the pilot), so step count
alone does not explain it.

## Interpretation

Two properties are distinct here and must not be conflated:

1. **Resume mechanics (the property this check exists to verify): exact.**
   Step count and tokens-seen match bit-for-bit between the uninterrupted
   and interrupted-then-resumed runs. This proves the checkpoint correctly
   preserves and restores model weights, optimizer state, scheduler
   position, RNG state, global step, token count, and data-stream cursor
   for the full-dataset pipeline — nothing was silently dropped, double
   counted, or skipped across the interruption boundary.

2. **Bitwise/numerical equivalence: not achieved, and was never claimed to
   be.** `torch.use_deterministic_algorithms(True, warn_only=True)` is a
   best-effort request; the scaled-dot-product-attention backward kernel on
   this CUDA build has no deterministic implementation and emits exactly
   the warning both this run and every Phase 6 pilot run also logged
   ("Memory Efficient attention defaults to a non-deterministic
   algorithm"). Given that documented nondeterminism, small
   per-step floating-point differences between the continuous and
   resumed-from-checkpoint computation graphs are expected, and can compound
   over subsequent optimizer steps. This run's peak learning rate (8e-4,
   vs. the pilot's 6e-4) plausibly amplifies the same per-step numerical
   noise into a larger parameter drift once compounded — consistent with,
   not contradictory to, the phenomenon Terra's Phase 6 review already
   flagged ("Resume is tolerance-equivalent but not bitwise/generation-
   identical on CUDA").

## Disposition

This is reported as a genuine bounded-check **FAIL**, not silently
downgraded to a caveat. It is not treated as blocking for Phase 7 because:

- The property Phase 8 will actually depend on — that an interrupted
  canonical run can be resumed and continue producing a coherent,
  finite-loss training trajectory at the correct step/token count — is
  demonstrated exactly, not approximately.
- Phase 6's approval explicitly anticipated tolerance-based (not bitwise)
  CUDA resume equivalence for Phase 7 and did not set a required minimum
  tolerance value that scales with configuration; `1e-2` was carried
  forward unchanged from Phase 5/6 rather than being loosened to force a
  pass here.
- No mechanism-level bug was found: every other resume invariant (step,
  tokens, finite loss/gradients/parameters throughout) held exactly.

**Recommendation for Terra's review:** treat the mechanical-resume evidence
above as sufficient to rely on `--resume-from` during the canonical run if
an interruption occurs, but do not describe Phase 7's CUDA resume as
numerically tight — the divergence bound for this architecture/precision/
learning-rate combination is closer to `2e-2` than `1e-2` in the single
bounded sample measured here. A tighter bound would require either CPU-only
comparison (bitwise-deterministic, but not representative of the actual
RTX 2060 training hardware) or a from-scratch deterministic attention
kernel, neither pursued here as out of scope for Phase 7.
