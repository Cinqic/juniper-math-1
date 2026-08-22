# Phase 1 → GPT-5.6 Terra Handoff Package

**Purpose:** self-contained handoff. Terra should not need any memory of the conversation that
produced this candidate — everything needed to independently review, audit, remediate if
necessary, and approve Phase 1 is either in this document or linked from it.

## Candidate identity

| | |
|---|---|
| Repository | `https://github.com/Cinqic/juniper-math-1` |
| Branch | `main` |
| Phase 1 review candidate commit | `6a49586ca04c97721c4cf0fad4938f2a2a3f2da5` (this handoff doc is committed in a follow-up commit on top of that commit; the tag `phase-1-review-candidate` points at the final commit including this file) |
| Candidate tag | `phase-1-review-candidate` (non-final — points at the Sonnet 5 candidate; Terra owns `phase-1-architecture`, the final tag) |
| Starting foundation | tag `phase-0-foundation`, commit `f9e3659efe37183bae60ffafc2762b3d342cd047` |

## Reports (read in this order)

1. [`reports/PHASE1_REPORT.md`](PHASE1_REPORT.md) — overall Phase 1 summary
2. [`reports/PHASE1_ARCHITECTURE_VALIDATION.md`](PHASE1_ARCHITECTURE_VALIDATION.md) — component-by-component evidence
3. [`reports/PHASE1_BENCHMARKS.md`](PHASE1_BENCHMARKS.md) — hardware benchmarks and methodology
4. [`reports/PHASE1_SELF_REVIEW.md`](PHASE1_SELF_REVIEW.md) — defects found and fixed during development, adversarial test matrix, security audit

## Exact parameter count

```
Expected: 5,004,032
Actual:   5,004,032
```

Reproduce: `python -m juniper_math model --device cpu` (or `--device cuda` on GPU hardware).

## Test commands (run all of these; all must pass)

```bash
python -m juniper_math validate-env
python -m juniper_math validate-config
python -m juniper_math hash verify
python -m juniper_math evals validate
python -m juniper_math evals verify
python -m juniper_math manifests-validate
python -m juniper_math deps-check
python -m juniper_math model --device cpu
pytest -v                    # 202 tests, includes 190 Phase 0 gates (no regressions)
ruff check .
ruff format --check .
mypy
```

GPU-specific tests (`tests/test_model_cuda.py`, 7 tests) are marked `@pytest.mark.gpu` and
auto-skip on CPU-only machines (`tests/conftest.py`). On CUDA-capable hardware, `pytest -v`
automatically includes them — no separate invocation needed.

## Recovery procedure (fresh clone)

```bash
git clone https://github.com/Cinqic/juniper-math-1.git /tmp/juniper_recovery_check
cd /tmp/juniper_recovery_check
git checkout phase-1-review-candidate   # or the exact SHA recorded above
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e . --no-deps
python -m juniper_math validate-env
pytest -v
python -m juniper_math model --device cpu
python -m juniper_math hash verify
```

This exact procedure was run by Claude Sonnet 5 against the pushed candidate before handoff —
see the Recovery section of `PHASE1_REPORT.md` for the result. Terra should re-run it
independently rather than trust that account.

## Checkpoint/resume reproduction

```bash
pytest -v tests/test_checkpoint.py::test_interrupted_resume_matches_uninterrupted_control
```

This is the strict CPU bitwise-equality resume test (control vs. checkpoint-interrupt-resume,
identical synthetic data ordering, `torch.equal` on final parameters — no tolerance).

```bash
pytest -v tests/test_model_cuda.py::test_cuda_checkpoint_save_restore_continue_smoke
```

CUDA operational smoke test (save → destroy → reload → continue; finite state, not bitwise
equality — CUDA kernels are not guaranteed bit-deterministic, documented in the test docstring).

## Tiny-overfit reproduction

```bash
python scripts/tiny_overfit.py --device cuda   # or --device cpu
```

Deterministic (fixed seed 5,004,032), 4 synthetic sequences × 32 tokens, AdamW lr=3e-3, 300
steps. Gate: final/initial loss ratio ≤ 0.05 AND next-token accuracy ≥ 0.99. Exits non-zero if
the gate fails. Raw output from the last run: `reports/artifacts/tiny_overfit_cuda.json`.

## CUDA validation commands (run on the RTX 2060 or equivalent CUDA hardware)

```bash
pytest -v tests/test_model_cuda.py
python scripts/benchmark_phase1.py --device cuda --json /tmp/bench.json
```

## Known limitations

- SDPA's fallback attention kernel path is what actually runs on this Turing-generation GPU, not
  a FlashAttention-2-specific kernel (ADR 0009). Behaviorally correct, not benchmarked against
  alternative kernel choices.
- Benchmarks measure architecture mechanics on synthetic random token IDs — not a projection of
  real pretraining throughput, which depends on data loading and scheduling overhead that don't
  exist yet.
- No KV cache; inference throughput numbers are full-forward reference numbers only.
- Two real defects were found and fixed by the test suite during development (duplicate
  weight-tying parameter registration; NaN on an all-labels-ignored batch) — see
  `PHASE1_SELF_REVIEW.md` for full root-cause accounts and the regression tests added.

## Deferred work (confirmed absent from this candidate)

Tokenizer training/artifacts, real mathematical dataset, Cinqic Calculator integration, SFT,
production pretraining/training loop, serious optimizer/scheduler configuration (Phase 1's
AdamW settings are explicitly labeled "architecture validation only," not a Phase 6/7 commitment).

## Terra's authority

Per the Phase 1 instructions governing this project, GPT-5.6 Terra is authorized to independently
review all Phase 1 work; rerun tests; inspect architecture mathematics; compare implementation to
the frozen config; challenge parameter accounting; inspect tensor semantics, causality,
checkpointing, and benchmarking; run adversarial tests; identify flaws; **directly modify and fix**
Phase 1 defects; add regression tests; repeat recovery; update reports; push remediation; perform
final audit; issue Phase 1 approval; and create the final `phase-1-architecture` tag. Terra does
not need to return defects to Sonnet 5 unless Terra chooses to.

Terra must not silently change the frozen architecture to fix an implementation defect — fix the
implementation, the tests, or the documentation, whichever is actually wrong. If the frozen
Phase 0 architecture itself is found to be fundamentally invalid, Terra should stop approval and
escalate to Cinqic rather than silently redesigning Juniper Math 1.

## Status

```
AWAITING_GPT_5_6_TERRA_REVIEW
```
