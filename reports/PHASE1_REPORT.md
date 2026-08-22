# Phase 1 Completion Report

## Identification

| | |
|---|---|
| Project | Juniper Math 1 |
| Phase | 1 — Architecture |
| Primary engineer | Claude Sonnet 5 |
| Independent reviewer / approval authority | GPT-5.6 Terra |
| Starting foundation | `phase-0-foundation` tag, commit `f9e3659efe37183bae60ffafc2762b3d342cd047` |
| Phase 1 candidate commit | recorded at push time — see `reports/PHASE1_TERRA_HANDOFF.md` |
| Date | 2026-08-22 |
| Environment | Ryzen 7 5700G / RTX 2060 6GB / 16GB RAM, Linux, Python 3.12.3, PyTorch 2.13.0+cu130, CUDA 13.0 |

## Architecture (frozen, per `config/architecture.yaml`)

```
Architecture:       Decoder-only causal Transformer
Parameters:         5,004,032 trainable parameters
Vocabulary:         4,096
Context:            1,024
d_model:            256
Layers:             5
Query heads:        4
KV heads:            4
Head dimension:     64
Attention:          Standard multi-head causal self-attention
FFN:                SwiGLU (d_ff=688)
Normalization:      RMSNorm, Pre-Norm placement
Position encoding:  RoPE (theta=10,000)
Biases:             None
Embedding tying:    Input embedding = LM output weights
Dropout:            0.0
```

No architecture value was changed during Phase 1. `n_query_heads == n_kv_heads` is enforced at
construction time, so this cannot silently become GQA/MQA.

## Parameter verification

```
Expected: 5,004,032
Actual:   5,004,032   (python -m juniper_math model)
```

Counted from actual instantiated `nn.Parameter` objects (`count_trainable_parameters`,
deduplicated by storage identity), not derived from the configured target. A deliberately
altered configuration (`d_ff=700`) was constructed and confirmed to produce a *different* count,
proving the check can fail.

## Components implemented

`src/juniper_math/model.py`: `RMSNorm`, `RotaryEmbedding` (+ `apply_rotary_pos_emb`),
`CausalSelfAttention`, `SwiGLU`, `TransformerBlock`, `JuniperMathModel`, `ModelOutput`,
`count_trainable_parameters`, `verify_parameter_count`, `build_model`.

`src/juniper_math/checkpoint.py`: full training-state checkpoint schema
(`TrainingCheckpoint`, `build_checkpoint`), atomic save (`save_checkpoint_atomic`), safe
metadata inspection (`inspect_checkpoint_metadata`), compatibility verification
(`verify_checkpoint_compatibility`), RNG capture/restore (`capture_rng_state`,
`restore_rng_state`).

`src/juniper_math/cli.py`: activated `model` (construct + verify + synthetic forward pass) and
`checkpoint inspect` (safe metadata-only inspection) commands. `tokenizer`, `dataset`, `train`,
`evaluate`, `infer`, `tool-test` remain honest "not implemented until Phase N" stubs.

`scripts/tiny_overfit.py`, `scripts/benchmark_phase1.py`: reproducible experiments (see below).

## Tests

```
$ pytest -v
202 passed, 2 warnings in ~11s

$ ruff check .
All checks passed!

$ ruff format --check .
73 files already formatted

$ mypy
Success: no issues found in 16 source files
```

Test breakdown: 190 Phase 0 tests (all still passing, no regressions) + 12 new files/additions
covering the Phase 1 architecture:

- `tests/test_model.py` — 50 tests (parameter count, RMSNorm, RoPE, SwiGLU, block, shapes,
  sequence boundaries, causal masking, padding masking, loss semantics, weight tying,
  forward/backward/gradients, deterministic init, save/load, numerical sanity)
- `tests/test_checkpoint.py` — 10 tests (save/load round-trip, incompatibility rejection,
  corrupted/missing file rejection, atomic-save-under-failure, metadata inspection, checkpoint
  size comparison, exact CPU resume equivalence)
- `tests/test_model_cuda.py` — 7 tests (GPU-only, `@pytest.mark.gpu`, auto-skipped without CUDA;
  ran and passed on the actual RTX 2060: FP32 forward/backward, full context length, device
  transfer including RoPE cache, FP16 mixed precision, peak VRAM budget, CUDA resume smoke test)
- `tests/test_tiny_overfit.py` — 1 test (cheap CI-safe wrapper around the tiny overfit script)
- `tests/test_cli.py` — additions for `model` and `checkpoint inspect`

## Tiny controlled overfit

Methodology: 4 fixed-seed synthetic sequences (32 tokens each), full-batch AdamW (lr=3e-3), 300
steps, deterministic project seed (5,004,032).

| | |
|---|---|
| Initial loss | 8.3545 |
| Final loss | 0.000847 |
| Loss ratio gate | ≤ 0.05 (actual: 0.000101) |
| Next-token accuracy gate | ≥ 0.99 (actual: 1.0) |
| Result | **GATE PASSED** (3.48s on CUDA; also passes on CPU, 10.2s) |

Gates were fixed before running the experiment (see `scripts/tiny_overfit.py`), not adjusted
after seeing results. Full methodology: `reports/PHASE1_ARCHITECTURE_VALIDATION.md` §9.

## Checkpointing

Full state coverage: model, optimizer, scheduler (when present), GradScaler (when present),
Python/NumPy/torch-CPU/torch-CUDA-all RNG state, step, tokens seen, architecture identity,
training config, synthetic data-stream position, seed, git commit. Atomic save (temp file +
`os.replace`). Schema-versioned (`CHECKPOINT_SCHEMA_VERSION`), architecture-identity-checked on
load, rejects incompatible/corrupted/missing files with a clear `CheckpointError`.

**Exact CPU resume equivalence:** control (6 uninterrupted steps) vs. resume (3 steps →
checkpoint → destroy → reload → 3 more steps), identical synthetic data ordering — final model
parameters are **bitwise identical** (`torch.equal`) between the two runs. No tolerance needed.

**CUDA resume:** save → destroy → reload → continue smoke test passes (finite loss, finite
parameters, parameters actually change on the post-resume step) — an operational check, since
CUDA kernels are not guaranteed bit-deterministic (documented, not glossed over).

## Hardware

| | |
|---|---|
| Peak training VRAM (T=1024, batch=4) | 757.6 MB allocated / 840 MB reserved (~12-14% of 6GB budget) |
| Training throughput (CUDA, T=1024, batch=4) | 64,054 tokens/sec |
| Inference throughput (CUDA, T=1024, batch=1, full-forward) | 199,381 tokens/sec |
| Model parameter memory (FP32) | 19.09 MB |
| Model-only checkpoint | 19.1 MB |
| Full training checkpoint | 57.3 MB |
| Process RAM (RSS, benchmark process) | ~1.2 GB |

Full methodology, all seq-len configurations (128/512/1024), and CPU numbers:
`reports/PHASE1_BENCHMARKS.md`.

## Artifacts

- Raw benchmark JSON: `reports/artifacts/phase1_benchmark_cuda.json`,
  `reports/artifacts/phase1_benchmark_cpu.json`
- Raw tiny-overfit JSON: `reports/artifacts/tiny_overfit_cuda.json`
- Frozen artifact hash manifest (Phase 0 + updated `project.yaml` hash):
  `manifests/artifacts.yaml`, verified by `python -m juniper_math hash verify`

## Recovery

Fresh-clone recovery was performed against the pushed candidate commit — see
`reports/PHASE1_TERRA_HANDOFF.md` for the exact procedure and result.

## Known limitations

- No FlashAttention-2-specific kernel confirmed on this Turing GPU (SDPA fallback path is what
  actually runs — see ADR 0009). Correct, not necessarily maximally fast.
- Benchmarks are architecture-mechanics microbenchmarks on synthetic data, not a projection of
  real pretraining throughput.
- No KV cache; inference numbers are full-forward reference throughput only (in-scope deferral,
  not a gap — see item 48 of the Phase 1 instructions).
- Two real defects were found and fixed during development (duplicate weight-tying parameter
  registration; NaN on all-labels-ignored loss) — see `reports/PHASE1_SELF_REVIEW.md` for full
  root-cause accounts. Both have regression test coverage.

## Deferred work (explicitly out of scope for Phase 1)

Tokenizer training and artifacts, real mathematical dataset construction, Cinqic Calculator
integration, SFT, production pretraining/training loop, serious optimizer/scheduler
configuration (Phase 1's AdamW config is explicitly labeled "architecture validation only").
None of this exists anywhere in the Phase 1 diff — verified by `grep` audit in
`reports/PHASE1_SELF_REVIEW.md`.

## Status

```
AWAITING_GPT_5_6_TERRA_REVIEW
```

Phase 2 remains `NOT_AUTHORIZED` until Phase 1 receives GPT-5.6 Terra's independent review,
remediation if necessary, and final approval (final tag `phase-1-architecture`, owned by Terra).
