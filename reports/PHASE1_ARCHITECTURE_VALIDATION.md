# Phase 1 Architecture Validation

**Project:** Juniper Math 1
**Phase:** 1 — Architecture
**Engineer:** Claude Sonnet 5
**Date:** 2026-08-22

This report is the quantitative evidence trail behind the Phase 1 architecture claims.
See [`reports/PHASE1_REPORT.md`](PHASE1_REPORT.md) for the overall Phase 1 summary and
[`reports/PHASE1_BENCHMARKS.md`](PHASE1_BENCHMARKS.md) for throughput/VRAM/RAM numbers.

## 1. Exact parameter verification

`juniper_math.model.count_trainable_parameters` counts actual instantiated `nn.Parameter`
objects (deduplicated by storage identity — required because weight tying makes
`embed_tokens.weight` and the LM output projection the same underlying tensor).

```
$ python -m juniper_math model --device cpu
Architecture:        decoder_only_causal_transformer v0.1.0
d_model=256 n_layers=5 n_heads=4 d_ff=688 vocab=4096 context=1024
Trainable parameters: 5,004,032
Parameter target:     5,004,032
Device:               cpu
PASS: parameter count matches frozen target exactly
Synthetic forward pass: logits shape=(1, 8, 4096), dtype=torch.float32, finite=True
PASS: synthetic forward pass succeeded
```

`tests/test_model.py::test_parameter_count_detects_mismatch` deliberately constructs a model
from an altered config (`d_ff=700` instead of the frozen `688`) and asserts the counter reports
a *different* number — proving the check can actually fail, not just self-confirm the configured
target (per item 22, "the configured target must not be allowed to validate itself").

## 2. Component tests (`tests/test_model.py`, 50 tests)

| Component | Coverage |
|---|---|
| RMSNorm | Independent reference implementation comparison; shape preservation across 3 shapes; explicit test that a constant-offset input is *not* driven to zero (distinguishing it from LayerNorm's mean subtraction) |
| RoPE | Position-dependence (identical content at different positions produces different rotated Q/K); no trainable parameters; rejects odd `head_dim`; covers the full 1,024-position context with finite output |
| SwiGLU | Shape preservation; exact parameter count (`3 * d_model * d_ff`, no bias); confirms zero bias parameters |
| Transformer block | Exactly 2 RMSNorm instances per block; residual shape preservation |
| Full model | Shapes across (B,T) ∈ {(1,1),(1,16),(2,32),(4,128)}; full T=1024 context; T=1025 rejected; T=0 rejected; wrong-rank input rejected; float token IDs rejected; negative token ID rejected; token ID == vocab_size rejected |

## 3. Causal masking — behavioral test, not mask inspection

`test_future_tokens_cannot_affect_past_logits`: runs a sequence, mutates every token strictly
after a cutoff position, reruns, and asserts logits at positions `<= cutoff` are numerically
unchanged (`atol=1e-5`) while logits after the cutoff *do* change. This catches an accidentally
non-causal implementation that a static triangular-mask inspection would miss (e.g. if RoPE or
attention wiring leaked future information despite a correct mask tensor existing).

**Result: PASS.**

## 4. Padding mask

`attention_mask` convention: `[B, T]`, `True`/`1` = real token (attend), `False`/`0` = padding
(never attended to), documented in the `model.py` module docstring. Tests:

- Shape mismatch between `attention_mask` and `input_ids` is rejected with a clear error.
- `test_padding_positions_not_attended`: mutating only the padded tail of a sequence leaves all
  valid-position logits unchanged — proves padding is actually excluded, not just permitted to
  be excluded by the mask shape.
- `test_all_padding_row_produces_finite_output`: an entirely-padded row still produces finite
  output (position 0 always attends to itself under causal masking regardless of the mask flag
  at position 0, so no NaN/inf from an empty attention distribution).

**Result: PASS.**

## 5. Loss semantics

- Standard next-token shift (`logits[:, :-1]` vs `labels[:, 1:]`), `ignore_index=-100`.
- `test_ignored_labels_excluded`: masking out later label positions changes the loss value,
  proving masking is live rather than a no-op.
- **All-ignored-labels edge case (item 82):** `F.cross_entropy(reduction="mean")` on a batch
  where every label is `-100` computes `0/0` and returns `NaN` on this PyTorch version — this
  was caught by `tests/test_model.py::test_all_ignored_labels_documented_behavior` during
  development (see `PHASE1_SELF_REVIEW.md`). The model now explicitly special-cases this to a
  defined `0.0` loss (`shift_logits.sum() * 0.0`, keeping the tensor connected to the
  computation graph so `.backward()` still succeeds with zero gradient) rather than letting an
  unexplained `NaN` reach training silently.
- Labels/input_ids shape mismatch is rejected with a clear error.

**Result: PASS (with one real bug found and fixed — see self-review report).**

## 6. Weight tying

- `model.embed_tokens.weight is model.lm_head_weight` — true storage-level identity via a
  read-only `@property`, not value copying (see ADR-equivalent note in `model.py`).
- Survives a `state_dict()` save/load round-trip (`test_weight_tying_survives_state_dict_roundtrip`).
- Updating the embedding in place changes the LM projection output identically
  (`test_updating_embedding_changes_lm_projection`).
- `state_dict()` contains exactly one weight tensor for this matrix — no duplicate
  `lm_head_weight` key (`test_state_dict_has_no_duplicate_output_weight`). This was also a real
  bug found during development: an earlier version assigned `self.lm_head_weight = self.embed_tokens.weight`
  as a plain instance attribute, which `nn.Module.__setattr__` intercepts and registers as a
  *second* `nn.Parameter` entry pointing at the same tensor — correct in `count_trainable_parameters`
  (which dedupes by `id()`) but incorrect in `state_dict()` (which would have serialized the same
  tensor under two keys, silently doubling that portion of every checkpoint). Fixed by making
  `lm_head_weight` a `@property` instead of an attribute assignment.

**Result: PASS (after fixing the duplicate-registration bug).**

## 7. Forward / backward / gradients / parameter update

- Full-context (T=1024) forward pass on both CPU and CUDA (RTX 2060) produces finite logits.
- Backward pass produces gradients for every parameter with `requires_grad=True`; spot-checked
  explicitly at the embedding, first block, middle block, last block, and final RMSNorm
  (`test_backward_produces_finite_gradients_across_all_layers`).
- `torch.isfinite(param.grad).all()` checked for **every** parameter, not inferred from loss
  finiteness alone (per item 29).
- No parameter is accidentally frozen (`test_no_parameter_is_accidentally_frozen` — every
  parameter has `requires_grad=True`).
- An `AdamW` optimizer step changes at least one value in every updated parameter tensor,
  produces no NaN/inf, and weight tying survives the update
  (`test_optimizer_step_updates_parameters`).

**Result: PASS.**

## 8. Deterministic initialization

- Two models constructed from an identical seed produce bitwise-identical initial weights on
  CPU (`test_deterministic_initialization_same_seed`).
- A different seed produces at least one differing weight
  (`test_different_seed_yields_different_weights`).
- Standard PyTorch initialization is used (`nn.init.normal_(std=0.02)` for `Linear`/`Embedding`
  weights) — no custom residual-scaling scheme; this is a deliberate simplicity choice for a
  5M-parameter model, not an oversight, and is documented here per item 23.

**Result: PASS.**

## 9. Tiny controlled overfit

Script: [`scripts/tiny_overfit.py`](../scripts/tiny_overfit.py). Deterministic synthetic dataset:
4 sequences of 32 random (fixed-seed) token IDs, memorized via full-batch AdamW training.

| | |
|---|---|
| Seed | 5,004,032 (project default) |
| Optimizer | AdamW, lr=3e-3 |
| Steps | 300 |
| Device | CUDA (RTX 2060) |
| Initial loss | 8.3545 |
| Final loss | 0.000847 |
| Loss ratio (final/initial) | 0.000101 (gate: ≤ 0.05) |
| Final next-token accuracy | 1.0 (gate: ≥ 0.99) |
| Wall-clock | 3.48 seconds |
| **Gate** | **PASSED** |

Also verified cheap and passing on CPU (10.2 seconds, same gates). Raw output:
[`reports/artifacts/tiny_overfit_cuda.json`](artifacts/tiny_overfit_cuda.json). Automated
regression coverage: `tests/test_tiny_overfit.py` (150 steps, CPU, cheap enough for CI).

**Result: PASSED — gates defined before running, not adjusted after seeing results.**

## 10. Checkpointing and resume

Full training-state checkpoint (`juniper_math.checkpoint`) covers: model state dict, optimizer
state dict, scheduler state dict (when present), GradScaler state dict (when present), Python
`random` state, NumPy RNG state, `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`
(all CUDA devices), global step, tokens seen, architecture identity, training config,
synthetic-data-stream position, seed, and git commit — see `tests/test_checkpoint.py`
(10 tests) for save/load round-trip coverage of every field.

**Exact CPU resume equivalence** (`test_interrupted_resume_matches_uninterrupted_control`):
a control run trains 6 steps uninterrupted; a resume run trains 3 steps, checkpoints, destroys
all Python objects, reloads fresh model/optimizer, and trains the remaining 3 steps using the
identical synthetic data ordering. Final parameters are asserted **bitwise identical**
(`torch.equal`) between control and resumed runs, with deterministic algorithms enabled.

**Result: PASSED — bitwise CPU equivalence achieved, no tolerance needed.**

**CUDA resume smoke test** (`tests/test_model_cuda.py::test_cuda_checkpoint_save_restore_continue_smoke`):
save → destroy → reload → continue training on CUDA. Verifies finite loss, finite parameters
after a further optimizer step, and that parameters actually changed post-resume — an
operational check, not bitwise equality (CUDA kernels are not guaranteed bit-deterministic;
this is documented explicitly rather than claimed away).

**Result: PASSED.**

Checkpoint compatibility/safety: incompatible architecture (different `d_model`) is rejected
with a clear `CheckpointError`; a file missing the `schema_version` marker is rejected; a wrong
`schema_version` is rejected; a corrupted file fails deserialization loudly instead of silently;
a missing checkpoint path is rejected; atomic save leaves the previous good checkpoint untouched
if `torch.save` is interrupted mid-write (simulated via a monkeypatched failure).

## 11. CPU / CUDA / mixed precision

- All architecture tests run and pass on CPU (`tests/test_model.py`, `tests/test_checkpoint.py`).
- GPU-specific tests (`tests/test_model_cuda.py`, 7 tests, marked `@pytest.mark.gpu`, auto-skipped
  when CUDA is unavailable per `tests/conftest.py`) run and pass on the actual RTX 2060: forward/
  backward FP32, full T=1024 context, CPU↔CUDA device transfer (including the RoPE cos/sin cache,
  a common cached-state device-transfer bug), FP16 mixed precision with `GradScaler` (finite loss,
  finite parameters after a scaled step), peak VRAM comfortably within the 6GB budget, and the
  CUDA checkpoint resume smoke test above.
- BF16 was not assumed; the RTX 2060 is Turing-generation hardware and was tested for FP16
  mixed-precision capability specifically (per item 27's guidance to detect actual support
  rather than guess).

**Result: PASSED on both CPU and the RTX 2060.**

## Summary

| Gate | Result |
|---|---|
| Exact parameter count (5,004,032) | PASS |
| Component tests (RMSNorm, RoPE, SwiGLU, block) | PASS |
| Causal masking (behavioral) | PASS |
| Padding masking | PASS |
| Loss shift/masking/all-ignored edge case | PASS |
| Weight tying (storage identity, survives save/load, no duplicate key) | PASS |
| Forward/backward/gradients/parameter update | PASS |
| Deterministic initialization | PASS |
| Tiny controlled overfit | PASS |
| Checkpoint save/load (all state) | PASS |
| Exact CPU resume equivalence (bitwise) | PASS |
| CUDA resume smoke test | PASS |
| CPU/CUDA/mixed-precision operation | PASS |

Two real defects were found and fixed during this validation pass (duplicate weight-tying
parameter registration; NaN on all-ignored-labels loss) — see `reports/PHASE1_SELF_REVIEW.md`
for the full account.
