# Architecture — Frozen Specification (Phase 0 design, Phase 1 implementation)

This document describes the frozen architecture for Juniper Math 1, design-frozen
in Phase 0 and implemented in Phase 1. Implementation (embeddings, RMSNorm,
RoPE, attention, SwiGLU, blocks, LM head, loss, and authoritative programmatic
parameter-count verification) lives in
[`src/juniper_math/model.py`](../src/juniper_math/model.py) — see
[`reports/PHASE1_ARCHITECTURE_VALIDATION.md`](../reports/PHASE1_ARCHITECTURE_VALIDATION.md)
for validation evidence. **No model has been trained.** Phase 1 proves the
architecture mechanics are correct, not that the model has any mathematical
capability.

The canonical, machine-readable copy of this specification lives at
[`config/architecture.yaml`](../config/architecture.yaml) and is loaded/validated
by [`src/juniper_math/architecture.py`](../src/juniper_math/architecture.py).
This document must stay consistent with that file — if they disagree, the
YAML is authoritative and this file has drifted and needs fixing.

## Specification

| Field | Value |
|---|---|
| Architecture class | Decoder-only causal Transformer |
| Parameter target | 5,004,032 (~5.00M) |
| `d_model` | 256 |
| Layers | 5 |
| Query heads | 4 |
| KV heads | 4 |
| Head dimension | 64 |
| Attention | Standard multi-head causal self-attention |
| FFN | SwiGLU |
| `d_ff` | 688 |
| Normalization | RMSNorm, Pre-Norm layout |
| Position encoding | RoPE, theta = 10,000 |
| Biases | None |
| Vocabulary size | 4,096 |
| Weight tying | Enabled |
| Max context length | 1,024 tokens |
| Dropout | 0.0 |

## Parameter count sanity check

A rough arithmetic estimate (`ArchitectureConfig.estimated_parameter_count`)
gives:

- Token embedding: `4096 * 256 = 1,048,576` (weight-tied, so no separate output projection)
- Per layer: attention `4 * 256² = 262,144` + SwiGLU FFN `3 * 256 * 688 = 528,384` + 2 RMSNorm `512` = `791,040`
- 5 layers: `3,955,200`
- Final RMSNorm: `256`
- **Total: `1,048,576 + 3,955,200 + 256 = 5,004,032`** — matches the declared target exactly.

This estimate is superseded by authoritative verification: Phase 1's
`juniper_math.model.count_trainable_parameters` counts actual instantiated
`nn.Parameter` objects (deduplicated by storage identity, so weight tying is
counted once) and confirms exactly 5,004,032 — run `python -m juniper_math model`
to reproduce.

## Change policy

This specification is frozen for Phase 0. Any change requires a new
Architecture Decision Record (see [`docs/adr/`](adr/)) and a version bump in
`config/architecture.yaml`'s `architecture_version` field. Do not silently
edit frozen values.
