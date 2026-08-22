# ADR 0009: `scaled_dot_product_attention` as the attention backend

**Context.** The frozen architecture specifies standard causal multi-head
attention (ADR 0003). PyTorch offers `torch.nn.functional.scaled_dot_product_attention`
(SDPA), which dispatches to fused kernels (FlashAttention, memory-efficient
attention, or a math fallback) depending on hardware/dtype, versus a
hand-written manual attention implementation.

**Decision.** Use `F.scaled_dot_product_attention` for both the pure-causal
path (`is_causal=True`, no mask) and the padding-mask path (explicit
additive float mask combining causal + padding, `is_causal=False` — SDPA
does not accept both `is_causal=True` and an explicit `attn_mask`
simultaneously). Do not depend on FlashAttention specifically being
available; SDPA falls back to a correct (if slower) kernel when the fused
paths are unavailable, which the RTX 2060 (Turing, no FlashAttention-2
support) exercises in practice.

**Consequences.** `tests/test_model.py::test_future_tokens_cannot_affect_past_logits`
and `test_padding_positions_not_attended` behaviorally verify causality and
masking independent of which SDPA backend dispatches — the tests would
fail identically whether SDPA used a manual reference implementation or a
fused kernel, since they check output behavior, not implementation
mechanics. No manual attention implementation is required as a fallback;
SDPA's math-backend path already serves as the reference.
