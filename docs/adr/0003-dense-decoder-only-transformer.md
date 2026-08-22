# ADR 0003: Dense decoder-only Transformer as the starting architecture

**Context.** Many architecture variants exist (MoE, state-space models,
hybrid attention). At 5M parameters, added architectural complexity has
limited room to pay for itself, and a well-understood baseline is needed
before experimenting.

**Decision.** Start with a standard dense decoder-only causal Transformer
(RMSNorm, Pre-Norm, RoPE, SwiGLU, standard MHA) as specified in
[`ARCHITECTURE.md`](../ARCHITECTURE.md), rather than a novel or exotic
architecture.

**Consequences.** Establishes a well-understood, debuggable baseline.
Alternative architectures are legitimate future experiments but must be
run as separate, clearly labeled experiments (ADR 0007), not silent
substitutions for the frozen baseline.
