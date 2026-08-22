# ADR 0002: ~5M parameter initial target

**Context.** Primary hardware is an RTX 2060 (6GB VRAM), 16GB system RAM.
Training must remain practical on this hardware.

**Decision.** Target approximately 5,000,000 parameters (frozen exact
target: 5,004,032) for the initial architecture, rather than a larger model
that would strain or exceed available VRAM for training with reasonable
batch sizes and context length.

**Consequences.** The research question is specifically about how capable a
*small* purpose-built model can become — not about matching large
general-purpose model capability. This constrains context length (1,024
tokens), vocabulary size (4,096), and layer count accordingly.
