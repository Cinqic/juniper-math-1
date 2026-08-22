# ADR 0008: Phase 0 precedes implementation and training

**Context.** It's tempting to skip straight to building the Transformer and
training it, especially for a small model where a first pass could be
written quickly. But foundation gaps (no recovery plan, no frozen eval
baseline, no reproducibility infrastructure) become expensive once training
data and checkpoints start accumulating on top of them.

**Decision.** Establish repository structure, packaging, environment
validation, seeding, configuration, logging, CLI, testing, the frozen
Phase 0 evaluation suite, manifests, and recovery validation *before* any
model implementation, tokenizer training, or dataset construction begins.

**Consequences.** Phase 0 has no trained model or user-facing capability —
by design. Its value is entirely in making every later phase reproducible,
recoverable, and auditable. Impatience to reach "real" model work is not a
reason to blur this boundary (see `docs/architecture.md`'s phase-ownership
notes and the CLI's honest "not implemented until Phase N" placeholders).
