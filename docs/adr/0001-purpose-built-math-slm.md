# ADR 0001: Juniper Math 1 is a purpose-built mathematical SLM

**Context.** General-purpose LLMs are unreliable at arithmetic and often
hallucinate confident-sounding wrong answers, because they were not built to
prefer verifiable computation over pattern completion.

**Decision.** Build a small, purpose-built model specialized for
understanding, decomposing, and solving math problems — paired with
deterministic tool execution — rather than a general-purpose model.

**Consequences.** Scope is narrower (math only) but calibration and
reliability targets are higher. Success is measured by truthful, verifiable
answers and honest refusal/clarification behavior, not breadth.
