# ADR 0004: Deterministic tool execution over neural arithmetic guessing

**Context.** Neural networks are unreliable at exact arithmetic, especially
for large numbers or long computations; a 5M-parameter model has even less
capacity to memorize arithmetic patterns.

**Decision.** For computations above a "direct answer" threshold, the model
is expected to decompose the problem and invoke deterministic mathematical
tools (Phase 3 — the "Cinqic Calculator") rather than produce a numeric
answer purely from learned weights.

**Consequences.** Correctness for tool-eligible problems is bounded by tool
correctness, not model arithmetic capability. The model's job shifts to
problem understanding, decomposition, tool selection, and result
verification — reflected in the `tool_required`/`invoke_tool` fields in the
Phase 0 evaluation schema.
