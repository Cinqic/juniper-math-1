# Research Notes

A durable place for hypotheses, decisions, observations, and unresolved
questions, kept distinct from established fact. Append entries chronologically
with a date heading; do not retroactively edit past entries except to fix
factual errors (note the correction, don't silently rewrite history).

## 2026-08-22 — Phase 0 kickoff

**Hypothesis.** A ~5M-parameter decoder-only Transformer, paired with
deterministic tool execution for arithmetic beyond direct-answer scale, can
achieve high truthful-calibration on the Phase 0 evaluation categories
(ambiguity detection, missing-information detection, error recognition) even
though raw arithmetic capacity is necessarily limited by parameter count.

**Decision.** See [ADR 0001–0008](adr/) for the frozen foundational
decisions made this phase.

**Observation.** The declared architecture's arithmetic parameter estimate
(`5,004,032`) matches the frozen `parameter_target` exactly (see
[`ARCHITECTURE.md`](ARCHITECTURE.md#parameter-count-sanity-check)) — no
inconsistency was found requiring escalation to independent review.

**Unresolved questions (deferred to later phases):**

- What vocabulary construction (Phase 2) best balances tokenization
  efficiency for mathematical notation (digits, operators, fraction bars)
  against the fixed 4,096-token budget?
- What is the right tool-invocation interface contract for Phase 3 (single
  deterministic calculator vs. a small set of typed operations)?
- How should ambiguity/missing-information/unsupported-capability behaviors
  be weighted relative to raw answer accuracy in eventual scoring (Phase 1+
  evaluation infrastructure)?
