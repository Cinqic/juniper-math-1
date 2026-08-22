# ADR 0007: Later architectural ideas are separate, labeled experiments

**Context.** It is tempting to tweak the frozen architecture opportunistically
during implementation ("just try MoE real quick"), which erodes the value of
having a frozen baseline to compare against.

**Decision.** Any architectural idea beyond the frozen Phase 0 specification
must be run as a distinct, named experiment (see
[`EXPERIMENT_NAMING.md`](../EXPERIMENT_NAMING.md)) with its own recorded
configuration and results — never as a silent substitution for the baseline.

**Consequences.** The baseline architecture stays a stable comparison point
across the project's lifetime. Promoting an experimental architecture to be
the new baseline requires an explicit ADR and version bump, not an
in-place edit.
