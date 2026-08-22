# ADR 0005: GitHub is canonical

**Context.** Local development happens on a single physical machine
(FLOWBOX) which could be wiped, fail, or be replaced at any time.

**Decision.** The Cinqic GitHub repository is the single source of truth
for project state. Anything required to understand, restore, or continue
the project must be committed and pushed there.

**Consequences.** Recovery must be tested against a real clone from the
remote, not by assuming local state (see
[`docs/RECOVERY.md`](../RECOVERY.md) and
[`reports/RECOVERY_TEST_REPORT.md`](../../reports/RECOVERY_TEST_REPORT.md)).
Local-only artifacts (large checkpoints, raw datasets) need a documented
alternative preservation path (see
[`CHECKPOINT_POLICY.md`](../CHECKPOINT_POLICY.md)).
