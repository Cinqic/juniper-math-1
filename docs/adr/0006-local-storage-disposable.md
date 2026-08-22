# ADR 0006: Local storage is treated as disposable

**Context.** Follows directly from ADR 0005. If local state is assumed
durable, recovery procedures atrophy and go untested until an actual loss
event, when it's too late to fix them.

**Decision.** Design every workflow (config, evaluation, manifests,
environment setup) as if the local disk could disappear at any moment.
Anything important gets committed; anything disposable (caches, raw
downloads, virtual environments) is explicitly `.gitignore`d and documented
as regenerable.

**Consequences.** Slightly more upfront discipline (explicit manifests,
hashing, recovery testing) in exchange for a project that survives hardware
loss without data loss of anything that matters.
