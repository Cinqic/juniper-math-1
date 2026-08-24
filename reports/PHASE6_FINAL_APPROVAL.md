# Phase 6 Final Approval

## APPROVED

Acceptance source commit: `ac2bbe9c6c40c9da9a4fc5dcb1dcd34cd458c6aa`.

Phase 6 Pilot Pretraining is independently verified and approved after
remediation. The original candidate experiment is invalidated and preserved
only as superseded historical evidence; the authoritative evidence is the
clean-shard reproduction recorded in `PHASE6_RESULTS.md` and
`experiments/phase6-pilot/`.

Frozen artifacts remained unchanged and hash verification passed. The pilot
selection is deterministic, manifest-reproducible, split-safe, and now fails
closed on stale shard bytes. Packed 1,024-token FP32 training is stable,
checkpointed, and practical on the RTX 2060. The frozen evaluation denominator
is complete and confirms that this pilot has not produced math or tool-use
capability. Phase 5 regressions remain valid.

Phase 7 Full Base Pretraining is authorized but has not started. It must use
fresh random initialization, never the pilot weights. Before committing its
full budget, run one bounded LR preflight around 6e-4, 8e-4, and 1e-3 and
record tolerance-based—not bitwise—CUDA resume expectations.
