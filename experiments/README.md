# experiments/

Experiment metadata and documentation, named per
[`docs/EXPERIMENT_NAMING.md`](../docs/EXPERIMENT_NAMING.md). Not an
uncontrolled junk drawer — every subdirectory here must have an
`experiment.yaml` describing its intent, configuration identities, and
outcome. Phase 0 defines the convention only; no experiments exist yet.

## experiments/phase5-smoke/

The first real experiment: Phase 5 smoke pretraining. See
`experiments/phase5-smoke/experiment.yaml` for identity/config metadata and
`reports/PHASE5_RESULTS.md` for full results. The raw per-step JSONL logs
(`train_log.jsonl`, `resume_test_log.jsonl`, a few tens of KB) are small
enough to commit directly as primary evidence rather than only summarizing
them — they record every logged training/validation/checkpoint/resume
event with step numbers, losses, and timestamps.
