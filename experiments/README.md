# experiments/

Experiment metadata and documentation, named per
[`docs/EXPERIMENT_NAMING.md`](../docs/EXPERIMENT_NAMING.md). Not an
uncontrolled junk drawer — every subdirectory here must have an
`experiment.yaml` describing its intent, configuration identities, and
outcome. Historical experiments are preserved rather than rewritten after
later findings.

## Phase 8 index

| Run ID | Intervention | Primary report | Disposition |
| --- | --- | --- | --- |
| `phase8-sft-v1` / `v2` | Original full SFT | [`PHASE8_RESULTS.md`](../reports/PHASE8_RESULTS.md) | Rejected evidence |
| `phase8-sft-v3-remediated` | Corrected tool-error continuation | [`PHASE8_REMEDIATION.md`](../reports/PHASE8_REMEDIATION.md) | Rejected evidence |
| `phase8-sft-v4-recovery` | Diversified recovery preflight | [`PHASE8_PREFLIGHT_V3.md`](../reports/PHASE8_PREFLIGHT_V3.md) | Preflight evidence |
| `phase8-sft-v5-full-recovery` | Low-LR diversified full run | [`PHASE8_RESULTS_V3_FULL.md`](../reports/PHASE8_RESULTS_V3_FULL.md) | Rejected evidence |
| `phase8-sft-v6`–`v10` | Direct curricula, high rate, replay, safety replay | [`PHASE8_PREFLIGHT_V4.md`](../reports/PHASE8_PREFLIGHT_V4.md)–[`V8`](../reports/PHASE8_PREFLIGHT_V8.md) | Preflight evidence |
| `phase8-sft-v11-safety-replay-milestones` | Safety replay milestone run | [`PHASE8_RESULTS_V4_MILESTONES.md`](../reports/PHASE8_RESULTS_V4_MILESTONES.md) | Rejected evidence |
| `phase8-sft-v12`–`v15` | Explicit tools, partial layers, staged-call preflight | [`PHASE8_PREFLIGHT_V9.md`](../reports/PHASE8_PREFLIGHT_V9.md)–[`V12`](../reports/PHASE8_PREFLIGHT_V12.md) | Preflight evidence |
| `phase8-sft-v16-call-stage-milestones` | Staged tool-call milestone run | [`PHASE8_RESULTS_V5_CALL_STAGE.md`](../reports/PHASE8_RESULTS_V5_CALL_STAGE.md) | Rejected evidence |

See the [final research conclusion](../reports/FINAL_RESEARCH_CONCLUSION.md)
for reconciled evaluation-version context and Pareto research checkpoints.

## experiments/phase5-smoke/

The first real experiment: Phase 5 smoke pretraining. See
`experiments/phase5-smoke/experiment.yaml` for identity/config metadata and
`reports/PHASE5_RESULTS.md` for full results. The raw per-step JSONL logs
(`train_log.jsonl`, `resume_test_log.jsonl`, a few tens of KB) are small
enough to commit directly as primary evidence rather than only summarizing
them — they record every logged training/validation/checkpoint/resume
event with step numbers, losses, and timestamps.
