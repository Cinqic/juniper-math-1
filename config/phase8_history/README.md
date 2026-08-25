# Phase 8 Recovery Configuration Index

`config/training_phase8_sft.yaml` is a historical final-research configuration,
not authorization for a new run or an approved Phase 8 release configuration.
The following immutable commits
contain the exact configuration used by each preserved post-recovery run;
use `git show <commit>:config/training_phase8_sft.yaml` to materialize one
without altering the active file.

| Run ID | Source commit | Result record |
| --- | --- | --- |
| `phase8-sft-v10-safety-replay-preflight` | `653a004` | `reports/PHASE8_PREFLIGHT_V8.md` |
| `phase8-sft-v11-safety-replay-milestones` | `4bd27e1` | `reports/PHASE8_RESULTS_V4_MILESTONES.md` |
| `phase8-sft-v12-explicit-tool-preflight` | `21bf09d` | `reports/PHASE8_PREFLIGHT_V9.md` |
| `phase8-sft-v13-safety-tool-lr2e-4-preflight` | `481c99a` | `reports/PHASE8_PREFLIGHT_V10.md` |
| `phase8-sft-v14-last2-safety-tool-preflight` | `dad3e62` | `reports/PHASE8_PREFLIGHT_V11.md` |
| `phase8-sft-v15-call-stage-preflight` | `905c0c0` | `reports/PHASE8_PREFLIGHT_V12.md` |
| `phase8-sft-v16-call-stage-milestones` | `9c1ea58` | `reports/PHASE8_RESULTS_V5_CALL_STAGE.md` |

Every listed run is rejected evidence. No row denotes an approved checkpoint,
release, or Phase 9 authorization. Historical source commits are immutable
remote Git history; experiment logs and report files are tracked alongside
their result commits.
