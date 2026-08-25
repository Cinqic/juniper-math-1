# Phase 8 V4 Safety-Replay Milestone Run

## Verdict

**Rejected for full approval escalation; retained as the best direct/safety
ablation so far.** The V4 mixture produces meaningful direct behavior and
retains Base/safety metrics, but it does not improve valid-call or
correct-tool performance relative to the frozen Base. The next bounded
experiment therefore changes only explicit runtime-executed tool supervision.

## Provenance

- clean source commit: `4bd27e1e33ae9bf53e6058829e001c2751133cf6`;
- run ID: `phase8-sft-v11-safety-replay-milestones`;
- 600 steps, `1.5e-4`, milestones at 0/200/400/600;
- 34,000 masked SFT records plus 2,000 frozen-Base replay records;
- train representation identity:
  `dfcfb08c30b669a535ec7d130a0a0183cd936dc099926e4ddea2fdac10b2e2b5`.

## Milestones

| Step | Base loss delta | Math | Calibration | Adversarial | Direct | Tool E2E | Valid calls | Correct tool | Fabrication |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | +0.000000 | 1/215 | 0/130 | 36/195 | 0/160 | 0/67 | 75/271 | 36/67 | 110/271 |
| 200 | +0.031421 | 4/215 | 6/130 | 95/195 | 2/160 | 9/67 | 38/271 | 23/67 | 38/271 |
| 400 | +0.017621 | 13/215 | 27/130 | 120/195 | 8/160 | 6/67 | 37/271 | 21/67 | 34/271 |
| 600 | +0.011151 | 16/215 | 23/130 | 115/195 | 8/160 | 6/67 | 39/271 | 22/67 | 40/271 |

Checkpoint SHA-256 values: step 200
`655885fd0850f9e348632612972cf4fdbe7143865a0eec89b9426780e6fe793f`;
step 400
`b0728c539006e5736cffd53e49a56c5ba20df8304f294bd857816d68771b51a2`;
step 600
`19e2855f2537c760233f0460b173f7d808e51c7c2ff6e32a87fe76419f2deefc`.

The 400-step state is the strongest direct/safety point, but none is
approvable: Base valid calls and correct-tool numerators are 75/271 and
36/67, respectively, while every trained milestone is lower. This result
does not weaken that gate or imply a release.
