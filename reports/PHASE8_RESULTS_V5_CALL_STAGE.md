# Phase 8 V5 Staged Tool-Call Milestone Run

## Verdict

**Rejected for approval and full escalation.** Staged call supervision improves
direct answers, end-to-end completion, terminal behavior, and fabricated-result
resistance while preserving Base loss. It does not improve the Base's
required-tool parsing or correct-tool selection, both explicit gates.

## Provenance

- clean source commit: `9c1ea58eb6d71f91a5f4dbdfa3dd49d00646cb1b`;
- run ID: `phase8-sft-v16-call-stage-milestones`;
- 600 steps at `1.5e-4`; 37,000 SFT + 2,000 frozen-Base replay trajectories;
- representation identity:
  `9057a0d774fbf45546c1b40af7ca05e893ca4cf8d75190413eb849151025d957`.

## Milestones

| Step | Base loss delta | Math | Adversarial | Direct | Required parsed | Correct tool | Tool E2E | Fabrication |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | +0.000000 | 1/215 | 36/195 | 0/160 | 51/67 | 36/67 | 0/67 | 110/271 |
| 200 | +0.049477 | 2/215 | 106/195 | 0/160 | 40/67 | 25/67 | 8/67 | 0/271 |
| 400 | +0.028165 | 3/215 | 76/195 | 3/160 | 40/67 | 25/67 | 9/67 | 0/271 |
| 600 | +0.019456 | 5/215 | 72/195 | 7/160 | 40/67 | 25/67 | 10/67 | 7/271 |

Checkpoint SHA-256 values: step 200
`f1c0a50eec5501c8c8a4f5844da84ab0fa1d3f8568cf5d705bed2ea650dd4175`;
step 400
`34f4d48f15694ce988cdf07450ab7c2131b24fd1614b8c3ea589ab30cb6f24fe`;
step 600
`03896f06f0e89a9096a9e948c6a4e3728c440814649cdbc9b4c7cc5edc9502bc`.

The last milestone is the strongest direct/end-to-end result, but remains
rejected because its 40/67 required parsed calls and 25/67 correct-tool calls
are below the Base's 51/67 and 36/67. No release, final tag, approval, or
Phase 9 authorization is justified.
