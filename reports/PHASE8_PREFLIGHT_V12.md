# Phase 8 V12 Staged Tool-Call Preflight

## Verdict

**Advance only to a 600-step bounded milestone run.** Separating
call-construction supervision eliminates fabricated-result attempts and
improves all-case call parsing, but it has not yet exceeded the Base on
required-tool parsing/correct-tool selection or reached meaningful direct
accuracy. A longer bounded run is justified; approval is not.

## Provenance

- clean source commit: `905c0c0`;
- run ID: `phase8-sft-v15-call-stage-preflight`;
- 300 steps at `1.5e-4`;
- 37,000 SFT records: selected parent data, independent direct/safety data,
  and 3,000 runtime-verified call-only records; plus 2,000 frozen-Base replay
  trajectories;
- representation identity:
  `9057a0d774fbf45546c1b40af7ca05e893ca4cf8d75190413eb849151025d957`;
- checkpoint SHA-256:
  `11b0732414c5f4e214d87f4ccfd6830ee01974f9561687bbce5d69ed2b88f954`.

## Results

| Metric | Frozen Phase 7 | V12 300-step |
| --- | ---: | ---: |
| Base loss delta | — | +0.046618 |
| Direct held-out correctness | 0/160 | 3/160 |
| Required tool calls parsed | 51/67 | 40/67 |
| Correct tool | 36/67 | 25/67 |
| Exact expected arguments | 6/67 | 8/67 |
| Tool end-to-end success | 0/67 | 8/67 |
| Fabricated-result attempts | 110/271 | **0/271** |
| Adversarial correctness | 36/195 | 85/195 |

The next run preserves every data and optimizer choice, evaluates 0/200/400/600,
and rejects any milestone that exceeds +0.05 Base loss or fails to improve the
required-tool metrics. It is not authorized to create a release or tag.
