# Phase 8 Recovery Preflight V3

## Run identity

- run ID: `phase8-sft-v4-recovery`;
- clean source commit at start: `b886db0ffc798f8c90bdf016d7f53a6ec144250b`;
- parent checkpoint: `checkpoints/phase7-full-v2/step_007483_final.pt`;
- parent SHA-256: `2e8098ab3a5db3c59a82fad19af2050154637fbe0628f3f6b0ca932d6cb6ea60`;
- SFT selection identity: `1fbcaf6afe623529badf2c2e2fd7faf5e541928e239359152b70ba2973681f1e`;
- SFT representation identity: `24502369b203ec3c7a17d57cb64bb358aa4cdcdaced58624662f527505a0ed47`.

This was a bounded 50-step preflight, not an approval candidate. Its log is
`experiments/phase8-sft-v4-recovery/train_log.jsonl`; the retained checkpoint
is ignored from Git by policy and hashes to
`a737d399180ce04f6f1143472aac194166cdd1e382839338db36fa02f5dbb07b`.

## Efficiency and Base preservation

The GPU run used 5,004,032 parameters, 39,000 train records, 644,369
supervised corpus tokens, 5.05% deterministic dynamic-padding overhead, and
489.8 MiB peak CUDA allocation. The rerun unmasked Base-validation loss was
0.6111139743 nats over 78,244 targets, a +0.0049305408-nat change from the
0.6061834334-nat Phase 7 baseline. It passes the +0.05 preflight gate.

## Complete evaluator result

| Metric | Phase 7 Base | 50-step preflight |
| --- | ---: | ---: |
| Phase 4 math | 1/215 | 1/215 |
| Phase 4 calibration | 0/130 | 0/130 |
| Phase 4 adversarial | 36/195 | 40/195 |
| Direct answer correctness | 0/160 | 1/160 |
| Tool end-to-end success | 0/67 | 10/67 |
| Exact expected tool arguments | 6/67 | 10/67 |
| Unnecessary tool calls | 59/204 | 54/204 |
| Fabricated-result attempts | 110/271 | 105/271 |

The run establishes that the corrected representation and low learning rate
can preserve Base validation while producing nonzero direct and tool metrics.
One direct case is not a meaningful direct-capability result, so this does not
meet the Phase 8 acceptance gate and must not be selected or released.
