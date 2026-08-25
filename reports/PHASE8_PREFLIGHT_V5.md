# Phase 8 V5 Balanced Independent-Curriculum Preflight

## Verdict

**Do not advance this mixture to a longer or full run.** Reducing the
independent-direct curriculum to 6,000 records restored some tool-call
emission, but it produced neither a held-out direct answer nor an end-to-end
tool success. It therefore does not meet the concrete mixture gate set after
the v4 rejection.

## Provenance

- clean source commit: `a33efcc69c224ab4d164254c97c680bff20dbd2e`;
- run ID: `phase8-sft-v7-balanced-curriculum-preflight`;
- SFT representation identity:
  `f6650269b440685fdb74b2910ac56be0eb7727230c88518be551c78cea528fce`
  (train; validation representation identity:
  `8dabbe9e259d8b6eb74815351b96b5c0e9796a8c09b92cbc148db9cc44a71d59`);
- checkpoint SHA-256:
  `2d74ef9d1716ccab45dd38a808d4d5198159367206839550d5afa70fc142e848`
  (local preflight evidence only);
- 50 optimizer steps; 30,000 training records: 14,000 selected direct
  records, 6,000 independent direct records, and 10,000 tool/safety records;
- dynamic-batch padding fraction: 0.050969; Base regression set: 78,244
  unmasked target tokens.

## Results

| Metric | Frozen Phase 7 | V5 50-step |
| --- | ---: | ---: |
| Base validation loss | 0.606183 | 0.609217 |
| Base loss delta | — | +0.003033 |
| Math correctness | 1/215 | 2/215 |
| Calibration correctness | 0/130 | 2/130 |
| Adversarial correctness | 36/195 | 39/195 |
| Direct held-out correctness | 0/160 | 0/160 |
| Tool end-to-end success | 0/67 | 0/67 |
| Exact expected arguments | 6/67 | 9/67 |
| Unnecessary tool calls | 59/204 | 47/204 |
| Fabricated-result attempts | 110/271 | 87/271 |

The Base loss remains within the 0.05 regression ceiling, but capability
gates are conjunctive: the zero direct and zero end-to-end numerators rule
out escalation. The candidate stays rejected; no Phase 8 approval, release,
or Phase 9 authorization is implied.

## Next test

The common 50-step failure mode is sparse supervised loss exposure over long
instruction prefixes. The next ablation should keep independent examples but
increase their answer-token density and use deterministic, token-accounted
sampling rather than only changing their record count. It must first pass a
bounded preflight with nonzero direct and end-to-end tool numerators.
