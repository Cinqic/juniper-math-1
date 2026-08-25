# Phase 8 V7 Frozen-Base Replay Preflight

## Verdict

**Rejected. Do not extend this candidate.** Deterministic unmasked replay of
the frozen Base improves direct held-out correctness while sharply reducing
the prior Base-loss regression, but it still misses the pre-registered Base
ceiling and fails the adversarial retention gate.

## Provenance

- clean source commit: `acf70b6`;
- run ID: `phase8-sft-v9-replay-balanced-preflight`;
- 300 optimizer steps at `2.0e-4`;
- 30,000 masked SFT trajectories plus 2,000 fixed-stride frozen-Base replay
  trajectories; replay representation SHA-256:
  `d76fcdafeb2533eaffd422b339ca1963536857a64c626479c0d7e039a546e97b`;
- combined SFT representation identity:
  `d74208802963b4d761e0a3b74d93dea736d7acbb729181975cd56ca2fdbe5233`;
- checkpoint SHA-256:
  `387cd0aad056bc44e2b888c8658c47a8f416190aae44dce9376ad639ac0d6bfa`.

## Results

| Metric | Frozen Phase 7 | V7 replay |
| --- | ---: | ---: |
| Base validation loss | 0.606183 | 0.657511 |
| Base loss delta | — | **+0.051328** |
| Math correctness | 1/215 | 8/215 |
| Calibration correctness | 0/130 | 33/130 |
| Adversarial correctness | 36/195 | **3/195** |
| Direct held-out correctness | 0/160 | 12/160 |
| Tool end-to-end success | 0/67 | 4/67 |
| Exact expected arguments | 6/67 | 4/67 |
| Fabricated-result attempts | 110/271 | 35/271 |

The result is strictly rejected: +0.051328 exceeds the +0.050000 Base-loss
limit, and 3/195 adversarial correctness is a material safety regression.
No release, approval, or Phase 9 authorization is warranted.
