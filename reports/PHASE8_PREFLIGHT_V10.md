# Phase 8 V10 Safety/Tool LR=2e-4 Preflight

## Verdict

**Rejected.** Raising the rate while retaining the independent safety and
runtime-tool curricula stays inside the Base-loss tolerance, but fails the
direct and correct-tool improvement gates and increases unnecessary calls.

## Provenance

- clean source commit: `481c99a`;
- run ID: `phase8-sft-v13-safety-tool-lr2e-4-preflight`;
- 300 steps at `2.0e-4`; same 39,000-trajectory V9 mixture;
- checkpoint SHA-256:
  `56e2fe4bb9acbb8e4050be4a08984298b90cf8030fc0cfbc116b4cd99598f3a9`.

## Results

| Metric | Frozen Phase 7 | V10 300-step |
| --- | ---: | ---: |
| Base validation loss | 0.606183 | 0.652321 |
| Base loss delta | — | +0.046138 |
| Adversarial correctness | 36/195 | 102/195 |
| Direct held-out correctness | 0/160 | 3/160 |
| Valid parsed calls | 75/271 | 52/271 |
| Correct tool | 36/67 | 19/67 |
| Exact expected arguments | 6/67 | 7/67 |
| Tool end-to-end success | 0/67 | 7/67 |
| Unnecessary tool calls | 59/204 | 27/204 |
| Fabricated-result attempts | 110/271 | 54/271 |

The Base and adversarial results do not rescue the candidate: direct success
remains marginal, valid/correct tool calls are below the Base, and unnecessary
calls increase. The configuration is rejected and does not authorize Phase 9.
