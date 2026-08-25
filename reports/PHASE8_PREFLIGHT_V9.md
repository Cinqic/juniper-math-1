# Phase 8 V9 Explicit Runtime-Tool Preflight

## Verdict

**Rejected.** Adding 3,000 independently re-identified, runtime-executed
tool trajectories improves V11's immediate tool-call counts, but not enough
to exceed the frozen Base. It also weakens the direct/safety balance. It is
not a candidate for a full approval run.

## Provenance

- clean source commit: `21bf09d`;
- run ID: `phase8-sft-v12-explicit-tool-preflight`;
- 300 steps at `1.5e-4`;
- 37,000 masked SFT records (including 3,000 real-runtime derived tool
  trajectories) plus 2,000 frozen-Base replay records;
- checkpoint SHA-256:
  `fd28bcded19a3e789b92febb25282094dde43d301ca14d21daef511670711ff0`.

## Results

| Metric | Frozen Phase 7 | V9 300-step |
| --- | ---: | ---: |
| Base validation loss | 0.606183 | 0.635674 |
| Base loss delta | — | +0.029491 |
| Adversarial correctness | 36/195 | 55/195 |
| Direct held-out correctness | 0/160 | 3/160 |
| Tool end-to-end success | 0/67 | 8/67 |
| Valid parsed calls | 75/271 | **50/271** |
| Correct tool | 36/67 | **25/67** |
| Exact expected arguments | 6/67 | 8/67 |
| Fabricated-result attempts | 110/271 | 60/271 |

Base retention is within the +0.05 limit, and tool end-to-end/non-fabrication
signals improve relative to the Base. But the pre-registered tool validity and
correct-tool requirements are conjunctive: both remain below the Base. No
release, approval, or Phase 9 authorization follows.
