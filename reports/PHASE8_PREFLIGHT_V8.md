# Phase 8 V8 Safety-Replay Preflight

## Verdict

**Advance to a bounded, milestone-evaluated 600-step run.** This is the first
recovery mixture to meet the Base-regression gate, retain strong adversarial
behavior, and show nonzero held-out direct and end-to-end tool success. Its
direct score is still too small for approval, so this is evidence for a
longer bounded run—not an approval claim.

## Provenance

- clean source commit: `653a004`;
- run ID: `phase8-sft-v10-safety-replay-preflight`;
- 300 optimizer steps at `1.5e-4`;
- 34,000 masked SFT records: the selected parent data, 6,000 independent
  direct records, and 4,000 independent answerless safety records; plus
  2,000 deterministic frozen-Base replay records;
- dynamic padding fraction: 0.0593; loss-bearing tokens: 668,260;
- checkpoint SHA-256:
  `ae189a1d2407e9e793cfea914c4144fb56ba77ad4c0eb64259a74a14d01eb4b2`.

## Results

| Metric | Frozen Phase 7 | V8 300-step |
| --- | ---: | ---: |
| Base validation loss | 0.606183 | 0.636475 |
| Base loss delta | — | +0.030291 |
| Math correctness | 1/215 | 3/215 |
| Calibration correctness | 0/130 | 2/130 |
| Adversarial correctness | 36/195 | 138/195 |
| Direct held-out correctness | 0/160 | 2/160 |
| Tool end-to-end success | 0/67 | 6/67 |
| Exact expected arguments | 6/67 | 6/67 |
| Unnecessary tool calls | 59/204 | 2/204 |
| Fabricated-result attempts | 110/271 | 25/271 |

The next run keeps this mixture and evaluator unchanged, evaluates separate
milestones, and stops if the Base-loss delta exceeds +0.05 or if adversarial
retention materially reverses. Approval remains contingent on a meaningfully
larger direct capability result and complete repository/remote evidence.
