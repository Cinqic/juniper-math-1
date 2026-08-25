# Phase 8 V4 Independent-Curriculum Preflight

## Verdict

**Do not advance this mixture to a full run.** The 50-step independent-direct
curriculum preflight preserves Base validation but is worse than the v3
preflight on end-to-end tool use and does not produce any held-out direct
correctness.

## Provenance

- clean source commit: `cdddc11`;
- run ID: `phase8-sft-v6-curriculum-preflight`;
- representation identity:
  `ebb7039bac1386e1108765f23c71585d4a4a0202e09b6242fe20fdbb86324f3c`;
- checkpoint SHA-256: `1093d42aba4baed564fa92afe45a43472377aca64aaec4d8a48719e6a0d81f4c`
  (local preflight evidence only);
- train composition: 51,000 records including 12,000 independent direct
  records.

## Results

| Metric | V3 50-step | V4 50-step |
| --- | ---: | ---: |
| Base loss delta | +0.004931 | +0.002265 |
| Direct held-out correctness | 1/160 | 0/160 |
| Tool end-to-end success | 10/67 | 0/67 |
| Exact expected arguments | 10/67 | 8/67 |
| Unnecessary tool calls | 54/204 | 21/204 |
| Fabricated-result attempts | 105/271 | 48/271 |

The result supports a mixture diagnosis: the extra direct curriculum teaches
the model to avoid unnecessary calls and fabricated results, but at this
weight it suppresses required tool invocation. It does not justify a full
training run or an approval claim.

## Next test

Keep the independent families but reduce their effective training weight and
separate direct and tool continuation sampling through a deterministic,
token-accounted mixture. A follow-up preflight must demonstrate nonzero direct
correctness without losing nonzero tool end-to-end completion.
