# Phase 8 V6 Balanced Curriculum LR=2e-4 Preflight

## Verdict

**Rejected. Do not extend this candidate.** The higher learning rate produces
the first nonzero direct and end-to-end tool scores in the independent
curriculum series, but it violates two non-negotiable retention gates: Base
regression is +0.118104 nats (ceiling: +0.050000) and adversarial correctness
falls from the frozen Base's 36/195 to 1/195.

## Provenance

- clean source commit: `8f6a1ca`;
- run ID: `phase8-sft-v8-balanced-curriculum-lr2e-4-preflight`;
- learning rate: `2.0e-4`; 300 optimizer steps;
- same 30,000-record balanced representation as V5 (train representation
  SHA-256 `f6650269b440685fdb74b2910ac56be0eb7727230c88518be551c78cea528fce`);
- checkpoint SHA-256:
  `524aee0d009580efdffc311627b51701afd1e83cf58c26240dddaa484643bb83`
  (local preflight evidence only);
- Base regression measured over the fixed 78,244 unmasked target tokens.

## Results

| Metric | Frozen Phase 7 | V6 300-step |
| --- | ---: | ---: |
| Base validation loss | 0.606183 | 0.724288 |
| Base loss delta | — | **+0.118104** |
| Math correctness | 1/215 | 4/215 |
| Calibration correctness | 0/130 | 9/130 |
| Adversarial correctness | 36/195 | **1/195** |
| Direct held-out correctness | 0/160 | 5/160 |
| Tool end-to-end success | 0/67 | 4/67 |
| Exact expected arguments | 6/67 | 4/67 |
| Unnecessary tool calls | 59/204 | 20/204 |
| Fabricated-result attempts | 110/271 | 44/271 |

This is a useful trade-off diagnosis, not an approval candidate: a rate high
enough to make the short supervised outputs appear in held-out generations
also causes catastrophic safety/retention damage at 300 steps. No release,
Phase 8 approval, or Phase 9 authorization follows from this result.

## Next test

Do not increase the global SFT rate further. A subsequent experiment must
seek the same answer-token learning signal while constraining parameter drift,
for example by using a deterministic replay mixture of frozen Base examples
with independently supervised instruction records. It must be fully
versioned and pass the existing Base and adversarial gates in a bounded run
before any full run is considered.
