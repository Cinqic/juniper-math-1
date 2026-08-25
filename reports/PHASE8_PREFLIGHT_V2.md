# Phase 8 Recovery Preflight V2

## Scope and status

This is a fresh baseline measurement from the approved Phase 7 release, not
evidence of a Phase 8 candidate. The source tree was clean at commit
`6fc8020c`, and the release asset was freshly verified against SHA-256
`2e8098ab3a5db3c59a82fad19af2050154637fbe0628f3f6b0ca932d6cb6ea60`.

The locally reconstructed frozen parent corpus contains 1,629,078 examples
across 34 shards and has identity
`bf9933f032a58b4eb618b32156783b8563097a5fc1c0ef26be4f76445128d25a`.

## Phase 7 baseline, rerun with evaluator v2

The deterministic 2,000-example unmasked Base-validation sample is
**0.6061834334 nats** over 78,244 shifted non-padding targets. The approval
ceiling for a candidate is therefore **0.6561834334 nats** (+0.0500000000).

| Surface | Result |
| --- | --- |
| Frozen Phase 4 math | 1/215 (0.47%) |
| Frozen Phase 4 calibration | 0/130 (0.00%) |
| Frozen Phase 4 adversarial | 36/195 (18.46%) |
| Frozen Phase 4 tool-format valid calls | 90/185 (48.65%) |
| Phase 8 routing correct | 197/271 (72.69%) |
| Phase 8 exact expected arguments | 6/67 (8.96%) |
| Phase 8 tool end-to-end success | 0/67 (0.00%) |
| Phase 8 concrete final-answer correctness | 0/217 (0.00%) |
| Phase 8 direct-answer correctness | 0/160 (0.00%) |
| Phase 8 unnecessary calls | 59/204 (28.92%) |
| Phase 8 fabricated-result attempts | 110/271 (40.59%) |

The direct-answer and end-to-end baselines are both zero. A recovery candidate
must materially exceed both while retaining Base regression within +0.050 nats.

## Derived-representation preflight

The current v3 selection contains 24,000 train examples and 540,713 supervised
train tokens. Its separate identities are:

- selection identity: `1fbcaf6afe623529badf2c2e2fd7faf5e541928e239359152b70ba2973681f1e`;
- representation identity: `ed7e71eab24e78e32c38bf35ea19997261640f15bc1f9360cd51059b2867168f`.

Deterministic length-bucketed dynamic padding measures 4.64% padding for the
first shuffled epoch at micro-batch size 8, down from the historical fixed
sequence design's reported roughly 74%. No conversations are packed together.

## Design implication

The equal 1,000-per-category selector is not sufficient for the recovery
candidate: direct mathematical tasks need substantially more supervised-token
weight and broader independent wording, while the low-frequency safety/tool
error categories remain represented but must not dominate central mathematical
instruction. No full SFT candidate has been run under this plan.
