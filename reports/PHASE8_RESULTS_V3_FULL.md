# Phase 8 V3 Full Recovery Result

## Verdict

**REJECTED — NOT APPROVED.** The v3 full run preserves Base validation and
improves several tool metrics, but it does not meet the mandatory direct
mathematical capability gate: held-out direct-answer correctness is 0/160 at
every meaningful full-run checkpoint.

## Provenance

- run ID: `phase8-sft-v5-full-recovery`;
- clean pushed source commit: `b8ff443c5d89fdb43ede91314bfc66309d443bed`;
- parent Phase 7 SHA-256: `2e8098ab3a5db3c59a82fad19af2050154637fbe0628f3f6b0ca932d6cb6ea60`;
- selection identity: `1fbcaf6afe623529badf2c2e2fd7faf5e541928e239359152b70ba2973681f1e`;
- representation identity: `24502369b203ec3c7a17d57cb64bb358aa4cdcdaced58624662f527505a0ed47`.

The retained run log is `experiments/phase8-sft-v5-full-recovery/train_log.jsonl`.
No checkpoint is added to a release, tag, or approval record.

## Milestone comparison

| Step | Base loss | Delta vs. 0.606183 | Direct held-out | Tool end-to-end | Exact arguments | Fabrication attempts |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.606183 | +0.000000 | 0/160 | 0/67 | 6/67 | 110/271 |
| 300 | 0.615541 | +0.009357 | 0/160 | 6/67 | 6/67 | 94/271 |
| 600 | 0.613397 | +0.007213 | 0/160 | 3/67 | 3/67 | 91/271 |
| 900 | 0.612462 | +0.006278 | 0/160 | 6/67 | 6/67 | 83/271 |

All checkpoints meet the preregistered +0.05-nat Base-regression ceiling.
None meets the direct-capability requirement. Tool success is improved over
the zero baseline but is not sufficient to offset the direct failure.

## Frozen-surface context

The best frozen Phase 4 math result is 2/215 at step 300; calibration reaches
2/130 at step 900; adversarial behavior reaches 40/195 at all trained
milestones. These limited improvements do not establish general mathematical
instruction following. Fixed familiar prompts demonstrate learned responses
(for example, `2x + 3 = 11` → `<final>4`), while independent held-out prompt
families remain unsolved. That discrepancy is the central generalization
failure.

## Checkpoint preservation

- `step_000300.pt`: `694cc1123240de41af058ca2c8a94bd6a5d19cdcebf149bfafd5709af8a42bc2`;
- `step_000600.pt`: `11326c090518d789ecdcd98d37303885bee9d621c4e4dc2d843457d5d922a656`;
- `step_000900.pt`: `a5c875c3c99df3be681ba48956ba856a239bd67d7a0b987fe9e57fd525d34aed`;
- final state: `step_000900_final.pt`,
  `a27787063740911c34c6776b11611bcd44fc85001d446670a7adefdd4f6221ab`.

These are ignored local research artifacts pending any separately authorized,
clearly labeled remote evidence-preservation release. They must not be
presented as an approved Phase 8 checkpoint.

## Next remediation requirement

The next corpus version must add genuinely independent mathematical prompt
families and structural constructions, not only instruction-frame rewrites of
the same frozen parent prompts. It must then repeat bounded preflight,
complete milestones, Base regression, and full held-out evaluation under new
run IDs before a further approval decision.
