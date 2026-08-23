# Phase 4 Final Approval

## Verdict

**APPROVED.**

Phase 4 — Dataset and Evaluation Freeze: **COMPLETE**  
Phase 5 — Smoke Pretraining: **AUTHORIZED — NOT STARTED**

## Source and review identity

- Approved foundation: `phase-3-tools` (`1cb070945dd1f968f0dd10e6ceb915e0d24fac0a`).
- Original Sonnet review candidate: `phase-4-review-candidate`
  (`84ce0762b71d8feb00636ab7fcb9eda45a99c767`).
- Final approved source: resolve `phase-4-dataset^{commit}`; the tag is the
  immutable source identity because a commit cannot correctly self-reference
  its own resulting SHA in this file.

## Dataset identity

| Field | Final value |
|---|---:|
| Dataset | `juniper-math-dataset-v1` |
| Whole-dataset identity | `bf9933f032a58b4eb618b32156783b8563097a5fc1c0ef26be4f76445128d25a` |
| Examples | 1,629,078 |
| Tokens | 62,421,215 |
| Train | 1,466,970 examples / 56,209,616 tokens |
| Validation | 81,094 examples / 3,103,913 tokens |
| Test | 81,014 examples / 3,107,686 tokens |
| Shards | 34 |
| Structural prompt forms | 1,965 |

The final build rejected 1,267,816 structurally repeated candidates and
259,912 diversity-cap candidates. No accepted family exceeded the configured
15% corpus token ceiling. Tool-error coverage is 1,553 examples (280,130
tokens) across real runtime division-by-zero, domain, resource-limit,
unsupported-unit, and unsupported-operation outcomes.

## Evaluation identity

| Suite | Cases | SHA-256 |
|---|---:|---|
| `phase4-math-v2` | 215 | `b2422402313691a0dfec8366190d55c51a6155213b9739633eeda3454bb4b83b` |
| `phase4-tool-use-v2` | 185 | `bf26d480f0eb1517740d3894d367064f088dc4293d3706fc377f3aec2ed9ab77` |
| `phase4-calibration-v2` | 130 | `6e975117598ef85701c2b1c13075fbabc6ff250bf3223c49c87d024e95cc33a5` |
| `phase4-adversarial-v2` | 195 | `8ab94b3510c30a07ada85ffff3f99de3be7f6b934abcab9da3fc4ffefc70b7da` |

All v2 cases use the dedicated `phase4_evaluation_only` constructor and
held-out family/template identities. They do not call the training registry.
The v1 suite files are retained as historical evidence and are not the active
evaluation surface.

## Verification

- Frozen Phase 1--3 paths: no diff from `phase-3-tools`.
- Dataset schema validation: pass for all 1,629,078 records.
- Dataset ground-truth verification: deterministic answers recomputed and
  tool traces re-executed against the real runtime.
- Artifact hash verification, configuration, manifests, dependency check,
  CPU model smoke, tokenizer validation, tool self-test, eval validation,
  pytest, Ruff, formatting, and mypy: pass.
- The fresh-clone recovery test is recorded after tag creation in the final
  commit's verification log.

## Review history

The original candidate was not approved. Terra findings F-01 (unenforced
family policy), F-02 (operand-substitution duplicate blind spot), and F-03
(seed-shifted rather than independent evaluations) were remediated as
documented in `TERRA_PHASE4_REVIEW.md` and `PHASE4_REMEDIATION.md`.
