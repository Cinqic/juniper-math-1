# Phase 1 Terra Remediation

**Date:** 2026-08-22  
**Candidate:** `phase-1-review-candidate` / `3ec0683c16f600cd4387f864d77500062cf7dca6`

| Finding | Root cause | Change | Regression coverage | Result |
|---|---|---|---|---|
| T1 — partial checkpoint restore | Restore operations mutated user-supplied objects sequentially without rollback. | Validate required payload fields first; snapshot model/optimizer/scheduler/scaler/RNG; roll back on any restore failure and raise `CheckpointError`. | Corrupt/incomplete model state proves parameter values remain unchanged; missing RNG field is rejected before restoration. | PASS |
| T2 — ambiguous invalid inputs | Shape-only mask/label checks deferred malformed values to framework internals. | Require masks to be bool/integer 0/1 and labels to be integer `-100` or vocabulary IDs, on the input device. | Invalid mask dtype/value and invalid label dtype/range tests. | PASS |
| T3 — insufficient independent reference coverage | Candidate lacked a standalone manual attention numerical oracle. | Added audit tests with an independent float64 RoPE calculation and manual scaled-dot-product causal+padding attention. | `tests/test_terra_phase1_audit.py`. | PASS |

No architecture dimensions, parameter target, weight tying, position encoding, bias policy, or
dropout policy changed. Focused checks passed before full regression.
