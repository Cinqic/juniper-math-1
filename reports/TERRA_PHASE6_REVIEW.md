# Terra Phase 6 Independent Review

## Verdict: APPROVED WITH REMEDIATION

Reviewed from a clean clone of `https://github.com/Cinqic/juniper-math-1`.
`phase-5-smoke` resolves to `73792c04f365c6f139a979f6950fa87be2af5d76`.
`phase-6-pilot-candidate` and remote `main` resolved to
`054e837ff0f61649b52e3fd7c6e67f29ba28fb6f`; its sole post-implementation
commit is legitimate candidate-SHA bookkeeping. No unreviewed remote commits
were present.

Frozen architecture, tokenizer, dataset configuration, tool protocol,
evaluation suites, and their manifests/hash records were unchanged from Phase
5. A clean source rebuild reproduced the frozen dataset manifest and identity;
all schema, ground-truth, and contamination checks passed.

The candidate's local pilot evidence did not reproduce because it selected
unverified ignored data shards. I corrected this fail-open path and regenerated
the experiment on verified shards. The remediated pilot is stable (8.3801 to
0.9777 validation loss), uses 5,051,139 loss-bearing tokens, evaluates every
frozen case at every milestone, and demonstrates structural-format learning
only—not mathematical or tool-use capability. See `PHASE6_RESULTS.md`.

Resume is tolerance-equivalent but not bitwise/generation-identical on CUDA;
Phase 7 must retain numerical resume gates and make no stronger claim.
Phase 5 smoke training/evaluation/checkpoint/restore/resume was rerun and
remains bitwise equivalent.

The learning-rate evidence supports `6e-4` as a conservative initial point,
but does not establish it as optimal because `1e-3` was faster in the short
screen. Phase 7 should run one bounded 6e-4/8e-4/1e-3 preflight before the
serious budget. `tool_error` is not scarce in the verified frozen train split.

Important limitation: this pilot is reproducible but disposable. Phase 7 starts
from fresh random initialization and must not inherit its checkpoint.
