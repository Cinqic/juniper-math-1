# Terra Phase 4 Independent Review

## Initial verdict

**CHANGES REQUIRED.** The Phase 4 candidate at `84ce076` and its subsequent
documentation/recovery update at `1b62597` were not approved as submitted.

## Verified strengths

- The Phase 1--3 frozen paths have no diff from `phase-3-tools`.
- Deterministic examples are recomputed through the closed operation
  allowlist; tool examples retain real Phase 3 runtime traces.
- The source manifest supports the 100% synthetic-data claim.
- Empty shard directories fail honestly rather than reporting a false pass.

## Findings

### F-01 — HIGH — unenforced family cap

`config/dataset.yaml` declared `max_family_share_of_corpus`, but
`build_dataset()` never consumed it. The original recorded corpus included
`arithmetic_core/arith_two_operand` at 411,649 examples. A declared control
that does nothing cannot support a diversity claim.

### F-02 — HIGH — lexical-only near-duplicate control

The original five-word Jaccard comparison did not normalize values, so
operand substitutions could evade it. The original full build reported zero
near-duplicate rejections despite large-scale template generation.

### F-03 — HIGH — evaluation suites were not independent

The original suite builder called `build_registry()` and selected the same
training generators with another seed namespace. This prevented instance
identity collisions, but did not provide template or generator independence.

## Independent checks performed

- Fetched canonical `origin/main` and compared frozen Phase 1--3 paths to
  `phase-3-tools` (no diff).
- Reproduced baseline configuration, manifest, model, tokenizer, tool, eval,
  static-analysis, and test checks.
- Rebuilt the dataset from repository state; directly inspected corpus-level
  family and structural-template statistics.
- Added regression tests for operand substitution and evaluation identity
  isolation.

## Current review state

Remediation is in progress. This report deliberately does not grant approval;
see `PHASE4_REMEDIATION.md` for the applied changes and the final approval
report only after all recovery gates pass.
