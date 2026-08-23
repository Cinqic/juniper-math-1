# Phase 4 Remediation Record

This record preserves the history of the original Sonnet candidate rather
than rewriting it as if it had passed independent review.

## F-01 — family share control

The configured cap is now enforced while building, against accepted token
counts (the resource the training budget consumes). Corpus statistics now
record tokens by generator/family so the policy is auditable.

## F-02 — structural repetition control

`structural_normalize()` replaces numeric literals before shingling.
`NearDeduplicator` maintains a deterministic, family-scoped repeat ceiling
and has a regression test demonstrating that `12 plus 13` and `87 plus 642`
are recognized as the same prompt shape. The cap is intentionally bounded:
it reduces unbounded numeric substitution without pretending that a corpus
can learn arithmetic from only one operand pair.

## F-03 — evaluation independence

The v1 evaluation files remain historical artifacts. The active v2 suites
are generated only by `dataset.eval_isolated`, which has no dependency on
the training generator registry. Every v2 case uses the evaluation-only
generator identity `phase4_evaluation_only`, a `held_out_*` family, and an
`eval_only_*` template. A regression test prevents reuse of any training
generator identity.

## Rebuild consequence

The corpus and manifest metadata must be regenerated after these changes.
The final dataset identity, evaluation hashes, fresh-clone result, and
approval decision are recorded only in `PHASE4_FINAL_APPROVAL.md` after the
remaining verification gates complete.
