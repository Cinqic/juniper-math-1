# Evaluations

## Purpose of the Phase 0 fixed suite

[`evals/phase0_v1.json`](../evals/phase0_v1.json) is a frozen baseline suite
authored **before any model exists**. Its Phase 0 purpose is narrower than a
real benchmark: it validates schema integrity, ID uniqueness, category
coverage, and deterministic-reference structure so that later-phase scoring
infrastructure has a stable, versioned target to build against. It is
intentionally compact (22 cases, one per category) and is **not** the final
exhaustive evaluation suite — later phases will add depth per category.

## Schema

Each case has: `id` (unique string), `category` (one of the fixed category
set below), `difficulty` (`trivial`/`easy`/`medium`/`hard`), `prompt`,
`expected_behavior`
(`answer`/`refuse_ambiguous`/`request_clarification`/`refuse_unsupported`/`flag_missing_information`/`flag_incorrect_answer`/`invoke_tool`),
`expected_answer` (numeric, string, boolean, or `null` for non-answer
behaviors), `tolerance` (numeric or `null`), `tool_required` (bool),
`verification` (see below), `provenance`, `notes`.

`expected_behavior` values: `answer`, `request_clarification`,
`refuse_unsupported`, `flag_missing_information`, `flag_undefined`,
`flag_incorrect_answer`, `invoke_tool`.

### `tolerance` semantics

`tolerance` is an **absolute** bound, never relative. A case passes when:

```
|computed - expected_answer| <= tolerance
```

`tolerance: 0` and `tolerance: null` both mean exact equality. All comparison
arithmetic uses `fractions.Fraction`, so decimals and fractions are compared
exactly rather than through binary floating point — `2.5 + 3.75` is exactly
`6.25` here. This was previously unspecified (Opus 5 review finding F-09),
which would have forced later scoring code to guess.

### `verification`

Every case carries a structured verification block:

```json
"verification": {
  "mode": "deterministic",
  "expression": {"op": "mul", "args": [84317, 9926]}
}
```

`mode` is either `deterministic` (the answer is recomputed from `expression`
and compared to `expected_answer`) or `semantic` (`expression` is `null` and
`expected_answer` must be `null` — ambiguity, missing information, undefined
mathematics, and unsupported requests have no single correct value).

Expressions are a closed structure, never source code. The evaluator in
`src/juniper_math/verification.py` walks explicit JSON nodes and dispatches on
an allowlist of exactly eight operations — `add`, `sub`, `mul`, `div`, `neg`,
`pow`, `percent_of`, `equals`. It never calls `eval`, `exec`, or `compile`, and
prompts are never executed. Unknown operations raise rather than being ignored.

## Categories covered

arithmetic_interpretation, operator_precedence, negative_values, decimals,
fractions, percentages, ratios, proportions, basic_algebra, units, currency,
scientific_notation, word_problem, estimation, ambiguity,
missing_information, undefined_operation, unsupported_capability,
tool_required, direct_answer, incorrect_supplied_answer, error_recognition.

18 of the 22 cases are `deterministic`; the 4 semantic cases are `amb-001`,
`miss-001`, `undef-001`, and `unsup-001`.

## Freezing and versioning

The suite's `suite_version` (`0.1.1`) and its SHA-256 (recorded in
`manifests/artifacts.yaml`) together define its frozen identity. **Editing a
frozen suite's cases without bumping `suite_version` and regenerating the
hash is not permitted.** A change to expected answers, categories, or case
count requires a new version.

## Validation

```bash
python -m juniper_math evals validate   # schema + deterministic ground truth
python -m juniper_math evals verify     # ground truth only
```

**Schema validation** checks: valid JSON, required fields present per case,
unique IDs, known categories/difficulties/expected_behaviors, `tool_required`
is boolean, `tolerance` is numeric or null, `verification` well-formed.

**Deterministic ground-truth validation** recomputes each `deterministic`
case's answer from its `verification.expression` and compares it against the
recorded `expected_answer` within `tolerance`.

### Why ground-truth validation exists

Suite version `0.1.0` had schema validation only, while the documentation
claimed answers had been "hand-verified". That claim was false. Case
`tool-001` recorded `84317 * 9926` as `837042742`; the correct product is
`836930542`, an error of 112,200. Schema validation passed it, the test suite
passed it, and CI passed it, because nothing checked the arithmetic. The
defect was found by the Opus 5 independent Phase 0 review (finding F-01).

Version `0.1.1` corrects the answer and adds `verification` metadata so the
error class is now caught automatically. `tests/test_verification.py`
re-injects the original wrong value and asserts that validation rejects it.

Ambiguity, missing-information, undefined, and unsupported cases deliberately
have `expected_answer: null` and `mode: semantic`, because no single answer is
correct. Those are checked for classification consistency, not recomputed.

## Version history

| Version | Change |
|---|---|
| `0.1.0` | Initial frozen baseline (22 cases). Contained invalid ground truth in `tool-001`. |
| `0.1.1` | Corrected `tool-001` (`837042742` → `836930542`); reclassified `undef-001` from `flag_missing_information` to the new `flag_undefined` behavior; removed the unused `refuse_ambiguous` behavior; added structured `verification` metadata to all 22 cases. |
