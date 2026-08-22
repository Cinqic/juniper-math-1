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
`provenance`, `notes`.

## Categories covered

arithmetic_interpretation, operator_precedence, negative_values, decimals,
fractions, percentages, ratios, proportions, basic_algebra, units, currency,
scientific_notation, word_problem, estimation, ambiguity,
missing_information, undefined_operation, unsupported_capability,
tool_required, direct_answer, incorrect_supplied_answer, error_recognition.

## Freezing and versioning

The suite's `suite_version` (`0.1.0`) and its SHA-256 (recorded in
`manifests/artifacts.yaml`) together define its frozen identity. **Editing a
frozen suite's cases without bumping `suite_version` and regenerating the
hash is not permitted.** A change to expected answers, categories, or case
count requires a new version.

## Validation

```bash
python -m juniper_math evals validate
```

Checks: valid JSON, required fields present per case, unique IDs, known
categories/difficulties/expected_behaviors, `tool_required` is boolean,
`tolerance` is numeric or null.

Where an answer is deterministically checkable (arithmetic, algebra,
conversions), it was verified by hand against the stated operations before
freezing. No uncertain ground truth was fabricated; ambiguity/missing-info
cases deliberately have `expected_answer: null` because no single answer is
correct.
