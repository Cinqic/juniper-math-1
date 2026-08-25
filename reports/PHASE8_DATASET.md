# Phase 8 Dataset — `juniper-math-sft-v1`

## Source and construction

Derived entirely from the frozen `juniper-math-dataset-v1` train/validation
splits (never test; never any of the four frozen `evals/phase4_*_v2.json`
suites, which were reserved out of the train split at Phase 4 build time).
No new generator module was written — every category in
`juniper_math.dataset.schema.VALID_CATEGORIES` already had verified
examples covering direct answers, tool use, error correction,
clarification, and unsupported requests (see ADR 0011 and
reports/PHASE8_PLAN.md Sec. 4/7).

Selection code: `juniper_math.sft_data` (`select_and_record_sft_subset`).
Method: deterministic per-category fixed-stride sampling
(`smoke_data.compute_stride_selection`, the same reviewed primitive Phase
5/6 use), with a **flattened** (uniform, not corpus-proportional) target
per category, floored/capped at that category's real availability. Every
candidate is masked-tokenized via `sft_rendering.tokenize_and_mask` at
selection time and **rejected** (never truncated) if its full BOS+body+EOS
length exceeds `max_sequence_length=256`.

## Counts

| Split | Target/category | Examples selected |
| --- | --- | --- |
| train | 1,000 | **24,000** (all 24 categories hit the full 1,000 target) |
| validation | 150 | **3,437** (drawn from the frozen validation split only, never trained on) |

`sft_identity` (this build): `1e55652407d3624a7e7c4d9d849ac6284fba4ce868a18cd59a3bf556a1a2d1b7`
concatenation basis (train `example_ids_sha256` + validation
`example_ids_sha256`, sorted by split name) — recorded in
`data/processed/phase8-sft/sft_manifest.json`.

Rejection count: 9 `incorrect_tool_call` train candidates were rejected as
oversized (out of ~1,155 candidates scanned for that category); every other
category had zero length-rejections. No category needed its availability
floor — all 24 hit the full requested target.

## Category / difficulty / behavior distribution (train)

- **Category counts**: exactly 1,000 per category across all 24 categories
  (ambiguity, arithmetic, basic_algebra, decimals, estimation,
  expression_translation, financial_math, fractions,
  incorrect_supplied_answer, incorrect_tool_call, missing_information,
  multi_step, negative_values, numerical_comparison, operator_precedence,
  percentages, ratios_proportions, scientific_notation, tool_error,
  tool_use, undefined_operation, unit_conversion, unsupported_capability,
  word_problem).
- **Difficulty**: trivial 1,678; easy 6,713; medium 9,418; hard 6,191.
- **Expected-behavior distribution**: `answer` 14,000; `invoke_tool` 4,000;
  `flag_incorrect_answer` 2,000; `flag_missing_information` 1,000;
  `flag_undefined` 1,000; `refuse_unsupported` 1,000;
  `request_clarification` 1,000.
- **Tool-required examples**: 5,000 of 24,000 (20.8%) — deliberately a
  minority, so the model is not trained to associate "any math request"
  with "call a tool" (Sec. 14's tool-collapse concern).
- **Family diversity**: 24 distinct families represented (one per
  category, matching the frozen corpus's family taxonomy).

## Sequence-length safety (Sec. 12)

Empirical pre-selection measurement (2% sample, 29,386 examples, full 24
categories, using the real `render_training_text` + tokenizer): median 29 /
p90 58 / p95 159 / p99 197 / p999 232 / max 256 tokens. `max_sequence_length
= 256` was chosen from this evidence, **and** the selection pipeline itself
independently verifies every selected example against that cap at
tokenization time (not trusting the sample) — 9 oversized
`incorrect_tool_call` examples were caught and rejected this way, not
silently truncated.

## Loss masking

Every selected example is rendered via `juniper_math.sft_rendering.
render_segments`/`tokenize_and_mask`: prompt and `<tool_result>` segments
are context-only (`label=-100`); `<tool_call>` and the terminal
`<final>`/`<unsupported>`/`<error>` tag are loss-bearing. Verified against
zero mismatches vs. the existing joint-string tokenization
(`pilot_data.tokenize_examples`) on 7,452 sampled examples; 15 unit tests in
`tests/test_sft_rendering.py` cover every item in Sec. 11's required list.

## What was explicitly excluded

- The frozen validation/test splits' remaining, non-selected examples
  (never touched).
- The four frozen `evals/phase4_*_v2.json` suites (structurally excluded —
  reserved from the train split at Phase 4 build time, verified zero
  example_id/prompt overlap in this session).
- The new held-out `evals/phase8_instruction_v1.json` suite (built with a
  seed namespace offset by `PHASE8_SEED_OFFSET = 10_000_000`, verified zero
  overlap with the training corpus and every existing eval suite).

## Known limitation surfaced during this build (not a Phase 8 defect)

While building the Phase 8 held-out suite, a latent bug was found already
present in the **frozen** `evals/phase4_calibration_v2.json`: its
`_make_case("incorrect_supplied_answer", ...)` constructor reuses
`_direct("arithmetic", index, seed)` internally and keeps that call's
`example_id` unchanged, so 30 of that suite's 130 cases collide pairwise
across the `arithmetic`/`incorrect_supplied_answer` categories onto only
100 truly unique example ids. This is a frozen Phase 4 artifact
(reports/PHASE8_PLAN.md Sec. 13 forbids modifying it) and is reported here,
not fixed. `juniper_math.dataset.eval_isolated.build_phase8_eval_suite`
gives its own suite a category-specific index offset specifically to avoid
reproducing this collision (verified: 271/271 example ids unique).
