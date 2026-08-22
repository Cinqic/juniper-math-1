# Phase 2 Tokenizer Benchmarks

Tokenizer: `juniper-math-tokenizer-v1` (4096-piece BPE, byte fallback,
`split_digits`). Baseline: `tiktoken` `gpt2` encoding (OpenAI, 50,257
tokens) — **informational only**, not vocabulary-size-matched (Sec. 56 of
the Phase 2 instructions: comparing a 4,096-token specialist against a
50,257-token general-purpose tokenizer is not apples-to-apples; it is
included to sanity-check that Juniper Math 1's specialization has not
produced pathological over-segmentation).

## Evaluation set

Frozen, deterministic, **held out from tokenizer training**: same category
generators as the training corpus (`src/juniper_math/tokenizer_corpus.py`),
but seeded with `771177` (training uses `20260201`) — 150 lines per
category, 1800 lines total. Reproduce with `python -m juniper_math tokenizer
benchmark`.

## Per-category results

| Category | tokens/char | tokens/sample | baseline (gpt2) tokens/char |
|---|---|---|---|
| algebra | 0.773 | 10.71 | 0.513 |
| arithmetic | 0.781 | 6.81 | 0.472 |
| errors | 0.163 | 4.41 | 0.227 |
| explanations | 0.198 | 9.07 | 0.207 |
| financial_math | 0.635 | 17.57 | 0.423 |
| general_english | 0.152 | 9.34 | 0.156 |
| natural_language_math_problems | 0.298 | 16.57 | 0.249 |
| percentages | 0.553 | 6.66 | 0.476 |
| ratios_proportions | 0.911 | 6.36 | 0.564 |
| scientific_notation | 0.948 | 8.77 | 0.700 |
| tool_syntax | 0.280 | 9.05 | 0.393 |
| units | 0.804 | 7.26 | 0.558 |

## Interpretation (Sec. 56-57)

- **Ordinary English is competitive**: `general_english` (0.152 tokens/char)
  and `explanations` (0.198) are close to or better than gpt2's
  general-purpose baseline (0.156 / 0.207) despite a 12x smaller vocabulary
  — the corpus mixture (Sec. 13) deliberately weighted natural-language
  categories heavily enough that the tokenizer did not become math-symbol-only.
- **Numeric/symbolic categories cost more tokens by design, not by accident.**
  `scientific_notation` (0.948), `ratios_proportions` (0.911), and `units`
  (0.804) are the least efficient categories, and consistently worse than
  the gpt2 baseline. This is the direct, intended consequence of digit
  atomicity (Sec. 57): gpt2's BPE is free to memorize multi-digit number
  chunks as single tokens, which Juniper Math 1's `split_digits` policy
  categorically forbids. An arbitrary integer like `84317` costs five
  tokens here (one per digit) rather than one opaque memorized token — this
  tradeoff is required by the numeric-consistency research design (Sec. 21,
  57) and is not something this benchmark should be used to argue against.
- **`tool_syntax` beats the baseline** (0.280 vs 0.393): the five required
  control strings are single atomic pieces here, while gpt2 (which has
  never seen `<tool_call>` as a unit) fragments them.
- **`errors` is unusually efficient** (0.163): the error vocabulary
  (`division by zero`, `invalid operation`, ...) is a small closed set that
  BPE merges heavily inside a 4096 budget.

No category collapses to pathological byte-level fallback segmentation
(all tokens/char values stay well under 1.0, i.e. tokenizer pieces are, on
average, coarser than single characters even for the least-efficient
categories).
