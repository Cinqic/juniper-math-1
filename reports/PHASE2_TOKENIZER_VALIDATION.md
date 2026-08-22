# Phase 2 Tokenizer Validation Report

Tokenizer identity: `juniper-math-tokenizer-v1` (SentencePiece BPE, byte
fallback, `split_digits`). Config: [`config/tokenizer.yaml`](../config/tokenizer.yaml).
Reproduce with:

```bash
python -m juniper_math tokenizer validate
```

or programmatically via `juniper_math.tokenizer_validation.run_full_validation`,
which backs both the CLI command and `tests/test_tokenizer.py`.

## Results

| Check | Result | Detail |
|---|---|---|
| `vocab_size` | **PASS** | tokenizer=4096, config=4096, expected=4096 |
| `id_range` | **PASS** | 0 vocab-range violations, 0 encoded violations across a broad sample |
| `special_token_ids` | **PASS** | all 19 special tokens (4 core + 5 control + 10 digit) stable between live encoder and the frozen `special_tokens.json` map |
| `digit_atomicity` | **PASS** | all 10 digits atomic in 9 surrounding contexts each (start of string, after whitespace, before `.`, after `-`, adjacent digits, before `e`, after `$`, before `%`) — 90/90 checks |
| `no_unauthorized_multi_digit_pieces` | **PASS** | 0 violations found scanning all 4096 vocabulary pieces (byte-fallback `<0xXX>` pieces correctly excluded from the scan) |
| `random_numeric_segmentation` | **PASS** | 2000/2000 deterministic random numeric strings (seed 909090) — integers, negatives, leading zeros, decimals, scientific notation, currency, percentages, digit runs up to 30 digits long |
| `byte_fallback_unk_rate` | **PASS** | unk_rate = 0.0000 (0/86 tokens) across the Unicode/emoji/rare-symbol test set |
| `round_trips` | **PASS** | all cases across arithmetic, algebra, percentages, ratios, units, currency, scientific notation, explanations, errors, tool syntax, and Unicode math |
| `random_round_trips` | **PASS** | 1000/1000 deterministic random combinations (seed 909091) of the round-trip pool |
| `empty_whitespace_input` | **PASS** | `""`, `" "`, `"   "`, tab, newline, and blank lines all encode/decode without raising |
| `long_input` | **PASS** | 108,500-token input (far beyond the model's 1024-token context) encoded and decoded with zero truncation and an exact round trip |

**All 11 checks pass.**

## Digit atomicity (Sec. 20-22)

Mechanism: SentencePiece's `split_digits=true` forces every decimal digit to
be split into its own symbol *before* BPE merge learning ever runs. This is
structural, not statistical — the trainer never sees a two-digit substring to
merge, so a multi-digit piece cannot be learned regardless of corpus
frequency. All ten digits are additionally listed in `user_defined_symbols`
(`config/tokenizer.yaml: digit_symbols`), guaranteeing each has a stable
vocabulary entry even if it were rare in the corpus (none are — digits 0-9
are heavily represented across arithmetic/algebra/units/currency/scientific
notation categories).

Verified: every digit 0-9 tested individually and in 9 surrounding contexts;
`84317` → `8 4 3 1 7` (five atomic digit pieces, not a memorized `84317`
token); random 2000-sample property test (including a 30-digit-long random
run) confirms no vocabulary piece ever contains more than one consecutive
decimal digit outside the excluded byte-fallback hex pieces.

## Numeric vocabulary audit (Sec. 60)

Every one of the 4096 vocabulary pieces was scanned with
`\d{2,}` (after stripping the SentencePiece whitespace marker `▁` and
excluding both the 19 special/digit tokens and the 256 byte-fallback
`<0xXX>` pieces, whose hex digits are not decimal digits). **Zero**
unauthorized multi-digit pieces exist in the frozen vocabulary.

## Byte fallback / Unicode (Sec. 26-28, 39)

Tested: `€42`, `₹500`, `你好`, `résumé`, `α + β`, `∂f/∂x`, `∫`, `🤖`, `π`,
`√2`, `x²`, `≤`, `≠`, `≈` — none produce `<unk>` (id 0). Rare/absent-from-corpus
characters correctly fall back to UTF-8 byte pieces (`<0xE2>`, `<0x82>`, ...),
which decode back to the exact original bytes.

## Special tokens (Sec. 17-19, 61)

Frozen ID table (also machine-readable at
`releases/tokenizer/juniper-math-tokenizer-v1.special_tokens.json`):

| ID | Token | Category |
|---|---|---|
| 0 | `<unk>` | core |
| 1 | `<s>` | core |
| 2 | `</s>` | core |
| 3 | `<pad>` | core |
| 4 | `<tool_call>` | control |
| 5 | `<tool_result>` | control |
| 6 | `<final>` | control |
| 7 | `<unsupported>` | control |
| 8 | `<error>` | control |
| 9-18 | `0`-`9` | digit |

`<tool_call>{"expression":"2+2"}</tool_call>` round-trips exactly, with
`<tool_call>` tokenizing as a single atomic piece embedded in surrounding
JSON-like text (the closing `</tool_call>` is not itself a reserved special
token per the Phase 2 spec — only the five listed opening/marker strings are
— and it tokenizes as ordinary text, which round-trips correctly).

## Model compatibility (Sec. 62-63)

`tokenizer.vocab_size == model.vocab_size == 4096` confirmed against
`config/architecture.yaml`. A forward pass through the frozen Phase 1 model
(`build_model`) on tokenizer-encoded IDs for representative math, algebra,
tool-syntax, and Unicode text produced finite logits with no negative or
out-of-range ID errors.

## Deterministic reconstruction (Sec. 51, 75)

SentencePiece embeds its `TrainerSpec` (including the literal `input` and
`model_prefix` paths it was invoked with) inside the serialized `.model`
file. Training directly against the repository's absolute path would make
the artifact depend on clone location, breaking cross-machine byte-identical
reproducibility. `train_tokenizer` in `src/juniper_math/tokenizer.py`
works around this by staging the corpus and training output at a
fixed-name path (`$TMPDIR/juniper_tokenizer_build/`) and copying the result
into place — so the embedded metadata (and therefore the artifact hash)
depends only on corpus bytes and training parameters, not on where the
repository happens to be checked out.

Verified: training twice from the same corpus/config/seed into two
different output directories produced **byte-identical** `.model` files
(`tests/test_tokenizer.py::test_deterministic_rebuild_produces_byte_identical_artifact`).
