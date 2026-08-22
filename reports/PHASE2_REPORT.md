# Phase 2 Engineering Report: Math Tokenizer

## Identification

- **Phase:** 2 — Math Tokenizer
- **Engineer:** Claude Sonnet 5
- **Starting Phase 1 commit/tag:** `83d6106bae2a465a1c166c3137b10b87eda22f91` (`phase-1-architecture`)
- **Candidate tag:** `phase-2-review-candidate` (created at handoff)
- **Environment:** exact validated lock, `requirements-lock.txt` (Python 3.12, torch 2.13.0, sentencepiece 0.2.2)

## Corpus

- **Categories (12):** natural-language math problems, arithmetic, algebra,
  percentages, ratios/proportions, units, financial math, scientific
  notation, explanations, tool syntax, errors, general English.
- **Size:** 200,000 lines (`config/tokenizer.yaml: training.corpus_total_lines`).
- **Composition:** see `CATEGORY_PROPORTIONS` in
  `src/juniper_math/tokenizer_corpus.py` — natural-language math problems
  (22%) and explanations (10%) plus general English (7%) dominate, so the
  tokenizer stays a math *specialist that understands ordinary English*
  rather than a symbol-only tokenizer; numeric/symbolic categories (36%
  combined: arithmetic, algebra, percentages, ratios, units, financial,
  scientific notation) get enough coverage to teach notation without
  numeric sludge dominating; tool syntax (6%) and errors (3%) get a small
  but guaranteed-nonzero share.
- **Provenance:** 100% synthetic, deterministic, project-generated — no
  external text acquired. Recorded in `manifests/sources.yaml` as a
  project-controlled synthetic source
  (`phase2-synthetic-tokenizer-corpus-v1`), with generator, seed
  (`20260201`), and line count. Reconstructible byte-for-byte via
  `python -m juniper_math tokenizer train`.

## Tokenizer

- **Algorithm:** BPE (Byte-Pair Encoding), byte fallback enabled.
- **Library:** SentencePiece 0.2.2 (Apache-2.0), pinned in `pyproject.toml`
  and `requirements-lock.txt`, licensed in `manifests/licenses.yaml`.
- **Vocabulary:** exactly 4096 pieces, verified programmatically
  (`tokenizer.vocab_size == 4096`).
- **Normalization:** `identity` (no NFKC folding) — conservative, preserves
  distinct Unicode math characters exactly. See
  `unicode_normalization_policy` in `config/tokenizer.yaml`.
- **Byte fallback:** enabled and validated (0% `<unk>` rate on a
  Unicode/emoji/rare-symbol test set).

## Digits

- **Atomicity method:** `split_digits=true` forces every decimal digit to
  be pre-split before BPE merge learning runs, making multi-digit merges
  structurally impossible to learn (not merely undertrained away). Digits
  0-9 are additionally listed as `user_defined_symbols` to guarantee stable
  vocabulary entries.
- **Vocabulary audit:** all 4096 pieces scanned for `\d{2,}` (excluding
  byte-fallback hex pieces and reserved specials) — **0 unauthorized
  multi-digit pieces**.

## Special tokens

Frozen IDs (also in `releases/tokenizer/juniper-math-tokenizer-v1.special_tokens.json`):

| ID | Token |
|---|---|
| 0 | `<unk>` |
| 1 | `<s>` |
| 2 | `</s>` |
| 3 | `<pad>` |
| 4 | `<tool_call>` |
| 5 | `<tool_result>` |
| 6 | `<final>` |
| 7 | `<unsupported>` |
| 8 | `<error>` |
| 9-18 | `0`-`9` |

## Validation

Full battery in `reports/PHASE2_TOKENIZER_VALIDATION.md` — all 11 checks
pass: vocab size, ID range, special-token ID stability, digit atomicity,
no-unauthorized-multi-digit-pieces, random numeric segmentation (2000
cases), byte-fallback/unk rate, round trips, random round trips (1000
cases), empty/whitespace input, long input (108,500 tokens, no truncation).

## Benchmarks

Full report in `reports/PHASE2_TOKENIZER_BENCHMARKS.md`. Per-category
tokens/char against a held-out evaluation set, plus an informational
comparison against `tiktoken` gpt2 (50,257 tokens). Ordinary English and
explanations are competitive with the general-purpose baseline despite a
12x smaller vocabulary; numeric/symbolic categories cost more tokens than
the baseline by design (digit atomicity forbids the multi-digit memorized
tokens gpt2 uses).

## Reproducibility

`train_tokenizer` stages training at a fixed-name temp path so the
serialized `.model` file's embedded metadata (and therefore its hash) does
not depend on the repository's clone location. Verified: two independent
rebuilds from the same corpus/config/seed into different output
directories produce **byte-identical** artifacts
(`tests/test_tokenizer.py::test_deterministic_rebuild_produces_byte_identical_artifact`).
Training itself completes in under 1 second on the target hardware.

## Hashes

| Artifact | SHA-256 |
|---|---|
| `config/tokenizer.yaml` | `5b426b046db53d8b8dc54e7f34a651d44d8a00c4cb74de53a4bf95988f35cdf4` |
| `releases/tokenizer/juniper-math-tokenizer-v1.model` | `a855b35bffbbd4b934c25f82afc7cad733f83a66f614e4fea40c112f3072fd97` |
| `releases/tokenizer/juniper-math-tokenizer-v1.vocab` | `0fe994195142e04974ec6ce75807a53e7f85bbab468b0825873dd8df0e26e3dc` |
| `releases/tokenizer/juniper-math-tokenizer-v1.special_tokens.json` | `e00cdb409a57546b191ccad198d052d4224f5cdca0f56179ee6815a8a1abba67` |
| `releases/tokenizer/juniper-math-tokenizer-v1.config_snapshot.json` | `941d3406f74e773557a6eb9ccb2700fe613cf8479d5226b47acf819aa77ecac4` |

All independently cross-checked with `sha256sum` and verified via
`python -m juniper_math hash verify` (all 11 registered artifacts PASS,
including corruption-detection spot-check: a manually flipped byte in the
committed model file was correctly caught, then the original was restored).

## Tests

**289 passed, 0 failed** (`pytest -v`), including all Phase 0/1 regression
tests (no regressions) plus new Phase 2 coverage: 81 tokenizer/corpus tests
covering vocab size, ID range, special tokens, digit atomicity, byte
fallback, round trips (including randomized), empty/long input,
deterministic reconstruction, overwrite protection, and negative/corruption
cases. `ruff check .`, `ruff format --check .`, and `mypy` all pass clean.

## Recovery

Fresh-clone recovery procedure executed and evidenced in
`reports/PHASE2_TERRA_HANDOFF.md`.

## Known limitations

- Numeric/symbolic categories (scientific notation, ratios, units) are
  less token-efficient than a general-purpose baseline — an intended
  consequence of digit atomicity, not a defect.
- The tokenizer-training corpus is synthetic and template-based; it teaches
  segmentation, not model capability, and is explicitly not the Phase 4
  training corpus.
- The deterministic-reconstruction workaround relies on a fixed staging
  path under the OS temp directory (`$TMPDIR/juniper_tokenizer_build/`),
  which assumes single-process, non-concurrent tokenizer training — a
  reasonable assumption for this project's workflow but worth flagging for
  Terra's review.

## Status

```
AWAITING_GPT_5_6_TERRA_REVIEW
```
