# Phase 2 Self-Review

Adversarial self-review performed by Claude Sonnet 5 against its own Phase 2
tokenizer engineering, before handoff to GPT-5.6 Terra (Sec. 82-84 of the
Phase 2 instructions). Approached as if reviewing someone else's tokenizer
pipeline, specifically hunting for the failure modes tokenizer work tends to
hide.

## Findings

### F-01 (HIGH, found and fixed): non-deterministic reconstruction across paths

**What was found:** SentencePiece embeds its `TrainerSpec` — including the
literal `input` and `model_prefix` filesystem paths it was invoked with —
inside the serialized `.model` file. The first implementation trained
directly at `config.model_path` (an absolute path under the repository's
clone location). Two honest rebuilds of the exact same
corpus/config/seed, but at two different output directories (as would
happen on two different machines, or two different clones), produced
byte-**different** `.model` files — caught by
`tests/test_tokenizer.py::test_deterministic_rebuild_produces_byte_identical_artifact`
during development, not after the fact.

**Root cause:** conflating "the tokenizer's semantic identity is
deterministic" (true — same corpus/config/seed always produces the same
vocabulary) with "the serialized artifact is byte-identical" (was false,
because of path-dependent metadata SentencePiece embeds regardless of
training parameters).

**Fix:** `train_tokenizer` (`src/juniper_math/tokenizer.py`) now stages the
corpus and training output at a fixed-name path
(`$TMPDIR/juniper_tokenizer_build/`) before training, then copies the
result into the real destination. The embedded metadata — and therefore
the artifact hash — now depends only on corpus bytes and training
parameters. Re-verified: two rebuilds into different directories now
produce byte-identical `.model` files.

### F-02 (MEDIUM, found and fixed): audit false-positives from byte-fallback pieces

**What was found:** The first version of the unauthorized-multi-digit-piece
scanner (`audit_vocabulary`) flagged 100 pieces as violations — every
byte-fallback piece from `<0x00>` through `<0x63>`. `\d{2,}` matches the hex
digits inside `<0x00>`, `<0x11>`, etc., which are not decimal digits at all.

**Root cause:** the regex scan didn't account for SentencePiece's
byte-fallback piece format (`<0xXX>`) — an oversight anticipated by the
Phase 2 spec itself (Sec. 21: "correctly accounting for SentencePiece
whitespace markers or reserved special tokens") but the byte-fallback case
specifically wasn't listed and was missed on the first pass.

**Fix:** added an explicit `_BYTE_FALLBACK_PIECE` exclusion pattern
(`^<0x[0-9A-Fa-f]{2}>$`) before the multi-digit scan. Re-verified: 0
violations across all 4096 pieces.

### F-03 (LOW, found and fixed): initial corpus too repetitive for target vocab size

**What was found:** the first corpus (40,000 lines, ~90-word general-English
vocabulary) could not sustain BPE merge learning up to 4096 pieces —
`SentencePieceTrainer.Train` raised `Vocabulary size too high (4096). Please
set it to a value <= 2210`, because the template-based generators produced
too little unique substring diversity once digits were split.

**Root cause:** underestimated how much lexical diversity split-digit BPE
training needs to reach a specific target vocab size; a smaller/more
repetitive corpus exhausts distinct mergeable substrings before reaching
the target.

**Fix:** expanded the general-English word bank to ~600 words spanning
Sec. 10's required surface areas (instructions, relationships, quantities,
objects, time, money, comparisons, ambiguity) and increased
`corpus_total_lines` to 200,000. Training now completes cleanly at exactly
4096.

## Checked and found clean

- **Normalization**: `normalization_rule_name: identity` is a real
  SentencePiece built-in (no-op) rule, confirmed by inspecting round-trip
  behavior on Unicode math symbols (×, √, π, ≤, ...) — none are folded or
  rewritten.
- **Special-token ID stability**: live `piece_to_id` lookups match the
  frozen `special_tokens.json` map exactly; covered by
  `check_special_token_ids` and re-run on every `tokenizer validate`.
- **`<unk>` leakage**: 0% unknown-token rate across a Unicode/emoji/rare-symbol
  test set; byte fallback activates correctly instead.
- **Corpus source licensing**: the corpus is 100% synthetic and
  project-generated (no external text acquired), recorded as a
  project-controlled source in `manifests/sources.yaml` — no third-party
  licensing exposure.
- **Dependency isolation**: `tiktoken` (used only for the informational
  baseline comparison) is a dev-only dependency; verified by uninstalling
  it and confirming `tokenizer train/inspect/encode/decode/validate` are
  unaffected and `tokenizer benchmark` degrades gracefully
  ("baseline unavailable") instead of crashing.
- **Corruption detection**: manually flipped a byte in the committed
  `.model` file and confirmed `juniper-math hash verify` fails with a
  precise mismatch report, then restored the original and confirmed it
  passes again.

## Remaining notes (not blockers)

- The tokenizer-training corpus generator (`tokenizer_corpus.py`) is not
  itself committed as a static corpus file — only the generator, seed, and
  line count are committed. This is deliberate (Sec. 78: preserve enough to
  reconstruct deterministically rather than committing a large generated
  text blob) and is exercised by the deterministic-rebuild test.
- Numeric-category token efficiency (scientific notation, ratios, units)
  trails the informational gpt2 baseline by design — this is the intended
  cost of digit atomicity, not a defect. See
  `reports/PHASE2_TOKENIZER_BENCHMARKS.md` for the full discussion.

No unresolved BLOCKER, HIGH, or material MEDIUM findings remain. This
candidate is ready for GPT-5.6 Terra independent review.
