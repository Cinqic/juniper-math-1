# Juniper Math 1 — Phase 2 Final Approval

## Identification

- Project: Juniper Math 1
- Phase: 2 — Math Tokenizer
- Foundation: `phase-1-architecture` — `83d6106bae2a465a1c166c3137b10b87eda22f91`
- Sonnet candidate: `phase-2-review-candidate` — `c6cf6b650cf6e6bdb84a9ff026a2457c07b02c7b`
- Reviewer/remediator: GPT-5.6 Terra
- Canonical final reference: `phase-2-tokenizer` (created only after this approved state was pushed and CI was green)

## Final tokenizer

`juniper-math-tokenizer-v1` is a SentencePiece 0.2.2 BPE model with 4,096 pieces, byte fallback, `split_digits=true`, identity normalization, and no dummy prefix or extra-whitespace removal. The model has exactly 4,096 contiguous IDs (0–4095); the frozen Phase 1 model has the same embedding vocabulary and exactly 5,004,032 trainable parameters.

| ID | Token |
|---:|---|
| 0 | `<unk>` |
| 1 | `<s>` |
| 2 | `</s>` |
| 3 | `<pad>` |
| 4–8 | `<tool_call>`, `<tool_result>`, `<final>`, `<unsupported>`, `<error>` |
| 9–18 | `0`–`9` |

The wrapper does not inject BOS/EOS. It accepts valid Python `str` values only; callers must decode bytes before encoding.

## Corpus and vocabulary

- 200,000 nonblank deterministic synthetic lines in 12 configured categories.
- Seed: `20260201`.
- Corpus SHA-256: `86c3afa92b2cc109b9d3ba340ce59e920e84092fd88347154b006529db7fd13f`.
- Provenance: project-authored templates and word banks only; no downloaded, scraped, or package-provided text.
- Vocabulary audit: 4,096/4,096 pieces inspected programmatically; 256 byte-fallback pieces; zero unauthorized multi-digit pieces; zero unusually long pieces.

The numeric policy forbids any ordinary vocabulary piece containing a run of two or more ASCII decimal digits, including alphanumeric forms. SentencePiece byte notation `<0xXX>` is explicitly exempt because it is a byte-fallback label, not numeric text. Digits 0–9 were atomic in all curated contexts and 40,000 independent unseen numeric cases, including 50+ digit strings.

## Validation and quality

- Serialized protobuf audit: PASS — BPE, IDs, size, `split_digits`, byte fallback, user symbols, and identity normalizer match config.
- Special token, injection-lookalike, Unicode math, byte fallback, identity-normalization, whitespace, empty input, and long-input checks: PASS.
- `<unk>` on valid out-of-distribution Unicode audit cases: 0.
- Deterministic reconstruction: PASS — two output paths produced byte-identical model, vocab, special map, config snapshot, and corpus-digest artifacts.
- Artifact hashes: PASS, both through `sha256sum` and the project manifest verifier.
- Benchmark: PASS as informational evidence. English/explanations remain compact for a 4,096-piece model; numeric categories deliberately cost more than GPT-2 because digits are atomic. GPT-2 baseline vocabulary is 50,257 and is not a like-for-like quality score.
- Security audit: PASS — no tokenizer network fetch, dynamic code execution, unsafe YAML, telemetry, or credentials.
- Dependency governance: PASS — SentencePiece is runtime; tiktoken and protobuf are development-only; direct dependencies have metadata-backed license entries.

## Findings and remediation

Four findings were resolved: unsafe fixed-staging cleanup (HIGH), missing corpus digest (MEDIUM), missing raw-protobuf/broad numeric audit (MEDIUM), and undocumented bytes boundary (LOW). See `TERRA_PHASE2_REVIEW.md` and `PHASE2_REMEDIATION.md`.

## Verdict

**APPROVED**

Phase 2 is **COMPLETE**. Phase 3 — Cinqic Calculator Tool Runtime — is **AUTHORIZED — NOT STARTED**. No Phase 3 runtime code was added.
