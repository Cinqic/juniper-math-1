# Terra Independent Phase 2 Review

## Scope and anchor

- Phase 1 foundation: `phase-1-architecture` — `83d6106bae2a465a1c166c3137b10b87eda22f91`
- Submitted candidate: `phase-2-review-candidate` — `c6cf6b650cf6e6bdb84a9ff026a2457c07b02c7b`
- Candidate tag was preserved and never moved.

The review independently inspected the Phase 2 diff, synthetic generator, wrapper, validation tests, serialized SentencePiece model, all 4,096 vocabulary IDs, manifests, CI workflow, and dependency metadata in a fresh Python 3.12 environment installed from `requirements-lock.txt`.

## Independent evidence

| Area | Result |
|---|---|
| Frozen Phase 1 compatibility | `vocab_size=4096`; instantiated model has exactly `5,004,032` trainable parameters. |
| Corpus | 200,000 generated nonblank lines; 12 configured categories; deterministic corpus SHA-256 `86c3afa92b2cc109b9d3ba340ce59e920e84092fd88347154b006529db7fd13f`; code-only synthetic provenance, no network/resource ingestion. |
| Serialized model | SentencePiece 0.2.2 BPE; 4,096 pieces; `split_digits=true`; byte fallback enabled; identity normalization; IDs 0–3 core; required symbols and digits serialized. |
| Vocabulary | 4,096/4,096 programmatically examined; IDs contiguous 0–4095; 256 byte pieces; no unauthorized run of two decimal digits; no unusually long piece. |
| Numeric policy | Any normal vocabulary piece containing a run of two or more ASCII decimal digits is forbidden. Byte fallback `<0xXX>` notation is exempt because its hex characters are metadata, not encoded decimal-number pieces. Alphanumeric runs such as `x42` are forbidden. |
| Digits | IDs 9–18 map exactly to `0`–`9`; all are atomic in adversarial contexts and 40,000 unseen deterministic numeric cases, including 50+ digit values. |
| Special tokens | IDs 0–8 exactly map to `<unk>`, `<s>`, `</s>`, `<pad>`, `<tool_call>`, `<tool_result>`, `<final>`, `<unsupported>`, `<error>`; exact special strings are atomic and near-misses are ordinary text. The wrapper does not inject BOS/EOS. |
| Unicode and fallback | Unicode math, emoji, CJK, Arabic, Cyrillic, accented text, currency, and combining marks round-trip without `<unk>` on valid `str` input. Bytes are intentionally rejected at the wrapper boundary. |
| Whitespace and normalization | Empty, spaces, tabs, newlines, repeated whitespace, NBSP, Unicode minus, and composed/decomposed accents were exercised. Identity normalization preserves distinctions; exact round-trip is the contract for valid Python strings. |
| Long input | 108,500-token input round-trips without tokenizer truncation; model context remains separately 1,024. |
| Security | No tokenizer network download, shell execution, unsafe YAML, dynamic evaluation, telemetry, or credential handling. |

## Findings

| ID | Severity | Component | Finding | Status |
|---|---|---|---|---|
| R-01 | HIGH | staging | Existing fixed staging directory was recursively deleted. | Resolved |
| R-02 | MEDIUM | provenance | Generated corpus had no frozen digest. | Resolved |
| R-03 | MEDIUM | tests | Protobuf and broad independent numeric coverage were missing. | Resolved |
| R-04 | LOW | API | Byte-input behavior was not explicit. | Resolved |

See `reports/PHASE2_REMEDIATION.md` for root causes, changes, and regression coverage.

## Reproducibility and recovery

Two clean output directories built from independently generated corpus bytes produced byte-identical `.model`, `.vocab`, special-token map, config snapshot, and corpus digest. The staging path remains fixed deliberately because SentencePiece serializes it; contention or stale state now fails safely rather than deleting data or reusing stale outputs.

## Verdict

All findings are resolved. Phase 2 is approved subject to the final regression, remote push, CI, and tag verification recorded in `PHASE2_FINAL_APPROVAL.md`.
