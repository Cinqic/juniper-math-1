# Phase 2 Remediation Record

## R-01 — HIGH — deterministic staging could delete unrelated files

**Root cause:** `train_tokenizer` used `shutil.rmtree($TMPDIR/juniper_tokenizer_build)` before training. A stale directory, a concurrent build, or a deliberately placed file at that fixed path was silently deleted.

**Fix:** staging creation is now atomic (`mkdir(mode=0o700)`). An existing directory causes a clear failure and is never removed by the process that did not create it. The directory created by a successful invocation is removed in `finally`, including on trainer failure.

**Regression test:** `test_existing_staging_directory_is_never_deleted` verifies that a sentinel file remains intact and the command fails clearly.

## R-02 — MEDIUM — corpus identity was described but not frozen

**Root cause:** the generator seed and line count were recorded, but no digest established that the reconstructed corpus bytes were the intended corpus.

**Fix:** training now writes `releases/tokenizer/juniper-math-tokenizer-v1.corpus.sha256`; the source and artifact manifests register it. The canonical generated corpus digest is `86c3afa92b2cc109b9d3ba340ce59e920e84092fd88347154b006529db7fd13f`.

**Regression test:** `test_corpus_digest_matches_a_fresh_generated_corpus` regenerates all 200,000 lines and checks the digest.

## R-03 — MEDIUM — serialized configuration and broad numeric behavior lacked independent assertions

**Fix:** tests now deserialize the SentencePiece protobuf and assert BPE, size, byte fallback, digit splitting, IDs, user-defined symbols, and identity normalization. An independent property suite exercises 40,000 deterministic unseen numeric strings, including 50+ digit cases.

## R-04 — LOW — byte inputs lacked a project-owned error contract

**Fix:** the wrapper explicitly accepts `str` only and raises a clear `TypeError` for bytes; callers must decode bytes before tokenization.

## Final result

The model and vocabulary bytes are unchanged from the submitted candidate. The configuration snapshot and manifests changed to record the new corpus identity; all remediation tests and full regression passed before approval.
