# Phase 6 Remediation

## Defects found

1. **P1 — pilot evidence used unverified ignored shards.** The candidate's
   reported selection (137,057 train examples and 26 train `tool_error`
   examples) cannot be recreated from the committed frozen shard manifest and
   whole-dataset identity. A clean rebuild reproduces the manifest hashes but
   selects 130,492 examples and finds 1,393 train `tool_error` records.
   The original experiment was therefore not reproducible evidence.
2. **P2 — `pack_sequences: false` was silently ignored.** The pipeline always
   packed despite accepting and recording the setting.

## Remediation

- Added `verify_parent_dataset_shards()` to fail closed when any JSONL shard
  differs in bytes or SHA-256 from the committed manifest, and verify the
  whole-dataset identity before pilot selection.
- Made `PackedPilotDataset` honor disabled packing and added regression tests
  for both defects.
- Rebuilt the frozen corpus, generated the pilot selection twice (identical
  manifests), reran the full pilot and resume test, and replaced the candidate
  experiment record/logs with the independently produced records.

## Verification

Focused Phase 6/Phase 5 trainer tests: 98 passed, 4 CUDA-only skipped. Ruff,
Ruff format, and mypy passed before rerunning the full suite. The clean dataset
build, schema validation, ground-truth verification, and contamination check
passed. The original candidate result and `tool_error` scarcity claim are
superseded; no frozen artifact was modified.
