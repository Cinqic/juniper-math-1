# Phase 7 Independent Review

## Scope and target

Reviewed `3b84d6d` through `7c7ea9f`; remediation is committed as `0442820`.
This is an initial independent review, not a final approval.

## Findings

### BLOCKER — canonical run used a non-frozen training population

The frozen shard manifest and `stats.json` specify 1,466,970 train examples
and 56,209,616 train tokens.  The raw `phase7-full-v1` `run_start` event
records 1,618,141 train examples and 61,726,503 train tokens: 151,171 extra
examples and 5,516,887 extra tokens.  Its `full_manifest` therefore cannot
describe the frozen train split.  The candidate checkpoint is ineligible as
the Phase 7 Base.

The historical training directory contains exactly four unmanifested train
shards, `00030` through `00033`, totaling 151,171 records and 5,516,887
tokens. Their exact totals match the run-log difference, proving that the
broad glob ingested them.  The three full shards contain 50,000 records each;
the final shard contains 1,171 records.

Root cause: `full_data.py` and inherited Phase 6 selection helpers globbed
`*.{split}.*.jsonl`; verification checked manifest entries but did not reject
additional JSONL shards.  An unmanifested stale shard could pass verification
and enter training.

### HIGH — canonical source provenance is insufficient

The run records `source_tree_state: dirty` and `git_commit: 3b84d6d`, while
the Phase 7 implementation was later committed as `3cf0f3b`.  The exact
training source is not proven reconstructible.  This independently requires
a clean-source rerun if the checkpoint were otherwise eligible.

### MEDIUM — metadata integrity/test inconsistency

`config/project.yaml`'s recorded artifact hash was stale and its test still
asserted a pre-Phase-7 state.  Corrected in `0442820`.

## Remote checkpoint audit

The candidate GitHub release asset exists, is 60,123,651 bytes, and a fresh
download hashes to
`0ed23a8262edcf123fc9cc29e5dbd74f9169cc8bf4922d85b5e982d124d47f8e`.
This establishes recoverability of the bytes only; it does not cure the
invalid training data or source provenance.

## Initial verdict

**BLOCKED.** A clean manifest-backed dataset reconstruction and a fresh-random
initialization Phase 7 rerun are required before checkpoint selection or final
approval. Phase 8 remains not authorized.
