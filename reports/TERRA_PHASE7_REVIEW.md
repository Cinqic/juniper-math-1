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

## Final audit update

The required remediation was completed before the replacement run. A clean
repository at `0a34581ccd07de12b229528de45abbf5cb5a3a5d` rebuilt the exact
34-shard frozen dataset (1,466,970 train / 81,094 validation / 81,014 test;
identity `bf9933f032a58b4eb618b32156783b8563097a5fc1c0ef26be4f76445128d25a`).
The replacement `phase7-full-v2` began from fresh initialization with a clean
source tree and consumed only those manifest entries.

Its reviewed budget is 7,483 steps: production rendering/tokenization and
first-fit packing measured 59,864 packed train sequences, 59,083,692
loss-bearing tokens per epoch, and 3.519% padding. Two epochs therefore give
119,728 sequences / effective batch 16 = 7,483 steps and 118,167,384
loss-bearing tokens. The raw run log records exactly those steps/tokens.

All milestone candidates were compared on identical full validation and frozen
suite settings. Overall validation loss was 0.706918 (20%), 0.643330 (40%),
0.622091 (60%), 0.607735 (80%), and 0.600296 (final). Nineteen of 24 category
losses improved monotonically; five had small nonmonotonic changes. The final
candidate is selected because it has the best overall validation loss and the
best final values for most categories. Its 62.2% tool-call emission is below
the 40% candidate's 98.4%, but both candidates have 0% valid parse and tool
name match; this diagnostic alone does not outweigh the broad held-out-loss
improvement. Base-pretraining capability remains limited: final math 1/215,
calibration 0/130, and adversarial 36/195.

The bounded CUDA resume comparison was independently rerun against the same
manifest-backed full-data path: exact final step 200 and tokens 3,157,988,
maximum logged-loss difference 0.0001933610, maximum parameter difference
0.0002411418, and identical fixed generations. This passes the existing
`< 1e-2` criterion without changing it and is consistent with the documented
non-bitwise CUDA attention warning.

Final approval and remote-artifact details are recorded in
`PHASE7_FINAL_APPROVAL.md`. The old candidate is permanently ineligible for
canonical Base use; its release remains historical evidence only.
