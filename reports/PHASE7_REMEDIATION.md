# Phase 7 Remediation

## Manifest-backed shard loading

Commit `0442820` replaces broad directory globbing in Phase 6/7 selection
with manifest-driven paths. Every manifest entry is checked for duplicate
filenames, required metadata, existence, size, and SHA-256; all unmanifested
JSONL files cause failure. Full-data loading also verifies that record split
metadata matches the requested split.

Regression coverage verifies exact split totality plus rejection of missing,
modified, duplicate-manifest, wrong-split, and unmanifested shards.

## Verification

Using the Phase 7 Python 3.12 environment with the reviewed clone explicitly
on `PYTHONPATH`, `pytest -q` completed with **653 passed, 7 skipped, 2 CUDA
non-deterministic-attention warnings**.

## Invalidated evidence

`phase7-full-v1` and its candidate checkpoint remain preserved historical
evidence, but are invalidated for canonical Base selection by the 151,171
extra train examples and unreconstructible dirty source state. No Phase 7
final approval has been issued.
