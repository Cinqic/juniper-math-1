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

## Clean replacement run

The clean reconstruction was validated against the committed frozen manifest:
34 exact hash-valid shards and dataset identity
`bf9933f032a58b4eb618b32156783b8563097a5fc1c0ef26be4f76445128d25a`.
Manifest-backed packing reproduced 1,466,970 train records, 59,864 packed
sequences, 59,083,692 loss-bearing tokens per epoch, and 3.519% padding. The
Phase 7 configuration was corrected from 8,218 to the exact two-epoch budget
of 7,483 steps in commit `0a34581` before training.

`phase7-full-v2` ran from a clean tree at that commit, fresh initialization,
and the rebuilt dataset. It completed 7,483 steps and 118,167,384
loss-bearing tokens, with five serious retained milestone checkpoints. Its raw
log supersedes the local v1 log while v1 remains recoverable through Git
history and its candidate release. The independent bounded CUDA resume rerun
passed at the existing `< 1e-2` thresholds (loss 0.0001933610; parameters
0.0002411418), unlike the invalid candidate-era check.

## Final verification

With the reviewed clone explicitly first on `PYTHONPATH`, Python 3.12, and
the RTX 2060 CUDA environment, `pytest -q` completed with **660 passed, 0
failed, 2 warnings**. The warnings are PyTorch's documented non-deterministic
memory-efficient-attention warning in two CUDA tests. Final checkpoint,
remote-preservation, and approval evidence are in `PHASE7_FINAL_APPROVAL.md`.
