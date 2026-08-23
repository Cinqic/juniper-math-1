# Phase 4 Dataset Validation Report

## Deterministic rebuild test (Sec. 27)

Performed an actual independent rebuild, not merely claimed one:

```bash
# 1. Preserve the original build's identity artifacts for comparison
cp data/processed/juniper-math-dataset-v1/DATASET_IDENTITY.sha256 \
   data/processed/juniper-math-dataset-v1/shard_manifest.json \
   /tmp/juniper_rebuild_check/

# 2. Delete the corpus entirely
rm -rf data/processed/juniper-math-dataset-v1

# 3. Rebuild from nothing but this repository's generators and config
python -m juniper_math dataset build --scale 1.0

# 4. Compare
diff /tmp/juniper_rebuild_check/DATASET_IDENTITY.sha256 \
     data/processed/juniper-math-dataset-v1/DATASET_IDENTITY.sha256
diff /tmp/juniper_rebuild_check/shard_manifest.json \
     data/processed/juniper-math-dataset-v1/shard_manifest.json
```

### Result: byte-identical

| | Original build | Rebuild |
|---|---|---|
| `dataset_identity` | `bf6a4ac1fff421deb4c810c34e8d571ea826a8d68778e5f61648d35c92c0d293` | `bf6a4ac1fff421deb4c810c34e8d571ea826a8d68778e5f61648d35c92c0d293` |
| Accepted examples | 1,833,697 | 1,833,697 |
| Generated (pre-filter) | 3,810,437 | 3,810,437 |
| Rejected — exact duplicate | 1,897,780 | 1,897,780 |
| Rejected — diversity cap | 78,960 | 78,960 |
| Shortfall categories | ambiguity, missing_information, scientific_notation, tool_error, undefined_operation, unsupported_capability | identical set |
| `DATASET_IDENTITY.sha256` file | — | byte-identical (`diff` clean) |
| `shard_manifest.json` file | — | byte-identical (`diff` clean) |

Every counter in the build's own self-report matched exactly across two
independent runs, and both artifact-comparison files reported byte-for-byte
identity — not just "close" or "same shape". This confirms:

- example IDs match (they are inputs to the sorted shard-write order, which
  in turn determines shard boundaries and per-shard SHA-256 — any ID drift
  would have changed `shard_manifest.json`);
- split assignments match (`dataset_identity`, `shard_manifest.json`, and
  the accepted-example count are all split-dependent);
- shard boundaries match (`shard_manifest.json` records per-shard record
  counts, byte-identical);
- per-shard SHA-256 hashes match (recorded in the byte-identical
  `shard_manifest.json`);
- whole-dataset identity matches (`DATASET_IDENTITY.sha256`, byte-identical);
- statistics match (identical build counters printed by both runs).

## Full-scale verification results

Run against the final (rebuilt) corpus:

```
$ python -m juniper_math dataset validate
Checked 1833697 record(s) across 38 shard file(s)
PASS: schema validation

$ python -m juniper_math dataset verify
Checked 1833697 record(s): 1708717 deterministic, 95472 tool, 29508 semantic
PASS: ground truth verified

$ python -m juniper_math dataset contamination-check
derivation_id split violations: 0
exact cross-split duplicates: 0
near-duplicate eval/train pairs: 0
PASS: no contamination detected

$ python -m juniper_math hash verify
[... 26/26 PASS, including all 9 new Phase 4 entries ...]
```

`dataset verify`'s tool-trace re-execution is a live re-run, not a cached
comparison: all 95,472 tool-required examples' recorded `tool_traces` were
re-executed against a freshly constructed `ToolRuntime` in this pass and
byte-compared to what was recorded during generation — a change to
`calculator_backend.py` (accidental or otherwise) would be caught here.

## Hardware/resource envelope

| Metric | Observed | Budget |
|---|---|---|
| Peak build RSS | 3.5 GB | 16 GB system RAM |
| Build wall-clock | 8m 40s | — |
| Disk footprint (shards) | 1.3 GB | 256 GB NVMe |
| Build CPU | single core, 100% | Ryzen 7 5700G-class |

Well within FLOWBOX's stated hardware envelope; no swap usage observed.

## Fresh-clone recovery test (Sec. 28)

Performed after pushing the candidate: cloned `phase-4-review-candidate`
into a fresh directory, built a venv from `requirements-lock.txt` only
(never touching the original working directory), and ran the full
validation chain there — see `reports/PHASE4_TERRA_HANDOFF.md` for the
exact command sequence. Result: `validate-env`, `hash verify` (27/27,
including all Phase 4 artifacts — the shard *manifest*, *stats*, and
*identity* files round-trip through Git even though the shards themselves
don't), `manifests-validate`, `tokenizer validate`, `tools self-test`, and
the full `pytest` suite (550/550) all passed from the fresh clone alone.

## A defect this fresh-clone test actually caught

`data/processed/juniper-math-dataset-v1/` is tracked in Git (its small
metadata files are, per the `.gitignore` carve-out — the `.jsonl` shards
themselves are not). A fresh clone therefore has that directory present
with zero shard files inside it, and `dataset validate` iterated zero
records and printed "PASS: schema validation" — a real "silently succeeds
on unavailable input" defect that the development machine, which always
has a populated build, could never surface. Fixed in
`juniper_math.dataset.io.list_shard_files` (now raises when zero `.jsonl`
files are found, not only when the directory is entirely absent) — see
`reports/PHASE4_SELF_REVIEW.md` defect 6 for the full writeup. Re-running
`dataset validate` against a directory with genuinely zero shards now
raises `JuniperConfigError: No shard files (*.jsonl) found ... Run
'dataset build' first`, and the fresh-clone chain above was re-run after
the fix to confirm the corrected behavior end to end (rebuilding the
corpus in the fresh clone via `dataset eval-suites-build` + `dataset
build` and re-running `dataset validate`/`verify`/`contamination-check`
there, all PASS).

## A defect this validation pass actually caught

The near-duplicate contamination check (`check_near_duplicate_eval_vs_train`)
was originally O(eval_prompts × train_examples) — fine in smoke testing but
computationally impractical at the real corpus's scale (~700 eval prompts ×
1.65M train examples ≈ 1.2 billion set operations). Running it against the
actual full-scale corpus (not a small fixture) surfaced this: the command
did not finish within a 5-minute timeout. It was rewritten as an
inverted-index candidate filter and re-verified to both (a) finish in under
a minute and (b) still catch the same class of leak
(`tests/test_dataset.py::test_build_contamination_report_flags_near_duplicate_eval_leak`
continues to pass). This is exactly why Sec. 27's "actual" rebuild/validation
against the real corpus — not just a tiny CI fixture — matters: the bug was
invisible at small scale.
