# Phase 4 Engineering Report — Dataset and Evaluation Freeze

**Engineer:** Claude Sonnet 5 (primary Phase 4 implementation, self-review)
**Starting foundation:** `phase-3-tools` (commit `1cb070945dd1f968f0dd10e6ceb915e0d24fac0a`)
**Status at handoff:** Phase 4 engineering complete; independent Phase 4
approval PENDING GPT-5.6 Terra (see `reports/PHASE4_TERRA_HANDOFF.md`).

## Scope decisions (documented, conservative)

1. **Synthetic-only corpus.** No external/third-party data was acquired for
   `juniper-math-dataset-v1`. See
   `reports/PHASE4_PROVENANCE_LICENSE_REVIEW.md` for the full reasoning —
   in short, responsibly establishing redistribution rights for a candidate
   external math corpus is a legal-judgment call this engineering session
   is not positioned to make, and the Phase 4 instructions themselves warn
   against including a dataset "merely because everyone else trains on it."
2. **Six categories intentionally undersized relative to their configured
   token target.** `ambiguity`, `missing_information`,
   `undefined_operation`, `unsupported_capability`, `scientific_notation`,
   and `tool_error` plateaued on exact-duplicate space during the full
   build — see "Dataset statistics" below and `docs/DATASET.md`. This is
   the Sec. 18 "don't force size past a quality plateau" signal, not a bug:
   these categories are deliberately small, curated, construction-verified
   template families with bounded combinatorial diversity, and
   `config/dataset.yaml`'s `category_mixture` already reflects that (their
   shares were reduced from an initial draft after the first full-scale
   build surfaced the plateau).

## What was built

- `config/dataset.yaml` — canonical Phase 4 dataset specification (identity,
  seed, token budget, 24-category mixture, split strategy, dedup
  thresholds, diversity caps, normalization rules, shard format).
- `src/juniper_math/dataset/` — the full pipeline package: deterministic
  seed/ID derivation (`idgen.py`), config loading (`config.py`), the
  `Example` record schema (`schema.py`), a closed-allowlist ground-truth
  verifier extending the Fraction-based pattern the frozen Phase 0
  evaluator already uses (`verify.py`), cleaning/normalization
  (`clean.py`), exact + near deduplication (`dedup.py`), deterministic
  family-aware split assignment (`split.py`), contamination checks
  (`contamination.py`), deterministic sharding/serialization/hashing
  (`shard.py`), streaming shard I/O (`io.py`), statistics computation
  (`stats.py`), full pipeline orchestration (`build.py`), the frozen
  eval-suite builder (`eval_suites.py`), and the CLI-facing operations
  layer (`pipeline.py`).
- 24 category generators across 6 modules under
  `src/juniper_math/dataset/generators/` — see `docs/DATASET.md` for the
  full category -> module table.
- The `dataset` CLI command group (`acquire`, `eval-suites-build`,
  `generate`, `build`, `validate`, `verify`, `stats`,
  `contamination-check`), replacing the honest Phase 0-3 placeholder.
- Four frozen Phase 4 evaluation suites (725 cases total) — see
  `reports/PHASE4_EVALUATION_FREEZE.md`.
- 43 new tests in `tests/test_dataset.py` covering config parsing,
  deterministic generation, schema validation, ground-truth verification
  (including a direct regression test for the Phase 0 tool-001 defect
  class), cleaning, dedup, split determinism/family isolation,
  contamination detection, the full generator registry's self-consistency,
  tokenizer rendering determinism, a reproducible tiny end-to-end build, and
  ongoing re-verification of all four frozen Phase 4 eval suites (so a
  hand-edited or stale suite file fails regular `pytest`, not just
  suite-generation time).
- Documentation: `docs/DATASET.md` (new), `docs/EVALUATIONS.md`,
  `docs/CLI.md`, `docs/RECOVERY.md` (all updated).
- Manifest updates: `manifests/sources.yaml` (new synthetic-source entry),
  `manifests/artifacts.yaml` (9 new frozen-artifact hash entries),
  `config/project.yaml` (Phase 4 status).

## Dataset statistics (final build)

| Metric | Value |
|---|---|
| Total examples | 1,833,697 |
| Total tokens (frozen tokenizer) | 66,905,612 |
| Total bytes (prompt text) | 90,237,224 |
| Shards | 38 (`records_per_shard: 50,000`) |
| Train examples / tokens | 1,651,171 / 60,246,503 |
| Validation examples / tokens | 91,246 / 3,327,757 |
| Test examples / tokens | 91,280 / 3,331,352 |
| Average tokens/example | 36.5 |
| Median tokens/example | 26 |
| p90 / p99 tokens/example | 52 / 189 |
| Max example tokens | 257 (well under the 1,024-token context limit) |
| Fraction exceeding context | 0.0 |
| Peak build RSS | 3.5 GB (well within the 16 GB system budget) |
| Build wall-clock | 8m 40s (single CPU core, Ryzen 7 5700G-class) |
| Disk footprint | 1.3 GB (shards, gitignored/disposable) |

**Cleaning/verification counters** (from the full build):

| Counter | Count |
|---|---|
| Generated | 3,810,437 |
| Rejected — ground-truth mismatch | 0 |
| Rejected — schema invalid | 0 |
| Rejected — exceeds context | 0 |
| Rejected — exact duplicate | 1,897,780 |
| Rejected — near duplicate | 0 |
| Rejected — diversity cap | 78,960 |
| **Accepted** | **1,833,697** |

Zero ground-truth mismatches and zero schema-invalid records means every
accepted example's `expected_answer` independently re-verified against its
`verification.expression` (or, for tool cases, the live `ToolRuntime`)
during generation — see `docs/DATASET.md` "Category schema". The exact
duplicate count is large in absolute terms because `generated` (3.81M) was
deliberately overshot relative to `accepted` (1.83M): the build loop keeps
drawing from a category's generator until its token target is met or its
attempt budget is exhausted, so a category that plateaus produces a long
tail of exact duplicates before the loop gives up — this is the same
Sec. 18 signal as the six shortfall categories, not a distinct defect.

Total tokens (66.9M) landed within the configured 50–100M envelope, close
to but under the 70M target — the six documented shortfall categories
account for the gap.

## Verification results

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
```

`dataset verify` re-executes every one of the 95,472 recorded tool traces
against the **live** Phase 3 `ToolRuntime` and byte-compares the result —
this is not trusting a cached execution from generation time.

## Full regression

```
python -m juniper_math validate-env        # PASS
python -m juniper_math validate-config     # PASS
python -m juniper_math hash verify         # PASS (26 artifacts, incl. 9 new Phase 4 ones)
python -m juniper_math manifests-validate  # PASS
python -m juniper_math deps-check          # PASS (no new dependency)
python -m juniper_math model --device cpu  # PASS (5,004,032 params, unchanged)
python -m juniper_math tokenizer validate  # PASS (unchanged)
python -m juniper_math tools self-test     # PASS (unchanged)
python -m juniper_math evals validate      # PASS (frozen Phase 0 suite, unchanged)
python -m juniper_math evals verify        # PASS

pytest -v          # 550 passed (507 pre-existing + 43 new)
ruff check .        # All checks passed
ruff format --check . # all files already formatted
mypy                # Success: no issues found in 51 source files
```

No regression from the Phase 3 approved baseline (507 tests, 2 pre-existing
CUDA determinism warnings — those two GPU-only warnings persist unchanged;
everything else is new-and-passing).

## Deterministic rebuild

See `reports/PHASE4_DATASET_VALIDATION.md` for the actual independent
rebuild evidence (delete the corpus, reconstruct from nothing but this
repository, compare identities).

## Frozen architecture / tokenizer / tool protocol

Unchanged. `config/architecture.yaml`, `config/tokenizer.yaml`,
`config/tools.yaml`, the tokenizer artifacts under `releases/tokenizer/`,
and the tool protocol schemas under `tools/schemas/` were not modified —
`git diff phase-3-tools -- config/architecture.yaml config/tokenizer.yaml
config/tools.yaml releases/ tools/schemas/` is empty. Phase 4 only adds
`config/dataset.yaml` alongside them.

## Known limitations

- Six categories (see above) are below their proportional token target by
  design, not by defect.
- No external data sources — a future dataset version may add them with a
  proper source-by-source license review (Terra is authorized to propose
  candidates; see `reports/PHASE4_TERRA_HANDOFF.md`).
- Near-duplicate detection is family-scoped with a bounded 200-example
  recent-window (documented trade-off, not full corpus-wide pairwise
  comparison — see `docs/DATASET.md` "Deduplication").
- No manual/human spot-audit of a large prompt sample was performed beyond
  automated ground-truth re-verification and the construction-time semantic
  labeling; see `reports/PHASE4_SELF_REVIEW.md` for what was and was not
  checked.

## Status

```
PHASE 4 ENGINEERING: COMPLETE
INDEPENDENT PHASE 4 APPROVAL: PENDING GPT-5.6 TERRA
PHASE 5: NOT AUTHORIZED
```
