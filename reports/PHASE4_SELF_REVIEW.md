# Phase 4 Self-Review

> **Historical record.** This self-review documents the original Sonnet
> candidate. Its candidate statistics are not the final approved Phase 4
> state; see `reports/PHASE4_FINAL_APPROVAL.md`.

Written from an adversarial stance after implementation appeared complete —
assume Phase 4 is wrong until demonstrated otherwise (Sec. 29). This
documents both defects found and fixed during this engineering session and
areas deliberately checked and found clean.

## Defects found and fixed during this session

1. **`arithmetic` category marked `tool_required=True` without executing a
   tool.** The original generator flagged large-magnitude arithmetic as
   `invoke_tool` but never called `ToolRuntime`, leaving `tool_traces`
   empty. `validate_example`'s own invariant ("tool_required=True but no
   tool_traces recorded") caught this immediately at smoke-test scale
   (~1% of generated examples in that category). **Fix:** the `arithmetic`
   category no longer marks anything tool-required — large-expression tool
   invocation is the dedicated `tool_use` category's job, which does
   execute the runtime. Regression coverage:
   `tests/test_dataset.py::test_generators_produce_valid_ground_truth`
   exercises every registered generator's ground truth, which would fail
   again if this recurred.

2. **Diversity cap mathematically infeasible for 2-template families.** An
   initial `max_template_share_within_family` of 0.34 makes convergence
   impossible for any family with exactly 2 templates (two shares summing
   to 1.0 cannot both stay under 0.34) — such families deadlocked against
   the attempt budget. **Fix:** raised to 0.60 (documented in
   `config/dataset.yaml` inline) — still blocks one template from
   dominating a family, but is achievable for every actual template count
   in the generator suite.

3. **Under-specified `derivation_id` in `basic_algebra`.** The original key
   was `(family_id, x, shape)` — omitting the equation's other coefficients
   meant two structurally different problems sharing the same solution `x`
   and shape were forced into the same split-isolation bucket. Not a
   correctness bug (over-grouping is conservative, not a leak), but
   imprecise. **Fixed** to key on the full rendered prompt instead.

4. **Real eval/train contamination**, caught by the very check built to
   catch it. Before eval-suite content was reserved in the corpus build's
   deduplicators, a full pipeline run produced ~20 *exact* prompt
   collisions between eval suites and the training corpus — concentrated in
   the same low-diversity categories that later needed reduced token
   targets (ambiguity, undefined_operation, tool_error, etc.), where small
   operand ranges made an independent seed offset alone insufficient to
   guarantee disjointness. **Fix:** `build_dataset` now accepts
   `eval_reserved_examples` and pre-seeds the exact/near deduplicators with
   every eval-suite example before generating a single training example —
   see `docs/DATASET.md` "Order matters" and "Contamination prevention".
   Final full-scale `dataset contamination-check`: zero violations across
   1,833,697 examples and 725 eval cases.

5. **`check_near_duplicate_eval_vs_train` was O(eval × train).** Fine at
   smoke-test scale, but at 1.65M train examples and ~700 eval prompts this
   is well over a billion set operations and did not finish in a reasonable
   time on the actual full-scale corpus (killed after 5+ minutes with no
   result). **Fix:** rewrote as an inverted-index candidate filter (build a
   shingle -> eval-prompt-index map once, then a train example only scores
   Jaccard against eval prompts it shares at least one shingle with).
   Re-ran successfully in under a minute on the real corpus, with identical
   catching behavior (verified by the existing
   `test_build_contamination_report_flags_near_duplicate_eval_leak` test,
   which still passes).

6. **`dataset validate`/`dataset verify` silently reported success on zero
   records.** Found during the Sec. 28 fresh-clone recovery test, not
   during development: `data/processed/juniper-math-dataset-v1/` is
   tracked in Git (its small `shard_manifest.json`/`stats.json`/
   `DATASET_IDENTITY.sha256` metadata files are — see `.gitignore`'s Phase
   4 section), so a fresh clone has that directory present with zero
   `.jsonl` shard files inside it. `list_shard_files` only checked whether
   the directory existed, not whether it contained any shards, so `dataset
   validate` iterated zero records and printed "PASS: schema validation" —
   exactly the "never print success while skipping unavailable inputs"
   failure Sec. 20 explicitly forbids. **Fix:** `list_shard_files` now
   raises when zero `.jsonl` files are found, whether or not the directory
   itself exists. Regression coverage:
   `tests/test_dataset.py::test_list_shard_files_fails_honestly_on_existing_but_empty_directory`.
   This is exactly why the fresh-clone test matters as an *actual* test,
   not a formality — this defect was invisible on the development machine
   (which always has a populated build) and only surfaced against a
   genuinely fresh checkout.

7. **Several mypy type errors** (dict-literal branch inference across
   if/elif/else, a tuple passed where `*args` was expected, a
   `Literal[...]` string parameter). Fixed; `mypy` now reports zero issues
   across all 51 source files including the new `dataset` package.

## Areas checked and found clean

- **Licensing/provenance**: v1 acquires zero external data — see
  `reports/PHASE4_PROVENANCE_LICENSE_REVIEW.md`. Nothing to misjudge.
- **Synthetic generator correctness**: every accepted example's ground
  truth was independently recomputed (deterministic cases) or actually
  executed (tool cases) at generation time — zero
  `rejected_ground_truth_mismatch` in the full build's counters would be
  suspicious if it meant "the check never ran"; it does not — the same
  check rejected 78,960 diversity-cap violations and 1,897,780 exact
  duplicates in the same run, so the pipeline demonstrably rejects things.
- **Accidental answer leakage**: prompts do not embed `expected_answer`
  anywhere (verified by inspection of `make_example`/`Example` construction
  — the answer is a separate field, never string-interpolated into
  `prompt`).
- **Template repetition**: `diversity_caps` enforced during generation (see
  defect 2 above for the one place this was initially wrong).
- **Wrong tool/result mismatches**: `dataset verify` re-executes all 95,472
  recorded tool traces live and byte-compares — zero mismatches on the full
  corpus.
- **Split leakage / template-family leakage**:
  `check_derivation_id_isolation` — zero violations on the full corpus.
- **Evaluation contamination**: see defect 4 above; zero after the fix.
- **Tokenizer incompatibility**: every token count comes from the actual
  frozen `JuniperTokenizer`, not an estimate; `fraction_exceeding_context`
  is 0.0 (max example is 257 tokens against a 1,024-token limit).
- **Unstable hashes / filesystem-order nondeterminism**: shards are sorted
  by `example_id` before writing (never insertion/iteration order); the
  deterministic-rebuild test (`reports/PHASE4_DATASET_VALIDATION.md`)
  independently confirms byte-identical reconstruction.
- **Python hash randomization dependence**: no `hash()` calls drive any
  reproducibility-critical path — `juniper_math.dataset.idgen` uses SHA-256
  exclusively for every seed/ID derivation. (One draft line in
  `eval_suites.py` briefly used `hash()` as a component of a seed offset
  during early development; removed before the eval suites were
  regenerated — the final code path uses only `sum(ord(c) for c in name)`.)
- **RNG misuse**: `random.Random` instances are always constructed from a
  SHA-256-derived seed (`derive_seed`), never left at global/unseeded
  state; no generator mutates shared RNG state across calls.
- **Locale/time dependence**: no generator reads wall-clock time, locale,
  or environment-dependent formatting.
- **RAM/disk usage**: peak build RSS 3.5 GB (16 GB system budget); on-disk
  footprint 1.3 GB (256 GB budget) — see `reports/PHASE4_REPORT.md`.
- **Stale documentation/hashes**: `docs/DATASET.md`, `docs/EVALUATIONS.md`,
  `docs/CLI.md`, `docs/RECOVERY.md`, and `README.md` were all updated in
  this session to describe the actual implemented pipeline; every hash in
  `manifests/artifacts.yaml` was regenerated from actual file bytes via
  `python -m juniper_math hash file <path>`, never hand-typed, and verified
  with `python -m juniper_math hash verify`.

## Not independently checked (documented limitation, not a hidden gap)

- No large-N manual human read-through of generated prompts beyond spot
  inspection during development and the construction-time semantic-reason
  labeling each ambiguity/missing/undefined/unsupported case carries. GPT-5.6
  Terra is explicitly authorized to sample and manually audit prompts as
  part of independent review (see `reports/PHASE4_TERRA_HANDOFF.md`).
- No adversarial fuzzing of the generator code beyond what the full-scale
  build itself exercised (3.8M generation attempts across all 24
  categories, all difficulty levels, both the happy path and the
  diversity-cap/dedup/context-length rejection paths).

## Outcome

All findings listed under "Defects found and fixed" were fixed and
verified during this session — none are outstanding. This report itself is
part of the record GPT-5.6 Terra reviews; Terra may disagree with any
judgment call here and is authorized to remediate directly.
