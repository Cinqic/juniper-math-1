# Phase 4 Terra Handoff Package

This package is self-contained: GPT-5.6 Terra should be able to clone the
repository with zero knowledge of this conversation and independently audit
Phase 4 end to end using only this document and the repository itself.

## Repository and candidate

- **Repository:** `https://github.com/Cinqic/juniper-math-1`
- **Candidate tag:** `phase-4-review-candidate` (non-final — see
  `docs/GIT_POLICY.md`)
- **Resolve the candidate commit:**
  ```bash
  git fetch origin
  git rev-parse phase-4-review-candidate^{commit}
  ```
  (Never trust a hardcoded SHA in this document — Phase 1 and Phase 3 both
  hit the "a commit cannot self-reference its own resulting SHA" issue; the
  tag is the source of truth, not a pasted hash.)
- **Approved starting foundation:** `phase-3-tools`
  (`1cb070945dd1f968f0dd10e6ceb915e0d24fac0a`). `git diff phase-3-tools --
  config/architecture.yaml config/tokenizer.yaml config/tools.yaml
  releases/ tools/schemas/` must be empty — Phase 4 does not touch the
  frozen architecture, tokenizer, or tool protocol.

## Identity

- **Dataset:** `juniper-math-dataset-v1` (`config/dataset.yaml`)
- **Tokenizer:** `juniper-math-tokenizer-v1` (frozen Phase 2, unchanged)
- **Tool protocol:** `juniper-tool-protocol-v1` v1.0.0 (frozen Phase 3, unchanged)
- **Final example/token counts, split counts, shard manifest identity:** see
  `reports/PHASE4_REPORT.md` "Dataset statistics" and
  `data/processed/juniper-math-dataset-v1/shard_manifest.json` /
  `DATASET_IDENTITY.sha256` (both tracked in Git despite living under the
  otherwise-gitignored `data/processed/` — see `.gitignore`'s Phase 4
  section).
- **Evaluation suite identities:** see
  `reports/PHASE4_EVALUATION_FREEZE.md` — four suites, 725 cases, hashes in
  `manifests/artifacts.yaml`.

## Reports to read (in order)

1. `reports/PHASE4_REPORT.md` — the main engineering report.
2. `reports/PHASE4_PROVENANCE_LICENSE_REVIEW.md` — why v1 has zero external
   sources.
3. `reports/PHASE4_EVALUATION_FREEZE.md` — the four eval suites.
4. `reports/PHASE4_DATASET_VALIDATION.md` — deterministic rebuild evidence.
5. `reports/PHASE4_SELF_REVIEW.md` — defects found and fixed during
   engineering, and what was and was not independently checked.
6. `docs/DATASET.md` — the pipeline reference (read alongside
   `config/dataset.yaml`).

## Complete command reference

### Environment setup (fresh clone)

```bash
git clone https://github.com/Cinqic/juniper-math-1.git
cd juniper-math-1
git checkout phase-4-review-candidate
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

### Test commands

```bash
python -m juniper_math validate-env
python -m juniper_math validate-config
python -m juniper_math hash verify
python -m juniper_math manifests-validate
python -m juniper_math deps-check
python -m juniper_math model --device cpu
python -m juniper_math tokenizer validate
python -m juniper_math tools self-test
python -m juniper_math evals validate
python -m juniper_math evals verify
pytest -v
ruff check .
ruff format --check .
mypy
```

### Dataset rebuild commands

```bash
# Order matters — eval suites must exist before the corpus build so the
# corpus build's dedup pass can reserve their content (contamination
# isolation, Sec. 13; see docs/DATASET.md "Order matters").
python -m juniper_math dataset eval-suites-build
python -m juniper_math dataset build            # ~9 minutes, ~3.5GB peak RAM, ~1.3GB disk
python -m juniper_math dataset validate
python -m juniper_math dataset verify            # re-executes every tool trace live
python -m juniper_math dataset stats
python -m juniper_math dataset contamination-check
```

### Contamination checks

`python -m juniper_math dataset contamination-check` — checks
`derivation_id` split isolation, exact cross-split duplicates, and
near-duplicate eval/train prompt pairs (inverted-index-filtered Jaccard,
not a naive quadratic scan — see `juniper_math.dataset.contamination`'s
module docstring for why that mattered at this corpus's actual scale).

### Hash verification

```bash
python -m juniper_math hash verify
```
Covers 26 frozen artifacts, including the 9 new Phase 4 entries added this
phase (dataset config, four eval suites, shard manifest, stats, dataset
identity file).

### Fresh-clone recovery procedure

See `docs/RECOVERY.md`, steps 1–13 (steps 1–12 are the pre-existing Phase
0–3 procedure, unmodified; step 13 is the new Phase 4 dataset
reconstruction, added this phase).

## Known limitations and risks (see PHASE4_SELF_REVIEW.md for detail)

- Six categories (`ambiguity`, `missing_information`, `undefined_operation`,
  `unsupported_capability`, `scientific_notation`, `tool_error`) are below
  their configured proportional token target by design — bounded
  combinatorial diversity in their template families, not a defect. Terra
  may disagree with this judgment call and expand their generators.
- Zero external data sources acquired for v1 (see
  `reports/PHASE4_PROVENANCE_LICENSE_REVIEW.md`) — a deliberate, documented
  scope decision, not an oversight. Terra is explicitly authorized to
  propose and evaluate candidate external sources for a future dataset
  version.
- Near-duplicate detection during corpus build is family-scoped with a
  bounded 200-example recent-window, not full corpus-wide pairwise
  comparison (documented trade-off — see `docs/DATASET.md`
  "Deduplication"). The independent `contamination-check` pass (used for
  eval/train isolation) uses full comparison against the small eval suites,
  which is where it actually matters for leakage; it does not check
  near-duplication *within* the training corpus itself at full pairwise
  granularity.
- No large-N manual human read-through of generated prompts beyond
  construction-time semantic labeling and development-time spot checks.

## Self-review findings (summary — full detail in PHASE4_SELF_REVIEW.md)

Six real defects were found and fixed during this engineering session,
including a genuine eval/train contamination leak (caught by the
contamination checker itself before this report was written) and a
performance bug in that same checker that made it impractical at full
corpus scale (also found and fixed before this report was written). See
`reports/PHASE4_SELF_REVIEW.md` for the complete list with root cause and
fix for each.

## Authority granted to Terra

Per the Phase 4 instructions (Sec. 34), GPT-5.6 Terra is authorized to:

- independently audit all Phase 4 work;
- reproduce dataset generation and independently verify the deterministic
  rebuild claim in `reports/PHASE4_DATASET_VALIDATION.md`;
- inspect external source licensing (none currently acquired — Terra may
  propose sources for a future version);
- challenge dataset composition, including the six intentionally-small
  categories and the overall category mixture;
- independently verify mathematical ground truth;
- run and extend contamination tests;
- inspect deduplication behavior and thresholds;
- inspect split leakage and template-family leakage;
- independently rebuild corpus artifacts and compare hashes;
- fix ordinary Phase 4 defects directly;
- regenerate Phase 4 artifacts and hashes;
- update any Phase 4 report;
- push remediation commits;
- perform the fresh-clone recovery procedure;
- issue final Phase 4 approval;
- create the `phase-4-dataset` final tag (reserved for Terra — not created
  by this engineering session, per `docs/GIT_POLICY.md`'s tag convention);
- authorize Phase 5.

Terra must not silently modify the frozen architecture (`config/
architecture.yaml`), tokenizer (`config/tokenizer.yaml` and
`releases/tokenizer/`), or Phase 3 tool protocol (`config/tools.yaml` and
`tools/schemas/`) — any defect discovered there should be documented and
escalated, not silently rewritten, exactly as this engineering session was
instructed.

## Explicit non-claim

This handoff does **not** claim independent Phase 4 approval. `config/
project.yaml`'s bare `phase_approval` block (the convention: `phase_N_approval`
keys are frozen once a phase is superseded, e.g. `phase_3_approval`; the
bare `phase_approval` key always describes whichever phase is currently in
progress — Phase 4 now) records `terra_independent_review: "pending"` and
`terra_final_approval: "pending"` — both literal, not placeholders that
were meant to be filled in and forgotten. Phase 5 remains `NOT_AUTHORIZED`
until Terra completes review and updates that file.
