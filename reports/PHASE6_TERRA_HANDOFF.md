# Phase 6 Terra Handoff Package

This package is self-contained: GPT-5.6 Terra should be able to clone the
repository with zero knowledge of this session and independently audit
Phase 6 end to end using only this document and the repository itself.

## Repository and candidate

- **Repository:** `https://github.com/Cinqic/juniper-math-1`
- **Candidate tag:** `phase-6-pilot-candidate` (non-final — see
  `docs/GIT_POLICY.md`)
- **Resolve the candidate commit:**
  ```bash
  git fetch origin
  git rev-parse phase-6-pilot-candidate^{commit}
  ```
  (Never trust a hardcoded SHA in this document — resolve the tag
  yourself; a commit cannot correctly self-reference its own resulting
  SHA, the same issue Phase 1, Phase 3, and Phase 4's handoffs each hit.)
- **Approved starting foundation:** `phase-5-smoke`
  (`73792c04f365c6f139a979f6950fa87be2af5d76`).
  `git diff phase-5-smoke -- config/architecture.yaml config/tokenizer.yaml
  config/dataset.yaml config/tools.yaml releases/ tools/schemas/ evals/`
  must be empty — Phase 6 does not touch the frozen architecture,
  tokenizer, dataset config, tool protocol, or evaluation suites.
  `config/training.yaml`, `src/juniper_math/training_config.py`, and
  `src/juniper_math/smoke_data.py` (Phase 5's own config/loader/selector)
  are also unmodified — `git diff phase-5-smoke -- config/training.yaml
  src/juniper_math/training_config.py src/juniper_math/smoke_data.py` must
  be empty.

## Identity

- **Architecture:** v0.1.0, 5,004,032 trainable parameters (unchanged)
- **Tokenizer:** `juniper-math-tokenizer-v1` (unchanged)
- **Dataset:** `juniper-math-dataset-v1`, whole-dataset identity
  `bf9933f032a58b4eb618b32156783b8563097a5fc1c0ef26be4f76445128d25a`
  (unchanged)
- **Tool protocol:** `juniper-tool-protocol-v1` v1.0.0 (unchanged)
- **Pilot config:** `config/training_phase6_pilot.yaml`, sha256
  `cb2eb8249fef4c8cc7b87c3e3e4f9807988aa230268780f1214df64a6d676343`
  (`manifests/artifacts.yaml` artifact_id `pilot_training_config`)
- **Pilot subset identity:** 137,057 train examples / 5,002,683 tokens,
  3,043 validation examples / 112,765 tokens, seed 5004032 —
  `experiments/phase6-pilot/experiment.yaml` and
  `data/processed/phase6-pilot/pilot_manifest.json` (not committed;
  reproducible from the frozen dataset + this config + seed — regenerate
  with `python -m juniper_math train pilot-run`, which writes the manifest
  as a side effect before training)
- **Final checkpoint:** `checkpoints/phase6-pilot/step_000320_final.pt`,
  sha256 `6087d10aa802080b45bf748c70496e2ea8bef64e06c827d3b3387f196032f81b`,
  60,123,779 bytes (not committed; disposable/reproducible — see
  `checkpoints/README.md`)

## Reports to read (in order)

1. `reports/PHASE6_PLAN.md` — the design/plan.
2. `reports/PHASE6_RESULTS.md` — the full run evidence (this is the
   primary report — everything else summarizes or points into it).
3. `reports/PHASE6_SELF_REVIEW.md` — defects found and fixed during
   engineering, and what was and was not independently checked.
4. `reports/PHASE6_COMPLETION.md` — acceptance-gate summary and the
   evidence-backed Phase 7 recommendation table.
5. `docs/PILOT_TRAINING.md` — the pipeline reference (read alongside
   `config/training_phase6_pilot.yaml`).

## Files added and modified

**New files:**
- `config/training_phase6_pilot.yaml`
- `src/juniper_math/pilot_training_config.py`
- `src/juniper_math/pilot_data.py`
- `src/juniper_math/pilot_eval.py`
- `src/juniper_math/pilot_pipeline.py`
- `tests/test_pilot_training_config.py`, `tests/test_pilot_data.py`,
  `tests/test_pilot_eval.py`, `tests/test_pilot_pipeline.py`,
  `tests/test_pilot_cli.py`
- `docs/PILOT_TRAINING.md`
- `reports/PHASE6_PLAN.md`, `reports/PHASE6_RESULTS.md`,
  `reports/PHASE6_SELF_REVIEW.md`, `reports/PHASE6_COMPLETION.md`, this
  file
- `experiments/phase6-pilot/experiment.yaml`,
  `experiments/phase6-pilot/train_log.jsonl`,
  `experiments/phase6-pilot/resume_test_log.jsonl`

**Modified files (all additive — no frozen behavior changed):**
- `src/juniper_math/trainer.py` — type hints broadened from
  `TrainingConfig`/`TokenizedSmokeDataset` to a structural
  `TrainingConfigLike` `Protocol`/`torch.utils.data.Dataset[Any]`, so
  Phase 5's and Phase 6's configs/datasets share one training loop. Pure
  static-typing change; `tests/test_trainer.py` (Phase 5's own trainer
  tests, unmodified) still pass unchanged, confirming no runtime behavior
  changed.
- `src/juniper_math/dataset/shard.py` — additive: exported the existing
  `_BEHAVIOR_TAG` mapping as public `BEHAVIOR_TAG` (kept as a back-compat
  alias) and added `expected_completion()`, reused by the new capability
  evaluator instead of a second copy of the same ground-truth mapping.
  `render_training_text`'s own output is provably unchanged —
  `tests/test_dataset.py`'s pre-existing `render_training_text` tests
  pass unmodified, and three new tests directly check
  `expected_completion` against every `BEHAVIOR_TAG` entry.
- `src/juniper_math/cli.py` — added `train pilot-run`, `train
  pilot-resume-test`, `pilot-evaluate`, `pilot-infer` subcommands
  (additive); corrected a stale module docstring (see
  `reports/PHASE6_SELF_REVIEW.md` defect 3).
- `manifests/artifacts.yaml` — added the `pilot_training_config` entry.
- `tests/test_metadata.py` — fixed two pre-existing broken tests (see
  `reports/PHASE6_SELF_REVIEW.md` defect 2) and updated for the new
  `phase_6_engineering` block.
- `tests/test_dataset.py` — added `expected_completion` tests.
- `config/project.yaml` — added `phase_6_engineering` block, updated
  `next_phase` (`current_phase` intentionally still 5 — see below).
- `README.md`, `docs/CLI.md`, `docs/TRAINING.md`, `docs/RECOVERY.md`,
  `checkpoints/README.md` — Phase 6 documentation additions, plus
  correction of pre-existing staleness found during self-review (see
  `reports/PHASE6_SELF_REVIEW.md` defect 3).

## Complete command reference

### Environment setup (fresh clone)

```bash
git clone https://github.com/Cinqic/juniper-math-1.git
cd juniper-math-1
git checkout phase-6-pilot-candidate
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
python -m juniper_math model --device cuda
python -m juniper_math tokenizer validate
python -m juniper_math tools self-test
python -m juniper_math evals validate
pytest -v
ruff check .
ruff format --check .
mypy
```

### Dataset rebuild (only if `data/processed/juniper-math-dataset-v1/` shards are absent)

```bash
python -m juniper_math dataset eval-suites-build
python -m juniper_math dataset build
python -m juniper_math dataset validate
python -m juniper_math dataset verify
python -m juniper_math dataset contamination-check
```

### Phase 5 regression check (must still pass, unmodified)

```bash
python -m juniper_math train run --evaluate
python -m juniper_math train resume-test
```

### Phase 6 pilot reproduction

```bash
# Full canonical run — ~7 minutes on the target RTX 2060 (406.9s measured):
python -m juniper_math train pilot-run

# Faster smoke check of the pipeline itself (few minutes):
python -m juniper_math train pilot-run --max-steps 4 --eval-sample-size 3

# Resume-equivalence gate — ~6-7 minutes (three full training passes):
python -m juniper_math train pilot-resume-test

# Score an arbitrary checkpoint against all four frozen v2 suites:
python -m juniper_math pilot-evaluate --checkpoint checkpoints/phase6-pilot/step_000320_final.pt

# Single-prompt inference:
python -m juniper_math pilot-infer --checkpoint checkpoints/phase6-pilot/step_000320_final.pt --prompt "2 + 2 ="
```

### Hash verification

```bash
python -m juniper_math hash verify
```
Covers 29 frozen artifacts, including the 1 new Phase 6 entry
(`pilot_training_config`) added this phase.

### Fresh-clone recovery procedure

See `docs/RECOVERY.md`, steps 1–15 (steps 1–14 are the pre-existing Phase
0–5 procedure, unmodified; step 15 is the new Phase 6 pilot
reconstruction, added this phase).

## Known limitations and risks (see `PHASE6_SELF_REVIEW.md` for detail)

- `tool_error` has only 26 examples in the entire frozen train split
  (out of 1,553 dataset-wide) — a consequence of Phase 4's
  family-grouped split assignment concentrating this category's few
  families outside train, discovered during Phase 6, not a Phase 6
  defect. Terra may wish to independently confirm this via
  `data/processed/phase6-pilot/pilot_selection_audit.json`'s
  `category_record_counts.tool_error` field and consider whether it
  affects Phase 7 planning.
- Resume equivalence at pilot scale is tolerance-based (max param diff
  ≈2.37e-4, max loss diff ≈2.90e-4, both against a `<1e-2` threshold), not
  bitwise like Phase 5's smoke run. This was run once; Terra may wish to
  repeat it for additional confidence (see
  `reports/PHASE6_SELF_REVIEW.md` "Not independently checked").
- Only peak learning rate was screened before the canonical run (Sec. 14
  explicitly discourages an exhaustive sweep); the follow-up 1.0e-3 data
  point was deliberately not adopted — Terra may disagree with that
  judgment call and either accept 6.0e-4 as final or request a fuller
  sweep.
- Capability accuracy (0-0.5% across all four suites at every milestone)
  is the expected result at this scale and is not, and must not be
  represented as, a capability claim.
- Pilot checkpoints are not archived to a remote store (disposable, per
  `checkpoints/README.md`) — reproducibility is demonstrated via the
  resume-comparison gate and the deterministic pilot-selection algorithm,
  not via checkpoint preservation. Terra is authorized to require
  archival if it disagrees with this judgment call.

## Self-review findings (summary — full detail in PHASE6_SELF_REVIEW.md)

Four defects were found and fixed during this engineering session: a
validation-dataset padding-waste bug caught only by an actual timed run
(not code review), two pre-existing broken tests left stale by the Phase 5
approval merge (unrelated to Phase 6's own code, fixed while establishing
the starting state), stale README/CLI documentation contradicting
`config/project.yaml`'s own already-approved state, and a `mypy`
Protocol/frozen-dataclass typing incompatibility. See
`reports/PHASE6_SELF_REVIEW.md` for the complete list with root cause and
fix for each, plus one deliberately-unfixed minor cosmetic observation
(duplicate milestone log lines only reachable via a tiny `--max-steps`
override no real run uses).

## Authority granted to Terra

Per Sec. 3 and Sec. 39 of the Phase 6 instructions, GPT-5.6 Terra is
authorized to:

- independently audit all Phase 6 work;
- reproduce pilot-subset selection and independently verify the
  determinism claim;
- independently re-run `train pilot-run` and `train pilot-resume-test`
  and compare results;
- challenge the pilot dataset's category stratification, token budget,
  sequence-length/packing decision, and the learning-rate screening
  methodology;
- independently verify capability-evaluation scoring logic
  (`juniper_math.pilot_eval`) against the frozen suites;
- inspect checkpoint metadata and hashes;
- fix ordinary Phase 6 defects directly;
- regenerate Phase 6 artifacts and hashes;
- update any Phase 6 report;
- push remediation commits;
- perform the fresh-clone recovery procedure;
- issue final Phase 6 approval;
- create the `phase-6-pilot` final tag (reserved for Terra — not created
  by this engineering session, per `docs/GIT_POLICY.md`'s tag
  convention);
- authorize Phase 7.

Terra must not silently modify the frozen architecture
(`config/architecture.yaml`), tokenizer (`config/tokenizer.yaml` and
`releases/tokenizer/`), dataset (`config/dataset.yaml` and the frozen
shard manifest/stats/identity files), tool protocol (`config/tools.yaml`
and `tools/schemas/`), or evaluation suites (`evals/phase4_*_v2.json`) —
any defect discovered there should be documented and escalated, not
silently rewritten, exactly as this engineering session was instructed.
Terra must also not silently treat Phase 6's pilot checkpoint as a Phase 7
starting point without an explicit, documented decision to do so (Sec.
28).

## Explicit non-claim

This handoff does **not** claim independent Phase 6 approval. `config/
project.yaml`'s `phase_6_engineering` block records
`terra_independent_review: "not_yet_performed"` — literal, not a
placeholder left to rot. Phase 7 remains **NOT AUTHORIZED** until Terra
completes review and updates that file.
