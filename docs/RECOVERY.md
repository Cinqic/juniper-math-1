# Recovery Guide

This is the authoritative procedure to restore Juniper Math 1 from nothing
but the canonical GitHub repository, assuming the local machine (including
OS, caches, virtual environments, and any local-only files) has been wiped.
See [`reports/RECOVERY_TEST_REPORT.md`](../reports/RECOVERY_TEST_REPORT.md)
for evidence this procedure was actually exercised, not just written.

## Prerequisites (system-level, not covered by this repo)

These are the **system layer**. `requirements-lock.txt` reproduces the Python
layer exactly, but it cannot reproduce a kernel, a GPU driver, or an
interpreter — install those first.

1. A fresh Linux installation.
2. Install Git, Python 3.12, **and Python's venv/pip support**:

   ```bash
   sudo apt update
   sudo apt install -y git python3 python3-venv python3-pip
   ```

   `python3-venv` is not optional and is **not** included with `python3` on
   Debian, Ubuntu, or Linux Mint. Without it, step 3 below fails with
   `ensurepip is not available`. This was a real defect in the Phase 0 review
   candidate, which documented only "install Python" and was reproduced
   failing on the actual development host — see
   `reports/OPUS5_PHASE0_REVIEW.md` (F-04). If `python3-venv` is not found,
   try the version-specific package `python3.12-venv`.

   Verify before continuing:

   ```bash
   python3 -c "import ensurepip; print('venv support OK')"
   ```

3. Install an NVIDIA driver compatible with your GPU (only required for
   GPU-accelerated work; CPU-only recovery/validation does not need this
   step). Verify with `nvidia-smi`. The Python lock does **not** install or
   pin drivers.
4. Python 3.12 specifically. The validated dependency set requires it
   (`numpy` 2.5.2 needs >= 3.12), which is why `pyproject.toml` declares
   `>=3.12,<3.13`.

## Restoration steps

```bash
# 1. Clone the canonical repository
git clone https://github.com/Cinqic/juniper-math-1.git
cd juniper-math-1

# 2. (Optional) check out a specific reviewed commit/tag
# git checkout <commit-or-tag>

# 3. Create a fresh virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 4. Install the EXACT validated Phase 0 environment.
#    requirements-lock.txt pins every package to the version that actually
#    passed the Phase 0 gate. pyproject.toml's ranges are compatibility
#    metadata, not a reproducible environment.
pip install -r requirements-lock.txt
pip install -e . --no-deps

#    (Alternatively `scripts/bootstrap.sh` performs steps 3-4 plus the
#     prerequisite preflight. To deliberately test newer dependencies
#     instead of the lock: JUNIPER_USE_RANGES=1 ./scripts/bootstrap.sh)

# 5. Validate the environment
python -m juniper_math validate-env

# 6. Validate configuration
python -m juniper_math validate-config

# 7. Run the automated test suite
pytest -v

# 8. Verify artifact hashes
python -m juniper_math hash verify

# 9. Validate the frozen evaluation suite
python -m juniper_math evals validate   # schema + deterministic ground truth

# 10. Validate manifests and dependency licensing
python -m juniper_math manifests-validate
python -m juniper_math deps-check

# 11. Run harmless status and architecture commands
python -m juniper_math status
python -m juniper_math model --device cpu
python -m juniper_math tokenizer validate
python -m juniper_math tools self-test
python -m juniper_math evals verify

# 12. Confirm the approved project status
python -m juniper_math status

# 13. (Phase 4+) Reconstruct the dataset and re-freeze the evaluation suites.
#     data/processed/ is gitignored and disposable — everything needed to
#     regenerate it byte-for-byte lives in this repository's generators and
#     config/dataset.yaml. Eval suites MUST be built before the corpus
#     (contamination isolation — see docs/DATASET.md "Order matters").
python -m juniper_math dataset eval-suites-build
python -m juniper_math dataset build
python -m juniper_math dataset validate
python -m juniper_math dataset verify
python -m juniper_math dataset contamination-check

# 14. (Phase 5+) Reconstruct the smoke-training pipeline. The smoke subset
#     manifest, checkpoints, and JSONL step logs (other than the small
#     committed ones under experiments/phase5-smoke/) are gitignored and
#     disposable — re-running against the same frozen dataset build and
#     config/training.yaml reproduces an equivalent run (the resume-test
#     gate demonstrates this reproducibility directly). See
#     docs/TRAINING.md and reports/PHASE5_RESULTS.md.
python -m juniper_math train run --evaluate
python -m juniper_math train resume-test
```

A successful recovery satisfies all of the following:

- Steps 5–10 report PASS (CUDA WARNING is acceptable on non-GPU recovery
  environments and must be separately validated on the actual target
  hardware — see the recovery test report for what was and was not
  exercised).
- `pytest` reports 0 failures.
- `status` reports Phase 4 as `COMPLETE`; `config/project.yaml` identifies
  Phase 5, Smoke Pretraining, as implementation-complete and pending
  independent review (`config/project.yaml:phase_5_engineering`).
- Step 14's `train run`/`train resume-test` both report PASS on the actual
  target GPU hardware (they run on CPU with an explicit WARNING otherwise,
  which does not validate the intended hardware fit).

## What this repository does not depend on

- No shell aliases.
- No machine-specific absolute paths (all paths resolve relative to the
  repository root via `juniper_math.paths.find_repo_root`).
- No undocumented environment variables.
- No hidden local caches required for correctness (caches under `data/`,
  `.venv/`, etc. are disposable and excluded from Git; nothing in this
  repository depends on their prior contents).

## What this repository explicitly does NOT reproduce

`requirements-lock.txt` pins the **Python layer** only. It does not pin, and
must never be described as pinning:

- the Linux kernel or distribution;
- the NVIDIA driver or CUDA driver stack;
- GPU firmware;
- system packages (`git`, `python3`, `python3-venv`).

Those are the system prerequisites above. A successful `pip install -r
requirements-lock.txt` reproduces the tested Python environment; it says
nothing about whether the GPU works.

## Note on installation mode

Install the project **editable** (`pip install -e .`). `juniper_math.paths`
resolves the repository root by walking up for `pyproject.toml`, so a
non-editable install into `site-packages` raises `RepositoryRootNotFoundError`
at import time. Editable install is the supported and tested path.
