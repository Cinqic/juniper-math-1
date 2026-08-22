# Recovery Guide

This is the authoritative procedure to restore Juniper Math 1 from nothing
but the canonical GitHub repository, assuming the local machine (including
OS, caches, virtual environments, and any local-only files) has been wiped.
See [`reports/RECOVERY_TEST_REPORT.md`](../reports/RECOVERY_TEST_REPORT.md)
for evidence this procedure was actually exercised, not just written.

## Prerequisites (system-level, not covered by this repo)

1. A fresh Linux installation.
2. Install Git: `sudo apt-get install -y git` (or your distribution's equivalent).
3. Install an NVIDIA driver compatible with your GPU (only required for
   GPU-accelerated work; CPU-only recovery/validation does not need this
   step). Verify with `nvidia-smi`.
4. Install Python 3.10–3.12 (e.g. via your distribution's package manager or
   `pyenv`).

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

# 4. Install project dependencies (repository-controlled, not ad hoc)
pip install -e ".[dev]"

# 5. Validate the environment
python -m juniper_math validate-env

# 6. Validate configuration
python -m juniper_math validate-config

# 7. Run the automated test suite
pytest -v

# 8. Verify artifact hashes
python -m juniper_math hash verify

# 9. Validate the frozen evaluation suite
python -m juniper_math evals validate

# 10. Validate manifests
python -m juniper_math manifests-validate

# 11. Run harmless dry-run/status commands
python -m juniper_math status
python -m juniper_math model      # expected: "not implemented until Phase 1", exit code 2

# 12. Confirm project status
python -m juniper_math status | grep "Phase status"
```

A successful recovery satisfies all of the following:

- Steps 5–10 report PASS (CUDA WARNING is acceptable on non-GPU recovery
  environments and must be separately validated on the actual target
  hardware — see the recovery test report for what was and was not
  exercised).
- `pytest` reports 0 failures.
- `status` reports `Phase status: AWAITING_OPUS_5_REVIEW` (or whatever the
  current authoritative status in `config/project.yaml` is).

## What this repository does not depend on

- No shell aliases.
- No machine-specific absolute paths (all paths resolve relative to the
  repository root via `juniper_math.paths.find_repo_root`).
- No undocumented environment variables.
- No hidden local caches required for correctness (caches under `data/`,
  `.venv/`, etc. are disposable and excluded from Git; nothing in this
  repository depends on their prior contents).
