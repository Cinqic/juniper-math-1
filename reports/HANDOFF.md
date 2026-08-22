# Phase 0 — Handoff to Opus 5

This is evidence for independent review, not a conclusion. Please form your
own judgment — disagreement with anything below is the point of this gate.

## Where things are

- **Repository:** `https://github.com/Cinqic/juniper-math-1`, branch `main`.
- **Candidate commit:** the `HEAD` of `main` at the time this file was
  pushed — run `git log -1 --format=%H` against `origin/main` for the exact
  hash, or see the commit history: `9954daf` and everything after it on
  `main` (this file's own commit included) constitute the full Phase 0
  candidate.
- **Phase 0 report:** `reports/PHASE0_REPORT.md`
- **Self-review report:** `reports/SELF_REVIEW.md`
- **Recovery test report:** `reports/RECOVERY_TEST_REPORT.md`

## How to reproduce everything yourself

```bash
git clone https://github.com/Cinqic/juniper-math-1.git
cd juniper-math-1
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m juniper_math validate-env
python -m juniper_math validate-config
pytest -v
python -m juniper_math hash verify
python -m juniper_math evals validate
python -m juniper_math manifests-validate
python -m juniper_math status
ruff check . && ruff format --check . && mypy
```

Every one of these is expected to report PASS / all-green as of the
candidate commit. If any does not reproduce, that is itself a finding worth
reporting.

## Key files to inspect

- Architecture: `config/architecture.yaml`, `docs/ARCHITECTURE.md`,
  `src/juniper_math/architecture.py`
- Project status: `config/project.yaml`, `src/juniper_math/metadata.py`
- Evaluation suite: `evals/phase0_v1.json`, `docs/EVALUATIONS.md`
- Manifests: `manifests/sources.yaml`, `manifests/licenses.yaml`,
  `manifests/artifacts.yaml`
- CLI: `src/juniper_math/cli.py`, `docs/CLI.md`
- Recovery: `docs/RECOVERY.md`
- ADRs: `docs/adr/0001`–`0008`

## Known limitations and non-blocking warnings (already disclosed in full in the reports above — not hidden here)

- Recovery test ran on the same host as development, not a separately
  wiped machine — see `reports/RECOVERY_TEST_REPORT.md`.
- Evaluation suite is a compact 22-case baseline, intentionally not
  exhaustive.
- No GPU-marked test exists yet (no GPU-bound code in Phase 0).
- `mypy` scoped to `src/` only, not `tests/`.

## What was NOT done, deliberately (later-phase scope, not gaps)

Transformer implementation, tokenizer training, dataset construction, tool
runtime, real training/checkpoints/releases. See `docs/adr/0008` for the
phase-boundary rationale.

## Explicit note on review integrity

No test was skipped to make this review easier, no warning was hidden or
relabeled, and CUDA was genuinely available on the machine that ran these
tests (a real RTX 2060) rather than assumed — see the environment section
of `reports/RECOVERY_TEST_REPORT.md` for exactly what was and wasn't
exercised.
