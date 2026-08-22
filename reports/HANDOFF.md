# Phase 0 — Handoff to Opus 5

This is evidence for independent review, not a conclusion. Please form your
own judgment — disagreement with anything below is the point of this gate.

## Where things are

- **Repository:** `https://github.com/Cinqic/juniper-math-1`, branch `main`.
- **Candidate commit (pinned):** `b39e60031e85822a293ca60b5676ce5b7286a66f`,
  tagged `phase-0-review-candidate`.

  > **Correction (F-11):** this originally described the candidate as "the
  > `HEAD` of `main` at the time this file was pushed" — a moving reference.
  > An audit handoff must pin the exact commit so the reviewed state is
  > unambiguous. The SHA above is the state that was actually reviewed.
  >
  > **That candidate did not pass review.** It returned CHANGES REQUIRED
  > (1 HIGH, 4 MEDIUM, 6 LOW). See `reports/OPUS5_PHASE0_REVIEW.md`. The
  > approved state is tagged `phase-0-foundation`.
- **Phase 0 report:** `reports/PHASE0_REPORT.md`
- **Self-review report:** `reports/SELF_REVIEW.md`
- **Recovery test report:** `reports/RECOVERY_TEST_REPORT.md`

## How to reproduce everything yourself

```bash
git clone https://github.com/Cinqic/juniper-math-1.git
cd juniper-math-1
# Debian/Ubuntu/Linux Mint: sudo apt install -y python3-venv python3-pip
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e . --no-deps

python -m juniper_math validate-env
python -m juniper_math validate-config
pytest -v
python -m juniper_math hash verify
python -m juniper_math evals validate
python -m juniper_math manifests-validate
python -m juniper_math deps-check
python -m juniper_math evals verify
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
