# Testing

Test runner: `pytest`, configured in [`pyproject.toml`](../pyproject.toml)
(`testpaths = ["tests"]`).

## Running tests

```bash
pip install -e ".[dev]"
pytest -v
```

## GPU-marked tests

Tests requiring a CUDA device are marked `@pytest.mark.gpu` and are skipped
automatically (via a `pytest_collection_modifyitems` hook in
`tests/conftest.py`) when `torch.cuda.is_available()` is `False`. This keeps
the suite runnable on CPU-only machines and CI, while still exercising GPU
paths on the RTX 2060 target hardware.

## Lint / format / type-check

```bash
ruff check .
ruff format --check .
mypy
```

## What Phase 0 tests cover

- Configuration loading and validation (architecture, project metadata) —
  including malformed/missing-field rejection.
- Deterministic seed helper behavior.
- SHA-256 hashing utility (including missing-file handling).
- Evaluation suite schema, uniqueness of IDs, category validity.
- Manifest loading and validation (sources, licenses, artifacts) including
  hash-mismatch detection.
- CLI argument parsing and exit codes, including honest "not implemented"
  behavior for later-phase commands.
- Environment reporting structure (without asserting a specific machine's
  hardware).

Tests assert real behavior — including specific error messages/types for
invalid input — not `assert True` placeholders.

## Regression tests for independent-review findings

Phase 0's independent review found defects that the original 63-test suite
could not detect. Each is now covered by a dedicated regression test:

| Finding | Test |
|---|---|
| F-01 wrong `tool-001` ground truth | `tests/test_verification.py::test_original_tool_001_defect_is_now_caught` re-injects the exact wrong value and asserts rejection |
| F-02 no ground-truth validation | `tests/test_verification.py` (whole module) |
| F-03 no dependency lock | `tests/test_bootstrap_and_lock.py` lock coverage/pinning tests |
| F-04 recovery venv prerequisite | `tests/test_bootstrap_and_lock.py` bootstrap preflight and docs tests |
| F-05 NumPy unlicensed | `tests/test_dependency_licenses.py` |
| F-07 git status honesty | `tests/test_git_state.py` (clean / dirty / non-repo / missing git / timeout) |
| F-08 behavior vocabulary | `tests/test_evals.py::test_undefined_operation_is_not_labelled_missing_information` |

Tests here exist to prevent specific defects, not to inflate a count.
