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
