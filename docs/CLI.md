# CLI Usage

Juniper Math 1 has one canonical command surface:

```bash
python -m juniper_math <command> [args...]
```

(An installed console script `juniper-math` resolves to the identical entry
point after `pip install -e .` — use whichever is convenient, they are the
same code path, not two separate implementations.)

## Functional in Phase 0

| Command | Purpose |
|---|---|
| `status` | Report current phase/status, architecture version, git commit |
| `validate-env` | PASS/WARNING/FAIL environment report (Python, PyTorch, CUDA, hardware, git) |
| `validate-config` | Validate `config/architecture.yaml` and `config/project.yaml` |
| `seed-test` | Exercise the deterministic seed helper, report what was seeded |
| `evals validate` | Validate the frozen evaluation suite's schema/IDs/categories |
| `hash file <path>` | Print the SHA-256 of a file |
| `hash verify` | Verify every artifact in `manifests/artifacts.yaml` against its recorded hash |
| `manifests-validate` | Validate the source and license manifests |

## Functional in Phase 1

| Command | Purpose |
|---|---|
| `model [--device cpu\|cuda] [--no-forward-check]` | Construct the frozen architecture, verify the exact trainable parameter count (5,004,032), run a synthetic forward pass |
| `checkpoint inspect <path>` | Safely report checkpoint metadata (step, tokens seen, architecture identity, schema version, size) without restoring model/optimizer/RNG state |

## Not yet implemented (later phases)

These commands exist as honest placeholders — they print an explicit
`not implemented until Phase N` message and exit with status 2. They never
silently succeed or fabricate output.

| Command | Owning phase |
|---|---|
| `train` | Phase 6/7 (real training) |
| `evaluate` | later |
| `infer` | later |
| `tokenizer` | Phase 2 |
| `tool-test` | Phase 3 |
| `dataset` | Phase 4 |

## Examples

```bash
python -m juniper_math status
python -m juniper_math validate-env
python -m juniper_math validate-config
python -m juniper_math seed-test --seed 5004032
python -m juniper_math evals validate
python -m juniper_math hash verify
python -m juniper_math model
python -m juniper_math checkpoint inspect path/to/checkpoint.pt
python -m juniper_math --version
```

## Commands added during Opus 5 Phase 0 remediation

| Command | Purpose |
|---|---|
| `evals verify` | Recompute every deterministic evaluation answer from its structured `verification` metadata and compare against the recorded `expected_answer`. Exits 1 on any mismatch. |
| `deps-check` | Cross-check the direct dependencies declared in `pyproject.toml` against `manifests/licenses.yaml`. Exits 1 if a declared dependency has no licensing entry, has the wrong scope, or if a license entry is stale. |

`evals validate` now runs schema validation **and** deterministic ground-truth
verification. `manifests-validate` now includes the dependency/license
cross-check. Both were previously schema-only — see
`reports/OPUS5_PHASE0_REVIEW.md` (F-02, F-05).
