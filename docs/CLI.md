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

## Not yet implemented (later phases)

These commands exist as honest placeholders — they print an explicit
`not implemented until Phase N` message and exit with status 2. They never
silently succeed or fabricate output.

| Command | Owning phase |
|---|---|
| `model` | Phase 1 |
| `train` | Phase 1 |
| `evaluate` | Phase 1 |
| `infer` | Phase 1 |
| `tokenizer` | Phase 2 |
| `tool-test` | Phase 3 |
| `dataset` | Phase 4 |
| `checkpoint` | Phase 1 |

## Examples

```bash
python -m juniper_math status
python -m juniper_math validate-env
python -m juniper_math validate-config
python -m juniper_math seed-test --seed 5004032
python -m juniper_math evals validate
python -m juniper_math hash verify
python -m juniper_math --version
```
