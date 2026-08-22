# Juniper Math 1

A purpose-built, approximately five-million-parameter mathematical language
model. **No model is trained yet.** This repository is currently in
**Phase 0: Foundation and Recovery** — engineering scaffolding only.

## Research question

> How capable, reliable, efficient, and truthfully calibrated can an
> approximately five-million-parameter purpose-built mathematical language
> model become at understanding natural-language math problems, decomposing
> them into verifiable operations, selecting and operating deterministic
> mathematical tools, detecting ambiguity and unsupported requests, verifying
> computational results, and producing accurate mathematical answers?

## Current status

```
Phase:  0 — Foundation and Recovery
Status: AWAITING_OPUS_5_REVIEW
```

Phase 0 implementation and self-review by Claude Sonnet 5 are complete.
The phase is awaiting independent review by Claude Opus 5 and final
inspection by Cinqic. **Phase 1 is not authorized.** See
[`config/project.yaml`](config/project.yaml) for the canonical,
machine-readable status and [`reports/PHASE0_REPORT.md`](reports/PHASE0_REPORT.md)
for the full engineering report.

## Principles

- **GitHub is canonical.** Local storage is disposable — see
  [`docs/GIT_POLICY.md`](docs/GIT_POLICY.md) and
  [`docs/RECOVERY.md`](docs/RECOVERY.md).
- **Deterministic tools over neural guessing.** Arithmetic correctness comes
  from verifiable tool execution, not memorized patterns — see
  [ADR 0004](docs/adr/0004-deterministic-tools-over-neural-guessing.md).
- **Phase discipline.** Each phase's scope is frozen and reviewed before the
  next begins — see [`docs/adr/`](docs/adr/).

## Architecture target (frozen Phase 0 design intent — not yet implemented)

Decoder-only causal Transformer, `d_model=256`, 5 layers, 4 query/KV heads
(head_dim 64), SwiGLU FFN (`d_ff=688`), RMSNorm Pre-Norm, RoPE (theta
10,000), no biases, vocab 4,096, weight-tied, 1,024-token context,
~5,004,032 parameters. Full details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
canonical config: [`config/architecture.yaml`](config/architecture.yaml).

## Target hardware

AMD Ryzen 7 5700G · NVIDIA RTX 2060 (6GB VRAM) · 16GB RAM · 256GB NVMe.
See [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

## Repository layout

```
config/       Structured, version-controlled configuration
data/         Data workspace (disposable; provenance tracked in manifests/)
docs/         Engineering and research documentation, ADRs
evals/        Frozen evaluation suites
manifests/    Source, license, and artifact provenance/integrity metadata
scripts/      Project setup and orchestration scripts
src/          Importable juniper_math Python package
tests/        Automated test suite
tools/        (reserved) deterministic math tool contracts — Phase 3
training/     (reserved) training entry points — Phase 1+
experiments/  Experiment metadata
checkpoints/  Checkpoint metadata (binaries stored externally — see policy)
releases/     Release metadata
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m juniper_math validate-env
```

See [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) and
[`scripts/bootstrap.sh`](scripts/bootstrap.sh).

## Common commands

```bash
python -m juniper_math status              # current phase/status
python -m juniper_math validate-env         # environment PASS/WARNING/FAIL
python -m juniper_math validate-config      # architecture + project config
python -m juniper_math evals validate       # frozen evaluation suite integrity
python -m juniper_math hash verify          # verify frozen artifact hashes
python -m juniper_math manifests-validate   # source/license manifests
```

Full command reference: [`docs/CLI.md`](docs/CLI.md). Commands for later
phases (`model`, `train`, `tokenizer`, `dataset`, `tool-test`, ...) exist as
honest placeholders that report "not implemented until Phase N" — they
never fake success.

## Testing

```bash
pytest -v
ruff check .
ruff format --check .
mypy
```

See [`docs/TESTING.md`](docs/TESTING.md).

## Evaluation

A frozen, versioned, hashed baseline evaluation suite
([`evals/phase0_v1.json`](evals/phase0_v1.json)) exists ahead of any model —
see [`docs/EVALUATIONS.md`](docs/EVALUATIONS.md). It validates schema/ID/
category integrity now; scoring a real model against it is later-phase work.

## Recovery

Full disaster-recovery procedure (clone → fresh environment → validate →
test) is documented in [`docs/RECOVERY.md`](docs/RECOVERY.md) and was
actually exercised — see
[`reports/RECOVERY_TEST_REPORT.md`](reports/RECOVERY_TEST_REPORT.md).

## Development phases

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundation and Recovery | AWAITING_OPUS_5_REVIEW |
| 1 | Transformer implementation and training loop | Not started |
| 2 | Tokenizer training | Not started |
| 3 | Deterministic tool integration ("Cinqic Calculator") | Not started |
| 4 | Dataset construction | Not started |

## License

MIT — see [`LICENSE`](LICENSE).
