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
Phase 0 — Foundation and Recovery: COMPLETE
Phase 1 — Architecture:            AUTHORIZED (not started)
```

Phase 0 was implemented and self-reviewed by Claude Sonnet 5, independently
audited by Claude Opus 5, reviewed by Cinqic, then remediated and
re-verified. The independent audit initially returned **CHANGES REQUIRED**
(1 HIGH, 4 MEDIUM, 6 LOW) — most importantly a wrong answer in the frozen
evaluation suite. Every finding was resolved and re-verified before
approval.

Phase 1 is authorized but **has not begun**: no model, attention, RoPE,
tokenizer, dataset, or training code exists in this repository.

- Canonical machine-readable status: [`config/project.yaml`](config/project.yaml)
- Independent review (incl. final re-review): [`reports/OPUS5_PHASE0_REVIEW.md`](reports/OPUS5_PHASE0_REVIEW.md)
- Remediation record: [`reports/PHASE0_REMEDIATION.md`](reports/PHASE0_REMEDIATION.md)
- Final approval: [`reports/PHASE0_FINAL_APPROVAL.md`](reports/PHASE0_FINAL_APPROVAL.md)
- Engineering report: [`reports/PHASE0_REPORT.md`](reports/PHASE0_REPORT.md)

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
tests/        Automated test suite (128 tests)
tools/        (reserved) deterministic math tool contracts — Phase 3
training/     (reserved) training entry points — Phase 1+
experiments/  Experiment metadata
checkpoints/  Checkpoint metadata (binaries stored externally — see policy)
releases/     Release metadata
```

## Setup

Requires Python 3.12. On Debian, Ubuntu, or Linux Mint, install venv support
first — it is **not** bundled with `python3`:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
```

Then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-lock.txt   # exact validated environment
pip install -e . --no-deps
python -m juniper_math validate-env
```

Or just run [`scripts/bootstrap.sh`](scripts/bootstrap.sh), which performs the
prerequisite preflight and the above. `requirements-lock.txt` pins the exact
Python environment that passed the Phase 0 gate; `pyproject.toml`'s ranges are
compatibility metadata, not a reproducible environment.

See [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) and
[`docs/RECOVERY.md`](docs/RECOVERY.md).

## Common commands

```bash
python -m juniper_math status              # current phase/status
python -m juniper_math validate-env         # environment PASS/WARNING/FAIL
python -m juniper_math validate-config      # architecture + project config
python -m juniper_math evals validate       # suite schema + deterministic ground truth
python -m juniper_math evals verify         # recompute deterministic answers only
python -m juniper_math hash verify          # verify frozen artifact hashes
python -m juniper_math manifests-validate   # source/license manifests + dependency cross-check
python -m juniper_math deps-check           # pyproject deps vs licenses.yaml
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
([`evals/phase0_v1.json`](evals/phase0_v1.json), `suite_version 0.1.1`, 22
cases) exists ahead of any model — see
[`docs/EVALUATIONS.md`](docs/EVALUATIONS.md). It validates schema/ID/category
integrity **and** recomputes every deterministic answer from structured
verification metadata. Scoring a real model against it is later-phase work.

Version 0.1.1 corrected an invalid ground-truth answer found during
independent review (`tool-001`: `84317 * 9926` was recorded as `837042742`;
the correct product is `836930542`) and added the verification infrastructure
that makes that class of error detectable automatically.

## Recovery

Full disaster-recovery procedure (clone → fresh environment → validate →
test) is documented in [`docs/RECOVERY.md`](docs/RECOVERY.md) and was
actually exercised — see
[`reports/RECOVERY_TEST_REPORT.md`](reports/RECOVERY_TEST_REPORT.md).

## Development phases

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundation and Recovery | **COMPLETE** |
| 1 | Architecture — Transformer implementation and training loop | **AUTHORIZED** (not started) |
| 2 | Tokenizer training | Not started |
| 3 | Deterministic tool integration ("Cinqic Calculator") | Not started |
| 4 | Dataset construction | Not started |

## License

MIT — see [`LICENSE`](LICENSE).
