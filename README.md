# Juniper Math 1

A purpose-built, approximately five-million-parameter mathematical language
model. **No model is trained yet.** This repository has an implemented,
mechanically-validated architecture (Phase 1) and a trained, validated
math-specific tokenizer (Phase 2, awaiting independent review) — not a
trained model, and not yet mathematical capability of any kind.

## Research question

> How capable, reliable, efficient, and truthfully calibrated can an
> approximately five-million-parameter purpose-built mathematical language
> model become at understanding natural-language math problems, decomposing
> them into verifiable operations, selecting and operating deterministic
> mathematical tools, detecting ambiguity and unsupported requests, verifying
> computational results, and producing accurate mathematical answers?

## Current status

```
Phase 0 — Foundation and Recovery:          COMPLETE
Phase 1 — Architecture:                     COMPLETE
Phase 2 — Math Tokenizer:                   COMPLETE
Phase 3 — Cinqic Calculator Tool Runtime:   AUTHORIZED — NOT STARTED
```

Phase 0 was implemented and self-reviewed by Claude Sonnet 5, independently
audited by Claude Opus 5, reviewed by Cinqic, then remediated and
re-verified. The independent audit initially returned **CHANGES REQUIRED**
(1 HIGH, 4 MEDIUM, 6 LOW) — most importantly a wrong answer in the frozen
evaluation suite. Every finding was resolved and re-verified before
approval.

Phase 1 implements and mechanically validates the frozen architecture: the
Transformer itself (exactly 5,004,032 trainable parameters, programmatically
verified), causal masking, weight tying, loss semantics, full training-state
checkpointing with bitwise-exact CPU resume, a tiny controlled
memorization experiment, and benchmarking on the actual RTX 2060/Ryzen 7
5700G target hardware. **This is not trained mathematical capability** — it
is proof that the architecture mechanics (forward pass, backward pass,
gradients, checkpointing, hardware fit) are correct. Phase 1 was
independently audited and remediated by GPT-5.6 Terra.

Phase 2 trains and validates the canonical Juniper Math 1 tokenizer: a
4,096-token, math-specialized, byte-fallback BPE model (SentencePiece),
built from a deterministic synthetic corpus. Digits 0-9 are guaranteed
atomic (`split_digits` pretokenization, not hoped-for BPE behavior),
unauthorized multi-digit vocabulary pieces are rejected by an automated
audit, required control tokens (`<tool_call>`, `<tool_result>`, `<final>`,
`<unsupported>`, `<error>`) are frozen at stable IDs, byte fallback and
Unicode math notation are validated, and the artifact is reproducible
byte-for-byte from the committed corpus generator and config. Real
dataset construction and Cinqic Calculator integration remain later-phase
work. Phase 2 was independently audited, remediated, and approved by GPT-5.6
Terra. Phase 3 is authorized but has not started.

- Canonical machine-readable status: [`config/project.yaml`](config/project.yaml)
- Phase 0 independent review (incl. final re-review): [`reports/OPUS5_PHASE0_REVIEW.md`](reports/OPUS5_PHASE0_REVIEW.md)
- Phase 0 remediation record: [`reports/PHASE0_REMEDIATION.md`](reports/PHASE0_REMEDIATION.md)
- Phase 0 final approval: [`reports/PHASE0_FINAL_APPROVAL.md`](reports/PHASE0_FINAL_APPROVAL.md)
- Phase 0 engineering report: [`reports/PHASE0_REPORT.md`](reports/PHASE0_REPORT.md)
- Phase 1 engineering report: [`reports/PHASE1_REPORT.md`](reports/PHASE1_REPORT.md)
- Phase 1 architecture validation: [`reports/PHASE1_ARCHITECTURE_VALIDATION.md`](reports/PHASE1_ARCHITECTURE_VALIDATION.md)
- Phase 1 benchmarks: [`reports/PHASE1_BENCHMARKS.md`](reports/PHASE1_BENCHMARKS.md)
- Phase 1 self-review: [`reports/PHASE1_SELF_REVIEW.md`](reports/PHASE1_SELF_REVIEW.md)
- Phase 1 Terra handoff package: [`reports/PHASE1_TERRA_HANDOFF.md`](reports/PHASE1_TERRA_HANDOFF.md)
- Phase 1 independent review: [`reports/TERRA_PHASE1_REVIEW.md`](reports/TERRA_PHASE1_REVIEW.md)
- Phase 1 remediation: [`reports/PHASE1_REMEDIATION.md`](reports/PHASE1_REMEDIATION.md)
- Phase 1 final approval: [`reports/PHASE1_FINAL_APPROVAL.md`](reports/PHASE1_FINAL_APPROVAL.md)
- Phase 2 engineering report: [`reports/PHASE2_REPORT.md`](reports/PHASE2_REPORT.md)
- Phase 2 tokenizer validation: [`reports/PHASE2_TOKENIZER_VALIDATION.md`](reports/PHASE2_TOKENIZER_VALIDATION.md)
- Phase 2 tokenizer benchmarks: [`reports/PHASE2_TOKENIZER_BENCHMARKS.md`](reports/PHASE2_TOKENIZER_BENCHMARKS.md)
- Phase 2 tokenizer manual inspection: [`reports/PHASE2_TOKENIZER_INSPECTION.md`](reports/PHASE2_TOKENIZER_INSPECTION.md)
- Phase 2 self-review: [`reports/PHASE2_SELF_REVIEW.md`](reports/PHASE2_SELF_REVIEW.md)
- Phase 2 Terra handoff package: [`reports/PHASE2_TERRA_HANDOFF.md`](reports/PHASE2_TERRA_HANDOFF.md)
- Phase 2 independent review: [`reports/TERRA_PHASE2_REVIEW.md`](reports/TERRA_PHASE2_REVIEW.md)
- Phase 2 remediation: [`reports/PHASE2_REMEDIATION.md`](reports/PHASE2_REMEDIATION.md)
- Phase 2 final approval: [`reports/PHASE2_FINAL_APPROVAL.md`](reports/PHASE2_FINAL_APPROVAL.md)

## Principles

- **GitHub is canonical.** Local storage is disposable — see
  [`docs/GIT_POLICY.md`](docs/GIT_POLICY.md) and
  [`docs/RECOVERY.md`](docs/RECOVERY.md).
- **Deterministic tools over neural guessing.** Arithmetic correctness comes
  from verifiable tool execution, not memorized patterns — see
  [ADR 0004](docs/adr/0004-deterministic-tools-over-neural-guessing.md).
- **Phase discipline.** Each phase's scope is frozen and reviewed before the
  next begins — see [`docs/adr/`](docs/adr/).

## Architecture (frozen design, implemented in Phase 1)

Decoder-only causal Transformer, `d_model=256`, 5 layers, 4 query/KV heads
(head_dim 64), SwiGLU FFN (`d_ff=688`), RMSNorm Pre-Norm, RoPE (theta
10,000), no biases, vocab 4,096, weight-tied, 1,024-token context, exactly
5,004,032 trainable parameters (programmatically verified, not estimated).
Full details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), canonical
config: [`config/architecture.yaml`](config/architecture.yaml),
implementation: [`src/juniper_math/model.py`](src/juniper_math/model.py).

```bash
python -m juniper_math model   # construct, verify param count, synthetic forward pass
```

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
training/     (reserved) real training entry points — Phase 6/7
experiments/  Experiment metadata
checkpoints/  Checkpoint metadata (binaries stored externally — see policy)
releases/     Release metadata; releases/tokenizer/ holds the frozen Phase 2
              tokenizer artifacts (small enough for ordinary Git)
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
python -m juniper_math model                # construct frozen architecture, verify param count
python -m juniper_math checkpoint inspect <path>  # safe checkpoint metadata inspection
python -m juniper_math tokenizer train      # generate corpus + train the tokenizer (refuses to overwrite)
python -m juniper_math tokenizer inspect    # vocabulary stats + special-token table
python -m juniper_math tokenizer encode "2x + 5 = 11"
python -m juniper_math tokenizer decode --ids 12,10,...
python -m juniper_math tokenizer validate   # full Phase 2 validation battery
python -m juniper_math tokenizer benchmark  # per-category token efficiency + baseline comparison
```

Full command reference: [`docs/CLI.md`](docs/CLI.md). Commands for later
phases (`train`, `dataset`, `evaluate`, `infer`, `tool-test`) exist as
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
| 1 | Architecture — Transformer implementation, validation, benchmarking | **COMPLETE** |
| 2 | Math Tokenizer | **COMPLETE** |
| 3 | Deterministic tool integration ("Cinqic Calculator") | **AUTHORIZED — NOT STARTED** |
| 4 | Dataset construction | Not started |

## License

MIT — see [`LICENSE`](LICENSE).
