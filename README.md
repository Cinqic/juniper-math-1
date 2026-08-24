# Juniper Math 1

A purpose-built, approximately five-million-parameter mathematical language
model. **No model with mathematical capability exists yet.** This
repository has an implemented, mechanically-validated architecture
(Phase 1), a trained, validated math-specific tokenizer (Phase 2), a
deterministic calculator tool runtime (Phase 3), a frozen dataset and
evaluation suite (Phase 4), a validated smoke-pretraining pipeline
(Phase 5) that trains a tiny checkpoint end to end purely to prove the
training mechanics work, and a pilot-pretraining engineering candidate
(Phase 6) that trains on a ~5M-token, category-stratified slice of the
frozen corpus to establish whether the model learns stably and to
recommend a Phase 7 configuration — **the pilot checkpoint is not a claim
of mathematical capability either**: it demonstrates a clear, attributable
learning signal (loss, generation structure), not correct arithmetic. The
Phase 3 runtime proves the tool surface a future trained model will call
into is correct and secure; it does not itself demonstrate learned
tool-use behavior. Phase 6 is implementation-complete and self-reviewed;
independent review by GPT-5.6 Terra is pending — see
[`reports/PHASE6_RESULTS.md`](reports/PHASE6_RESULTS.md).

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
Phase 3 — Cinqic Calculator Tool Runtime:   COMPLETE
Phase 4 — Dataset and Evaluation Freeze:    COMPLETE
Phase 5 — Smoke Pretraining:                COMPLETE — INDEPENDENTLY APPROVED
Phase 6 — Pilot Pretraining:                ENGINEERING COMPLETE — PENDING INDEPENDENT REVIEW
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
dataset construction remained later-phase work. Phase 2 was independently
audited, remediated, and approved by GPT-5.6 Terra.

Phase 3 implements a versioned, strictly validated, deterministic tool
runtime (`juniper-tool-protocol-v1` v1.0.0) providing `calculator.evaluate`,
`calculator.convert`, and `calculator.finance` through a security-hardened,
narrowly-adapted integration with the pinned platform-independent core of
[Cinqic Calculator](https://github.com/Cinqic/Cinqic-Calculator) (commit
`8024cf107d6240386fa42b6c5193dd8b34848032`, MIT licensed) — no GUI or
platform-specific code. Model-generated tool calls are treated as untrusted:
strict JSON parsing (duplicate-key/NaN/trailing-content rejection), a
Python-`ast`-sandboxed expression evaluator with no `eval`/`exec` and
explicit resource limits, Decimal-based conversion/finance math, and a
closed tool registry with no dynamic dispatch. Fabricated
`<tool_result>...`-shaped model text is never trusted as a real execution
outcome. See [`docs/TOOLS.md`](docs/TOOLS.md) for the full protocol and
security model. Phase 3 was implemented and self-reviewed by Claude Sonnet
5 and was independently audited, remediated, and approved by GPT-5.6 Terra.

Phase 4 builds and freezes `juniper-math-dataset-v1`: a 100%-synthetic,
deterministically generated, ground-truth-verified corpus of 1,629,078
examples (62.4M frozen-tokenizer tokens) across 24 categories — arithmetic
through algebra, unit conversion and financial math via the real Phase 3
tool runtime, and adversarial cases (ambiguity, missing information,
undefined mathematics, unsupported capability, incorrect answers, incorrect
tool calls). Every deterministic answer is recomputed from a closed
arithmetic-operation allowlist and every tool-required example's ground
truth comes from actually executing `ToolRuntime`, never a fabricated
result. Splits are grouped by problem derivation (never row-shuffled) so
related variants cannot leak across train/validation/test, and four
evaluation-only frozen v2 suites (725 cases: core mathematics, tool use,
calibration/truthfulness, adversarial/error handling) are verified
contamination-free against the training corpus. See
[`docs/DATASET.md`](docs/DATASET.md) for the full pipeline. Phase 4 was
implemented and self-reviewed by Claude Sonnet 5, then independently
remediated and approved by GPT-5.6 Terra.

Phase 5 (Smoke Pretraining) proves the complete training pipeline works end
to end on the target hardware — not that the model is capable. A
deterministic tiny subset (2,048 train / 256 validation examples) is
selected from the frozen dataset by fixed-stride sampling, tokenized with
the frozen tokenizer, and trained for 200 optimizer steps on the RTX 2060:
validation loss fell from 8.38 to 2.23, gradients and parameters stayed
finite throughout, and generation demonstrably changed from initialization
(the model learned the `<final>` answer-tag format, though not correct
answers). Checkpoint save/restore, an interrupted-vs-uninterrupted resume
comparison (bitwise-identical on this run), and the tool-format evaluation
infrastructure against the frozen `evals/phase4_tool_use_v2.json` suite all
executed successfully. See [`docs/TRAINING.md`](docs/TRAINING.md) and
[`reports/PHASE5_RESULTS.md`](reports/PHASE5_RESULTS.md) for full results.
Phase 5 was implemented and self-reviewed by Claude Sonnet 5, then
independently reviewed, remediated, and approved by GPT-5.6 Terra.

Phase 6 (Pilot Pretraining) is the first phase where model behavior is
meant to matter — not to produce a capable model, but to establish
whether Juniper Math 1 learns stably from a meaningful slice of the frozen
corpus and to recommend a Phase 7 configuration backed by measurement. A
deterministic, category-stratified subset (137,057 train / 3,043
validation examples, 5,002,683 train tokens — 8.0% of the full corpus)
extends Phase 5's own approved selection algorithm to guarantee every one
of the 24 frozen categories is represented, then packs examples to the
full 1,024-token architectural context (3.3% padding waste, vs. Phase 5's
unpacked smoke approach) for training efficiency. Over 320 optimizer steps
on the RTX 2060 (peak 904.5 MiB VRAM, ~6.8 minutes total including
milestone evaluation): validation loss fell monotonically from 8.38 to
0.98, generation moved from degenerate token repetition to structured,
prompt-dependent `<final>`/`<tool_call>`/`<unsupported>` control-token
usage, and all four frozen v2 evaluation suites (725 cases) were scored in
full at every milestone — capability accuracy stayed at 0-0.5% throughout,
the expected and honestly-reported result at this scale, not a claim of
mathematical capability. A pilot-scale interrupted-vs-uninterrupted resume
comparison passed within the existing tolerance (not bitwise-identical,
unlike Phase 5's smoke run — expected at larger scale). See
[`docs/PILOT_TRAINING.md`](docs/PILOT_TRAINING.md) and
[`reports/PHASE6_RESULTS.md`](reports/PHASE6_RESULTS.md) for full results
and the evidence-backed Phase 7 recommendation. Phase 6 was implemented
and self-reviewed by Claude Sonnet 5; independent review by GPT-5.6 Terra
is pending.

### Phase 2 release verification

The approved tokenizer is frozen at [`phase-2-tokenizer`](https://github.com/Cinqic/juniper-math-1/tree/phase-2-tokenizer),
which resolves to `eaf8fd33837b7bb73c41f2f21bc81386d09dc516`. Its deterministic
200,000-line corpus has SHA-256
`86c3afa92b2cc109b9d3ba340ce59e920e84092fd88347154b006529db7fd13f`.
The final audit recorded 294 passing tests, a green
[GitHub Actions run](https://github.com/Cinqic/juniper-math-1/actions/runs/32588215850),
and successful fresh-clone reconstruction of all tokenizer artifacts.

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
- Phase 3 tool protocol documentation: [`docs/TOOLS.md`](docs/TOOLS.md)
- Phase 3 engineering report: [`reports/PHASE3_REPORT.md`](reports/PHASE3_REPORT.md)
- Phase 3 self-review: [`reports/PHASE3_SELF_REVIEW.md`](reports/PHASE3_SELF_REVIEW.md)
- Phase 3 security review: [`reports/PHASE3_SECURITY.md`](reports/PHASE3_SECURITY.md)
- Phase 3 tool validation: [`reports/PHASE3_TOOL_VALIDATION.md`](reports/PHASE3_TOOL_VALIDATION.md)
- Phase 3 Terra handoff package: [`reports/PHASE3_TERRA_HANDOFF.md`](reports/PHASE3_TERRA_HANDOFF.md)
- Phase 3 Terra review: [`reports/TERRA_PHASE3_REVIEW.md`](reports/TERRA_PHASE3_REVIEW.md)
- Phase 3 remediation: [`reports/PHASE3_REMEDIATION.md`](reports/PHASE3_REMEDIATION.md)
- Phase 3 final approval: [`reports/PHASE3_FINAL_APPROVAL.md`](reports/PHASE3_FINAL_APPROVAL.md)
- Phase 4 dataset documentation: [`docs/DATASET.md`](docs/DATASET.md)
- Phase 4 engineering report: [`reports/PHASE4_REPORT.md`](reports/PHASE4_REPORT.md)
- Phase 4 dataset validation: [`reports/PHASE4_DATASET_VALIDATION.md`](reports/PHASE4_DATASET_VALIDATION.md)
- Phase 4 evaluation freeze: [`reports/PHASE4_EVALUATION_FREEZE.md`](reports/PHASE4_EVALUATION_FREEZE.md)
- Phase 4 provenance/license review: [`reports/PHASE4_PROVENANCE_LICENSE_REVIEW.md`](reports/PHASE4_PROVENANCE_LICENSE_REVIEW.md)
- Phase 4 self-review: [`reports/PHASE4_SELF_REVIEW.md`](reports/PHASE4_SELF_REVIEW.md)
- Phase 4 Terra handoff package: [`reports/PHASE4_TERRA_HANDOFF.md`](reports/PHASE4_TERRA_HANDOFF.md)
- Phase 4 independent review: [`reports/TERRA_PHASE4_REVIEW.md`](reports/TERRA_PHASE4_REVIEW.md)
- Phase 4 remediation: [`reports/PHASE4_REMEDIATION.md`](reports/PHASE4_REMEDIATION.md)
- Phase 4 final approval: [`reports/PHASE4_FINAL_APPROVAL.md`](reports/PHASE4_FINAL_APPROVAL.md)
- Pre-Phase-5 consistency audit: [`reports/PRE_PHASE5_REPOSITORY_AUDIT.md`](reports/PRE_PHASE5_REPOSITORY_AUDIT.md)
- Phase 5 training pipeline documentation: [`docs/TRAINING.md`](docs/TRAINING.md)
- Phase 5 engineering report: [`reports/PHASE5_REPORT.md`](reports/PHASE5_REPORT.md)
- Phase 5 smoke-run results: [`reports/PHASE5_RESULTS.md`](reports/PHASE5_RESULTS.md)
- Phase 5 completion report: [`reports/PHASE5_COMPLETION.md`](reports/PHASE5_COMPLETION.md)
- Phase 5 independent review: [`reports/TERRA_PHASE5_REVIEW.md`](reports/TERRA_PHASE5_REVIEW.md)
- Phase 5 remediation: [`reports/PHASE5_REMEDIATION.md`](reports/PHASE5_REMEDIATION.md)
- Phase 5 final approval: [`reports/PHASE5_FINAL_APPROVAL.md`](reports/PHASE5_FINAL_APPROVAL.md)
- Phase 6 pilot-pretraining documentation: [`docs/PILOT_TRAINING.md`](docs/PILOT_TRAINING.md)
- Phase 6 plan: [`reports/PHASE6_PLAN.md`](reports/PHASE6_PLAN.md)
- Phase 6 pilot-run results: [`reports/PHASE6_RESULTS.md`](reports/PHASE6_RESULTS.md)
- Phase 6 self-review: [`reports/PHASE6_SELF_REVIEW.md`](reports/PHASE6_SELF_REVIEW.md)
- Phase 6 completion report: [`reports/PHASE6_COMPLETION.md`](reports/PHASE6_COMPLETION.md)
- Phase 6 Terra handoff package: [`reports/PHASE6_TERRA_HANDOFF.md`](reports/PHASE6_TERRA_HANDOFF.md)

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
tools/        Phase 3 deterministic tool protocol schemas (tools/schemas/)
training/     Training-run artifacts; the Phase 5 pipeline itself lives in src/ (see docs/TRAINING.md)
experiments/  Experiment metadata (experiments/phase5-smoke/ is the first real experiment)
checkpoints/  Checkpoint metadata (binaries are disposable/reproducible — see docs/CHECKPOINT_POLICY.md)
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
python -m juniper_math tools list           # Phase 3 canonical tools + availability
python -m juniper_math tools schemas        # print generated JSON Schemas
python -m juniper_math tools call '{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{"expression":"2+2"}}'
python -m juniper_math tools self-test      # fast happy-path + security battery
python -m juniper_math train run            # Phase 5 smoke pretraining
python -m juniper_math train resume-test    # interrupted-vs-uninterrupted resume equivalence gate
python -m juniper_math evaluate --checkpoint <path>   # smoke tool-format evaluation
python -m juniper_math infer --checkpoint <path> --prompt "2 + 2 ="
python -m juniper_math train pilot-run              # Phase 6 pilot pretraining
python -m juniper_math train pilot-resume-test      # pilot-scale resume equivalence gate
python -m juniper_math pilot-evaluate --checkpoint <path>  # all four frozen v2 suites
python -m juniper_math pilot-infer --checkpoint <path> --prompt "2 + 2 ="
```

Full command reference: [`docs/CLI.md`](docs/CLI.md). No placeholder
commands remain — see [`docs/TRAINING.md`](docs/TRAINING.md) for the Phase 5
smoke-pretraining pipeline and [`docs/PILOT_TRAINING.md`](docs/PILOT_TRAINING.md)
for Phase 6 pilot pretraining.

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
| 3 | Deterministic tool integration ("Cinqic Calculator") | **COMPLETE** |
| 4 | Dataset and Evaluation Freeze | **COMPLETE** |
| 5 | Smoke Pretraining | **COMPLETE — INDEPENDENTLY APPROVED** |
| 6 | Pilot Pretraining | **ENGINEERING COMPLETE — PENDING INDEPENDENT REVIEW** |

## License

MIT — see [`LICENSE`](LICENSE).
