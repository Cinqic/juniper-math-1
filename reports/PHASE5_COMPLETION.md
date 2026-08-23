# Phase 5 Completion Report — Smoke Pretraining

## Status

**PASS — Phase 5 implementation complete and ready for independent
review.**

Independent review (the GPT-5.6 Terra review chain used for Phases 1-4)
has **not** been performed. `config/project.yaml`'s `current_phase` stays
`4` and `phase_5_engineering.terra_independent_review` stays
`not_yet_performed` until that review completes — this report does not
claim independent approval, only that the implementation is complete and
self-reviewed by Claude Sonnet 5.

## Evidence

| Requirement | Result |
|---|---|
| Repository | `https://github.com/Cinqic/juniper-math-1` |
| Branch | `main` |
| Implementation commit SHA | `b70abb97f3588678ef2dc2b2f050a4dfb6f5183c` |
| Final tagged commit SHA | Not hardcoded here deliberately — a commit cannot correctly self-reference its own resulting SHA (recording it here would change the SHA). Resolve via `git rev-parse phase-5-smoke-candidate^{commit}`, matching the same resolution used in `reports/PHASE3_TERRA_HANDOFF.md`. |
| Candidate tag | `phase-5-smoke-candidate` |
| Architecture identity | v0.1.0, 5,004,032 trainable parameters (verified) |
| Tokenizer identity/hash | `juniper-math-tokenizer-v1` (unchanged; `hash verify` PASS) |
| Dataset identity | `juniper-math-dataset-v1`, `bf9933f0...` (unchanged; `hash verify` PASS) |
| Smoke subset identity | 2,048 train / 256 validation, seed 5004032 (see `experiments/phase5-smoke/experiment.yaml`) |
| Environment | Linux, Python 3.12.3, PyTorch 2.13.0+cu130, CUDA 13.0, RTX 2060 6GB |
| Training steps | 200 optimizer steps, effective batch 16, seq_len 256 |
| Token count | 117,828 loss-bearing tokens |
| Initial → final training loss | validation 8.3804 → 2.2346 |
| Gradient finiteness | PASS, every step |
| Parameter finiteness | PASS, every step |
| Checkpoint save | PASS (`checkpoint inspect` confirms full metadata) |
| Checkpoint restore | PASS (`train resume-test` restores into fresh process-simulated objects) |
| Resume equivalence | PASS — bitwise-identical vs. uninterrupted control, 3 independent runs |
| Generation changed from init | PASS — see `reports/PHASE5_RESULTS.md` §Generation |
| Evaluation execution | PASS — 40/40 tool-use cases processed without a crash |
| Tool-format evaluation | PASS (infrastructure); 0% valid-call rate (expected at smoke scale) |
| Test-suite result | 586 passed, `ruff`/`mypy` clean |
| Artifact hashes | `manifests/artifacts.yaml` updated and `hash verify` PASS (incl. new `training_config` entry) |
| Remote push verified | Yes — `git push origin main` fast-forwarded `1f5bc90..b70abb9`; `git ls-remote origin main` confirms the remote HEAD matches |
| Repository cleanliness | Working tree clean after final commit (verified via `git status`) |
| Recovery assessment | `docs/RECOVERY.md` step 14 added and exercised (see below) |
| Frozen artifacts unmodified | Architecture, tokenizer, dataset, tool protocol byte-identical (`hash verify` PASS) |
| Phase 6 started | No |

## Mandatory acceptance gates (Sec. 41 of the Phase 5 instructions)

All gates pass. Full detail in `reports/PHASE5_RESULTS.md`. Two notes on
interpretation, stated plainly rather than glossed over:

1. **Resume bitwise equivalence on CUDA** is a genuinely strong result for
   this run (5M params, 200 steps, fp32, `use_deterministic_algorithms(...,
   warn_only=True)`), not a hardware guarantee that will hold at every
   future scale — the tool's own output documents this caveat, and future
   resume tests should be independently re-verified rather than assumed.
2. **current_phase in `config/project.yaml` intentionally stays 4.** Every
   prior phase (1-4) only advanced `current_phase` after its own Terra
   independent review completed. Phase 5 follows the same discipline:
   `phase_5_engineering` records the implementation as complete and
   self-reviewed, with independent review explicitly `not_yet_performed`.
   This is not an oversight; it is the project's established phase
   discipline applied consistently.

## Defects discovered and fixed during implementation

- None required fixing *inherited* code — Phase 5 is new code. The only
  self-review finding was in Phase 5's own development: an early revision
  of `dict[str, float]`-typed training metrics failed `mypy` because
  `torch.nn.utils.clip_grad_norm_` returns a `Tensor`; fixed by calling
  `.item()` explicitly and adding an explicit `dict[str, float]`
  annotation (`src/juniper_math/trainer.py`). Caught by the mandatory
  `mypy` gate before any run, not discovered at runtime.

## Known limitations (see also `reports/PHASE5_RESULTS.md` §Known limitations)

- Smoke checkpoints (~60MB) are not archived to a remote store; they are
  disposable and reproducible from the committed config + frozen dataset.
- Generation has no KV cache/batching — adequate at smoke scale only.
- Tool-format accuracy is 0% at smoke scale, as expected; not a capability
  signal.

## Readiness for independent review

Ready. `reports/PHASE5_REPORT.md` (design/implementation),
`reports/PHASE5_RESULTS.md` (full run evidence), and this report together
give a reviewer everything needed to audit the pipeline without re-running
it, though re-running `train run --evaluate` and `train resume-test`
against a fresh `dataset build` is the recommended verification step (both
complete in well under a minute on the target hardware).

## Phase 6

Not started. Not authorized by this report — that decision belongs to the
same review/authorization chain used for every prior phase transition.
