# Phase 6 Completion Report — Pilot Pretraining

> Superseded by the independent remediation: the candidate dataset selection
> evidence in this historical report used unverified ignored shards. Use
> `reports/PHASE6_RESULTS.md`, `reports/PHASE6_REMEDIATION.md`, and
> `reports/TERRA_PHASE6_REVIEW.md` for the authoritative Phase 6 record.

## Status

**PASS — Phase 6 implementation complete and ready for independent
review.**

Independent review (the GPT-5.6 Terra review chain used for Phases 1-5)
has **not** been performed. `config/project.yaml`'s `current_phase` stays
`5` and `phase_6_engineering.terra_independent_review` stays
`not_yet_performed` until that review completes — this report does not
claim independent approval, only that the implementation is complete and
self-reviewed by Claude Sonnet 5.

## Evidence

| Requirement | Result |
|---|---|
| Repository | `https://github.com/Cinqic/juniper-math-1` |
| Branch | `main` |
| Starting foundation commit / tag | `73792c04f365c6f139a979f6950fa87be2af5d76` / `phase-5-smoke` |
| Candidate tag | `phase-6-pilot-candidate` |
| Architecture identity | v0.1.0, 5,004,032 trainable parameters (verified, unchanged) |
| Tokenizer identity/hash | `juniper-math-tokenizer-v1` (unchanged; `hash verify` PASS) |
| Dataset identity | `juniper-math-dataset-v1`, `bf9933f0...` (unchanged; `hash verify` PASS) |
| Pilot subset identity | 137,057 train / 3,043 validation, category-stratified, seed 5004032 (see `experiments/phase6-pilot/experiment.yaml`) |
| Environment | Linux, Python 3.12.3, PyTorch 2.13.0+cu130, CUDA 13.0, RTX 2060 6GB |
| Training steps | 320 optimizer steps, effective batch 16 (packed sequences), seq_len 1024 |
| Token count | 5,062,958 loss-bearing tokens |
| Initial → final validation loss | 8.3834 → 0.9807 |
| Gradient finiteness | PASS, every step |
| Parameter finiteness | PASS, every step |
| Checkpoint save | PASS (`checkpoint inspect` confirms full metadata) |
| Checkpoint restore | PASS (`train pilot-resume-test` restores into fresh process-simulated objects) |
| Resume equivalence | PASS — within tolerance (`<1e-2`), not bitwise (expected at this scale — see below) |
| Generation changed from init | PASS — degenerate repetition → structured `<final>`/`<tool_call>`/`<unsupported>` usage; see `reports/PHASE6_RESULTS.md` §Generation |
| Capability evaluation execution | PASS — all 4 frozen v2 suites (725 cases), full suites, at all 5 milestones (3,625 total scored calls), zero crashes/skips |
| Capability accuracy | 0-0.5% across all suites at all milestones — expected and correctly reported, not a capability claim |
| Test-suite result | 634 passed, `ruff`/`mypy` clean |

## Mandatory acceptance gates (Sec. 40 of the Phase 6 instructions)

All applicable gates pass. Full detail in `reports/PHASE6_RESULTS.md`.
Notes on interpretation, stated plainly rather than glossed over:

1. **Resume equivalence at pilot scale is tolerance-based, not
   bitwise**, unlike Phase 5's smoke run (which happened to land on
   bitwise-exact equivalence). Max parameter abs diff ≈2.37e-4, max
   loss-history abs diff ≈2.90e-4 — both ~40x inside the pre-existing
   `<1e-2` threshold, which was not loosened. This is exactly the outcome
   Phase 5's own report predicted might happen at larger scale, and
   demonstrates the tolerance-based gate does its job rather than being
   decorative.
2. **`current_phase` in `config/project.yaml` intentionally stays 5.**
   Every prior phase only advanced `current_phase` after its own Terra
   independent review completed. Phase 6 follows the same discipline:
   `phase_6_engineering` records the implementation as complete and
   self-reviewed, with independent review explicitly
   `not_yet_performed`.
3. **Rare-category floor was capped by real availability, not a Phase 6
   algorithm defect.** `tool_error` selected at 26/26 available train
   examples (below the 60-example floor because only 26 exist in train)
   — the strongest guarantee possible given the frozen split; flagged to
   Terra as a dataset characteristic in `reports/PHASE6_RESULTS.md`, not
   silently smoothed over.
4. **No hyperparameter beyond peak learning rate was swept.** Sec. 14
   explicitly discourages an exhaustive search; the one screen performed
   (three learning rates, one follow-up point) is documented in full in
   `reports/PHASE6_RESULTS.md` §Controlled experiments, including the
   follow-up point that was deliberately *not* adopted.

## Defects discovered and fixed during implementation

Four defects were found and fixed this session — see
`reports/PHASE6_SELF_REVIEW.md` for full detail on each:

1. Validation dataset reusing the training sequence length (1024)
   reintroduced padding waste, caught by an actual timed trial run, not
   code review — fixed with a dedicated, shorter validation sequence
   length.
2. Two pre-existing tests in `tests/test_metadata.py`, already broken
   before this session started (confirmed via `git stash`), left stale by
   the Phase 5 approval merge — fixed to assert the actual current state.
3. `README.md` and `cli.py`'s module docstring were stale relative to
   `config/project.yaml`'s own already-approved Phase 5 state — fixed and
   cross-checked.
4. `mypy` rejected broadening `trainer.py`'s config type hint to a
   `Protocol` until its members were declared as read-only properties
   (frozen dataclasses are not structurally settable) — fixed, a pure
   typing change with no runtime behavior change.

## Known limitations (see also `reports/PHASE6_RESULTS.md` §Known limitations)

- Pilot checkpoints (~57MB each) are not archived to a remote store; they
  are disposable and reproducible from the committed config + frozen
  dataset + resume-test evidence (see `checkpoints/README.md`).
- Generation has no KV cache/batching (unchanged from Phase 5) — adequate
  at pilot milestone-evaluation scale (~57s/milestone including
  validation), not for interactive/high-throughput use.
- Capability accuracy is 0-0.5% at pilot scale, as expected; not a
  capability signal for Phase 7 or any later phase.
- `tool_error` has only 26 examples in the entire frozen train split —
  a Phase 7 model trained on the full train split will still see very few
  of them, independent of any pilot-scale sampling decision.

## Readiness for independent review

Ready. `reports/PHASE6_PLAN.md` (design), `reports/PHASE6_RESULTS.md`
(full run evidence), `reports/PHASE6_SELF_REVIEW.md` (defects and clean
areas), and this report together give a reviewer everything needed to
audit the pipeline without re-running it — though re-running `train
pilot-run` and `train pilot-resume-test` against a fresh `dataset build`
is the recommended verification step (both complete in well under 15
minutes on the target hardware; see `reports/PHASE6_TERRA_HANDOFF.md` for
the exact command sequence).

## Recommended Phase 7 configuration

Every entry below cites the Phase 6 measurement that justifies it (Sec.
27) — none is asserted as a generic default.

| Parameter | Recommendation | Evidence |
|---|---|---|
| Initialization | Fresh random init, same architecture (5,004,032 params) | Sec. 28: nothing frozen changed during Phase 6; default is a clean start, not silent continuation from the pilot checkpoint — see `reports/PHASE6_RESULTS.md` §Phase 7 restart rule |
| Architecture identity | `0.1.0` (frozen, unchanged) | `hash verify` confirms unchanged throughout Phase 6 |
| Tokenizer identity | `juniper-math-tokenizer-v1` (frozen, unchanged) | `hash verify` |
| Dataset identity | `juniper-math-dataset-v1` (frozen, unchanged) | `hash verify` |
| Sequence length | 1024 (full architectural context), packed | `scripts/benchmark_phase1.py`: 757.6 MiB VRAM at batch 4/seq 1024 (12.3% of 6,144 MiB budget); pilot's actual packed run measured only 904.5 MiB peak at effective batch 16; packing measured 3.34% padding waste vs. an estimated >85% for unpacked short examples at this seq_len |
| Micro-batch size | 4 (or higher — pilot showed large VRAM headroom) | Pilot's 904.5 MiB peak leaves >80% of the 6,144 MiB budget unused; Terra/Phase 7 engineer should re-benchmark at the actual Phase 7 token budget's batch size before finalizing |
| Gradient accumulation | Tune for desired effective batch given Phase 7's much larger token budget | Pilot used 4 (effective batch 16); this was sized for a 5M-token pilot, not validated as optimal at Phase 7 scale |
| Effective batch size | Re-derive for Phase 7's token budget — not directly transferable from the pilot's 16 | Pilot's 320-step, effective-batch-16 run is far too short a horizon to recommend a specific value for a 50-100M-token run |
| Precision | fp32 | No VRAM or throughput pressure observed at pilot scale (904.5 MiB of 6,144 MiB); Sec. 12 requires the added complexity (GradScaler, resume/finite-check plumbing) be justified by measured benefit, and none was measured — re-evaluate only if Phase 7's larger batch/sequence combination approaches the VRAM budget |
| Optimizer | AdamW, weight_decay 0.01, beta1 0.9, beta2 0.95, eps 1e-8 | Inherited unchanged from Phase 5; not independently re-screened this phase (Sec. 14 discourages an exhaustive sweep) |
| Peak learning rate | 6.0e-4 as a starting point, with headroom to test higher | Controlled screen (`reports/PHASE6_RESULTS.md` §Controlled experiments): 6.0e-4 clearly outperformed 3.0e-4 with no stability cost; 1.0e-3 outperformed 6.0e-4 further but was not adopted at pilot scale pending a proper screen at Phase 7's actual scale |
| Gradient clipping | 1.0 | Kept applied updates bounded at both 6.0e-4 and 1.0e-3 pre-clip norms up to 10.6; no instability observed |
| Warmup | Ratio-based (≈5% of total steps), not a fixed step count | Pilot used a fixed 16 steps *derived* from a 5% ratio of its 320-step schedule; Sec. 15 recommends encoding this as an actual ratio in Phase 7's config schema so it scales automatically with a much larger `total_steps` |
| Scheduler | Cosine decay to 10% of peak lr | Inherited unchanged from Phase 5; produced a smooth, monotonic loss curve at pilot scale |
| Token budget | Evidence-based scaling from pilot throughput — see runtime estimate below, final number is a Terra/Phase-7-engineer scoping decision | Pilot consumed 8.0% of the full corpus for 5.0M tokens; Phase 7 is expected to consume substantially more of the ~70M-token corpus, scoped against the runtime estimate below |
| Validation cadence | Every 25% of total steps (5 milestones minimum), category-broken-out | Pilot's 5-milestone cadence surfaced a real, interpretable category-difficulty spread (0.34 to 2.94 final loss) that a single end-of-run number would have hidden |
| Checkpoint cadence | Aligned with validation cadence | Pilot's checkpoint-every-80-steps (= every milestone) added negligible overhead (checkpoint write time included in the 406.9s total run) |
| Evaluation cadence | Full frozen suites (not a sample) at every milestone, if the token budget's step count keeps per-milestone eval overhead proportionate | Pilot's full-suite milestone eval cost ~57s/milestone (725 generations + validation) — trivial relative to a Phase 7 run's expected total duration; re-benchmark this ratio if Phase 7's step count is dramatically higher, since eval cost scales with milestone *count*, not token budget |
| Generation inspection cadence | Same fixed 13-prompt set, every milestone | Pilot's set already covers every category Sec. 20 requires; reuse it as-is for direct before/after comparability |
| Random seed strategy | Same project seed (5004032) for the primary run; a documented different seed for any controlled comparison | Matches the pilot's own choice, made for direct comparability with Phase 5/6's initialization identity |
| Expected VRAM | Well under 6,144 MiB at seq_len 1024/batch 4 (904.5 MiB pilot peak); re-benchmark at Phase 7's actual batch size before committing | `scripts/benchmark_phase1.py` + pilot's measured peak |
| Expected RAM | Pilot's two-pass category-stratified selection scan (`count_categories` + `select_pilot_examples`) completed in ~23s wall time on the full 1.47M-record train split at 14 GiB system RAM; RAM was not the binding constraint | Direct measurement during pilot subset construction |
| Estimated checkpoint storage | ~57.3 MiB/checkpoint (model + optimizer + scheduler + RNG state) at this architecture's fixed parameter count — checkpoint size does not scale with token budget, only with checkpoint *count* | Pilot's measured `step_000320_final.pt` size (60,123,779 bytes) |
| Estimated Phase 7 runtime | Pilot's pure-training throughput was ~41,500 loss-bearing tokens/sec on this hardware (5,062,958 tokens / ~122s training-only). A Phase 7 run consuming, for illustration, 50M tokens would take roughly 50,000,000 / 41,500 ≈ 1,205s (~20 min) of pure training time at this same batch/sequence configuration, plus milestone-evaluation overhead scaling with milestone count (not token budget) at ~57s/milestone — the actual Phase 7 token budget is a scoping decision outside this report's authority, but this throughput number is the basis for estimating it | Directly measured, this report |

## Phase 7

Not started. Not authorized by this report — that decision belongs to the
same review/authorization chain used for every prior phase transition, and
requires GPT-5.6 Terra's independent review of this Phase 6 candidate
first.
