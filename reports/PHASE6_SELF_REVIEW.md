# Phase 6 Self-Review

Written from an adversarial stance after implementation appeared complete
— assume Phase 6 is wrong until demonstrated otherwise (Sec. 34). This
documents both defects found and fixed during this engineering session and
areas deliberately checked and found clean. Inspected as though another
engineer produced it, not merely re-read from memory of writing it.

## Defects found and fixed during this session

1. **Validation dataset reused `pilot_subset.max_sequence_length` (1024)
   for per-example padding, reintroducing the exact padding waste packing
   exists to avoid.** Not caught by code review or unit tests — caught by
   an actual timed trial run (`train pilot-run --max-steps 4
   --eval-sample-size 3`), which took 206.5s for 4 training steps plus 5
   milestones, almost entirely validation overhead: the unpacked
   `TokenizedSmokeDataset` was padding ~30-token validation examples out to
   1024 individually, then doing that 25 times per milestone (overall +
   24 per-category passes) × 5 milestones. **Fix:** a dedicated
   `VALIDATION_MAX_SEQUENCE_LENGTH = 256` constant in `pilot_pipeline.py`
   (matching Phase 5's own smoke validation length, comfortably covering
   the p99 example length of 194 tokens), used only for the unpacked
   validation dataset — training still uses the full packed 1024-token
   sequences. Re-measured after the fix: 5 milestones at
   `--eval-sample-size 40` (160 generations/milestone) completed in 120.8s
   total, and the full canonical run (725 cases/milestone × 5 milestones,
   full frozen suites) completed in 406.9s. This is exactly the kind of
   thing an actual execution catches that static review does not — see
   Sec. 10's instruction to benchmark, not assume.

2. **Two pre-existing, already-broken tests in `tests/test_metadata.py`.**
   Discovered while establishing the exact starting state (Sec. 4), not
   introduced by Phase 6: `test_loads_current_status` asserted
   `current_phase == 4` and `test_phase_5_implementation_complete_pending_independent_review`
   asserted `next_phase.number == 5`/`started is True` — both stale
   assertions from *before* Phase 5's independent-review merge (`docs:
   approve Phase 5 smoke pretraining`, commit `73792c04f`), which advanced
   `current_phase` to 5 and `next_phase` to describe Phase 6 without the
   corresponding test file being updated in the same commit. Confirmed via
   `git stash` that both failures pre-existed and had nothing to do with
   Phase 6 changes. **Fix:** updated both tests to assert the actual
   post-approval state, and renamed the second one
   (`test_phase_5_approved_phase_6_engineering_complete_pending_review`)
   to also cover the new `phase_6_engineering` block this session adds.
   Left unfixed, this would have been a false "clean baseline" claim in
   Sec. 4's "verify starting state" gate.

3. **Stale documentation inconsistent with `config/project.yaml`'s own
   already-approved state.** `README.md` still said "Phase 5 was
   implemented and self-reviewed by Claude Sonnet 5; independent review is
   pending" and omitted links to `reports/TERRA_PHASE5_REVIEW.md`,
   `reports/PHASE5_REMEDIATION.md`, and `reports/PHASE5_FINAL_APPROVAL.md`
   (all three files existed, just weren't linked) — directly contradicting
   `config/project.yaml`'s `phase_5_engineering.terra_final_approval:
   "approved"`. `src/juniper_math/cli.py`'s module docstring still said
   "Not yet implemented (Phase 5 and later): train, evaluate, infer" even
   though Phase 5 implemented all three. Both predate Phase 6 (same root
   cause as defect 2 — the Phase 5 approval merge did not update every
   reference to Phase 5's status). **Fix:** updated both, and added the
   corresponding Phase 6 references so the same drift does not recur once
   Phase 6 is independently reviewed — Terra should update these on
   approval rather than leave them describing the engineering-complete
   state.

4. **mypy rejected the `TrainingConfigLike` `Protocol` when its members
   were declared as plain mutable attributes.** `TrainingConfig` and
   `PilotTrainingConfig` are both frozen dataclasses; mypy structurally
   requires a `Protocol`'s plain-attribute members to be settable on the
   implementing type, which a frozen dataclass field is not. **Fix:**
   declared every `TrainingConfigLike` member as a read-only `@property`
   stub instead of a plain annotation — a pure typing-declaration change,
   no behavior change. `mypy` now reports zero issues across all 62 source
   files (up from 58 before this phase).

## Areas checked and found clean

- **Frozen artifacts unchanged**: `git diff phase-5-smoke --
  config/architecture.yaml config/tokenizer.yaml config/dataset.yaml
  config/tools.yaml releases/tokenizer/ tools/schemas/ evals/` is empty —
  Phase 6 does not touch the frozen architecture, tokenizer, dataset
  config, tool protocol, or evaluation suites. `hash verify` confirms
  every one of the 8 relevant artifact hashes still matches.
- **No cross-split leakage introduced by pilot selection**: pilot
  train-subset selection reads only the `train` split's shards; the
  validation pilot subset reads only the `validation` split's shards.
  Neither ever touches `test` or the eval-suite-seed-isolated frozen
  evaluation cases. `select_pilot_examples`'s own split-filtering
  (identical mechanism to Phase 5's `select_smoke_examples`, already
  reviewed) makes cross-split selection structurally impossible, not just
  policy.
- **Deterministic selection**: `tests/test_pilot_data.py` directly
  verifies same-seed-same-selection and different-seed-can-differ for the
  category-stratified selector, plus that it reuses Phase 5's exact
  `compute_stride_selection` primitive (not a reimplementation) via a
  test that recomputes the expected stride/offset independently and
  compares against the selector's own audit output.
- **Rare-category floor correctness**: `compute_category_targets` was
  found, by test and by the real run, to correctly floor a category at
  `min(min_category_examples, available)` — the real run's `tool_error`
  category (26 available in train, floor requested 60) is exactly this
  cap firing correctly, not a bug; see `reports/PHASE6_RESULTS.md` §Rare
  category finding for the full discussion of why this is a genuine
  dataset characteristic worth flagging to Terra, not a Phase 6 defect.
- **Packing correctness**: `test_pack_sequences_never_splits_an_example`
  verifies every tokenized example appears whole in exactly one packed bin
  (by reconstructing the full example-ID set from all bins and comparing
  against the input set) and that no bin exceeds `max_sequence_length`.
  Token accounting (`total_loss_tokens`, `total_real_tokens`,
  `total_padding_tokens`) is exercised by
  `test_packed_pilot_dataset_shapes_and_masking`, which checks label
  masking (`-100` on padding only) matches the attention mask exactly.
- **No evaluation contamination**: capability scoring
  (`pilot_eval.run_capability_evaluation`) reads the same frozen
  `evals/phase4_*_v2.json` files Phase 4 froze and Phase 5 already used
  for `tool_use` — nothing in Phase 6 regenerates, edits, or re-derives
  those files. `hash verify` confirms all four are byte-identical to the
  approved Phase 4 artifacts.
- **No accidental Phase 7 work**: no code path initializes a training run
  against the full (non-pilot) train split, no config references a
  non-pilot token budget above the enforced 3-10M envelope
  (`validate_pilot_training_config` hard-rejects out-of-envelope values —
  covered by `test_pilot_config_rejects_token_budget_outside_envelope`),
  and `config/project.yaml`'s `next_phase` explicitly states Phase 7 is
  NOT authorized.
- **No silent skip / false PASS**: `run_capability_evaluation` raises
  `JuniperConfigError` on an empty suite rather than reporting "0/0 = 100%"
  (`test_run_capability_evaluation_rejects_empty_suite`); every one of the
  3,625 real evaluation calls across the canonical run's 5 milestones
  scored and counted a result, verified by cross-checking
  `n_cases`/`len(results)` in the actual run's logged milestone events.
- **Checkpoint metadata completeness**: `checkpoint inspect` on the real
  final checkpoint (`checkpoints/phase6-pilot/step_000320_final.pt`)
  confirms schema version, architecture identity, step, tokens_seen, seed,
  git commit, full training config, and optimizer/scheduler state are all
  present — same schema Phase 1 froze, unmodified by Phase 6.
- **No hardcoded local absolute paths**: `grep`'d all new
  `src/juniper_math/pilot_*.py` files and `config/training_phase6_pilot.yaml`
  for `/home/` — none found; all paths resolve through
  `juniper_math.paths.REPO_ROOT`, same as every other module.
- **No accidentally-tracked large files**: `git ls-files | grep '\.pt$'`
  is empty; `git status --porcelain` shows no checkpoint binaries staged.
- **Stale documentation cross-checked**: `docs/TRAINING.md`,
  `docs/RECOVERY.md`, `docs/CLI.md`, `checkpoints/README.md`, and
  `README.md` were all updated in this session to describe the actual
  implemented Phase 6 pipeline (not just the plan); every new hash in
  `manifests/artifacts.yaml` was generated via `python -m juniper_math
  hash file <path>`, never hand-typed, and re-verified with `hash verify`.

## Minor, non-blocking observation (not fixed — deliberately out of scope)

`run_pilot_train`'s milestone loop (`_maybe_milestone`) can emit duplicate
milestone reports at the same step if `--max-steps` is small enough that
multiple `milestone_fractions` round to the same step (observed only
during ad hoc `--max-steps 2`/`--max-steps 4` sanity checks during
development, never during the canonical 320-step run, whose 5 milestone
fractions map to 5 strictly distinct steps: 0/80/160/240/320). Cosmetic —
an extra identically-valued log line, not an incorrect one — and only
reachable with a deliberately tiny `--max-steps` override that no real run
uses. Not fixed, because the fix (deduplicating by step rather than by
fraction) would add complexity for a code path only exercised by manual
smoke checks, not by any real pilot or resume-test run.

## Not independently checked (documented limitation, not a hidden gap)

- No large-N manual human read-through of the 3,625 individual generation
  results beyond the aggregate accuracy numbers and the 13-prompt
  fixed-seed qualitative set quoted in full in
  `reports/PHASE6_RESULTS.md`. GPT-5.6 Terra is explicitly authorized to
  sample and manually audit individual capability-suite generations as
  part of independent review.
- Only one controlled hyperparameter (peak learning rate) was screened
  (Sec. 14 explicitly discourages an exhaustive sweep); warmup ratio,
  weight decay, and betas were inherited unchanged from Phase 5 without
  their own dedicated screening runs.
- The pilot's resume-equivalence gate was run once (three training passes:
  A, B-to-interrupt, B-resumed); Phase 5's report ran the equivalent gate
  three independent times to build confidence in bitwise-exactness. Phase
  6's result is inherently *not* bitwise-exact (see
  `reports/PHASE6_RESULTS.md` §Resume verification), so repeated runs
  would show tolerance-bounded variation rather than a repeatable
  bitwise constant — a single run establishing "within tolerance" is
  sufficient evidence for that specific claim, but Terra may wish to
  repeat it for additional confidence.

## Outcome

All four defects listed under "Defects found and fixed" were fixed and
verified during this session — none are outstanding. This report itself is
part of the record GPT-5.6 Terra reviews; Terra may disagree with any
judgment call here (including the one deliberately-unfixed minor
observation above) and is authorized to remediate directly.
