# Phase 8 Self-Review (Sec. 30)

Adversarial review of this session's own Phase 8 implementation, assuming
mistakes exist until checked. Findings are recorded honestly, including
ones not fully resolved.

## Checked and clean

- **SFT split leakage**: `sft_data.select_and_record_sft_subset` reads only
  the frozen train split for `train` and only the frozen validation split
  for `validation`; `verify_parent_dataset_shards` fails closed against the
  true manifest first. No test/train crossover is possible by construction
  (each call scans exactly one named split's shard files).
- **Evaluation contamination**: verified in this session (not assumed) —
  zero `example_id`/prompt overlap between `evals/phase8_instruction_v1.json`
  and (a) the full frozen train/validation/test corpus, (b) all four
  frozen Phase 4 v2 suites.
- **Data duplication / target leakage**: the frozen schema guarantees one
  category per example; stride-selection per category cannot select the
  same underlying record twice into one split.
- **Masking off-by-one errors**: 15 dedicated tests
  (`tests/test_sft_rendering.py`) plus a direct byte-for-byte comparison
  against the existing, already-reviewed joint-string tokenization
  (`pilot_data.tokenize_examples`) on 7,452 sampled examples — zero
  mismatches.
- **Tool-result supervision mistakes**: covered by the same test file
  (`test_tool_result_tokens_are_masked`, `test_no_target_shifted_onto_
  context_only_token`).
- **Fabricated tool outputs / runtime bypasses**: `tool_interaction.py`'s
  trust boundary is unit-tested directly
  (`test_fabricated_tool_result_is_discarded_and_never_trusted`) with a
  scripted model that free-generates a false result; the harness is shown
  to feed back the real runtime's result, not the fabrication. No new code
  path calls anything other than the existing `ToolRuntime.execute_call`/
  `execute_text`.
- **Checkpoint lineage**: every Phase 8 checkpoint's `extra` field records
  `parent_checkpoint_path`, `parent_checkpoint_sha256`,
  `parent_phase7_tag`, and `sft_identity` (verified present in the saved
  final checkpoint's metadata).
- **Stale hashes**: `manifests/artifacts.yaml`'s new entries
  (`phase8_instruction_eval_suite_v1`, `phase8_sft_training_config`) were
  computed from the actual on-disk files and re-verified by
  `tests/test_manifests.py::test_artifacts_manifest_hashes_verify`.
- **Configuration drift**: `sft_training_config.verify_parent_checkpoint`
  and `sft_pipeline._load_common` fail loudly (raise `JuniperConfigError`)
  on any architecture/tokenizer/tool-protocol/parent-checkpoint-hash
  mismatch; unit-tested (`tests/test_sft_training_config.py`).
- **Last-checkpoint bias**: explicitly avoided — `step_004500_final.pt` is
  NOT the selected checkpoint; `step_002700.pt` was chosen on a documented
  composite of tool metrics (`reports/PHASE8_RESULTS.md`).
- **Cherry-picking**: the results table in `reports/PHASE8_RESULTS.md`
  reports every measured metric for every candidate, including several
  where the Base outperforms every SFT candidate
  (`tool_use_format.valid_rate`, `argument_execution_successful`,
  `terminal_tag_correct`) — these are not omitted.
- **Local-only artifacts**: checkpoints are correctly gitignored
  (`checkpoints/**/*.pt`, matching the existing project convention);
  experiment JSONL logs (including the rejected run and preflight
  candidates) follow the existing precedent of being git-tracked as
  evidence (`experiments/phase7-lr-preflight/*/train_log.jsonl` is already
  tracked the same way).

## Found and corrected during this session

- **Missing Sec. 22 regression metric in the preflight**: the original
  bounded preflight (`reports/PHASE8_PREFLIGHT.md`) selected a learning
  rate using only tool-format metrics and the SFT-masked loss, never the
  unmasked full-corpus validation loss that is the actual Base-regression
  gate. This let a severely regressive configuration (8e-4, >5x loss
  increase) look like the preflight winner. Found only after the first
  full run completed, by deliberately computing the missing metric.
  Corrected: the run was redone at a rate (2e-4) that does not show this
  regression, and the rejected run's evidence was preserved rather than
  discarded. See `reports/PHASE8_REGRESSION.md`.
- **`tool_interaction.py` context-duplication bug**: `generate()` returns
  the full decoded prompt+continuation text, not just new tokens; an early
  version of the multi-turn loop naively string-concatenated this,
  duplicating the prompt every turn. Found via direct inspection of the
  harness's own output during development (visible garbled repeated text),
  fixed with token-level slicing, and now covered by a regression test
  (`test_no_context_duplication_across_turns`).
- **`tool_name_correct` false-positive on `None == None`**: an early
  version of `sft_eval.py` counted a direct (no-tool) case where the model
  also failed to emit any parseable call as a "correct" tool-name match,
  because both `expected_tool` and the failed attempt's `tool_name` were
  `None`. Found by manual inspection of an inconsistent-looking metric
  output, fixed, and covered by
  `test_tool_name_correct_excludes_not_applicable_cases`.
- **`generation_max_new_tokens=48` truncating tool-call JSON**: the
  original config value was too small for a `<tool_call>{...}` block to
  finish before generation stopped, artificially suppressing
  `call_parsed_valid` regardless of training. Found by inspecting 0%
  valid-parse results that looked suspicious even for an untrained
  checkpoint, corrected to 200 across all Phase 8 configs.
- **Duplicate `example_id`s across categories, an existing latent bug**:
  found while building the Phase 8 eval suite — `evals/
  phase4_calibration_v2.json` (a frozen Phase 4 artifact) has 30 of 130
  cases colliding pairwise across categories onto only 100 unique ids,
  because its `incorrect_supplied_answer` constructor reuses `_direct`'s
  id unchanged. This is **not fixed** (frozen-artifact boundary, Sec. 13)
  but is documented here and in `reports/PHASE8_DATASET.md`, and the new
  Phase 8 suite is built with an explicit per-category index offset to
  avoid reproducing it (verified: 271/271 unique ids).

## Not fully resolved — flagged for Terra

- **The corrected run's own regression is still non-zero and exceeds the
  pre-committed tolerance** (+12.5% to +30.7% depending on checkpoint,
  against a stated ≤0.05-absolute-nat gate). This is reported plainly in
  `reports/PHASE8_RESULTS.md`, not minimized.
- **The larger-sample (n=200) evaluation shows smaller, more mixed gains
  than the smaller-sample (n=40-100) milestone/preflight numbers
  suggested**, including some metrics where the Base outperforms every
  SFT candidate. The overall verdict in `reports/PHASE8_RESULTS.md` is
  explicitly a "marginal, mixed outcome," not a clean success — this
  self-review confirms that characterization is not overly harsh: it
  matches the numbers.
- **No candidate achieves nonzero `end_to_end_success_on_tool_required` or
  `direct_answer_correct` at any evaluated checkpoint, including the
  Base.** Phase 8 has not demonstrated that the model can complete a full
  tool-mediated task or answer directly with a correct final value — only
  that its routing/format behavior shifted somewhat.
- **Only one corrective rerun was performed** (LR 8e-4 → 2e-4). A more
  thorough sweep (e.g. an intermediate LR, more warmup, fewer total steps
  to test whether forgetting is monotonic with step count within the
  2e-4 regime) was not run, given the bounded-preflight discipline (Sec.
  19 explicitly warns against "a giant hyperparameter sweep on an RTX
  2060"). This is a legitimate avenue for follow-up, not concealed.
- **The mixture-ablation preflight candidate (C) was run at the losing LR
  (2e-4) before the regression issue was known**, so its finding ("mixture
  reweighting didn't rescue the lower-LR candidate") was never re-tested
  at the LR actually used for the corrected full run. Noted as unexplored
  in `reports/PHASE8_PREFLIGHT.md`'s addendum.

## Dirty working tree / stale hash check

Repository state was verified clean at the start of this session
(`git status`: nothing to commit); every new/modified file introduced
during Phase 8 is listed explicitly in `reports/PHASE8_COMPLETION.md`'s
file manifest before commit.
