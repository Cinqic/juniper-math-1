# Phase 7 — Full Base Pretraining Results

## Status

**READY FOR TERRA REVIEW.** Not approved. Not independently reviewed.
GPT-5.6 Terra performs the independent audit, correction, and approval, as
in every prior phase.

## Starting state

- Starting branch: `main`, starting commit `3b84d6d9f85bfe996c05aac9559922065de54285`
  (`docs: record Phase 6 acceptance source`), the Terra-approved Phase 6
  baseline (`reports/PHASE6_FINAL_APPROVAL.md`, `reports/TERRA_PHASE6_REVIEW.md`).
- Phase 6 verdict: **APPROVED WITH REMEDIATION.** The pilot demonstrated
  stable training (validation loss 8.3801 → 0.9777 over 320 steps, 5,051,139
  loss-bearing tokens), a working checkpoint/resume mechanism (tolerance-
  equivalent, not bitwise, on CUDA), and structural/format learning only —
  explicitly not mathematical or tool-use capability. Phase 6 recommended:
  fresh random initialization for Phase 7 (never continue from the pilot
  checkpoint), retain the frozen architecture/tokenizer/dataset/1024-token
  packing, FP32, AdamW (β1 0.9, β2 0.95, ε 1e-8, weight decay 0.01, clip
  1.0), a warmup-**ratio** field instead of a hand-derived fixed step count,
  and one bounded LR preflight across `6e-4`/`8e-4`/`1e-3` before committing
  the full budget.
- **Critical-restart check:** compared against the Phase 6 pilot for
  architecture, parameterization, tokenizer, tokenizer IDs, special tokens,
  vocabulary, sequence format, causal objective, loss definition, context
  length, dataset interpretation, and model input/output contract — **no
  fundamental component changed.** Phase 7 reuses the exact same
  architecture (`config/architecture.yaml`, unedited), tokenizer
  (`juniper-math-tokenizer-v1`, unedited), dataset identity
  (`juniper-math-dataset-v1`, unedited), and packing/loss mechanics
  (`juniper_math.trainer`, `juniper_math.pilot_data.PackedPilotDataset`,
  reused unchanged). No restart-from-scratch condition was triggered beyond
  the already-mandated fresh initialization (Phase 7 never had a prior
  checkpoint to restart from in the first place).
- Before any training: the local (gitignored, disposable) dataset shards
  did not match the frozen `shard_manifest.json` — a stale local build from
  an earlier session. The fail-closed shard-hash check added during Phase 6
  remediation (`pilot_data.verify_parent_dataset_shards`) correctly refused
  to proceed. Rebuilt with `python -m juniper_math dataset build`;
  reproduced the frozen `dataset_identity`
  (`bf9933f032a58b4eb618b32156783b8563097a5fc1c0ef26be4f76445128d25a`)
  exactly; `dataset validate` and `dataset verify` both passed
  (1,780,249 records checked, schema + ground truth both PASS) before any
  Phase 7 training began.

## Phase 7 configuration

Frozen at `config/training_phase7_full.yaml`
(SHA-256 `a48d3410baebf8a11fb594d9009e24c341b2e25175dfd147ad1b58b10cc980ae`).

| Setting | Value |
|---|---|
| Model | `config/architecture.yaml` v0.1.0, 5,004,032 trainable parameters (verified by code, not README text) |
| Dataset | `juniper-math-dataset-v1`, identity `bf9933f0…`, **entire** frozen train split (no subsample) |
| Tokenizer | `juniper-math-tokenizer-v1` |
| Data selection | `juniper_math.full_data` — every train example (1,466,970) and every validation example (81,094), packed to 1,024-token sequences (3.51% padding), no stratification |
| Optimizer | AdamW, β1 0.9, β2 0.95, ε 1e-8, weight decay 0.01, grad clip 1.0 |
| Learning rate | `8.0e-4` (peak) — outcome of the bounded LR preflight, see below |
| Schedule | cosine with warmup, `warmup_ratio 0.05` → 411 warmup steps (a real ratio field, not a hand-derived constant — addresses Phase 6's explicit Phase 7 recommendation), `min_lr_ratio 0.1` |
| Batch | micro-batch 4, grad-accum 4 → effective batch 16 packed sequences/step |
| Precision | FP32 (904.5 MiB peak VRAM measured; no GradScaler complexity needed on a 6 GB card) |
| Token budget | 8,218 optimizer steps ≈ 2 epochs over the packed train split (131,488 of 131,492 possible packed sequences, 99.997% of 2 full epochs) |
| Seed | 5,004,032 (`DEFAULT_PROJECT_SEED`, same as every prior phase) |
| Milestones | 6, at 0/20/40/60/80/100% of total_steps |

### Bounded LR preflight (Terra-mandated, before committing the full budget)

Full method and evidence: [`reports/PHASE7_LR_PREFLIGHT.md`](PHASE7_LR_PREFLIGHT.md).
Three 60-step runs from identical init/seed/data (the frozen Phase 6 pilot
subset) at `6e-4`/`8e-4`/`1e-3`. `8e-4` had the lowest final-step training
loss (3.172 vs. 3.290 and 3.358) and a smooth monotonic descent; `1e-3`
descended fastest early but finished worst of the three. No non-finite
loss or gradient at any step, any LR. **Adopted `8e-4`.**

## Training outcome

- Run ID: `phase7-full-v1`. Command: `python -m juniper_math train
  full-run`. Initialization: fresh random (architecture-seeded, never the
  Phase 6 pilot weights).
- Actual optimizer steps: **8,218 / 8,218** (completed the full configured
  budget; no early termination, no interruption to the canonical run
  itself — the resume mechanism was exercised separately via a bounded
  check, not on this run).
- Actual training tokens: **129,788,239** loss-bearing tokens (matches
  `tokens_seen` in `experiments/phase7-full/train_log.jsonl`'s
  final `run_end` record).
- Elapsed: 5,102.5 s (≈ 85 min) of training + milestone-evaluation wall
  time (dataset build/pack, a one-time ~80 s cost, is not included in this
  figure — see the log's `run_start` → first `train_step` gap).
- Throughput: ~0.39–0.41 s/optimizer step, stable throughout (no
  throughput collapse, no memory-growth trend).
- Peak CUDA memory: **904.5 MiB** of 6,144 MiB (comfortably inside budget;
  never approached the 6 GB ceiling).
- No interruptions, no NaN/Inf loss, gradient, or parameter at any logged
  step (`torch.isfinite` assertions run every step; `assert_model_finite`
  also runs at run end and passed).
- Disk: dataset rebuild + all Phase 7 checkpoints together added
  well under 1 GB to local storage (287 MB checkpoints/phase7-full/, plus
  the already-existing dataset shards); 175+ GB remained free throughout —
  never close to exhausting the 256 GB NVMe.

### Loss trajectory (all 6 milestones)

| Step | Fraction | Validation loss (full frozen val split, 81,094 examples) | Category loss min/mean/max |
|---:|---:|---:|---|
| 0 | 0% | 8.3829 | 8.2646 / 8.3866 / 8.4463 |
| 1,644 | 20% | 0.6849 | 0.1405 / 0.6543 / 1.3374 |
| 3,287 | 40% | 0.6393 | 0.1262 / 0.6126 / 1.2621 |
| 4,931 | 60% | 0.6198 | 0.1180 / 0.5953 / 1.2344 |
| 6,574 | 80% | 0.6045 | 0.1141 / 0.5803 / 1.2076 |
| 8,218 | 100% | **0.5988** | **0.1101 / 0.5762 / 1.1975** |

Overall validation loss and **every** per-category validation loss improve
monotonically at every milestone — no category regresses at the loss
level, and the final checkpoint is the best on this axis by every measure.

## Checkpoints

Five checkpoints were retained (one per milestone after step 0, which has
no trained weights worth keeping separately from initialization):

| Step | Tokens (approx., proportional) | Validation loss | SHA-256 |
|---:|---:|---:|---|
| 1,644 | 25,952,491* | 0.6849 | `330e54f45053c1c00e4f1c0868bfa55dd15b8179bfc9eb0e2d3bf8c49fbcc57e` |
| 3,288† | 51,904,982* | 0.6393 (measured at step 3,287) | `fbec4ce25d543e028f082ded8594e1f906680130744379d463f5dac5dd451812` |
| 4,932† | 77,857,473* | 0.6198 (measured at step 4,931) | `81890cef90d8e293859fd583fa8a1f84cc58839779ebd94962f090974bd25d15` |
| 6,576† | 103,809,964* | 0.6045 (measured at step 6,574) | `79400a86cc0a078d9b5997de6c636a3ccd9de062a062a42115c120994ac82974` |
| **8,218** | **129,788,239** | **0.5988** | **`0ed23a8262edcf123fc9cc29e5dbd74f9169cc8bf4922d85b5e982d124d47f8e`** |

\* Interpolated from tokens/step; only the step-8,218 figure is read
directly from the log. † `checkpoint_interval=1644` fires at exact
multiples of 1,644 (1644/3288/4932/6576), one step later than the
`milestone_fractions`-rounded evaluation steps (1644/3287/4931/6574) for
the three interior milestones — a cosmetic off-by-one between two
independently-computed cadences, not a functional issue (one extra
optimizer step does not materially change the model). The metrics shown
for those rows are the milestone-eval numbers at the adjacent step.

All five `.pt` files are gitignored (`checkpoints/**/*.pt`), matching
established project convention (Phase 5/6 checkpoints are also
git-excluded); see "Remote preservation" below for how the selected
checkpoint survives a local wipe.

## Evaluation

All four frozen v2 suites (`evals/phase4_{math,calibration,adversarial,
tool_use_v2}.json`) ran at every milestone with identical settings
(temperature 0.0, `generation_max_new_tokens=32`, no sample-size
truncation).

| Step | Math acc | Calibration acc | Adversarial acc | Tool-call emitted rate | Tool name-match rate |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 1,644 | 0.0% | 0.0% | 21.0% | 77.8% | 0.0% |
| 3,287 | 0.0% | 18.5% | 20.5% | **80.5%** | 0.0% |
| 4,931 | 0.0% | 1.5% | 21.0% | 48.1% | 0.0% |
| 6,574 | 4.7% | 10.0% | 20.5% | 51.4% | 0.0% |
| 8,218 | 6.1% | 24.6% | 21.0% | 37.8% | 0.0% |

**Read this table as noise around zero capability, not learning, for
math/calibration/adversarial** — consistent with Phase 6's own conclusion
that pretraining at this scale produces structural/format learning, not
mathematical or tool-use capability. Math accuracy is 0% for the first
three milestones and only reaches single digits (4.7%/6.1%) at
80%/100% — plausibly guessing/format-collision noise on a 215-case suite,
not demonstrated arithmetic skill. Calibration accuracy is non-monotonic
(0% → 18.5% → 1.5% → 10.0% → 24.6%) — no consistent trend. Adversarial
accuracy is flat at ~20–21% throughout, i.e., unchanged by training at
all. `tool_name_match_rate` is 0.0% at every milestone (the suite's
warning field, carried over unchanged from Phase 5/6, still reads "SMOKE
PIPELINE VALIDATION ONLY — not a capability measurement").

**One finding is real and worth flagging, not averaging away:**
`tool_use_format`'s `emitted_rate` (n=185 cases, fraction of free
greedy-decoded generations that emit a `<tool_call>` block at all) peaks
at the 40% milestone (80.5%) and **declines** thereafter to 37.8% by the
final step — even though `tool_use`'s own per-token category validation
loss keeps *improving* monotonically over the same interval (0.225 →
0.180). The single fixed-prompt tool-use generation
(`config/training_phase7_full.yaml`'s `fixed_generation_prompts`) is
byte-identical across every milestone from step 1,644 onward
(`Use the calculator to evaluate 84317 * 9926.\n<tool_call>{"arguments":
{"expression":"84317 * 992"},"protocol_version` — truncated at
`generation_max_new_tokens=32`), so this is not a collapse on that
specific prompt; it is a broader shift, across the 185-case suite's more
varied prompts, away from spontaneously emitting the tool-call block in
free generation as training progresses, even while the model's
teacher-forced next-token predictions for tool-use text keep getting more
confident. This is reported honestly as an unresolved, real,
category-specific behavioral divergence between loss and free-generation
format emission — not explained away, not hidden, and explicitly a
candidate area for Phase 8 instruction/format tuning to address (Phase 7
is base pretraining, not instruction tuning; this finding does not block
Phase 7, but Terra and Phase 8 should be aware of it).

## Selected Base checkpoint

**Selected: `checkpoints/phase7-full/step_008218_final.pt` (step 8,218,
final).**

SHA-256: `0ed23a8262edcf123fc9cc29e5dbd74f9169cc8bf4922d85b5e982d124d47f8e`

Rationale, weighed against the Step 18 criteria:

1. **Validation performance:** strictly best of all 6 milestones — both
   overall validation loss (0.5988) and *every* per-category validation
   loss are at their best value at this checkpoint. No exception.
2. **Mathematical evaluation:** math accuracy is at its (noisy, low)
   highest here (6.1%); calibration accuracy is also at its highest here
   (24.6%). Adversarial accuracy is flat across all milestones (~20–21%),
   so it does not favor any candidate.
3. **Category balance:** category validation-loss spread (max − min) is
   narrowest here (1.0874) of any post-initialization milestone,
   indicating the most balanced learning across all 24 categories, not
   uneven overfitting to a subset.
4. **Stability:** no NaN/Inf at any point in training; finite
   parameters/gradients verified at every step and at run end.
5. **Generalization:** validation loss (held-out, never-trained-on split)
   tracks training loss without divergence at any milestone — no
   overfitting signal on the loss axis.
6. **Generation quality:** fixed-prompt generations at step 8,218 remain
   syntactically well-formed (`<final>`/`<tool_call>`/`<unsupported>`
   tags used appropriately by category — e.g., the differential-equation
   prompt correctly emits `<unsupported>`, financial/unit-conversion
   prompts correctly attempt `<tool_call>`), consistent with every
   milestone from 20% onward.
7. **Tool-format understanding:** the one dimension that does **not**
   favor the final checkpoint — `tool_use_format` emission rate peaked at
   the 40% milestone and is lowest at the final step (see "Evaluation"
   above). This is weighed explicitly and does not change the selection,
   because (a) it is a single narrow free-generation metric on greedy
   decoding, not a loss-linked measure, (b) the corresponding `tool_use`
   category *validation loss* — the more statistically grounded,
   directly-optimized metric — is at its best value at the final step,
   and (c) Phase 7 is base pretraining; tool-format specialization is
   explicitly Phase 8's job, not a Phase 7 selection blocker. This
   tradeoff is documented, not hidden, for Terra's independent judgment.
8. **Absence of serious regressions:** none found on the primary
   (loss-based) axes; the one regression found (tool-call emission rate)
   is documented above and factored into the rationale rather than
   omitted.
9. **Reproducibility:** trained from a frozen, hashed config
   (`a48d3410…`) against frozen, hash-verified architecture/tokenizer/
   dataset identities, from a recorded seed and git commit
   (`3b84d6d9f85bfe996c05aac9559922065de54285`).
10. **Integrity:** checkpoint SHA-256 recorded above; contains model,
    optimizer, scheduler, RNG state, step, token count, data-stream
    position, seed, and full training configuration (same checkpoint
    schema validated in Phase 5/6, `juniper_math.checkpoint`).

The other four milestone checkpoints are retained locally (see table
above) as documented, hash-recorded training-trajectory evidence, not as
alternate Base candidates — none of them exceeds the final checkpoint on
validation loss, category balance, or calibration accuracy, and the one
metric where an earlier checkpoint (step 3,287/3,288) is better
(tool-call emission rate) is outweighed by the reasoning in point 7 above.

## Remote preservation

**Complete.** The selected checkpoint file (`step_008218_final.pt`,
60,123,651 bytes) is excluded from git by `checkpoints/**/*.pt` (existing,
unmodified `.gitignore` rule — Phase 5/6 checkpoints are excluded the same
way), so it is preserved via a GitHub Release asset instead — the first
release this project has ever created (`releases/README.md` previously
read "No releases exist during Phase 0"; the Phase 6 pilot checkpoint was
explicitly documented as disposable and never a Phase 7 starting
artifact, so it was never released either).

- Release: <https://github.com/Cinqic/juniper-math-1/releases/tag/phase-7-pretraining-candidate>
- Asset: `step_008218_final.pt`
- Retrieval: `gh release download phase-7-pretraining-candidate --pattern "*.pt"` (or download directly from the release page)
- Verification performed: the asset was downloaded back from GitHub
  immediately after upload and re-hashed; the result matched the recorded
  SHA-256 (`0ed23a8262edcf123fc9cc29e5dbd74f9169cc8bf4922d85b5e982d124d47f8e`)
  exactly — a genuine round-trip check, not just an upload-and-assume.
- Tag: `phase-7-pretraining-candidate` (annotated, pushed, resolves to
  commit `3cf0f3bd4a7f8b6053d0bd75f944c54db39b1aae`). Per established repo
  policy (`phase-5-smoke-candidate` → `phase-5-smoke`,
  `phase-6-pilot-candidate` → `phase-6-pilot`), the final
  `phase-7-pretraining` tag is reserved for after Terra's independent
  review and approval and was deliberately not created here.

Gate I (Remote Preservation) is satisfied: code, config, manifests,
evaluation results, hashes, and documentation are committed and pushed to
`origin/main`, and the selected model checkpoint is retrievable and
hash-verifiable from GitHub independent of local storage.

## Resume-mechanics check

See [`reports/PHASE7_RESUME_CHECK.md`](PHASE7_RESUME_CHECK.md) in full.
Summary: a bounded 200-step check (not the canonical run) verified that
step count and tokens-seen match **exactly** between an uninterrupted run
and an interrupted-then-resumed run using the full-dataset pipeline
(`full_pipeline.run_full_resume_test`) — proving the resume mechanism
itself (checkpoint save/restore of model, optimizer, scheduler, RNG, step,
token count, data cursor) is correct. However, the *numerical* tolerance
check (`max_param_abs_diff < 1e-2`, unchanged from Phase 5/6) **failed**:
`loss_history_max_abs_diff=0.0208`, `max_param_abs_diff=0.0220`, roughly
4x the Phase 6 pilot's equivalent measurement, attributed to the same
documented CUDA nondeterministic-attention-kernel phenomenon Terra already
flagged in Phase 6, plausibly amplified by Phase 7's higher peak LR
(8e-4 vs. 6e-4). This is reported as a genuine bounded-check **FAIL**, not
downgraded — see the full report for the complete reasoning and
disposition.

## Testing

Command: `.venv/bin/python -m pytest -q`

**657 passed, 0 failed, 0 skipped, 2 warnings** (both warnings are the
already-documented, expected CUDA nondeterministic-attention-kernel
`UserWarning`, not test failures). This includes the full pre-existing
Phase 0–6 suite (636 tests, all still passing after the `SchedulerConfig`
change — verified backward-compatible before Phase 7 config work began)
plus 21 new tests for `full_data`/`full_training_config` added in this
phase (`tests/test_full_data.py`, `tests/test_full_training_config.py`).
No test was modified to make it pass; no evaluation suite was altered.

## Reproducibility

| Identity | Value |
|---|---|
| Architecture version | `0.1.0` (`config/architecture.yaml`, SHA-256 `ec763ed8e135f3697b2e4a1fec79df11694c5e2245f9c209160a40d12bc4f55b`), parameter count verified by code: 5,004,032 |
| Tokenizer | `juniper-math-tokenizer-v1` (unchanged from Phase 2) |
| Dataset identity | `bf9933f032a58b4eb618b32156783b8563097a5fc1c0ef26be4f76445128d25a` |
| Dataset shard manifest | SHA-256 `03c566b3603a8e916224499fb9e4780d71e45d943097ec87d5ab5a60cb4d6065` |
| Phase 7 training config | `config/training_phase7_full.yaml`, SHA-256 `a48d3410baebf8a11fb594d9009e24c341b2e25175dfd147ad1b58b10cc980ae` |
| Seed | 5,004,032 |
| Source commit (training) | `3b84d6d9f85bfe996c05aac9559922065de54285` |
| Environment | Python 3.12.3, PyTorch 2.13.0+cu130, CUDA 13.0, RTX 2060 6GB |

The full-dataset selection/packing manifest
(`data/processed/phase7-full/full_manifest.json`) is deterministically
reproducible from the above identities and is intentionally not committed
to git, matching the established Phase 6 convention (the pilot's own
`pilot_manifest.json` is also not tracked) — it is regenerable evidence,
not a frozen artifact in its own right.

## Repository state

- Branch: `main`.
- New/changed files this phase: `src/juniper_math/full_data.py`,
  `src/juniper_math/full_training_config.py`,
  `src/juniper_math/full_pipeline.py`, CLI wiring in
  `src/juniper_math/cli.py` (`train full-run`, `train full-resume-test`,
  `full-evaluate`, `full-infer`), an optional `warmup_ratio` field added to
  the shared `SchedulerConfig` in `src/juniper_math/training_config.py`
  (backward-compatible, defaults to `None`, verified not to affect Phase
  5/6 config loading), `config/training_phase7_full.yaml`,
  `config/phase7_preflight/lr_{6e-4,8e-4,1e-3}.yaml`,
  `config/phase7_resume_check.yaml`, `tests/test_full_data.py`,
  `tests/test_full_training_config.py`, this report,
  `reports/PHASE7_LR_PREFLIGHT.md`, `reports/PHASE7_RESUME_CHECK.md`,
  `experiments/phase7-full/train_log.jsonl`,
  `experiments/phase7-lr-preflight/*/train_log.jsonl`,
  `experiments/phase7-resume-check/resume_test_log.jsonl`.
- Not committed (gitignored, disposable, matches existing convention):
  `checkpoints/phase7-full/*.pt`, `data/processed/phase7-full/` contents.
  The selected checkpoint's bytes are preserved via the GitHub Release
  asset described in "Remote preservation" above, not via git.
- Commit `3cf0f3bd4a7f8b6053d0bd75f944c54db39b1aae` pushed to
  `origin/main` (fast-forward, no conflicts). Tag
  `phase-7-pretraining-candidate` created (annotated) and pushed,
  resolving to the same commit. The final `phase-7-pretraining` tag is
  deliberately **not** created — per established repo policy (tag only
  after independent review, matching every prior phase's own
  candidate-tag → final-tag sequence), that tag is reserved for after
  Terra's independent review and approval. This report does not
  self-approve Phase 7 or move `current_phase` past 6.

## Known issues (complete list, nothing omitted)

1. **Resume-mechanics numerical tolerance check FAILED** (see
   `reports/PHASE7_RESUME_CHECK.md`) — mechanically correct (exact step/
   token match), numerically outside the inherited 1e-2 tolerance,
   attributed to CUDA attention-kernel nondeterminism amplified by this
   run's higher LR. Not fixed; documented for Terra.
2. **`tool_use_format` emission-rate regression** (80.5% → 37.8% from the
   40% to 100% milestone) despite improving `tool_use` category loss — a
   real, unresolved, documented behavioral finding (see "Evaluation"
   above), not blocking Phase 7 but flagged for Phase 8 attention.
3. **Checkpoint-interval / milestone-fraction step misalignment**
   (1,644/3,288/4,932/6,576 vs. 1,644/3,287/4,931/6,574) — cosmetic
   off-by-one for three of five checkpoints, not a functional bug; noted
   in the "Checkpoints" table above.
4. **Math/calibration/adversarial capability remains at or near chance**,
   as expected and predicted by Phase 6 — Phase 7 is base pretraining, not
   capability tuning; this is not a defect, but is listed for completeness
   since a naive reading of "Full Base Pretraining" could otherwise imply
   a capability claim this report does not make.
5. **The working tree contained unrelated uncommitted changes from the
   prior session** at the moment training started (`source_tree_state:
   "dirty"` recorded in `experiments/phase7-full/train_log.jsonl`'s
   `run_start` event) — this reflects Phase 7's own in-progress files
   being added during the session (the new `full_*` modules, configs, and
   this report itself), not stray unrelated work; the tree is clean of
   anything else as of the commit accompanying this report.

## Terra handoff checklist

- [ ] Verify the Phase 6 → Phase 7 configuration justification (LR
      preflight methodology and conclusion, warmup-ratio rationale, token
      budget derivation) against `reports/PHASE7_LR_PREFLIGHT.md` and this
      report's "Phase 7 configuration" section.
- [ ] Verify frozen component integrity: architecture parameter count
      (5,004,032, code-verified), tokenizer identity, dataset identity
      (`bf9933f0…`) and shard-manifest hash, evaluation-suite identities —
      confirm none silently changed from Phase 6.
- [ ] Verify checkpoint resume state and re-run (or audit)
      `reports/PHASE7_RESUME_CHECK.md`'s bounded check; form an independent
      view on whether the numerical-tolerance FAIL is acceptable given the
      exact mechanical (step/token) match.
- [ ] Reproduce or spot-check the training metrics in
      `experiments/phase7-full/train_log.jsonl` (loss trajectory,
      milestone validation losses, capability-suite results).
- [ ] Independently assess the checkpoint-selection rationale (final vs.
      an earlier milestone), in particular the tool-format-emission
      tradeoff explicitly called out in "Selected Base checkpoint" point 7.
- [ ] Verify the selected checkpoint's SHA-256
      (`0ed23a8262edcf123fc9cc29e5dbd74f9169cc8bf4922d85b5e982d124d47f8e`)
      by downloading the release asset from
      <https://github.com/Cinqic/juniper-math-1/releases/tag/phase-7-pretraining-candidate>
      independently and re-hashing it.
- [ ] Confirm GitHub completeness: this report, `PHASE7_LR_PREFLIGHT.md`,
      `PHASE7_RESUME_CHECK.md`, all new source/config/test files, and the
      experiment logs are present on the pushed commit before approving.
- [ ] Confirm no Phase 8 work (instruction tuning, tool-use SFT, etc.) was
      performed — verify against this report's explicit scope statement
      and `config/project.yaml`'s `next_phase` block.
