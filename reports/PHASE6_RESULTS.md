# Phase 6 Pilot-Pretraining Results

This is the first training phase where model behavior is meant to matter.
**This is not a claim of mathematical capability.** A 5M-parameter model
trained on 8% of the frozen corpus for 320 optimizer steps is not expected
to solve math reliably — see §Interpretation at the end of this report and
Sec. 41 of the Phase 6 instructions.

## Experiment identity

- Run ID: `phase6-pilot-v1` (`experiments/phase6-pilot/train_log.jsonl`)
- Starting commit / tag: `73792c04f365c6f139a979f6950fa87be2af5d76` / `phase-5-smoke`
- Config: `config/training_phase6_pilot.yaml` (sha256
  `cb2eb8249fef4c8cc7b87c3e3e4f9807988aa230268780f1214df64a6d676343`)
- Seed: `5004032` (same project seed as Phase 5, for a directly comparable
  initialization identity)

## Frozen artifact identities (unchanged from Phase 5)

| Artifact | Identity |
|---|---|
| Architecture | v0.1.0, 5,004,032 trainable parameters (programmatically verified) |
| Tokenizer | `juniper-math-tokenizer-v1` |
| Dataset | `juniper-math-dataset-v1`, whole-dataset identity `bf9933f032a58b4eb618b32156783b8563097a5fc1c0ef26be4f76445128d25a` |
| Tool protocol | `juniper-tool-protocol-v1` v1.0.0 |
| Evaluation suites | `phase4_math_v2` (215), `phase4_tool_use_v2` (185), `phase4_calibration_v2` (130), `phase4_adversarial_v2` (195) — 725 cases total |

`hash verify` and `manifests-validate` confirm every one of these matches
the approved Phase 5 baseline byte-for-byte — Phase 6 did not touch any of
them.

## Pilot corpus

Selected by `juniper_math.pilot_data` (category-stratified, per-category
fixed-stride sampling extending Phase 5's `compute_stride_selection`), from
the train split only — `data/processed/phase6-pilot/pilot_manifest.json`
and `pilot_selection_audit.json` (not committed; reproducible from the
frozen dataset + this config + seed):

| Split | Examples | Total tokens | Packed sequences (seq_len 1024) | Padding fraction |
|---|---|---|---|---|
| train | 137,057 | 5,002,683 | 5,331 | 3.34% |
| validation | 3,043 | 112,765 | (unpacked, per-category) | n/a — see below |

- **8.01% of the full frozen corpus** (5,002,683 / 62,421,215 tokens),
  **8.90% of the train split alone** (5,002,683 / 56,209,616 tokens).
- All 24 frozen categories present in the train pilot subset. Every
  category is floored at `min(60, available-in-train)` — see §Rare
  category finding below for one category where availability, not the
  selection algorithm, was the binding constraint.
- `tool_required_count`: 7,156 of 137,057 train examples (5.2%).
- `family_count`: 24 distinct families represented (family/derivation
  boundaries preserved automatically — this subset never crosses split
  boundaries, so no train/validation/test leakage is introduced by pilot
  selection).
- Validation subset is category-stratified the same way, frozen for the
  entire run (never re-selected between milestones, never trained on).

### Rare category finding

`tool_error` — one of the six adversarial/error-handling categories Sec.
7 specifically calls out to protect from disappearing — was selected at
only **26 examples**, well under the 60-example floor. This is not a
selection-algorithm shortfall: `pilot_selection_audit.json` shows the
frozen train split contains **exactly 26 `tool_error` records in total**
(out of 1,553 dataset-wide). Because Phase 4's split assignment groups by
`(generator_id, family_id, derivation_id)` rather than row-shuffling, this
category's few families happened to fall mostly into validation/test. The
pilot subset includes **100% of what is actually available** in train for
this category — the strongest guarantee possible given availability — but
this is a genuine, worth-flagging characteristic of the frozen dataset: a
Phase 7 model training on the full train split will still see very few
`tool_error` examples specifically, independent of any pilot-scale
sampling decision. All five other rare categories in Sec. 7's list
(`missing_information` 1,109 available/92 selected [floor did not bind: 92
> floor because token-proportional target already exceeded the 60 floor],
`undefined_operation` 8,298 available/689 selected, `ambiguity` 8,470
available/703 selected, `unsupported_capability` 8,672 available/720
selected, `incorrect_tool_call` 5,886 available/488 selected) had ample
train-split availability and were sampled well above the floor.

## Sequence length and packing

Benchmarked using the existing, unmodified Phase 1 benchmark script
(`scripts/benchmark_phase1.py`, already measured seq_len 128/512/1024 —
reused rather than rewritten) on this exact RTX 2060:

| seq_len | Peak VRAM allocated (batch 4, training) | Training tokens/sec |
|---|---|---|
| 128 | 159.3 MiB | 41,951 |
| 512 | 417.7 MiB | 67,923 |
| 1024 | 757.6 MiB | 63,543 |

1024 (the full frozen architectural context) costs a small fraction of the
6,144 MiB budget and has throughput comparable to 512 — no VRAM or
throughput reason to avoid it. Frozen corpus examples are short (median 27
tokens, p99 194 — `data/processed/juniper-math-dataset-v1/stats.json`), so
naively padding each one to 1024 individually (Phase 5's approach) would
waste the large majority of every step's compute; **packing** (deterministic
single-pass first-fit, `juniper_math.pilot_data.pack_sequences`) reduces
that to 3.34% measured padding waste at the actual pilot scale. Decision:
use the full 1024-token context, packed. See `docs/PILOT_TRAINING.md` for
why validation deliberately does not use packing.

## Environment / hardware

| | |
|---|---|
| OS | Linux 7.0.0-30-generic |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| CUDA | 13.0 |
| GPU | NVIDIA GeForce RTX 2060 (6,144 MiB), driver 595.84 |
| CPU | AMD Ryzen 7 5700G (16 logical cores) |
| System RAM | 14 GiB total (8.4 GiB available at measurement time) |
| Free storage before run | 176 GiB (`/`) |
| Free storage after run | see §Storage below |

## Pilot training configuration

Effective batch 16 packed sequences/step (micro-batch 4 × grad-accum 4),
sequence length 1024, fp32, AdamW (lr 6e-4, weight_decay 0.01, betas
0.9/0.95, grad-clip-norm 1.0), cosine schedule with 16-step (5%) warmup
down to 10% of peak lr, 320 optimizer steps (≈0.96 epoch over the 5,331
packed train sequences).

## Controlled experiments

A short (60-optimizer-step) peak-learning-rate screen, holding
architecture/seed/pilot-data/ordering/token-budget fixed, was run before
committing to the canonical pilot config (Sec. 14):

| Learning rate | Final-10-step mean training loss | Max pre-clip gradient norm | Non-finite loss observed |
|---|---|---|---|
| 1.5e-4 | 5.06 | — | No |
| 3.0e-4 (Phase 5 baseline) | 3.82 | — | No |
| 6.0e-4 | 2.77 | 6.0 | No |
| 1.0e-3 (follow-up, not adopted) | 2.20 | 10.6 | No |

All four points were stable (finite loss and gradients throughout, grad
clip at 1.0 kept applied updates bounded regardless of pre-clip norm).
6.0e-4 was adopted: it clearly outperformed the Phase 5 baseline at this
horizon with no stability cost. 1.0e-3 converged faster still but was
deliberately **not** adopted — it was a single follow-up point, not part of
the original screening design, and pushes further from the validated
Phase 5 operating point than a single conservative bump justifies for a
first capability run. This is the only hyperparameter that was swept;
Sec. 14 explicitly discourages an exhaustive search, and every other
optimizer/scheduler value was inherited unchanged from the Phase 5
baseline.

## Training behavior

| Milestone | Step | Fraction | Validation loss |
|---|---|---|---|
| Init | 0 | 0% | 8.3834 |
| 1 | 80 | 25% | 2.1168 |
| 2 | 160 | 50% | 1.1740 |
| 3 | 240 | 75% | 1.0265 |
| Final | 320 | 100% | 0.9807 |

Monotonic decrease at every milestone — 8.38 → 0.98, a clear, non-trivial
downward trend across the full pilot, not a single lucky checkpoint.
Gradient and parameter finiteness: PASS at every one of the 320 optimizer
steps (checked before/after `optimizer.step()`, per Sec. 22 of the
frozen `trainer.py` mechanics Phase 5 already validated).

Total tokens_seen (loss-bearing, post-shift): **5,062,958**.

### Category validation loss (init → final)

The full per-category breakdown is in
`experiments/phase6-pilot/train_log.jsonl` (`milestone` events); notable
points:

- Lowest final loss: `financial_math` (8.33 → 0.34), `scientific_notation`
  (8.33 → 0.53), `tool_use` (8.36 → 0.49) — all categories with more
  structurally repetitive prompt/answer patterns.
- Highest final loss: `missing_information` (8.37 → 2.94),
  `unsupported_capability` (8.45 → 2.32), `ambiguity` (8.44 → 2.33) — the
  three "no concrete numeric answer" behaviors, which require the model to
  reliably emit a specific non-`<final>` control tag rather than any
  plausible-looking number; consistent with these being the hardest
  categories both semantically and by training-signal density (fewest
  examples).
- This spread (0.34 to 2.94 final loss across categories, all from the
  same 8.38 initialization) is itself informative for Phase 7: uniform
  token-proportional sampling under-serves the categories that are
  hardest to learn from few examples.

## Performance / resource observations

| | |
|---|---|
| Peak CUDA memory | 904.5 MiB (14.7% of the 6,144 MiB budget) |
| Pure training throughput | ~41,500 loss-bearing tokens/sec (320 steps' logged sample: mean 0.381s/optimizer step) |
| Training-only wall time (320 steps) | ~122s |
| Total wall time (`train pilot-run`, training + all 5 milestone evaluations across all 4 frozen suites) | 406.9s (6.8 min) |
| Milestone evaluation overhead | ~285s / 5 milestones ≈ 57s/milestone (725 generations + validation across 24 categories) |
| Final checkpoint size | 60,123,779 bytes (~57.3 MiB) — same order as Phase 5's checkpoint format, this is model+optimizer+scheduler+RNG state, not model weights alone |
| Checkpoint write overhead | Included in the 406.9s total; 5 checkpoints written (4 milestone + 1 final-duplicate at step 320) |

## Generation: qualitative trajectory across milestones

Fixed 13-prompt set (`fixed_generation_prompts`, one per required category
— arithmetic, word problem, algebra, percentages, units, financial math,
multi-step, tool-required, no-tool-needed, ambiguous, missing-information,
unsupported, incorrect-supplied-answer), greedy decoding, 32 max new
tokens, identical prompts at every milestone:

| Step | Representative behavior |
|---|---|
| 0 (init) | Degenerate repetition of a single BPE token (`ckckckck...`, `ImmediateImmediate...`, `ActivityActivity...`) — untrained-model garbage, no structure at all. |
| 80 (25%) | Learned to consistently emit `<final>NNN` after every prompt — structurally correct format, but the model has collapsed to emitting the same placeholder digits (`111`, `110`) regardless of the actual prompt content. |
| 160 (50%) | `<final>` values start varying by prompt (`-3`, `10.2`, `-1`) and `<tool_call>{...}`-shaped JSON begins appearing for tool-relevant prompts (unit conversion, financial math, ambiguity) — though the JSON is not yet protocol-valid (malformed key names, duplicate/garbled fields). |
| 240 (75%) | Continued differentiation; `<final>True` appears correctly in *format* (though not correctness) for the incorrect-supplied-answer boolean-style prompt; `unsupported_capability` prompt starts drifting into unrelated invented word-problem text rather than a clean `<unsupported>` tag. |
| 320 (final) | The `multi_step` prompt now emits a bare `<unsupported>` tag — structurally the right *kind* of response for a behavior category, even though `multi_step` is not actually an unsupported-capability case (a miscalibration, not evidence of correct behavior). `<tool_call>{...}` JSON is denser and closer to well-formed for several prompts but still not schema-valid. |

**Interpretation:** clear, unambiguous behavioral change across training —
the model moved from raw-token-repetition garbage to consistently
structured output (`<final>`/`<tool_call>`/`<unsupported>` control tokens
in roughly the right positions), and by the final milestone shows
prompt-dependent variation rather than a single collapsed response. It has
**not** learned correct arithmetic, correct tool-call syntax, or reliable
behavior-tag calibration — none of that is expected at this scale (Sec.
41), and none of it is claimed here.

## Capability evaluation (all four frozen v2 suites, full suites at every milestone)

| Milestone | math (215) | tool_use format-valid (185) | calibration (130) | adversarial (195) |
|---|---|---|---|---|
| 0 (init) | 0.00% | 0.00% | 0.00% | 0.00% |
| 80 (25%) | 0.00% | 0.00% | 0.00% | 0.00% |
| 160 (50%) | 0.00% | 0.00% | 0.77% (1/130) | 0.00% |
| 240 (75%) | 0.00% | 0.00% | 0.77% (1/130) | 0.51% (1/195) |
| 320 (final) | 0.00% | 0.00% | 0.00% | 0.51% (1/195) |

**This is the expected and correct result at this scale, not a defect.**
Every generation across all 3,625 (725 cases × 5 milestones) evaluation
calls was scored — never skipped, never excluded from the denominator
(Sec. 19, Sec. 30). 0% math accuracy is consistent with the generation
trace above: the model reliably emits *a* number after `<final>`, but not
the *correct* number — 320 steps and 137K training examples is nowhere
near enough for a 5M-parameter model to internalize arithmetic. 0%
tool-call format validity is consistent with the malformed JSON observed
in the generation trace (duplicate/garbled keys) — Phase 5's existing
`tool_format_eval` module scores this the same way it did at smoke scale,
unmodified. The two isolated 1-case hits (calibration @ step 160,
adversarial @ steps 240/320) are single correct-tag emissions among
hundreds of cases — statistical noise at this sample size, not a trend;
reported honestly rather than described as "the model is starting to
calibrate."

## Resume verification (Sec. 24 gate, pilot scale)

`python -m juniper_math train pilot-resume-test` — Run A (uninterrupted,
320 steps) vs. Run B (interrupted at step 160, checkpointed, restored into
brand-new model/optimizer/scheduler objects simulating a new process,
resumed to 320):

| | Run A (uninterrupted) | Run B (interrupted @ step 160, resumed) |
|---|---|---|
| Final step | 320 | 320 |
| Tokens seen | 5,062,958 | 5,062,958 |
| Final training loss | 1.0024919797201934 | 1.0025223562941916 |
| Loss-history max abs diff (common steps) | 0.0009533531856931177 | |
| Max parameter abs diff | 0.0034764092415571213 | |
| Fixed-prompt generations | identical | identical |
| Result | **PASS — equivalent** (tolerance `<1e-2` on both loss and parameter diff) | |

**This is the honest, expected result at pilot scale, and it differs from
Phase 5 in exactly the way Phase 5's own report predicted it might**: Run A
and Run B are **not bitwise-identical** (max param diff ≈3.5e-3, max loss
diff ≈9.5e-4) — unlike Phase 5's smoke run, which happened to land on
bitwise-exact equivalence at 200 steps on this GPU. Both differences are
comfortably inside the `<1e-2` tolerance the comparison actually enforces
(run twice independently during this session — the first run measured
≈2.4e-4/≈2.9e-4, this one ≈3.5e-3/≈9.5e-4 — both well within tolerance
despite the run-to-run variance, itself evidence that CUDA kernel
nondeterminism, not a code defect, is the source), and every other
equivalence criterion (final step, tokens seen, generation text) matched
exactly on both runs. `torch.use_deterministic_algorithms(...,
warn_only=True)` is a best-effort request, not a hardware guarantee, and
this run demonstrates precisely why the gate is tolerance-based rather than
assuming bitwise identity will always hold — the tolerance was not
loosened to convert a failure into a pass; the pre-existing `1e-2`
threshold (identical to Phase 5's) was met without adjustment on both
independent runs.

## Test suite / lint / type / hash gates (full, from a clean state)

```
pytest -v            # 634 passed
ruff check .          # All checks passed
ruff format --check . # all files already formatted
mypy                  # Success: no issues found in 62 source files
python -m juniper_math hash verify         # PASS, including new pilot_training_config entry
python -m juniper_math manifests-validate  # PASS
```

## Storage

| | Before | After |
|---|---|---|
| Free storage (`/`) | 176 GiB | 176 GiB (rounds equal at GiB resolution; ~1 GB actually consumed) |
| Used storage (`/`) | 46 GB | 47 GB |

Growth breakdown: `checkpoints/phase6-pilot/` 593 MB (5 milestone/final
checkpoints from the canonical run + init/interrupt/resume checkpoints
from the resume-test, all `.gitignore`d and disposable per
`checkpoints/README.md`), `experiments/phase6-pilot/` 84 KB (`train_log.jsonl`
+ `resume_test_log.jsonl`, committed), `data/processed/phase6-pilot/` 40 KB
(pilot manifest + selection audit, `.gitignore`d, reproducible). Well
within the practical storage budget for the target FLOWBOX hardware
(256 GB NVMe, per `config/project.yaml:primary_hardware`).

## Phase 7 restart rule (Sec. 28)

Nothing about the frozen architecture, tokenizer, special tokens, sequence
representation, or fundamental training objective changed during Phase 6.
Per Sec. 28's default, Phase 7 should still begin from a clean, deliberately
chosen initialization rather than silently continuing from this pilot's
final checkpoint — the pilot checkpoint's existence must not by itself
determine the Phase 7 research design. See `checkpoints/README.md` for the
resulting checkpoint-preservation decision (disposable, not archived).

## Known limitations

- `tool_error` availability in the train split (26 examples total) limits
  what any train-split-only model — pilot or full Phase 7 — can learn
  about that specific category; see §Rare category finding.
- Capability accuracy at 0-0.5% across all four suites is a real
  measurement at this scale, not evidence about what a full Phase 7 run
  would achieve — see §Interpretation.
- Generation still has no KV cache and is not batched (unchanged from
  Phase 5) — fine at the pilot's per-milestone case counts (up to 725
  sequential generate() calls, ~57s/milestone including validation), would
  need real work before any interactive/high-throughput usage.
- The two single-case capability hits (calibration, adversarial) should
  not be read as an emerging trend without a larger, longer run — flagged
  explicitly above.
- Pilot checkpoints are disposable, not archived to a remote store (see
  §Phase 7 restart rule and `checkpoints/README.md`); reproducibility is
  demonstrated via the resume-comparison gate, not via checkpoint
  preservation.

## Interpretation

Observed: loss decreased monotonically and substantially (8.38 → 0.98)
across 320 steps on 5M real training tokens; generation moved from
degenerate token repetition to structured, prompt-dependent control-token
usage; category-level validation loss spread is internally consistent
with category difficulty/example density. Measured: capability accuracy on
all four frozen suites remained at 0-0.5% throughout — a tiny model
trained on 8% of the corpus for a few hundred steps producing plausible-
looking but incorrect numbers is not evidence it can do math, and is not
claimed as such here. Inferred: the training configuration (packing,
lr=6e-4, effective batch 16, seq_len 1024) is stable and produces a clear,
attributable learning signal at pilot scale — this is the basis for the
Phase 7 recommendation below, not a claim about the pilot model's
usefulness. Speculative: whether this configuration scales cleanly to a
much larger Phase 7 token budget is not directly tested by this pilot and
should be treated as the highest-priority open question for Phase 7
itself, not as settled by this report.

## Recommended Phase 7 configuration

See `reports/PHASE6_COMPLETION.md` for the consolidated recommendation
table and the measurement each entry cites.
