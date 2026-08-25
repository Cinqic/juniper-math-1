# Phase 8 Plan — Mathematical Instruction and Tool Training

Status: engineering plan, committed before any Phase 8 training. Phase 8 is
`AUTHORIZED — NOT STARTED` in `config/project.yaml` at the time this plan is
written. This document freezes the goals, data design, masking design,
training/evaluation strategy, and success/failure gates *before* any
expensive run, per the Phase 8 instructions Sec. 6.

## 1. Starting repository state

- `origin/main` at `79c14afb291ac04123db16e90d64828ec62d959f` (tag
  `phase-7-pretraining`), working tree clean at the start of Phase 8.
- Approved Phase 7 Base: `phase7-full-v2`, checkpoint
  `step_007483_final.pt`, SHA-256
  `2e8098ab3a5db3c59a82fad19af2050154637fbe0628f3f6b0ca932d6cb6ea60`,
  independently verified in this session by downloading the asset fresh from
  the `phase-7-pretraining` GitHub release and re-hashing it (matches
  exactly). The local `checkpoints/phase7-full/` directory contains only
  leftover files from the *invalidated* v1 candidate (different step counts,
  different hashes) — not used; the verified Base lives at
  `checkpoints/phase7-full-v2/step_007483_final.pt`.
- Architecture identity `0.1.0` (5,004,032 params, d_model 256, 5 layers, 4
  Q/KV heads, SwiGLU, RMSNorm, RoPE θ=10,000, no biases, tied 4,096-vocab
  embedding, 1,024 context), tokenizer `juniper-math-tokenizer-v1`, tool
  protocol `juniper-tool-protocol-v1` v1.0.0 — all frozen, all unchanged by
  Phase 8.
- Frozen dataset `juniper-math-dataset-v1`: train 1,466,970 / validation
  81,094 / test 81,014 (`config/dataset.yaml`, `master_seed: 20260401`).
  During this session's pre-flight measurement, four unmanifested,
  contaminated shard files were found in
  `data/processed/juniper-math-dataset-v1/` — the same four shards Phase 7's
  remediation already documented as invalid v1-build contamination
  (`reports/PHASE7_REMEDIATION.md`). They were quarantined out of the
  processed directory (not part of the frozen 34-shard manifest; moved, not
  deleted) so `verify_parent_dataset_shards` passes against the true frozen
  34-shard manifest. This is local-storage cruft, not a change to the frozen
  dataset itself.

## 2. Phase 7 baseline evaluations (frozen, historical — Sec. 15 re-runs these against the Base before any SFT)

From `reports/PHASE7_FINAL_APPROVAL.md`: validation loss 0.600296; frozen
math 1/215 (0.47%); calibration 0/130; adversarial 36/195 (18.46%);
tool-call emission 115/185 (62.16%); valid parse 0%; tool-name match 0%.
Per `reports/PHASE7_RESULTS.md`, free-generation `<tool_call>` emission on
the frozen `tool_use_format` suite peaked at 80.5% at the 40% milestone and
*declined* to 37.8% by the final step even as `tool_use`-category validation
loss kept improving — i.e., the Base has representations for tool-shaped
text but has not learned *reliable, well-formed, well-timed* tool
invocation. This is exactly the format/instruction-following gap Phase 8
targets. These are baseline diagnostics, not capability claims (Sec. 2 of
the Phase 8 instructions).

## 3. Phase 8 goals

Teach the Base, via supervised fine-tuning only (no RLHF/DPO/preference
optimization — Sec. 35), the interaction loop: interpret → decide
direct-vs-tool → (call tool | answer | clarify | refuse) → (if tool)
interpret the real runtime result → produce a concise final answer. Success
is evaluated on: valid tool-call rate, correct tool-name rate, argument
correctness, end-to-end tool completion, unnecessary/missed-tool rates,
direct-answer correctness, clarification/unsupported handling, and
non-regression of Base validation/frozen-suite performance (Sec. 22).

## 4. Supervised-data design

**Source**: the frozen `juniper-math-dataset-v1` **train split only**
(never validation/test, never the four frozen v2 eval-suite files, which
were already reserved out of the train split at Phase 4 build time via
`ExactDeduplicator.seed()`). All 24 frozen categories
(`juniper_math.dataset.schema.VALID_CATEGORIES`) already cover the Sec. 8
supervision requirements end to end — direct answers (`arithmetic`,
`decimals`, `fractions`, `percentages`, `ratios_proportions`,
`numerical_comparison`, `estimation`, `basic_algebra`,
`expression_translation`, `word_problem`, `operator_precedence`,
`negative_values`, `scientific_notation`), tool categories
(`tool_use`, `unit_conversion`, `financial_math`, `multi_step`), error
correction (`incorrect_supplied_answer`, `incorrect_tool_call`,
`tool_error`), clarification (`ambiguity`, `missing_information`),
unsupported (`undefined_operation`, `unsupported_capability`). **No new
generator module is needed** — this satisfies Sec. 7's "may derive
supervised examples from approved Phase 4 training records... existing
verified generators" and avoids inventing a second, potentially-drifting
data-generation path. Every tool-required example's ground truth is a real
`ToolRuntime` execution trace recorded at Phase 4 build time (`tool_traces`
on the frozen `Example`), never re-synthesized — preserving ADR 0004's
"deterministic tools over neural guessing" invariant and Sec. 8's
fabricated-result-resistance requirement structurally, not just by policy.

**Selection method** (new module `juniper_math.sft_data`, mirroring
`pilot_data.select_pilot_examples`'s two-pass count/select structure, but
**flattened category targets instead of corpus-proportional ones**): a
uniform per-category target with a floor/cap by availability, specifically
*to counter* the tool-collapse risk in Sec. 14 — the frozen corpus's
category mixture is weighted toward direct arithmetic categories, so
naively proportional sampling would under-represent rarer-but-critical
categories like `tool_error` (~1,400 available) and `missing_information`
(~750 available) relative to `arithmetic` (~374,000 available). Flattening
gives the model comparably strong signal for "don't call a tool" and "call
the tool correctly" alike. Same deterministic (stride, offset) selection
primitive as Phase 6 (`derive_seed`-backed, no `random`/row-order
dependence).

**Target sizes**: ~1,000 train examples/category floor (min(1,000,
available)) → **~24,000 train examples**, ~150 validation examples/category
drawn from the frozen **validation** split only (never trained on) → **~3,600
validation examples**. Deliberately modest relative to the 1.46M-example
corpus per Sec. 7 ("do not blindly copy... quality over raw token count") —
enough for category/behavior coverage and diminishing-returns balance at a
5M-parameter model's capacity, small enough to bound RTX 2060 wall-clock.

**Sequence length**: empirical measurement this session (2% sampled scan,
29,386 examples, using the real `render_training_text` + tokenizer) gives
median 29 / p90 58 / p95 159 / p99 197 / p999 232 / max 256 tokens
(BOS+EOS included) across all 24 categories; the longest categories are
`incorrect_tool_call` (max 256), `unit_conversion` (max 234),
`financial_math`/`tool_error`/`tool_use` (~180-211). Phase 8 sets
`max_sequence_length = 256` for the SFT corpus, **and** the selection
pipeline explicitly **rejects** (counts, never truncates) any selected
example whose full rendered+tokenized length exceeds 256 — since the 256
figure comes from a 2% sample, not an exhaustive scan, silent truncation of
an unseen longer tail example is not an acceptable failure mode (Sec. 12).
**No packing**: each SFT example occupies its own sequence (right-padded),
unlike Phase 6/7's first-fit packing. Packing two unrelated supervised
trajectories into one causal sequence has no tested boundary-safety story
for *masked* loss (Sec. 12 explicitly warns packing needs "a tested,
semantically safe boundary strategy" it does not assume by default); at
SFT's modest scale the padding-compute cost of one-example-per-sequence is
acceptable and removes an entire class of masking bugs.

## 5. Loss-masking design

Assistant-focused loss (Sec. 11): prompt tokens and `<tool_result>{...}`
tokens are context-only (label = `-100`); `<tool_call>{...}` tokens and the
terminal `<final>`/`<unsupported>`/`<error>` tokens are loss-bearing. New
module `juniper_math.sft_rendering` renders each example as an ordered list
of `(text, role)` segments — reusing the exact tag conventions
`dataset.shard.render_training_text`/`expected_completion` already froze
(`<tool_call>`, `<tool_result>`, `<final>`, `<unsupported>`, `<error>`), then
**tokenizes each segment independently** (never re-tokenizes the joined
string) so segment boundaries are exact at the token level by construction,
not recovered after the fact by string-matching. BOS is context, EOS is
supervised (it is the correct "stop" signal after a supervised segment).
Padding is `-100`. This exact scheme is unit-tested against every item in
Sec. 11's required list (`tests/test_sft_rendering.py`) before any training
run, including a check that concatenated segment-wise token ids equal
`pilot_data.tokenize_examples`' joint-string tokenization for a
representative sample (proves segment-wise tokenization introduces no
drift at the tag boundaries used here).

## 6. Training architecture

New `juniper_math.sft_config`/`juniper_math.sft_pipeline`, following
`full_training_config.py`/`full_pipeline.py`'s exact house style
(frozen-dataclass config, `_load_common` identity-mismatch guard, milestone
loop). Reuses `juniper_math.trainer` **completely unchanged** — the trainer
loop is already generic over `-100`-masked labels (it computes
loss-bearing-token counts and cross-entropy directly from whatever labels a
`Dataset.__getitem__` returns), so assistant-focused masking requires zero
trainer changes, only a new `Dataset` (`MaskedSftDataset`) that emits
`{input_ids, labels, attention_mask}` with real masked labels instead of
`labels == input_ids`.

**Parent weights, fresh optimizer** (Sec. 4): `sft_pipeline.init_sft_state`
loads `state.model` from the verified Base checkpoint via
`checkpoint.load_checkpoint(..., optimizer=None, scheduler=None,
restore_rng=False)` (model weights only), then builds a brand-new AdamW +
warmup/cosine schedule via `trainer.init_state`'s optimizer construction —
Phase 8 never resumes Phase 7's optimizer/scheduler/RNG trajectory. Every
Phase 8 checkpoint's `extra` field records `parent_checkpoint_path`,
`parent_checkpoint_sha256`, `parent_phase7_tag: "phase-7-pretraining"`, plus
the standard architecture/tokenizer/dataset/config/seed/source-commit
identities `checkpoint.py` already threads through `training_config.raw`.

**Config** (`config/training_phase8_sft.yaml`, drafted here, values
finalized after the Sec. 19 preflight): `dataset_identity:
"juniper-math-sft-v1"` (Phase 8's own frozen SFT-selection identity, never
overwriting `juniper-math-dataset-v1`), `architecture_identity: "0.1.0"`,
`tokenizer_identity: "juniper-math-tokenizer-v1"`,
`tool_protocol_identity: "juniper-tool-protocol-v1"`, `parent_checkpoint_
sha256` (checked at load, fail loudly on mismatch — same idiom as
`full_pipeline._load_common`'s architecture/dataset checks).

## 7. Evaluation strategy

1. **Pre-SFT baseline** (Sec. 15): re-run, not recalled from a document —
   full frozen-suite evaluation (math/calibration/adversarial/tool-format)
   plus a new end-to-end tool-interaction pass against the verified Base,
   before touching any weight.
2. **New Phase 8 held-out eval suite** (`evals/phase8_instruction_v1.json`,
   Sec. 16): built by a new `dataset.eval_isolated`-style function using a
   seed namespace disjoint from both the training-corpus generators and the
   existing v2 suites, covering direct-answer, tool-routing (incl.
   tool-negative), tool-argument-correctness, tool-result-interpretation,
   error-correction, clarification, and unsupported-request cases. Frozen
   (hashed, committed) before the real training run.
3. **New end-to-end tool-interaction harness** (`juniper_math.
   tool_interaction`, Sec. 10): generate → detect `<tool_call>` → parse via
   the frozen protocol parser → execute via the real `ToolRuntime` → append
   the *runtime's* `<tool_result>` (never a model-generated one) → continue
   generation → extract final answer. Reused at every milestone and for
   final checkpoint selection.
4. **Regression suite** (Sec. 22): the four frozen v2 suites + full
   validation loss + category losses, run identically on the Base and every
   milestone/candidate checkpoint (`juniper_math.full_pipeline.
   run_capability_suites` reused unchanged).
5. **Sec. 23's 18 numerator/denominator tool metrics**, computed by
   `juniper_math.sft_eval` over the held-out suite at every milestone.

## 8. Checkpoint-selection criteria (fixed now, before results exist — Sec. 21)

A candidate is preferred over another only when it improves at least one of
{valid tool-call rate, tool-name match rate, argument correctness, tool
end-to-end success rate, unnecessary-tool rate, missed-tool rate, direct
mathematical accuracy on the held-out suite} **without** regressing frozen
validation loss by more than a fixed absolute tolerance of 0.05 nats and
without regressing any frozen-suite accuracy below the Base's own value. If
no later milestone strictly dominates, the milestone with the best
composite of (valid-call rate + tool-name-match rate + end-to-end success
rate + direct-answer accuracy) minus (unnecessary-tool-rate +
missed-tool-rate) wins, with the exact numbers and reasoning recorded in
`reports/PHASE8_RESULTS.md`. The final training step is never auto-selected
merely for being final (Sec. 21).

## 9. Planned preflight experiments (Sec. 19)

Bounded, one-variable-at-a-time, using the verified Base:
- **Candidate A**: learning rate carried over from Phase 7 (8e-4), full SFT
  mixture, 5% warmup.
- **Candidate B**: lower learning rate (2e-4) — SFT fine-tuning from a
  pretrained base conventionally needs a much smaller LR than
  from-scratch pretraining; testing whether 8e-4 destabilizes/overwrites
  Base representations.
- **Candidate C**: same LR as the winner of A/B, but a flattened-vs-default
  mixture ablation (double the weight on `tool_error`/`incorrect_tool_call`/
  `missing_information`/`ambiguity` — the rarest, highest-value-per-example
  categories) to check whether under-represented-category weighting
  measurably changes routing/error-handling metrics at this bounded scale.

Each candidate trains for a small fixed step count (order of a few hundred
steps, chosen so a full preflight round completes in well under an hour on
the RTX 2060) and is scored on the held-out suite's tool metrics + direct
accuracy + frozen-suite regression, exactly per Sec. 19's evidence list.
Results and the winning configuration are recorded in
`reports/PHASE8_PREFLIGHT.md` before the config is frozen for full training.

## 10. Hardware constraints / compute budget

RTX 2060 6 GB, 16 GB system RAM, Ryzen 7 5700G. Phase 7's full pretraining
measured ~904 MiB peak VRAM at micro-batch 4 / seq-len 1024 / fp32; Phase
8's shorter sequences (256 vs 1024) and smaller per-step batch keep VRAM
well inside budget with no precision change needed. Full SFT run budget:
on the order of a few thousand optimizer steps over the ~24,000-example SFT
train set (roughly 1-3 epochs at the frozen batch size), bounded so the
complete preflight + full run + resume-check + evaluation fits in a single
extended session.

## 11. Non-regression methodology

Every milestone and the pre-SFT baseline are evaluated with byte-identical
suite files, decoding settings (`temperature=0.0`, greedy), and sample sizes
— never a moving target, matching Phase 6/7's own discipline. Base-vs-
candidate comparison is a literal side-by-side table in
`reports/PHASE8_REGRESSION.md` (Sec. 22), not a narrative claim.

## 12. Success gates / failure conditions

**Success** (Sec. 34): the selected candidate must show measurable, numeric
improvement over the Base on at least valid-tool-call rate, tool-name-match
rate, and direct-vs-tool routing, while frozen validation loss and
frozen-suite accuracies do not regress beyond the Sec. 8 tolerance.

**Failure**: if no SFT candidate improves tool-call validity/routing over
the Base without unacceptable regression, Phase 8 is not declared complete
by manufacturing a favorable metric; the failure, its likely cause (data
mixture, LR, mask correctness), and a corrective bounded rerun are
documented instead (Sec. 34). The architecture is never changed to rescue
Phase 8 (Sec. 34, Sec. 3 boundary).

## 13. Frozen-component boundary (restated)

Phase 8 must not modify: model architecture/parameter count, tokenizer
model/vocabulary/special-token identities, the frozen `juniper-math-
dataset-v1` corpus or its shard manifest, the frozen Phase 4 eval suites,
the tool protocol/calculator semantics, or the approved Phase 7 checkpoint.
Phase 8 may add: SFT data-selection code, SFT config, loss-masking/
rendering code, the tool-interaction evaluation harness, Phase 8 eval
suites, Phase 8 tests/reports/manifests, and checkpoint-selection logic —
per Sec. 3.
