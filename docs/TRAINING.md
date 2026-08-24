# Phase 5: Smoke Pretraining

## Purpose

Phase 5 answers one question: **does the complete Juniper Math 1 training
pipeline actually work end to end under realistic execution conditions?**
It is not a capability run, not a benchmark, and not an opportunity to
redesign the architecture, tokenizer, dataset, or tool protocol — all four
remain exactly as frozen in Phases 1-4. See
[`reports/PHASE5_RESULTS.md`](../reports/PHASE5_RESULTS.md) for the actual
run's numbers and
[`reports/PHASE5_COMPLETION.md`](../reports/PHASE5_COMPLETION.md) for the
phase completion report.

The smoke model produced by this pipeline is expected to be mathematically
useless. What must work is every mechanical step of the pipeline: frozen
dataset artifacts → a deterministic tiny subset → tokenization → batching
→ forward/backward → optimization → validation → checkpointing →
checkpoint restoration → resumed training → generation → evaluation →
tool-format evaluation → logging → artifact preservation → repository
recovery.

## Configuration

[`config/training.yaml`](../config/training.yaml) is the single source of
truth for smoke-training parameters, loaded by
[`src/juniper_math/training_config.py`](../src/juniper_math/training_config.py).
It never overrides `config/architecture.yaml`, `config/tokenizer.yaml`, or
`config/dataset.yaml` — it only layers training-time parameters (smoke
subset size, batch/optimizer/scheduler settings, run length, device,
precision) on top of those frozen identities, and the loader fails loudly
if `architecture_identity`/`dataset_identity` in `config/training.yaml`
disagree with the frozen configs.

## Smoke subset selection

The dataset itself (`juniper-math-dataset-v1`, 1,629,078 examples) is
frozen and unmodified. Phase 5 trains on a deterministic tiny subset
selected by fixed-stride sampling over each split's shard files, in their
committed on-disk order — see
[`src/juniper_math/smoke_data.py`](../src/juniper_math/smoke_data.py):

```
stride = floor(split_record_count / target_examples)
offset = seed % stride
select every line whose zero-based global index satisfies
  (index - offset) % stride == 0
```

This reproduces byte-identically from `(dataset_identity, split
record_count, seed, target_examples)` with a single sequential scan (most
lines are never JSON-parsed). Selection is recorded to
`data/processed/phase5-smoke/smoke_manifest.json` (parent dataset identity,
per-split stride/offset, selected-example-id hash, token counts) — this
manifest is the reproducibility record; it is not committed (disposable,
like the parent dataset's own shard files), but is fully reconstructible
by re-running `train run` against the same frozen dataset build.

Each selected example is rendered into training text with the *same*
`render_training_text` the frozen dataset's own token-count pass uses
(`juniper_math.dataset.shard.render_training_text`), so smoke training sees
exactly the text the dataset declares that example to be — prompt, then
any `<tool_call>`/`<tool_result>` traces, then `<final>`/`<unsupported>`/
`<error>`.

## Training loop

[`src/juniper_math/trainer.py`](../src/juniper_math/trainer.py) implements
a deliberately simple synchronous loop: no mixed precision, no multi-GPU,
no gradient checkpointing, no DataLoader worker processes. AdamW +
cosine-with-warmup schedule, gradient accumulation, gradient-norm clipping.
Every optimizer step's loss and gradients are checked for finiteness
before the step is applied; every parameter is checked for finiteness
after. A non-finite value anywhere raises `TrainingNumericalError` rather
than silently continuing.

**Token accounting** (`tokens_seen`) counts exactly the loss-bearing
(shifted-by-one, non-`-100`) token positions the model's own loss
computation consumes — not raw input length, and it is not reset across a
checkpoint resume.

**Data stream determinism**: each epoch's example order is
`torch.randperm` seeded by `(training seed + epoch number)` only — never by
wall-clock time or process state — so the exact same sequence of batches is
reproduced from a checkpoint's recorded `(epoch, position_in_epoch)`.

## Checkpointing and resume

Uses the Phase 1 checkpoint format (`src/juniper_math/checkpoint.py`)
unchanged: full training state (model, optimizer, scheduler, RNG, step,
tokens_seen, data-stream position, training config, architecture identity,
git commit). `train resume-test` (see below) is the automated gate proving
resume is not merely "doesn't crash" but numerically equivalent to an
uninterrupted run.

Checkpoint binaries are **not** committed to Git (`checkpoints/**/*.pt` is
`.gitignore`d, per `docs/CHECKPOINT_POLICY.md`) — only metadata and this
documentation are. See `reports/PHASE5_RESULTS.md` for the disposition of
the actual smoke checkpoints produced.

## Generation and evaluation

[`src/juniper_math/generation.py`](../src/juniper_math/generation.py)
implements minimal uncached greedy/temperature autoregressive generation —
sufficient to demonstrate that training changed model behavior and to
drive tool-format evaluation. It makes no claim about generation quality.

[`src/juniper_math/tool_format_eval.py`](../src/juniper_math/tool_format_eval.py)
runs generations against the frozen `evals/phase4_tool_use_v2.json` suite
and checks whether output contains a well-formed `<tool_call>{...}` block
per the approved protocol (`juniper_math.tools.protocol.parse_tool_call`).
**This is smoke pipeline validation only** — a smoke model is expected to
score near zero; what must not happen is the evaluator crashing on
malformed generations.

## CLI

```bash
python -m juniper_math train run [--config PATH] [--max-steps N] [--evaluate]
python -m juniper_math train resume-test [--config PATH]
python -m juniper_math evaluate --checkpoint PATH [--config PATH] [--sample-size N]
python -m juniper_math infer --checkpoint PATH --prompt TEXT [--max-new-tokens N]
```

`train run` prints fixed-prompt generations from both the freshly
initialized model and the final trained model side by side, so a human can
see training changed behavior without re-deriving it from logs.

`train resume-test` implements the Sec. 22 gate: it builds one shared
initial checkpoint, trains it straight through to `schedule.total_steps`
(Run A), and separately trains a checkpoint restored from the same init to
`resume_test.interrupt_step`, saves, restores into brand-new model/
optimizer/scheduler objects (simulating a new process), and continues to
the same final step (Run B). It reports step/token/loss/parameter
equivalence and exits non-zero if the comparison fails a tolerance check.

## Scope boundary

Phase 5 does not: retrain the tokenizer, modify the frozen dataset or
evaluation suites, change the architecture, or run a hyperparameter sweep.
`config/training.yaml`'s smoke-scale parameters (2,048 train examples, 200
optimizer steps) are deliberately small — see `reports/PHASE5_RESULTS.md`
for why they are sufficient to validate the pipeline without doing Phase
6's job.

Pilot Pretraining is Phase 6, documented separately in
[`docs/PILOT_TRAINING.md`](PILOT_TRAINING.md) — it reuses this phase's
training loop (`juniper_math.trainer`) unchanged but adds a
category-stratified, packed pilot dataset and multi-suite milestone
evaluation on top of it. Nothing in this file was changed to enable that;
`config/training.yaml`, `training_config.py`, and `smoke_data.py` remain
exactly as Phase 5 approved them.
