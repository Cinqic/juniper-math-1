# Phase 5 Engineering Report — Smoke Pretraining

**Engineer:** Claude Sonnet 5. **Independent review:** not yet performed
(see `config/project.yaml:phase_5_engineering`). This report documents the
implementation; it is not itself an approval.

## 1. Scope

Prove the complete Juniper Math 1 training pipeline works end to end on
the target hardware (AMD Ryzen 7 5700G / NVIDIA RTX 2060 6GB): frozen
dataset → deterministic tiny subset → tokenization → batching → forward/
backward → AdamW optimization → validation → checkpointing → checkpoint
restoration → resumed training → generation → evaluation → tool-format
evaluation → logging → recovery. Explicitly **not** in scope: capability
training, hyperparameter search, architecture/tokenizer/dataset/tool
protocol changes, or Phase 6 (Pilot Pretraining). See
[`docs/TRAINING.md`](../docs/TRAINING.md) for the full design and
[`reports/PHASE5_RESULTS.md`](PHASE5_RESULTS.md) for the actual run's
numbers.

## 2. New code

| Module | Purpose |
|---|---|
| `config/training.yaml` + `src/juniper_math/training_config.py` | Frozen smoke-training configuration and loader (same house style as `dataset/config.py`) |
| `src/juniper_math/smoke_data.py` | Deterministic fixed-stride smoke-subset selection over the frozen dataset's shards; tokenized, padded, loss-masked `Dataset` |
| `src/juniper_math/trainer.py` | Training loop: optimizer step, gradient accumulation, gradient clipping, finite-value checks, validation, checkpoint save/load, deterministic epoch cursor |
| `src/juniper_math/generation.py` | Minimal uncached greedy/temperature autoregressive generation |
| `src/juniper_math/tool_format_eval.py` | Runs generations against the frozen `evals/phase4_tool_use_v2.json` suite and checks `<tool_call>` well-formedness via the real Phase 3 protocol parser |
| `src/juniper_math/train_pipeline.py` | Orchestration layer the CLI calls into (`run_smoke_train`, `run_resume_test`, `evaluate_tool_use_suite`, `run_infer`) — mirrors `dataset/pipeline.py`'s house style |
| `src/juniper_math/cli.py` | `train run`, `train resume-test`, `evaluate`, `infer` replace the Phase 5 placeholder entries in `_NOT_IMPLEMENTED` |

## 3. Design decisions and why

**Smoke subset selection is fixed-stride, not random sampling or a
generator re-run.** `stride = floor(split_record_count / target_examples)`,
`offset = seed % stride`, selecting every line whose global index satisfies
`(index - offset) % stride == 0`. This reproduces byte-identically from
`(dataset_identity, split record_count, seed, target_examples)` with a
single sequential scan that JSON-parses only the ~2,300 selected lines out
of ~1.5M, rather than requiring a second pass to count examples or
re-deriving anything from the dataset generators (which would risk
silently drifting from the frozen v1 corpus).

**Loss is computed over the entire rendered sequence** (prompt + tool
traces + final answer), matching `render_training_text` — the same text
the frozen dataset's own token-count pass uses. No instruction-tuning-style
prompt masking was introduced; Phase 4's dataset design already treats
this as a single training text per example.

**fp32, no mixed precision.** At 5M parameters the extra complexity of a
`GradScaler` buys nothing for a smoke pipeline check and would have added
a numerical-instability axis unrelated to what Phase 5 is actually testing.

**No KV cache in generation.** Recomputing the full forward pass per step
is fine at `max_new_tokens<=32` and keeps the generation path simple and
easy to verify; a cached path is later-phase optimization work.

**Resume-equivalence gate (Sec. 22) uses three real, independent process
states**, not a shortcut: Run A trains straight through from a saved init
checkpoint; Run B trains a *separate* set of freshly constructed model/
optimizer/scheduler objects from the same init checkpoint to the interrupt
step, saves, and a *third* freshly constructed set of objects (simulating
a new process) restores and finishes. Comparing A against this third
object set is what actually tests restoration, not merely "training
continued after a save call."

**Token accounting** counts exactly the shifted, non-`-100` label
positions the model's own `forward()` uses for its loss — not raw input
length — so it can never silently drift from what the optimizer actually
saw, including across a resume.

## 4. What Phase 5 deliberately does not do

- Does not retrain or modify the tokenizer, dataset, or tool protocol.
- Does not run any hyperparameter search; `config/training.yaml`'s values
  (200 steps, effective batch 16, lr 3e-4) were chosen once for a fast,
  stable smoke run on a 6GB GPU and not tuned further.
- Does not claim the resulting checkpoint has mathematical capability —
  it doesn't (see `reports/PHASE5_RESULTS.md` §Generation).
- Does not begin Phase 6.

## 5. Testing

Six new test files (`test_training_config.py`, `test_smoke_data.py`,
`test_generation.py`, `test_trainer.py`, `test_tool_format_eval.py`,
`test_train_cli.py`) plus updates to `test_cli.py` and `test_metadata.py`.
All build synthetic tiny datasets in `tmp_path` (except `test_train_cli.py`,
which requires and honestly skips without a local `dataset build`, matching
the existing `test_dataset.py` convention) so the suite passes on a fresh
clone without the 1.6M-example corpus. `pytest -v` reports 586 passed;
`ruff check .`, `ruff format --check .`, and `mypy` all pass. See
`reports/PHASE5_RESULTS.md` §Test suite for the exact commands and output.
