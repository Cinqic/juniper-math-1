# training/

Training entry points and training-specific supporting infrastructure.

Phase 5 (Smoke Pretraining) is implemented: `python -m juniper_math train
run` (and `train resume-test`) are real commands, not placeholders. See
[`docs/TRAINING.md`](../docs/TRAINING.md) for the full pipeline and
[`reports/PHASE5_RESULTS.md`](../reports/PHASE5_RESULTS.md) for the actual
smoke run's results.

The training pipeline's real implementation lives under
`src/juniper_math/` (`training_config.py`, `smoke_data.py`, `trainer.py`,
`generation.py`, `tool_format_eval.py`, `train_pipeline.py`) rather than
here — this directory is reserved for training-run artifacts and any
future training-specific scripts that don't belong in the importable
package. No such artifacts are committed here yet (checkpoints and logs
are disposable/gitignored per `docs/CHECKPOINT_POLICY.md`).
