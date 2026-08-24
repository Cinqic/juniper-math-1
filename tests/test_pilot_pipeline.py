"""Phase 6 pilot orchestration tests.

Runs entirely on CPU with a tiny synthetic dataset (never the real
1.6M-example dataset build), mirroring tests/test_trainer.py's approach.
Verifies the trainer's Protocol-typed loop (juniper_math.trainer) actually
accepts PilotTrainingConfig/PackedPilotDataset at runtime, not just that it
type-checks, and that Phase 6's category-specific validation loss logic is
correct.
"""

from __future__ import annotations

import pytest
import torch

from juniper_math.architecture import load_architecture_config
from juniper_math.dataset.schema import Example
from juniper_math.pilot_data import PackedPilotDataset
from juniper_math.pilot_pipeline import PilotDatasets, compute_validation_metrics
from juniper_math.pilot_training_config import (
    PilotOutputPaths,
    PilotSubsetConfig,
    PilotTrainingConfig,
)
from juniper_math.seed import set_global_seed
from juniper_math.smoke_data import TokenizedSmokeDataset
from juniper_math.tokenizer import JuniperTokenizer
from juniper_math.trainer import init_state, train_one_optimizer_step
from juniper_math.training_config import (
    DataConfig,
    OptimizerConfig,
    ResumeTestConfig,
    ScheduleConfig,
    SchedulerConfig,
)


def _example(i: int, category: str) -> Example:
    return Example(
        example_id=f"ex{category}{i:04d}",
        generator_id="test",
        generator_version="1.0.0",
        family_id="f",
        template_id="t0",
        derivation_id=f"d{i}",
        seed=i,
        category=category,
        difficulty="easy",
        synthetic=True,
        split="train",
        prompt=f"What is {i} + 1?",
        expected_behavior="answer",
        expected_answer=str(i + 1),
        tolerance=0,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "deterministic", "expression": {"op": "add", "args": [i, 1]}},
        provenance="test",
        notes="",
        token_count=8,
    )


def _make_pilot_config(tmp_path, total_steps=6, seed=777) -> PilotTrainingConfig:
    raw = {"placeholder": True}
    return PilotTrainingConfig(
        run_id="test-pilot-run",
        architecture_identity=load_architecture_config().architecture_version,
        tokenizer_identity="juniper-math-tokenizer-v1",
        dataset_identity="test-dataset-v1",
        seed=seed,
        pilot_subset=PilotSubsetConfig(
            target_train_tokens=5_000_000,
            validation_examples=8,
            max_sequence_length=32,
            min_category_examples=1,
            min_category_examples_validation=1,
            pack_sequences=True,
        ),
        data=DataConfig(micro_batch_size=2, gradient_accumulation_steps=1, shuffle=True),
        optimizer=OptimizerConfig(
            name="adamw",
            learning_rate=3e-3,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            grad_clip_norm=1.0,
        ),
        scheduler=SchedulerConfig(name="cosine_with_warmup", warmup_steps=1, min_lr_ratio=0.1),
        schedule=ScheduleConfig(
            total_steps=total_steps,
            validation_interval=0,
            checkpoint_interval=0,
            generation_interval=0,
            logging_interval=1,
        ),
        resume_test=ResumeTestConfig(interrupt_step=max(1, total_steps // 2)),
        device="cpu",
        precision="fp32",
        output=PilotOutputPaths(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            experiment_dir=str(tmp_path / "experiments"),
            pilot_dataset_dir=str(tmp_path / "data"),
        ),
        fixed_generation_prompts=[{"category": "arithmetic", "prompt": "1 + 1 ="}],
        generation_max_new_tokens=4,
        milestone_fractions=[0.0, 0.5, 1.0],
        raw=raw,
    )


@pytest.fixture(scope="module")
def tokenizer():
    return JuniperTokenizer.load()


def test_pilot_config_is_a_trainer_config_like(tmp_path, tokenizer):
    """The whole point of the trainer.py Protocol refactor: this must run without a second loop."""
    cfg = _make_pilot_config(tmp_path)
    architecture = load_architecture_config()
    train_examples = [_example(i, "arithmetic") for i in range(6)]
    train_ds = PackedPilotDataset(train_examples, tokenizer, max_sequence_length=32)

    set_global_seed(cfg.seed, deterministic_algorithms=False)
    state = init_state(architecture, cfg, torch.device("cpu"))
    metrics = train_one_optimizer_step(state, train_ds, cfg)
    assert state.step == 1
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_compute_validation_metrics_breaks_out_by_category(tmp_path, tokenizer):
    cfg = _make_pilot_config(tmp_path)
    architecture = load_architecture_config()
    set_global_seed(cfg.seed, deterministic_algorithms=False)
    state = init_state(architecture, cfg, torch.device("cpu"))

    val_examples = [_example(i, "arithmetic") for i in range(4)] + [
        _example(i, "word_problem") for i in range(4, 8)
    ]
    val_ds = TokenizedSmokeDataset(val_examples, tokenizer, max_sequence_length=32)
    datasets = PilotDatasets(
        train=PackedPilotDataset([_example(0, "arithmetic")], tokenizer, 32),
        validation=val_ds,
        validation_examples=val_examples,
        manifest=None,
    )

    overall, category_losses = compute_validation_metrics(state, datasets, batch_size=2)
    assert torch.isfinite(torch.tensor(overall["validation_loss"]))
    assert set(category_losses) == {"arithmetic", "word_problem"}
    for loss in category_losses.values():
        assert torch.isfinite(torch.tensor(loss))
