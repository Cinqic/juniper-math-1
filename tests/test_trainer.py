"""Phase 5 trainer mechanics tests: loss computation, optimizer step, checkpoint save/resume.

Runs entirely on CPU with a tiny synthetic dataset so it is fast and does
not require the real 1.6M-example dataset build.
"""

from __future__ import annotations

import pytest
import torch

from juniper_math.architecture import load_architecture_config
from juniper_math.dataset.schema import Example
from juniper_math.seed import set_global_seed
from juniper_math.smoke_data import TokenizedSmokeDataset
from juniper_math.tokenizer import JuniperTokenizer
from juniper_math.trainer import (
    TrainingNumericalError,
    assert_model_finite,
    data_stream_position,
    init_state,
    load_state,
    loss_bearing_tokens,
    run_training,
    save_state,
    train_one_optimizer_step,
    validate,
)
from juniper_math.training_config import (
    DataConfig,
    OptimizerConfig,
    OutputPaths,
    ResumeTestConfig,
    ScheduleConfig,
    SchedulerConfig,
    SmokeSubsetConfig,
    TrainingConfig,
)


def _example(i: int) -> Example:
    return Example(
        example_id=f"ex{i:04d}",
        generator_id="test",
        generator_version="1.0.0",
        family_id="f",
        template_id="t0",
        derivation_id=f"d{i}",
        seed=i,
        category="arithmetic",
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


def _make_training_config(tmp_path, total_steps=6, checkpoint_interval=3, seed=777) -> TrainingConfig:
    raw = {"placeholder": True}
    return TrainingConfig(
        run_id="test-run",
        architecture_identity=load_architecture_config().architecture_version,
        tokenizer_identity="juniper-math-tokenizer-v1",
        dataset_identity="test-dataset-v1",
        seed=seed,
        smoke_subset=SmokeSubsetConfig(train_examples=8, validation_examples=4, max_sequence_length=32),
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
            checkpoint_interval=checkpoint_interval,
            generation_interval=0,
            logging_interval=1,
        ),
        resume_test=ResumeTestConfig(interrupt_step=max(1, total_steps // 2)),
        device="cpu",
        precision="fp32",
        output=OutputPaths(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            experiment_dir=str(tmp_path / "experiments"),
            smoke_dataset_dir=str(tmp_path / "data"),
        ),
        fixed_generation_prompts=["1 + 1 ="],
        generation_max_new_tokens=4,
        raw=raw,
    )


@pytest.fixture(scope="module")
def tokenizer():
    return JuniperTokenizer.load()


@pytest.fixture
def train_dataset(tokenizer):
    return TokenizedSmokeDataset([_example(i) for i in range(8)], tokenizer, max_sequence_length=32)


@pytest.fixture
def val_dataset(tokenizer):
    return TokenizedSmokeDataset([_example(i) for i in range(8, 12)], tokenizer, max_sequence_length=32)


def test_loss_bearing_tokens_matches_shift_semantics():
    labels = torch.tensor([[1, 2, 3, -100], [1, -100, -100, -100]])
    # shift-by-one: labels[:, 1:] -> [[2,3,-100],[-100,-100,-100]] -> 2 real tokens
    assert loss_bearing_tokens(labels) == 2


def test_train_one_optimizer_step_advances_state_and_produces_finite_loss(tmp_path, train_dataset):
    set_global_seed(1, deterministic_algorithms=False)
    architecture = load_architecture_config()
    cfg = _make_training_config(tmp_path)
    state = init_state(architecture, cfg, torch.device("cpu"))

    metrics = train_one_optimizer_step(state, train_dataset, cfg)
    assert state.step == 1
    assert state.tokens_seen > 0
    assert metrics["loss"] == metrics["loss"]  # not NaN
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_training_reduces_loss_on_tiny_fixed_dataset(tmp_path, train_dataset):
    set_global_seed(2, deterministic_algorithms=False)
    architecture = load_architecture_config()
    cfg = _make_training_config(tmp_path, total_steps=40, checkpoint_interval=0)
    state = init_state(architecture, cfg, torch.device("cpu"))

    first_loss = train_one_optimizer_step(state, train_dataset, cfg)["loss"]
    for _ in range(39):
        last_loss = train_one_optimizer_step(state, train_dataset, cfg)["loss"]

    assert last_loss < first_loss
    assert_model_finite(state.model)


def test_assert_model_finite_detects_injected_nan(tmp_path):
    architecture = load_architecture_config()
    cfg = _make_training_config(tmp_path)
    state = init_state(architecture, cfg, torch.device("cpu"))
    assert_model_finite(state.model)  # sane by default

    with torch.no_grad():
        first_param = next(state.model.parameters())
        first_param[0] = float("nan")
    with pytest.raises(TrainingNumericalError):
        assert_model_finite(state.model)


def test_checkpoint_save_and_resume_reproduces_continuation(tmp_path, train_dataset, val_dataset):
    architecture = load_architecture_config()
    cfg = _make_training_config(tmp_path, total_steps=8, checkpoint_interval=4)
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    log_path = tmp_path / "log.jsonl"

    set_global_seed(cfg.seed, deterministic_algorithms=True)
    state_a = init_state(architecture, cfg, torch.device("cpu"))
    run_training(state_a, train_dataset, val_dataset, architecture, cfg, 8, ckpt_dir, log_path, "test-commit")

    set_global_seed(cfg.seed, deterministic_algorithms=True)
    state_b1 = init_state(architecture, cfg, torch.device("cpu"))
    run_training(
        state_b1, train_dataset, val_dataset, architecture, cfg, 4, ckpt_dir, log_path, "test-commit"
    )
    interrupt_path = ckpt_dir / "interrupt.pt"
    save_state(state_b1, architecture, cfg, interrupt_path, "test-commit")

    set_global_seed(999, deterministic_algorithms=True)  # different seed: proves load_state overrides it
    state_b2 = init_state(architecture, cfg, torch.device("cpu"))
    load_state(state_b2, architecture, interrupt_path)
    assert data_stream_position(state_b2) == data_stream_position(state_b1)
    assert state_b2.step == 4
    run_training(
        state_b2, train_dataset, val_dataset, architecture, cfg, 8, ckpt_dir, log_path, "test-commit"
    )

    assert state_a.step == state_b2.step == 8
    assert state_a.tokens_seen == state_b2.tokens_seen
    for p_a, p_b in zip(state_a.model.parameters(), state_b2.model.parameters(), strict=True):
        assert torch.allclose(p_a, p_b, atol=1e-5), "resumed run diverged from uninterrupted run"


def test_validate_returns_finite_average_loss(tmp_path, train_dataset, val_dataset):
    architecture = load_architecture_config()
    cfg = _make_training_config(tmp_path)
    state = init_state(architecture, cfg, torch.device("cpu"))
    metrics = validate(state, val_dataset, batch_size=2)
    assert torch.isfinite(torch.tensor(metrics["validation_loss"]))
    assert metrics["validation_tokens"] > 0
