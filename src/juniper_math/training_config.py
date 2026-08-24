"""Loader for the canonical Phase 5 smoke-training configuration (config/training.yaml).

Follows the same shape as `juniper_math.dataset.config`: one frozen
dataclass per YAML section, one `load_training_config()` entry point, no
hardcoded duplication of these values anywhere else in the training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from juniper_math.errors import JuniperConfigError
from juniper_math.paths import CONFIG_DIR, REPO_ROOT

TRAINING_CONFIG_PATH = CONFIG_DIR / "training.yaml"


@dataclass(frozen=True)
class SmokeSubsetConfig:
    train_examples: int
    validation_examples: int
    max_sequence_length: int


@dataclass(frozen=True)
class DataConfig:
    micro_batch_size: int
    gradient_accumulation_steps: int
    shuffle: bool

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    eps: float
    grad_clip_norm: float


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    warmup_steps: int
    min_lr_ratio: float
    # Optional, defaults to None for Phase 5/6 configs (which set warmup_steps
    # directly and never populate this). Phase 7 introduces a genuine
    # warmup-ratio field (recommended in reports/PHASE6_RESULTS.md, "Phase 7
    # recommendation") so warmup length scales automatically with a much
    # larger total_steps; its loader (full_training_config) computes
    # warmup_steps = round(warmup_ratio * total_steps) and populates both
    # fields, so this dataclass's shape does not need to change per phase.
    warmup_ratio: float | None = None


@dataclass(frozen=True)
class ScheduleConfig:
    total_steps: int
    validation_interval: int
    checkpoint_interval: int
    generation_interval: int
    logging_interval: int


@dataclass(frozen=True)
class ResumeTestConfig:
    interrupt_step: int


@dataclass(frozen=True)
class OutputPaths:
    checkpoint_dir: str
    experiment_dir: str
    smoke_dataset_dir: str

    @property
    def checkpoint_path(self) -> Path:
        return REPO_ROOT / self.checkpoint_dir

    @property
    def experiment_path(self) -> Path:
        return REPO_ROOT / self.experiment_dir

    @property
    def smoke_dataset_path(self) -> Path:
        return REPO_ROOT / self.smoke_dataset_dir


@dataclass(frozen=True)
class TrainingConfig:
    run_id: str
    architecture_identity: str
    tokenizer_identity: str
    dataset_identity: str
    seed: int
    smoke_subset: SmokeSubsetConfig
    data: DataConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    schedule: ScheduleConfig
    resume_test: ResumeTestConfig
    device: str
    precision: str
    output: OutputPaths
    fixed_generation_prompts: list[str]
    generation_max_new_tokens: int
    raw: dict[str, Any]


def _positive_int(value: Any, name: str, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise JuniperConfigError(f"{name} must be {'non-negative' if allow_zero else 'positive'} integer.")


def validate_training_config(config: TrainingConfig) -> None:
    """Reject syntactically valid configurations that cannot be run truthfully."""
    import math

    for name in ("run_id", "architecture_identity", "tokenizer_identity", "dataset_identity"):
        if not isinstance(getattr(config, name), str) or not getattr(config, name).strip():
            raise JuniperConfigError(f"{name} must be a non-empty string.")
    _positive_int(config.seed, "seed", allow_zero=True)
    for name, value in (
        ("smoke_subset.train_examples", config.smoke_subset.train_examples),
        ("smoke_subset.validation_examples", config.smoke_subset.validation_examples),
        ("smoke_subset.max_sequence_length", config.smoke_subset.max_sequence_length),
        ("data.micro_batch_size", config.data.micro_batch_size),
        ("data.gradient_accumulation_steps", config.data.gradient_accumulation_steps),
        ("schedule.total_steps", config.schedule.total_steps),
        ("generation_max_new_tokens", config.generation_max_new_tokens),
    ):
        _positive_int(value, name)
    for name, value in (
        ("schedule.validation_interval", config.schedule.validation_interval),
        ("schedule.checkpoint_interval", config.schedule.checkpoint_interval),
        ("schedule.generation_interval", config.schedule.generation_interval),
        ("scheduler.warmup_steps", config.scheduler.warmup_steps),
    ):
        _positive_int(value, name, allow_zero=True)
    _positive_int(config.schedule.logging_interval, "schedule.logging_interval")
    if config.optimizer.name != "adamw" or config.scheduler.name != "cosine_with_warmup":
        raise JuniperConfigError("Unsupported optimizer or scheduler.")
    if config.device not in {"cpu", "cuda"} or config.precision != "fp32":
        raise JuniperConfigError("Unsupported device or precision.")
    values = (
        ("optimizer.learning_rate", config.optimizer.learning_rate, True),
        ("optimizer.weight_decay", config.optimizer.weight_decay, False),
        ("optimizer.eps", config.optimizer.eps, True),
        ("optimizer.grad_clip_norm", config.optimizer.grad_clip_norm, True),
    )
    for number_name, number_value, positive in values:
        if (
            isinstance(number_value, bool)
            or not isinstance(number_value, (int, float))
            or not math.isfinite(number_value)
            or (positive and number_value <= 0)
            or (not positive and number_value < 0)
        ):
            raise JuniperConfigError(f"{number_name} has an invalid value.")
    if not 0 <= config.optimizer.beta1 < 1 or not 0 <= config.optimizer.beta2 < 1:
        raise JuniperConfigError("optimizer betas must be in [0, 1).")
    if not 0 <= config.scheduler.min_lr_ratio <= 1:
        raise JuniperConfigError("scheduler.min_lr_ratio must be in [0, 1].")
    if config.scheduler.warmup_steps > config.schedule.total_steps:
        raise JuniperConfigError("scheduler.warmup_steps cannot exceed schedule.total_steps.")
    if not 0 < config.resume_test.interrupt_step < config.schedule.total_steps:
        raise JuniperConfigError("resume_test.interrupt_step must be strictly between zero and total_steps.")
    if not config.fixed_generation_prompts or not all(
        isinstance(p, str) and p.strip() for p in config.fixed_generation_prompts
    ):
        raise JuniperConfigError("fixed_generation_prompts must contain non-empty strings.")


def load_training_config(path: Path | None = None) -> TrainingConfig:
    source = path or TRAINING_CONFIG_PATH
    if not source.is_file():
        raise JuniperConfigError(f"Training config not found at {source}.")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise JuniperConfigError(f"{source}: invalid YAML ({exc}).") from exc

    try:
        config = TrainingConfig(
            run_id=raw["run_id"],
            architecture_identity=raw["architecture_identity"],
            tokenizer_identity=raw["tokenizer_identity"],
            dataset_identity=raw["dataset_identity"],
            seed=raw["seed"],
            smoke_subset=SmokeSubsetConfig(**raw["smoke_subset"]),
            data=DataConfig(**raw["data"]),
            optimizer=OptimizerConfig(**raw["optimizer"]),
            scheduler=SchedulerConfig(**raw["scheduler"]),
            schedule=ScheduleConfig(**raw["schedule"]),
            resume_test=ResumeTestConfig(**raw["resume_test"]),
            device=raw["device"],
            precision=raw["precision"],
            output=OutputPaths(**raw["output"]),
            fixed_generation_prompts=list(raw["fixed_generation_prompts"]),
            generation_max_new_tokens=raw["generation_max_new_tokens"],
            raw=raw,
        )
        validate_training_config(config)
        return config
    except (KeyError, TypeError) as exc:
        raise JuniperConfigError(f"{source}: missing or malformed field ({exc}).") from exc


__all__ = [
    "TRAINING_CONFIG_PATH",
    "DataConfig",
    "OptimizerConfig",
    "OutputPaths",
    "ResumeTestConfig",
    "ScheduleConfig",
    "SchedulerConfig",
    "SmokeSubsetConfig",
    "TrainingConfig",
    "load_training_config",
    "validate_training_config",
]
