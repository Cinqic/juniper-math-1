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


def load_training_config(path: Path | None = None) -> TrainingConfig:
    source = path or TRAINING_CONFIG_PATH
    if not source.is_file():
        raise JuniperConfigError(f"Training config not found at {source}.")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise JuniperConfigError(f"{source}: invalid YAML ({exc}).") from exc

    try:
        return TrainingConfig(
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
]
