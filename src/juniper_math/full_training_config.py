"""Loader for the Phase 7 full base-pretraining configuration
(config/training_phase7_full.yaml).

Same house style as `juniper_math.pilot_training_config`: reuses
`training_config`'s shared section dataclasses unchanged where the shape is
identical, and only defines a new dataclass for the section that differs in
shape (`full_subset` — no stratified-subsampling fields, because Phase 7
trains/validates on the ENTIRE frozen train/validation splits, not a
subsample; see `juniper_math.full_data`).

Introduces a genuine `scheduler.warmup_ratio` field (recommended by
reports/PHASE6_RESULTS.md, "Phase 7 recommendation": "Phase 7 should use a
true warmup-ratio field ... so warmup length scales automatically with a
much larger total_steps rather than needing to be re-derived by hand").
`warmup_steps` is therefore NOT read from the YAML for this config; it is
computed here as `round(warmup_ratio * schedule.total_steps)` and stored on
the same shared `SchedulerConfig` Phase 5/6 use (its `warmup_steps` field is
still what `juniper_math.trainer.build_lr_lambda` actually consumes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from juniper_math.errors import JuniperConfigError
from juniper_math.paths import CONFIG_DIR, REPO_ROOT
from juniper_math.training_config import (
    DataConfig,
    OptimizerConfig,
    ResumeTestConfig,
    ScheduleConfig,
    SchedulerConfig,
)

FULL_TRAINING_CONFIG_PATH = CONFIG_DIR / "training_phase7_full.yaml"


def _positive_int(value: Any, name: str, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise JuniperConfigError(f"{name} must be {'non-negative' if allow_zero else 'positive'} integer.")


@dataclass(frozen=True)
class FullSubsetConfig:
    max_sequence_length: int
    pack_sequences: bool


@dataclass(frozen=True)
class FullOutputPaths:
    checkpoint_dir: str
    experiment_dir: str
    full_dataset_dir: str

    @property
    def checkpoint_path(self) -> Path:
        return REPO_ROOT / self.checkpoint_dir

    @property
    def experiment_path(self) -> Path:
        return REPO_ROOT / self.experiment_dir

    @property
    def full_dataset_path(self) -> Path:
        return REPO_ROOT / self.full_dataset_dir


@dataclass(frozen=True)
class FullTrainingConfig:
    run_id: str
    architecture_identity: str
    tokenizer_identity: str
    dataset_identity: str
    seed: int
    full_subset: FullSubsetConfig
    data: DataConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    schedule: ScheduleConfig
    resume_test: ResumeTestConfig
    device: str
    precision: str
    output: FullOutputPaths
    fixed_generation_prompts: list[dict[str, str]]
    generation_max_new_tokens: int
    milestone_fractions: list[float]
    raw: dict[str, Any]


def validate_full_training_config(config: FullTrainingConfig) -> None:
    """Mirrors `pilot_training_config.validate_pilot_training_config`'s shared invariants;
    drops the pilot-only [3M, 10M] target_train_tokens envelope check (no such field here) and
    adds the warmup_ratio consistency check."""
    for name in ("run_id", "architecture_identity", "tokenizer_identity", "dataset_identity"):
        if not isinstance(getattr(config, name), str) or not getattr(config, name).strip():
            raise JuniperConfigError(f"{name} must be a non-empty string.")
    _positive_int(config.seed, "seed", allow_zero=True)

    fs = config.full_subset
    _positive_int(fs.max_sequence_length, "full_subset.max_sequence_length")
    if not isinstance(fs.pack_sequences, bool):
        raise JuniperConfigError("full_subset.pack_sequences must be a boolean.")

    for name, value in (
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
    if config.scheduler.warmup_ratio is None or not (0 <= config.scheduler.warmup_ratio <= 1):
        raise JuniperConfigError("scheduler.warmup_ratio must be set and within [0, 1] for Phase 7.")
    expected_warmup_steps = round(config.scheduler.warmup_ratio * config.schedule.total_steps)
    if config.scheduler.warmup_steps != expected_warmup_steps:
        raise JuniperConfigError(
            "scheduler.warmup_steps does not match round(warmup_ratio * schedule.total_steps) "
            f"({config.scheduler.warmup_steps} != {expected_warmup_steps})."
        )
    if config.scheduler.warmup_steps > config.schedule.total_steps:
        raise JuniperConfigError("scheduler.warmup_steps cannot exceed schedule.total_steps.")
    if not 0 < config.resume_test.interrupt_step < config.schedule.total_steps:
        raise JuniperConfigError("resume_test.interrupt_step must be strictly between zero and total_steps.")
    if not config.fixed_generation_prompts or not all(
        isinstance(p, dict) and isinstance(p.get("prompt"), str) and p["prompt"].strip()
        for p in config.fixed_generation_prompts
    ):
        raise JuniperConfigError("fixed_generation_prompts must be a non-empty list of {category, prompt}.")
    if not config.milestone_fractions or list(config.milestone_fractions) != sorted(
        config.milestone_fractions
    ):
        raise JuniperConfigError("milestone_fractions must be a non-empty, non-decreasing list.")
    if config.milestone_fractions[0] != 0.0 or config.milestone_fractions[-1] != 1.0:
        raise JuniperConfigError("milestone_fractions must start at 0.0 and end at 1.0.")
    if any(not (0.0 <= f <= 1.0) for f in config.milestone_fractions):
        raise JuniperConfigError("milestone_fractions entries must be within [0.0, 1.0].")


def load_full_training_config(path: Path | None = None) -> FullTrainingConfig:
    source = path or FULL_TRAINING_CONFIG_PATH
    if not source.is_file():
        raise JuniperConfigError(f"Full training config not found at {source}.")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise JuniperConfigError(f"{source}: invalid YAML ({exc}).") from exc

    try:
        schedule = ScheduleConfig(**raw["schedule"])
        scheduler_raw = dict(raw["scheduler"])
        warmup_ratio = scheduler_raw.pop("warmup_ratio", None)
        if warmup_ratio is None:
            raise JuniperConfigError(f"{source}: scheduler.warmup_ratio is required for Phase 7.")
        scheduler_raw["warmup_steps"] = round(warmup_ratio * schedule.total_steps)
        scheduler_raw["warmup_ratio"] = warmup_ratio
        scheduler = SchedulerConfig(**scheduler_raw)

        config = FullTrainingConfig(
            run_id=raw["run_id"],
            architecture_identity=raw["architecture_identity"],
            tokenizer_identity=raw["tokenizer_identity"],
            dataset_identity=raw["dataset_identity"],
            seed=raw["seed"],
            full_subset=FullSubsetConfig(**raw["full_subset"]),
            data=DataConfig(**raw["data"]),
            optimizer=OptimizerConfig(**raw["optimizer"]),
            scheduler=scheduler,
            schedule=schedule,
            resume_test=ResumeTestConfig(**raw["resume_test"]),
            device=raw["device"],
            precision=raw["precision"],
            output=FullOutputPaths(**raw["output"]),
            fixed_generation_prompts=list(raw["fixed_generation_prompts"]),
            generation_max_new_tokens=raw["generation_max_new_tokens"],
            milestone_fractions=list(raw["milestone_fractions"]),
            raw=raw,
        )
        validate_full_training_config(config)
        return config
    except (KeyError, TypeError) as exc:
        raise JuniperConfigError(f"{source}: missing or malformed field ({exc}).") from exc


__all__ = [
    "FULL_TRAINING_CONFIG_PATH",
    "FullOutputPaths",
    "FullSubsetConfig",
    "FullTrainingConfig",
    "load_full_training_config",
    "validate_full_training_config",
]
