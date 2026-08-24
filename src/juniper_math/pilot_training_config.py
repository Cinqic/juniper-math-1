"""Loader for the Phase 6 pilot-pretraining configuration (config/training_phase6_pilot.yaml).

Deliberately a *separate* config/loader from `juniper_math.training_config`
(Phase 5's smoke config), not a mutation of it — Phase 5's config, loader,
and dataset selector must keep working unchanged after Phase 6 lands (see
docs/TRAINING.md and reports/PHASE5_RESULTS.md, Sec. 6 of the Phase 6
instructions). Where a config section has identical shape to Phase 5's
(optimizer, scheduler, schedule, resume_test, data, output paths), this
module reuses `training_config`'s frozen dataclasses directly rather than
redefining an identical shape a second time. Only the subset-selection
section differs in shape (category-stratified pilot sampling instead of a
flat per-split stride sample), so that section gets its own dataclass.
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

PILOT_TRAINING_CONFIG_PATH = CONFIG_DIR / "training_phase6_pilot.yaml"


def _positive_int(value: Any, name: str, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise JuniperConfigError(f"{name} must be {'non-negative' if allow_zero else 'positive'} integer.")


@dataclass(frozen=True)
class PilotSubsetConfig:
    target_train_tokens: int
    validation_examples: int
    max_sequence_length: int
    min_category_examples: int
    min_category_examples_validation: int
    pack_sequences: bool


@dataclass(frozen=True)
class PilotOutputPaths:
    checkpoint_dir: str
    experiment_dir: str
    pilot_dataset_dir: str

    @property
    def checkpoint_path(self) -> Path:
        return REPO_ROOT / self.checkpoint_dir

    @property
    def experiment_path(self) -> Path:
        return REPO_ROOT / self.experiment_dir

    @property
    def pilot_dataset_path(self) -> Path:
        return REPO_ROOT / self.pilot_dataset_dir


@dataclass(frozen=True)
class PilotTrainingConfig:
    run_id: str
    architecture_identity: str
    tokenizer_identity: str
    dataset_identity: str
    seed: int
    pilot_subset: PilotSubsetConfig
    data: DataConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    schedule: ScheduleConfig
    resume_test: ResumeTestConfig
    device: str
    precision: str
    output: PilotOutputPaths
    fixed_generation_prompts: list[dict[str, str]]
    generation_max_new_tokens: int
    milestone_fractions: list[float]
    raw: dict[str, Any]


def validate_pilot_training_config(config: PilotTrainingConfig) -> None:
    """Reject syntactically valid configurations that cannot be run truthfully.

    Mirrors `training_config.validate_training_config`'s invariants where
    the section shape is shared, and adds pilot-subset-specific checks.
    """
    for name in ("run_id", "architecture_identity", "tokenizer_identity", "dataset_identity"):
        if not isinstance(getattr(config, name), str) or not getattr(config, name).strip():
            raise JuniperConfigError(f"{name} must be a non-empty string.")
    _positive_int(config.seed, "seed", allow_zero=True)

    ps = config.pilot_subset
    for name, value in (
        ("pilot_subset.target_train_tokens", ps.target_train_tokens),
        ("pilot_subset.validation_examples", ps.validation_examples),
        ("pilot_subset.max_sequence_length", ps.max_sequence_length),
        ("data.micro_batch_size", config.data.micro_batch_size),
        ("data.gradient_accumulation_steps", config.data.gradient_accumulation_steps),
        ("schedule.total_steps", config.schedule.total_steps),
        ("generation_max_new_tokens", config.generation_max_new_tokens),
    ):
        _positive_int(value, name)
    for name, value in (
        ("pilot_subset.min_category_examples", ps.min_category_examples),
        ("pilot_subset.min_category_examples_validation", ps.min_category_examples_validation),
        ("schedule.validation_interval", config.schedule.validation_interval),
        ("schedule.checkpoint_interval", config.schedule.checkpoint_interval),
        ("schedule.generation_interval", config.schedule.generation_interval),
        ("scheduler.warmup_steps", config.scheduler.warmup_steps),
    ):
        _positive_int(value, name, allow_zero=True)
    _positive_int(config.schedule.logging_interval, "schedule.logging_interval")
    if not isinstance(ps.pack_sequences, bool):
        raise JuniperConfigError("pilot_subset.pack_sequences must be a boolean.")
    if not (3_000_000 <= ps.target_train_tokens <= 10_000_000):
        raise JuniperConfigError(
            "pilot_subset.target_train_tokens must be within the Phase 6 pilot budget envelope "
            f"[3,000,000, 10,000,000] tokens; got {ps.target_train_tokens}."
        )

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


def load_pilot_training_config(path: Path | None = None) -> PilotTrainingConfig:
    source = path or PILOT_TRAINING_CONFIG_PATH
    if not source.is_file():
        raise JuniperConfigError(f"Pilot training config not found at {source}.")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise JuniperConfigError(f"{source}: invalid YAML ({exc}).") from exc

    try:
        config = PilotTrainingConfig(
            run_id=raw["run_id"],
            architecture_identity=raw["architecture_identity"],
            tokenizer_identity=raw["tokenizer_identity"],
            dataset_identity=raw["dataset_identity"],
            seed=raw["seed"],
            pilot_subset=PilotSubsetConfig(**raw["pilot_subset"]),
            data=DataConfig(**raw["data"]),
            optimizer=OptimizerConfig(**raw["optimizer"]),
            scheduler=SchedulerConfig(**raw["scheduler"]),
            schedule=ScheduleConfig(**raw["schedule"]),
            resume_test=ResumeTestConfig(**raw["resume_test"]),
            device=raw["device"],
            precision=raw["precision"],
            output=PilotOutputPaths(**raw["output"]),
            fixed_generation_prompts=list(raw["fixed_generation_prompts"]),
            generation_max_new_tokens=raw["generation_max_new_tokens"],
            milestone_fractions=list(raw["milestone_fractions"]),
            raw=raw,
        )
        validate_pilot_training_config(config)
        return config
    except (KeyError, TypeError) as exc:
        raise JuniperConfigError(f"{source}: missing or malformed field ({exc}).") from exc


__all__ = [
    "PILOT_TRAINING_CONFIG_PATH",
    "PilotOutputPaths",
    "PilotSubsetConfig",
    "PilotTrainingConfig",
    "load_pilot_training_config",
    "validate_pilot_training_config",
]
