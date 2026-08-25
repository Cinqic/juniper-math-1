"""Loader for the Phase 8 SFT training configuration
(config/training_phase8_sft.yaml).

Same house style as `full_training_config.py`: reuses `training_config`'s
shared section dataclasses unchanged, defines a new dataclass only for the
section that differs in shape (`sft_subset` — category-flattened selection
targets + category weight overrides, instead of Phase 7's "train on
everything" full_subset or Phase 6's token-budget pilot_subset).

Adds the Phase 8-specific identity fields Sec. 17/27 require: the parent
checkpoint path/SHA-256/tag being fine-tuned from (checked at load time,
fail loudly on mismatch — same idiom as `full_pipeline._load_common`'s
architecture/dataset checks) and the tool-protocol identity being targeted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from juniper_math.errors import JuniperConfigError
from juniper_math.hashing import sha256_file
from juniper_math.paths import CONFIG_DIR, REPO_ROOT
from juniper_math.training_config import (
    DataConfig,
    OptimizerConfig,
    ResumeTestConfig,
    ScheduleConfig,
    SchedulerConfig,
)

SFT_TRAINING_CONFIG_PATH = CONFIG_DIR / "training_phase8_sft.yaml"


def _positive_int(value: Any, name: str, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise JuniperConfigError(f"{name} must be {'non-negative' if allow_zero else 'positive'} integer.")


@dataclass(frozen=True)
class SftSubsetConfig:
    train_target_per_category: int
    validation_target_per_category: int
    max_sequence_length: int
    category_weight_overrides: dict[str, float] = field(default_factory=dict)
    direct_prompt_variants: int = 0
    independent_direct_examples_per_category: int = 0
    independent_safety_examples_per_category: int = 0
    base_replay_examples: int = 0


@dataclass(frozen=True)
class SftOutputPaths:
    checkpoint_dir: str
    experiment_dir: str
    sft_dataset_dir: str

    @property
    def checkpoint_path(self) -> Path:
        return REPO_ROOT / self.checkpoint_dir

    @property
    def experiment_path(self) -> Path:
        return REPO_ROOT / self.experiment_dir

    @property
    def sft_dataset_path(self) -> Path:
        return REPO_ROOT / self.sft_dataset_dir


@dataclass(frozen=True)
class SftTrainingConfig:
    run_id: str
    architecture_identity: str
    tokenizer_identity: str
    dataset_identity: str
    tool_protocol_identity: str
    parent_checkpoint_path: str
    parent_checkpoint_sha256: str
    parent_phase7_tag: str
    seed: int
    sft_subset: SftSubsetConfig
    data: DataConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    schedule: ScheduleConfig
    resume_test: ResumeTestConfig
    device: str
    precision: str
    output: SftOutputPaths
    fixed_generation_prompts: list[dict[str, str]]
    generation_max_new_tokens: int
    base_regression_validation_examples: int
    milestone_fractions: list[float]
    raw: dict[str, Any]


def verify_parent_checkpoint(config: SftTrainingConfig) -> None:
    """Fail loudly if the Base checkpoint on disk doesn't match the frozen
    lineage this config declares (Sec. 4/27's checkpoint-lineage requirement,
    same fail-closed spirit as `pilot_data.verify_parent_dataset_shards`)."""
    path = REPO_ROOT / config.parent_checkpoint_path
    if not path.is_file():
        raise JuniperConfigError(
            f"Parent checkpoint not found at {path}. Retrieve it from the "
            f"{config.parent_phase7_tag!r} GitHub release before running Phase 8 training."
        )
    actual = sha256_file(path)
    if actual != config.parent_checkpoint_sha256:
        raise JuniperConfigError(
            f"Parent checkpoint at {path} has SHA-256 {actual!r}, expected "
            f"{config.parent_checkpoint_sha256!r}. Refusing to fine-tune from an unverified Base."
        )


def validate_sft_training_config(config: SftTrainingConfig) -> None:
    for name in (
        "run_id",
        "architecture_identity",
        "tokenizer_identity",
        "dataset_identity",
        "tool_protocol_identity",
        "parent_checkpoint_path",
        "parent_checkpoint_sha256",
        "parent_phase7_tag",
    ):
        if not isinstance(getattr(config, name), str) or not getattr(config, name).strip():
            raise JuniperConfigError(f"{name} must be a non-empty string.")
    if len(config.parent_checkpoint_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in config.parent_checkpoint_sha256.lower()
    ):
        raise JuniperConfigError("parent_checkpoint_sha256 must be a 64-character hex SHA-256 digest.")
    _positive_int(config.seed, "seed", allow_zero=True)

    ss = config.sft_subset
    _positive_int(ss.train_target_per_category, "sft_subset.train_target_per_category")
    _positive_int(ss.validation_target_per_category, "sft_subset.validation_target_per_category")
    _positive_int(ss.max_sequence_length, "sft_subset.max_sequence_length")
    _positive_int(ss.direct_prompt_variants, "sft_subset.direct_prompt_variants", allow_zero=True)
    _positive_int(
        ss.independent_direct_examples_per_category,
        "sft_subset.independent_direct_examples_per_category",
        allow_zero=True,
    )
    _positive_int(
        ss.independent_safety_examples_per_category,
        "sft_subset.independent_safety_examples_per_category",
        allow_zero=True,
    )
    _positive_int(ss.base_replay_examples, "sft_subset.base_replay_examples", allow_zero=True)
    if not isinstance(ss.category_weight_overrides, dict) or not all(
        isinstance(k, str) and isinstance(v, (int, float)) and v > 0
        for k, v in ss.category_weight_overrides.items()
    ):
        raise JuniperConfigError(
            "sft_subset.category_weight_overrides must map category name -> positive number."
        )

    for name, value in (
        ("data.micro_batch_size", config.data.micro_batch_size),
        ("data.gradient_accumulation_steps", config.data.gradient_accumulation_steps),
        ("schedule.total_steps", config.schedule.total_steps),
        ("generation_max_new_tokens", config.generation_max_new_tokens),
        ("base_regression_validation_examples", config.base_regression_validation_examples),
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
        raise JuniperConfigError("scheduler.warmup_ratio must be set and within [0, 1] for Phase 8.")
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


def load_sft_training_config(path: Path | None = None) -> SftTrainingConfig:
    source = path or SFT_TRAINING_CONFIG_PATH
    if not source.is_file():
        raise JuniperConfigError(f"SFT training config not found at {source}.")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise JuniperConfigError(f"{source}: invalid YAML ({exc}).") from exc

    try:
        schedule = ScheduleConfig(**raw["schedule"])
        scheduler_raw = dict(raw["scheduler"])
        warmup_ratio = scheduler_raw.pop("warmup_ratio", None)
        if warmup_ratio is None:
            raise JuniperConfigError(f"{source}: scheduler.warmup_ratio is required for Phase 8.")
        scheduler_raw["warmup_steps"] = round(warmup_ratio * schedule.total_steps)
        scheduler_raw["warmup_ratio"] = warmup_ratio
        scheduler = SchedulerConfig(**scheduler_raw)

        config = SftTrainingConfig(
            run_id=raw["run_id"],
            architecture_identity=raw["architecture_identity"],
            tokenizer_identity=raw["tokenizer_identity"],
            dataset_identity=raw["dataset_identity"],
            tool_protocol_identity=raw["tool_protocol_identity"],
            parent_checkpoint_path=raw["parent_checkpoint_path"],
            parent_checkpoint_sha256=raw["parent_checkpoint_sha256"],
            parent_phase7_tag=raw["parent_phase7_tag"],
            seed=raw["seed"],
            sft_subset=SftSubsetConfig(**raw["sft_subset"]),
            data=DataConfig(**raw["data"]),
            optimizer=OptimizerConfig(**raw["optimizer"]),
            scheduler=scheduler,
            schedule=schedule,
            resume_test=ResumeTestConfig(**raw["resume_test"]),
            device=raw["device"],
            precision=raw["precision"],
            output=SftOutputPaths(**raw["output"]),
            fixed_generation_prompts=list(raw["fixed_generation_prompts"]),
            generation_max_new_tokens=raw["generation_max_new_tokens"],
            base_regression_validation_examples=raw["base_regression_validation_examples"],
            milestone_fractions=list(raw["milestone_fractions"]),
            raw=raw,
        )
        validate_sft_training_config(config)
        return config
    except (KeyError, TypeError) as exc:
        raise JuniperConfigError(f"{source}: missing or malformed field ({exc}).") from exc


__all__ = [
    "SFT_TRAINING_CONFIG_PATH",
    "SftOutputPaths",
    "SftSubsetConfig",
    "SftTrainingConfig",
    "load_sft_training_config",
    "validate_sft_training_config",
    "verify_parent_checkpoint",
]
