"""Loader for the canonical Phase 4 dataset configuration (config/dataset.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from juniper_math.errors import JuniperConfigError
from juniper_math.paths import CONFIG_DIR, REPO_ROOT

DATASET_CONFIG_PATH = CONFIG_DIR / "dataset.yaml"


@dataclass(frozen=True)
class TokenBudget:
    envelope_min: int
    envelope_max: int
    target: int
    max_example_tokens: int


@dataclass(frozen=True)
class DiversityCaps:
    max_template_share_within_family: float
    max_family_share_of_corpus: float


@dataclass(frozen=True)
class SplitConfig:
    train: float
    validation: float
    test: float
    eval_suite_seed_offset: int


@dataclass(frozen=True)
class DedupConfig:
    exact_key: str
    near_duplicate_method: str
    near_duplicate_shingle_size: int
    near_duplicate_jaccard_threshold: float


@dataclass(frozen=True)
class NormalizationConfig:
    unicode_form: str
    collapse_repeated_whitespace: bool
    strip_surrounding_whitespace: bool
    preserve_math_unicode: list[str]
    reject_control_characters: bool


@dataclass(frozen=True)
class ShardConfig:
    format: str
    records_per_shard: int
    filename_pattern: str


@dataclass(frozen=True)
class OutputConfig:
    processed_dir: str
    manifest_file: str
    stats_file: str
    dataset_identity_file: str

    @property
    def processed_path(self) -> Path:
        return REPO_ROOT / self.processed_dir

    @property
    def manifest_path(self) -> Path:
        return REPO_ROOT / self.manifest_file

    @property
    def stats_path(self) -> Path:
        return REPO_ROOT / self.stats_file

    @property
    def dataset_identity_path(self) -> Path:
        return REPO_ROOT / self.dataset_identity_file


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    dataset_schema_version: str
    master_seed: int
    tokenizer_identity: str
    tool_protocol_identity: str
    tool_protocol_version: str
    token_budget: TokenBudget
    category_mixture: dict[str, float]
    diversity_caps: DiversityCaps
    split: SplitConfig
    dedup: DedupConfig
    normalization: NormalizationConfig
    shard: ShardConfig
    output: OutputConfig
    raw: dict[str, Any]


def load_dataset_config(path: Path | None = None) -> DatasetConfig:
    source = path or DATASET_CONFIG_PATH
    if not source.is_file():
        raise JuniperConfigError(f"Dataset config not found at {source}.")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise JuniperConfigError(f"{source}: invalid YAML ({exc}).") from exc

    try:
        mixture = dict(raw["category_mixture"])
        total = sum(mixture.values())
        if abs(total - 1.0) > 1e-6:
            raise JuniperConfigError(f"{source}: category_mixture must sum to 1.0, got {total}.")

        split_raw = raw["split"]
        split_total = split_raw["train"] + split_raw["validation"] + split_raw["test"]
        if abs(split_total - 1.0) > 1e-9:
            raise JuniperConfigError(f"{source}: split proportions must sum to 1.0, got {split_total}.")

        return DatasetConfig(
            dataset_id=raw["dataset_id"],
            dataset_schema_version=raw["dataset_schema_version"],
            master_seed=raw["master_seed"],
            tokenizer_identity=raw["tokenizer_identity"],
            tool_protocol_identity=raw["tool_protocol_identity"],
            tool_protocol_version=raw["tool_protocol_version"],
            token_budget=TokenBudget(**raw["token_budget"]),
            category_mixture=mixture,
            diversity_caps=DiversityCaps(**raw["diversity_caps"]),
            split=SplitConfig(**split_raw),
            dedup=DedupConfig(**raw["dedup"]),
            normalization=NormalizationConfig(**raw["normalization"]),
            shard=ShardConfig(**raw["shard"]),
            output=OutputConfig(**raw["output"]),
            raw=raw,
        )
    except (KeyError, TypeError) as exc:
        raise JuniperConfigError(f"{source}: missing or malformed field ({exc}).") from exc
