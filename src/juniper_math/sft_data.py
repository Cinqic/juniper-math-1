"""Phase 8 SFT-subset selection: a deterministic, category-flattened,
length-safe subset of the frozen `juniper-math-dataset-v1` train/validation
splits, tokenized and masked via `juniper_math.sft_rendering` instead of
`dataset.shard.render_training_text`'s single undifferentiated blob.

Selection method
-----------------
Uses the exact same reviewed selection primitive Phase 5/6 use
(`smoke_data.compute_stride_selection`: fixed-stride sampling over a split's
records in committed on-disk shard order) applied once *per category*
(same structural approach as `pilot_data.select_pilot_examples`), but with
**flattened** category targets — a uniform per-category floor/cap by
availability — instead of `pilot_data`'s corpus-token-proportional targets.
See reports/PHASE8_PLAN.md Sec. 4/14 for why: the frozen corpus's own
category mixture is weighted toward direct arithmetic, so proportional
sampling would under-represent exactly the categories (`tool_error`,
`missing_information`, `incorrect_tool_call`, ...) whose supervision signal
matters most for teaching correct-vs-unnecessary tool use.

Length safety
--------------
Every candidate example is masked-tokenized via
`sft_rendering.tokenize_and_mask` at selection time; any example whose full
BOS+body+EOS length exceeds `max_sequence_length` is *rejected* (counted,
never truncated) — see reports/PHASE8_PLAN.md Sec. 4/12.

Reuses `pilot_data.verify_parent_dataset_shards`/`manifest_shard_files`
unchanged (never a second shard-manifest-verification implementation).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from juniper_math.dataset.config import DatasetConfig, load_dataset_config
from juniper_math.dataset.io import example_from_dict
from juniper_math.dataset.schema import Example
from juniper_math.hashing import sha256_file
from juniper_math.pilot_data import manifest_shard_files, verify_parent_dataset_shards
from juniper_math.sft_rendering import SftRenderingError, tokenize_and_mask
from juniper_math.smoke_data import compute_stride_selection
from juniper_math.tokenizer import JuniperTokenizer

# v2 changes only the Phase-8-derived representation for tool-error cases:
# it adds a supervised response derived from the trusted runtime error rather
# than training EOS directly after a context-only tool result.  The frozen
# Phase 4 parent corpus is unchanged.
SFT_DATASET_ID = "juniper-math-sft-v2"
SFT_MANIFEST_SCHEMA_VERSION = "2.0.0"


class SftDataError(ValueError):
    """Raised for invalid SFT selection, length-rejection, or manifest state."""


@dataclass(frozen=True)
class CategoryCounts:
    record_count: dict[str, int]


def count_categories(dataset_config: DatasetConfig, split: str) -> CategoryCounts:
    record_count: dict[str, int] = {}
    for shard_path in manifest_shard_files(dataset_config, split):
        with shard_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                cat = json.loads(line)["category"]
                record_count[cat] = record_count.get(cat, 0) + 1
    if not record_count:
        raise SftDataError(f"Dataset split {split!r} contains zero records.")
    return CategoryCounts(record_count=record_count)


def compute_flattened_targets(
    counts: CategoryCounts,
    target_per_category: int,
    category_weight_overrides: dict[str, float] | None = None,
) -> dict[str, int]:
    """Uniform per-category target, floored/capped by availability.

    `category_weight_overrides` (e.g. {"tool_error": 1.5}) multiplies a
    single category's target relative to the uniform baseline — used by the
    Sec. 19 preflight's mixture-ablation candidate. Every category still
    gets at least min(target_per_category, available).
    """
    overrides = category_weight_overrides or {}
    targets: dict[str, int] = {}
    for cat, available in counts.record_count.items():
        weight = overrides.get(cat, 1.0)
        wanted = round(target_per_category * weight)
        targets[cat] = max(0, min(wanted, available))
    return targets


@dataclass(frozen=True)
class SelectionOutcome:
    examples: list[Example]
    audit: dict[str, Any]


def select_sft_examples(
    dataset_config: DatasetConfig,
    split: str,
    targets: dict[str, int],
    seed: int,
    tokenizer: JuniperTokenizer,
    max_sequence_length: int,
) -> SelectionOutcome:
    """Pass 1 (counts, via `count_categories`) is the caller's job (needed to
    compute `targets`); this function is the single-scan selection + length
    gate pass. Over-selects slightly per category to absorb length
    rejections without a second scan: takes `ceil(target * oversample)`
    stride-selected candidates, keeps the first `target` that pass the
    length gate, and raises if a category still falls short."""
    stride_by_cat: dict[str, tuple[int, int]] = {}
    counts = count_categories(dataset_config, split)
    oversample_targets: dict[str, int] = {}
    for cat, target in targets.items():
        if target <= 0:
            continue
        available = counts.record_count.get(cat, 0)
        oversampled = min(available, int(target * 1.15) + 5)
        oversample_targets[cat] = oversampled
        stride_by_cat[cat] = compute_stride_selection(available, oversampled, seed)

    running_index: dict[str, int] = {cat: 0 for cat in counts.record_count}
    candidates_by_category: dict[str, list[Example]] = {cat: [] for cat in oversample_targets}
    for shard_path in manifest_shard_files(dataset_config, split):
        with shard_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                cat = d["category"]
                idx = running_index[cat]
                running_index[cat] = idx + 1
                if cat not in stride_by_cat:
                    continue
                stride, offset = stride_by_cat[cat]
                want = oversample_targets[cat]
                if len(candidates_by_category[cat]) >= want:
                    continue
                if (idx - offset) % stride == 0:
                    candidates_by_category[cat].append(example_from_dict(d))

    selected_by_category: dict[str, list[Example]] = {}
    rejected_oversized: dict[str, int] = {}
    for cat, candidates in candidates_by_category.items():
        target = targets[cat]
        kept: list[Example] = []
        n_rejected = 0
        for ex in candidates:
            if len(kept) >= target:
                break
            try:
                tokenize_and_mask(ex, tokenizer, max_sequence_length)
            except SftRenderingError:
                n_rejected += 1
                continue
            kept.append(ex)
        selected_by_category[cat] = kept
        rejected_oversized[cat] = n_rejected

    shortfall = {
        cat: {"target": targets[cat], "selected": len(selected_by_category.get(cat, []))}
        for cat in targets
        if targets[cat] > 0 and len(selected_by_category.get(cat, [])) < targets[cat]
    }
    if shortfall:
        raise SftDataError(
            f"SFT selection under-filled categories for split {split!r} after length rejection "
            f"(oversample factor may be too small): {shortfall}"
        )

    all_selected: list[Example] = []
    for cat in sorted(selected_by_category):
        all_selected.extend(selected_by_category[cat])

    audit = {
        "split": split,
        "seed": seed,
        "max_sequence_length": max_sequence_length,
        "category_record_counts": counts.record_count,
        "category_targets": targets,
        "category_selected_counts": {cat: len(v) for cat, v in selected_by_category.items()},
        "category_rejected_oversized": rejected_oversized,
        "total_selected_examples": len(all_selected),
        "total_rejected_oversized": sum(rejected_oversized.values()),
    }
    return SelectionOutcome(examples=all_selected, audit=audit)


def _read_dataset_identity(dataset_config: DatasetConfig) -> str:
    path = dataset_config.output.dataset_identity_path
    if not path.is_file():
        raise SftDataError(f"Dataset identity file not found at {path}. Run `dataset build` first.")
    return path.read_text(encoding="utf-8").split()[0]


def _ids_sha256(examples: list[Example]) -> str:
    ids = sorted(e.example_id for e in examples)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _difficulty_counts(examples: list[Example]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in examples:
        counts[e.difficulty] = counts.get(e.difficulty, 0) + 1
    return counts


def _behavior_counts(examples: list[Example]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in examples:
        counts[e.expected_behavior] = counts.get(e.expected_behavior, 0) + 1
    return counts


@dataclass(frozen=True)
class SftManifest:
    schema_version: str
    sft_dataset_id: str
    parent_dataset_id: str
    parent_dataset_identity: str
    parent_manifest_sha256: str
    tokenizer_identity: str
    seed: int
    max_sequence_length: int
    splits: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sft_dataset_id": self.sft_dataset_id,
            "parent_dataset_id": self.parent_dataset_id,
            "parent_dataset_identity": self.parent_dataset_identity,
            "parent_manifest_sha256": self.parent_manifest_sha256,
            "tokenizer_identity": self.tokenizer_identity,
            "seed": self.seed,
            "max_sequence_length": self.max_sequence_length,
            "splits": self.splits,
        }

    @property
    def sft_identity(self) -> str:
        """SHA-256 over the sorted example-id hashes of every split — Phase
        8's own frozen SFT-selection identity (analogous to
        DATASET_IDENTITY.sha256 for the parent corpus)."""
        parts = "\n".join(
            f"{split}:{info['example_ids_sha256']}" for split, info in sorted(self.splits.items())
        )
        return hashlib.sha256(parts.encode("utf-8")).hexdigest()


def build_sft_manifest(
    dataset_config: DatasetConfig,
    tokenizer_identity: str,
    seed: int,
    max_sequence_length: int,
    selections: dict[str, list[Example]],
    audits: dict[str, dict[str, Any]],
) -> SftManifest:
    splits: dict[str, dict[str, Any]] = {}
    for split, examples in selections.items():
        splits[split] = {
            "example_count": len(examples),
            "example_ids_sha256": _ids_sha256(examples),
            "category_counts": audits[split]["category_selected_counts"],
            "category_targets": audits[split]["category_targets"],
            "category_rejected_oversized": audits[split]["category_rejected_oversized"],
            "difficulty_counts": _difficulty_counts(examples),
            "behavior_counts": _behavior_counts(examples),
            "tool_required_count": sum(1 for e in examples if e.tool_required),
            "family_count": len({e.family_id for e in examples}),
        }
    return SftManifest(
        schema_version=SFT_MANIFEST_SCHEMA_VERSION,
        sft_dataset_id=SFT_DATASET_ID,
        parent_dataset_id=dataset_config.dataset_id,
        parent_dataset_identity=_read_dataset_identity(dataset_config),
        parent_manifest_sha256=sha256_file(dataset_config.output.manifest_path),
        tokenizer_identity=tokenizer_identity,
        seed=seed,
        max_sequence_length=max_sequence_length,
        splits=splits,
    )


def write_sft_manifest(manifest: SftManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.as_dict()
    payload["sft_identity"] = manifest.sft_identity
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


class MaskedSftDataset(Dataset):
    """Eagerly tokenizes+masks an SFT subset into fixed-length tensors, one
    example per sequence (no packing — see reports/PHASE8_PLAN.md Sec. 4).
    Same `{input_ids, labels, attention_mask}` tensor contract as
    `smoke_data.TokenizedSmokeDataset`/`pilot_data.PackedPilotDataset`, so it
    is a drop-in Dataset for the unmodified `juniper_math.trainer` loop."""

    def __init__(
        self, examples: list[Example], tokenizer: JuniperTokenizer, max_sequence_length: int
    ) -> None:
        if not examples:
            raise SftDataError("MaskedSftDataset requires at least one example.")
        self.max_sequence_length = max_sequence_length
        self.pad_id = tokenizer.token_to_id("<pad>")

        self._input_ids: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._attention_mask: list[torch.Tensor] = []
        self.total_loss_tokens = 0
        self.total_real_tokens = 0
        self.total_padding_tokens = 0

        for ex in examples:
            mt = tokenize_and_mask(ex, tokenizer, max_sequence_length)
            n_real = len(mt.ids)
            pad_needed = max_sequence_length - n_real
            input_ids = mt.ids + [self.pad_id] * pad_needed
            attention_mask = [1] * n_real + [0] * pad_needed
            labels = mt.labels + [-100] * pad_needed

            self._input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            self._labels.append(torch.tensor(labels, dtype=torch.long))
            self._attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))
            self.total_loss_tokens += sum(1 for x in mt.labels[1:] if x != -100)
            self.total_real_tokens += n_real
            self.total_padding_tokens += pad_needed

    def __len__(self) -> int:
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self._input_ids[idx],
            "labels": self._labels[idx],
            "attention_mask": self._attention_mask[idx],
        }

    @property
    def padding_fraction(self) -> float:
        total = self.total_real_tokens + self.total_padding_tokens
        return self.total_padding_tokens / total if total else 0.0


def select_and_record_sft_subset(
    tokenizer_identity: str,
    seed: int,
    train_target_per_category: int,
    validation_target_per_category: int,
    max_sequence_length: int,
    output_dir: Path,
    tokenizer: JuniperTokenizer,
    dataset_config: DatasetConfig | None = None,
    category_weight_overrides: dict[str, float] | None = None,
) -> tuple[dict[str, list[Example]], SftManifest]:
    dataset_config = dataset_config or load_dataset_config()
    verify_parent_dataset_shards(dataset_config)

    train_counts = count_categories(dataset_config, "train")
    train_targets = compute_flattened_targets(
        train_counts, train_target_per_category, category_weight_overrides
    )
    train_outcome = select_sft_examples(
        dataset_config, "train", train_targets, seed, tokenizer, max_sequence_length
    )

    val_counts = count_categories(dataset_config, "validation")
    val_targets = compute_flattened_targets(
        val_counts, validation_target_per_category, category_weight_overrides
    )
    val_outcome = select_sft_examples(
        dataset_config, "validation", val_targets, seed, tokenizer, max_sequence_length
    )

    selections = {"train": train_outcome.examples, "validation": val_outcome.examples}
    audits = {"train": train_outcome.audit, "validation": val_outcome.audit}
    manifest = build_sft_manifest(
        dataset_config, tokenizer_identity, seed, max_sequence_length, selections, audits
    )
    write_sft_manifest(manifest, output_dir / "sft_manifest.json")
    (output_dir / "sft_selection_audit.json").write_text(
        json.dumps(audits, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return selections, manifest


__all__ = [
    "SFT_DATASET_ID",
    "SFT_MANIFEST_SCHEMA_VERSION",
    "CategoryCounts",
    "MaskedSftDataset",
    "SelectionOutcome",
    "SftDataError",
    "SftManifest",
    "build_sft_manifest",
    "compute_flattened_targets",
    "count_categories",
    "select_and_record_sft_subset",
    "select_sft_examples",
    "write_sft_manifest",
]
