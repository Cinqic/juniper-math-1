"""Phase 6 deterministic, category-stratified pilot-subset selection and packing.

Selection method
-----------------
Builds on the exact same reviewed, approved selection primitive Phase 5 uses
(`juniper_math.smoke_data.compute_stride_selection`: fixed-stride sampling
over a split's shards in their committed on-disk order, `stride = floor(
population / target)`, `offset = seed % stride`) but applies it once
*per category* instead of once for the whole split. This is the smallest
change that adds category stratification without inventing a second
selection algorithm: each category gets its own deterministic (stride,
offset) pair derived from (that category's exact train-split record count,
that category's target example count, seed), and the same "scan shards in
sorted order, take every index satisfying the stride test" rule Phase 5's
review already covers.

Two full sequential scans of the train split are required and are each
O(n) in the number of train records (no shard is read twice per scan):
  Pass 1 (`_count_categories`) — count exact per-category record counts and
    per-category token totals in the train split (needed because target
    example counts are computed from a token budget, and per-category
    average token length varies).
  Pass 2 (`select_pilot_examples`) — using the per-category (stride, offset)
    computed from pass 1's counts, take the fixed-stride sample per
    category.
A rare category never disappears to proportional rounding: every category's
target is floored at `min_category_examples` (capped at that category's
actual availability), per Sec. 7 of the Phase 6 instructions.

Packing
-------
Frozen corpus examples are short (median 27 tokens, p99 194 — see
data/processed/juniper-math-dataset-v1/stats.json) relative to the
architecture's 1024-token context. Padding every example out to a fixed
sequence length (Phase 5's approach, smoke-scale only) would waste the
large majority of every training step's compute on `<pad>` positions at
pilot scale. `pack_sequences()` performs simple, deterministic, single-pass
first-fit packing: each example is independently rendered/tokenized/
BOS-EOS-wrapped and length-clipped to fit within `max_sequence_length`
exactly as Phase 5's `TokenizedSmokeDataset` does (never split mid-example),
then examples are appended to the current packed sequence in selection
order until the next one would overflow, at which point the current
sequence is closed (right-padded) and a new one starts. Causal attention
already only lets each position see strictly earlier positions, so packing
introduces no non-causal leakage; loss is computed over every real
(non-pad) token exactly like Phase 5, including the boundary between two
packed examples (an EOS token's target is simply the next example's BOS,
which is standard practice for packed causal LM corpora and does not
require a special "reset" mask at pilot scale) — this is the documented,
deliberately simple choice; see reports/PHASE6_RESULTS.md Sec. 10.
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
from juniper_math.dataset.shard import render_training_text
from juniper_math.smoke_data import compute_stride_selection
from juniper_math.tokenizer import JuniperTokenizer

PILOT_MANIFEST_SCHEMA_VERSION = "1.0.0"


class PilotDataError(ValueError):
    """Raised for invalid pilot-subset selection, packing, or tokenization state."""


def _split_shard_files(processed_dir: Path, split: str) -> list[Path]:
    files = sorted(processed_dir.glob(f"*.{split}.*.jsonl"))
    if not files:
        raise PilotDataError(f"No shard files found for split {split!r} under {processed_dir}.")
    return files


def _read_shard_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise PilotDataError(
            f"Dataset shard manifest not found at {manifest_path}. Run `dataset build` first."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class CategoryCounts:
    record_count: dict[str, int]
    token_count: dict[str, int]


def count_categories(dataset_config: DatasetConfig, split: str) -> CategoryCounts:
    """Pass 1: exact per-category record/token counts for `split` (single sequential scan)."""
    record_count: dict[str, int] = {}
    token_count: dict[str, int] = {}
    for shard_path in _split_shard_files(dataset_config.output.processed_path, split):
        with shard_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                cat = d["category"]
                record_count[cat] = record_count.get(cat, 0) + 1
                token_count[cat] = token_count.get(cat, 0) + int(d.get("token_count") or 0)
    if not record_count:
        raise PilotDataError(f"Dataset split {split!r} contains zero records.")
    return CategoryCounts(record_count=record_count, token_count=token_count)


def compute_category_targets(
    counts: CategoryCounts, target_total_tokens: int, min_category_examples: int
) -> dict[str, int]:
    """Per-category example-count targets, proportional to that category's share of the split's
    tokens, with every category floored at `min(min_category_examples, available)` so a rare
    category can never round to zero (Sec. 7)."""
    total_tokens = sum(counts.token_count.values())
    if total_tokens <= 0:
        raise PilotDataError("Split has zero total tokens; cannot compute category targets.")
    targets: dict[str, int] = {}
    for cat, cat_tokens in counts.token_count.items():
        available = counts.record_count[cat]
        avg_tokens = cat_tokens / available if available else 0.0
        proportional_tokens = (cat_tokens / total_tokens) * target_total_tokens
        proportional_examples = round(proportional_tokens / avg_tokens) if avg_tokens > 0 else 0
        floor = min(min_category_examples, available)
        targets[cat] = max(floor, min(proportional_examples, available))
    return targets


def select_pilot_examples(
    dataset_config: DatasetConfig,
    split: str,
    target_total_tokens: int,
    min_category_examples: int,
    seed: int,
) -> tuple[list[Example], dict[str, Any]]:
    """Pass 1 (counts) + Pass 2 (per-category stride selection). Returns (examples, audit_info)."""
    counts = count_categories(dataset_config, split)
    targets = compute_category_targets(counts, target_total_tokens, min_category_examples)

    strides: dict[str, tuple[int, int]] = {}
    for cat, target in targets.items():
        strides[cat] = compute_stride_selection(counts.record_count[cat], target, seed)

    running_index: dict[str, int] = {cat: 0 for cat in counts.record_count}
    selected_by_category: dict[str, list[Example]] = {cat: [] for cat in counts.record_count}
    for shard_path in _split_shard_files(dataset_config.output.processed_path, split):
        with shard_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                cat = d["category"]
                idx = running_index[cat]
                running_index[cat] = idx + 1
                stride, offset = strides[cat]
                target = targets[cat]
                if len(selected_by_category[cat]) >= target:
                    continue
                if (idx - offset) % stride == 0:
                    selected_by_category[cat].append(example_from_dict(d))

    shortfall = {
        cat: {"target": targets[cat], "selected": len(selected_by_category[cat])}
        for cat in targets
        if len(selected_by_category[cat]) < targets[cat]
    }
    if shortfall:
        raise PilotDataError(f"Pilot selection under-filled categories for split {split!r}: {shortfall}")

    all_selected: list[Example] = []
    for cat in sorted(selected_by_category):
        all_selected.extend(selected_by_category[cat])

    audit = {
        "split": split,
        "seed": seed,
        "target_total_tokens": target_total_tokens,
        "min_category_examples": min_category_examples,
        "category_record_counts": counts.record_count,
        "category_token_counts": counts.token_count,
        "category_targets": targets,
        "category_strides": {cat: {"stride": s, "offset": o} for cat, (s, o) in strides.items()},
        "category_selected_counts": {cat: len(v) for cat, v in selected_by_category.items()},
        "total_selected_examples": len(all_selected),
        "total_selected_tokens": sum(e.token_count or 0 for e in all_selected),
    }
    return all_selected, audit


def _read_dataset_identity(dataset_config: DatasetConfig) -> str:
    path = dataset_config.output.dataset_identity_path
    if not path.is_file():
        raise PilotDataError(f"Dataset identity file not found at {path}. Run `dataset build` first.")
    return path.read_text(encoding="utf-8").split()[0]


@dataclass(frozen=True)
class PilotManifest:
    schema_version: str
    parent_dataset_id: str
    parent_dataset_identity: str
    parent_manifest_sha256: str
    tokenizer_identity: str
    seed: int
    max_sequence_length: int
    pack_sequences: bool
    splits: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "parent_dataset_id": self.parent_dataset_id,
            "parent_dataset_identity": self.parent_dataset_identity,
            "parent_manifest_sha256": self.parent_manifest_sha256,
            "tokenizer_identity": self.tokenizer_identity,
            "seed": self.seed,
            "max_sequence_length": self.max_sequence_length,
            "pack_sequences": self.pack_sequences,
            "splits": self.splits,
        }


def _ids_sha256(examples: list[Example]) -> str:
    ids = sorted(e.example_id for e in examples)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def build_pilot_manifest(
    dataset_config: DatasetConfig,
    tokenizer_identity: str,
    seed: int,
    max_sequence_length: int,
    pack_sequences: bool,
    selections: dict[str, list[Example]],
    audits: dict[str, dict[str, Any]],
) -> PilotManifest:
    from juniper_math.hashing import sha256_file

    splits: dict[str, dict[str, Any]] = {}
    for split, examples in selections.items():
        splits[split] = {
            "example_count": len(examples),
            "total_token_count": sum(e.token_count or 0 for e in examples),
            "example_ids_sha256": _ids_sha256(examples),
            "category_counts": audits[split]["category_selected_counts"],
            "category_targets": audits[split]["category_targets"],
            "category_token_counts_source": audits[split]["category_token_counts"],
            "difficulty_counts": _difficulty_counts(examples),
            "tool_required_count": sum(1 for e in examples if e.tool_required),
            "family_count": len({e.family_id for e in examples}),
        }
    return PilotManifest(
        schema_version=PILOT_MANIFEST_SCHEMA_VERSION,
        parent_dataset_id=dataset_config.dataset_id,
        parent_dataset_identity=_read_dataset_identity(dataset_config),
        parent_manifest_sha256=sha256_file(dataset_config.output.manifest_path),
        tokenizer_identity=tokenizer_identity,
        seed=seed,
        max_sequence_length=max_sequence_length,
        pack_sequences=pack_sequences,
        splits=splits,
    )


def _difficulty_counts(examples: list[Example]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in examples:
        counts[e.difficulty] = counts.get(e.difficulty, 0) + 1
    return counts


def write_pilot_manifest(manifest: PilotManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


@dataclass(frozen=True)
class TokenizedExample:
    example: Example
    ids: list[int]  # bos + clipped body + eos


def tokenize_examples(
    examples: list[Example], tokenizer: JuniperTokenizer, max_sequence_length: int
) -> list[TokenizedExample]:
    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    budget = max_sequence_length - 2
    if budget < 1:
        raise PilotDataError(f"max_sequence_length={max_sequence_length} leaves no room for content.")
    out: list[TokenizedExample] = []
    for ex in examples:
        body = tokenizer.encode(render_training_text(ex))[:budget]
        out.append(TokenizedExample(example=ex, ids=[bos_id, *body, eos_id]))
    return out


def pack_sequences(
    tokenized: list[TokenizedExample], max_sequence_length: int
) -> list[list[TokenizedExample]]:
    """Deterministic single-pass first-fit packing in the given (already-deterministic) order.

    Every individual tokenized example already fits within `max_sequence_length`
    (by construction — see `tokenize_examples`), so a fresh bin can always
    accept the next example; no example is ever split across two packed
    sequences.
    """
    bins: list[list[TokenizedExample]] = []
    current: list[TokenizedExample] = []
    current_len = 0
    for item in tokenized:
        n = len(item.ids)
        if current and current_len + n > max_sequence_length:
            bins.append(current)
            current = []
            current_len = 0
        current.append(item)
        current_len += n
    if current:
        bins.append(current)
    return bins


class PackedPilotDataset(Dataset):
    """Eagerly tokenizes and packs a pilot subset into fixed-length tensors.

    Same tensor contract as `juniper_math.smoke_data.TokenizedSmokeDataset`
    (`input_ids`/`labels`/`attention_mask`, labels = -100 on padding) so it
    is a drop-in `Dataset` for the existing `juniper_math.trainer` training
    loop — no trainer changes are needed for Phase 6.
    """

    def __init__(
        self, examples: list[Example], tokenizer: JuniperTokenizer, max_sequence_length: int
    ) -> None:
        if not examples:
            raise PilotDataError("PackedPilotDataset requires at least one example.")
        self.max_sequence_length = max_sequence_length
        self.pad_id = tokenizer.token_to_id("<pad>")

        tokenized = tokenize_examples(examples, tokenizer, max_sequence_length)
        self.bins = pack_sequences(tokenized, max_sequence_length)

        self._input_ids: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._attention_mask: list[torch.Tensor] = []
        self.total_loss_tokens = 0
        self.total_real_tokens = 0
        self.total_padding_tokens = 0

        for bin_items in self.bins:
            ids: list[int] = []
            for item in bin_items:
                ids.extend(item.ids)
            n_real = len(ids)
            pad_needed = max_sequence_length - n_real
            input_ids = ids + [self.pad_id] * pad_needed
            attention_mask = [1] * n_real + [0] * pad_needed
            labels = ids + [-100] * pad_needed

            self._input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            self._labels.append(torch.tensor(labels, dtype=torch.long))
            self._attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))
            self.total_loss_tokens += max(0, n_real - 1)
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


def select_and_record_pilot_subset(
    dataset_id: str,
    tokenizer_identity: str,
    seed: int,
    target_train_tokens: int,
    validation_examples: int,
    min_category_examples: int,
    min_category_examples_validation: int,
    max_sequence_length: int,
    pack_sequences_flag: bool,
    output_dir: Path,
    dataset_config: DatasetConfig | None = None,
) -> tuple[dict[str, list[Example]], PilotManifest]:
    dataset_config = dataset_config or load_dataset_config()
    if dataset_config.dataset_id != dataset_id:
        raise PilotDataError(
            f"pilot config dataset_identity {dataset_id!r} does not match config/dataset.yaml "
            f"dataset_id {dataset_config.dataset_id!r}."
        )
    train_examples, train_audit = select_pilot_examples(
        dataset_config, "train", target_train_tokens, min_category_examples, seed
    )
    # Validation token budget is sized proportionally to validation_examples via the same
    # average-tokens-per-example logic, by passing an explicit token target derived from the
    # average example length observed in the train pass (kept small and fixed; never trained on).
    val_counts = count_categories(dataset_config, "validation")
    val_avg_tokens = sum(val_counts.token_count.values()) / sum(val_counts.record_count.values())
    val_target_tokens = round(val_avg_tokens * validation_examples)
    val_examples, val_audit = select_pilot_examples(
        dataset_config, "validation", val_target_tokens, min_category_examples_validation, seed
    )

    selections = {"train": train_examples, "validation": val_examples}
    audits = {"train": train_audit, "validation": val_audit}
    manifest = build_pilot_manifest(
        dataset_config, tokenizer_identity, seed, max_sequence_length, pack_sequences_flag, selections, audits
    )
    write_pilot_manifest(manifest, output_dir / "pilot_manifest.json")
    (output_dir / "pilot_selection_audit.json").write_text(
        json.dumps(audits, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return selections, manifest


__all__ = [
    "PILOT_MANIFEST_SCHEMA_VERSION",
    "CategoryCounts",
    "PackedPilotDataset",
    "PilotDataError",
    "PilotManifest",
    "TokenizedExample",
    "build_pilot_manifest",
    "compute_category_targets",
    "count_categories",
    "pack_sequences",
    "select_and_record_pilot_subset",
    "select_pilot_examples",
    "tokenize_examples",
    "write_pilot_manifest",
]

# `PackedPilotDataset` is a drop-in `torch.utils.data.Dataset` for
# `juniper_math.trainer`'s existing loop: `trainer._next_micro_batches` uses
# `juniper_math.smoke_data.epoch_order`/`collate_smoke_batch`, both of which
# only depend on `len(dataset)`/`dataset[i]` returning the same
# `{input_ids, labels, attention_mask}` shape this module already produces —
# so this module deliberately does not redefine either.
