"""Phase 5 deterministic smoke-subset selection and tokenized dataset.

Selection method (documented, reproducible, no re-derivation from generator
internals): for a given split, compute
``stride = floor(split_record_count / target_examples)`` and
``offset = seed % stride`` from the split's committed record count in
``shard_manifest.json``. Then scan that split's shard files in their
on-disk sorted order and take every line whose zero-based global index
satisfies ``(index - offset) % stride == 0``, stopping once
``target_examples`` have been collected. This requires only a single
sequential scan (JSON-parsing only the selected lines) and reproduces
byte-identically from (dataset_identity, split record_count, seed,
target_examples) alone.

Tokenization renders each selected example with the same
``render_training_text`` used by the frozen dataset's own token-count pass
(``juniper_math.dataset.shard.render_training_text``), so smoke training
sees exactly the text the frozen dataset declares that example to be.
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
from juniper_math.dataset.schema import Example
from juniper_math.dataset.shard import render_training_text
from juniper_math.tokenizer import JuniperTokenizer
from juniper_math.training_config import TrainingConfig

SMOKE_MANIFEST_SCHEMA_VERSION = "1.0.0"


class SmokeDataError(ValueError):
    """Raised for invalid smoke-subset selection or tokenization state."""


def _split_shard_files(processed_dir: Path, split: str) -> list[Path]:
    files = sorted(processed_dir.glob(f"*.{split}.*.jsonl"))
    if not files:
        raise SmokeDataError(f"No shard files found for split {split!r} under {processed_dir}.")
    return files


def _read_shard_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise SmokeDataError(
            f"Dataset shard manifest not found at {manifest_path}. Run `dataset build` first."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _split_record_count(manifest: dict[str, Any], split: str) -> int:
    total = sum(s["record_count"] for s in manifest["shards"] if s["split"] == split)
    if total == 0:
        raise SmokeDataError(f"Dataset shard manifest records zero examples for split {split!r}.")
    return total


def compute_stride_selection(record_count: int, target_examples: int, seed: int) -> tuple[int, int]:
    if target_examples <= 0:
        raise SmokeDataError(f"target_examples must be positive, got {target_examples}.")
    if target_examples > record_count:
        raise SmokeDataError(
            f"target_examples ({target_examples}) exceeds available records ({record_count})."
        )
    stride = max(1, record_count // target_examples)
    offset = seed % stride
    return stride, offset


def select_smoke_examples(
    dataset_config: DatasetConfig, split: str, target_examples: int, seed: int
) -> list[Example]:
    """Scan `split`'s shards in sorted-filename order and take the fixed-stride sample."""
    manifest = _read_shard_manifest(dataset_config.output.manifest_path)
    record_count = _split_record_count(manifest, split)
    stride, offset = compute_stride_selection(record_count, target_examples, seed)

    from juniper_math.dataset.io import example_from_dict

    selected: list[Example] = []
    global_index = 0
    for shard_path in _split_shard_files(dataset_config.output.processed_path, split):
        with shard_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if (global_index - offset) % stride == 0 and len(selected) < target_examples:
                    selected.append(example_from_dict(json.loads(line)))
                    if len(selected) >= target_examples:
                        return selected
                global_index += 1
    if len(selected) < target_examples:
        raise SmokeDataError(
            f"Selected only {len(selected)} of {target_examples} requested examples for split {split!r} "
            f"(record_count={record_count}, stride={stride}, offset={offset})."
        )
    return selected


def _read_dataset_identity(dataset_config: DatasetConfig) -> str:
    path = dataset_config.output.dataset_identity_path
    if not path.is_file():
        raise SmokeDataError(f"Dataset identity file not found at {path}. Run `dataset build` first.")
    return path.read_text(encoding="utf-8").split()[0]


@dataclass(frozen=True)
class SmokeManifest:
    schema_version: str
    parent_dataset_id: str
    parent_dataset_identity: str
    parent_manifest_sha256: str
    tokenizer_identity: str
    seed: int
    splits: dict[str, dict[str, Any]]  # split -> {target_examples, record_count, stride, offset,
    #                                              example_ids (sorted), example_ids_sha256}

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "parent_dataset_id": self.parent_dataset_id,
            "parent_dataset_identity": self.parent_dataset_identity,
            "parent_manifest_sha256": self.parent_manifest_sha256,
            "tokenizer_identity": self.tokenizer_identity,
            "seed": self.seed,
            "splits": self.splits,
        }


def build_smoke_manifest(
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
    selections: dict[str, list[Example]],
) -> SmokeManifest:
    from juniper_math.hashing import sha256_file

    splits: dict[str, dict[str, Any]] = {}
    for split, examples in selections.items():
        manifest = _read_shard_manifest(dataset_config.output.manifest_path)
        record_count = _split_record_count(manifest, split)
        target = len(examples)
        stride, offset = compute_stride_selection(record_count, target, training_config.seed)
        ids = sorted(e.example_id for e in examples)
        ids_digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        splits[split] = {
            "target_examples": target,
            "parent_record_count": record_count,
            "stride": stride,
            "offset": offset,
            "example_count": len(ids),
            "example_ids_sha256": ids_digest,
            "total_token_count": sum(e.token_count or 0 for e in examples),
        }

    return SmokeManifest(
        schema_version=SMOKE_MANIFEST_SCHEMA_VERSION,
        parent_dataset_id=dataset_config.dataset_id,
        parent_dataset_identity=_read_dataset_identity(dataset_config),
        parent_manifest_sha256=sha256_file(dataset_config.output.manifest_path),
        tokenizer_identity=training_config.tokenizer_identity,
        seed=training_config.seed,
        splits=splits,
    )


def write_smoke_manifest(manifest: SmokeManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def select_and_record_smoke_subset(
    training_config: TrainingConfig, dataset_config: DatasetConfig | None = None
) -> tuple[dict[str, list[Example]], SmokeManifest]:
    dataset_config = dataset_config or load_dataset_config()
    selections = {
        "train": select_smoke_examples(
            dataset_config, "train", training_config.smoke_subset.train_examples, training_config.seed
        ),
        "validation": select_smoke_examples(
            dataset_config,
            "validation",
            training_config.smoke_subset.validation_examples,
            training_config.seed,
        ),
    }
    manifest = build_smoke_manifest(dataset_config, training_config, selections)
    manifest_path = training_config.output.smoke_dataset_path / "smoke_manifest.json"
    write_smoke_manifest(manifest, manifest_path)
    return selections, manifest


class TokenizedSmokeDataset(Dataset):
    """Eagerly tokenizes a small (thousands-of-examples) smoke subset into fixed-length tensors.

    Loss is computed over the entire rendered sequence (prompt + tool
    interactions + final answer), matching the dataset's own
    `render_training_text`; padding positions are masked with label -100
    so they never contribute to the loss or the reported token count.
    """

    def __init__(
        self, examples: list[Example], tokenizer: JuniperTokenizer, max_sequence_length: int
    ) -> None:
        if not examples:
            raise SmokeDataError("TokenizedSmokeDataset requires at least one example.")
        self.max_sequence_length = max_sequence_length
        self.bos_id = tokenizer.token_to_id("<s>")
        self.eos_id = tokenizer.token_to_id("</s>")
        self.pad_id = tokenizer.token_to_id("<pad>")

        self._input_ids: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._attention_mask: list[torch.Tensor] = []
        self.total_loss_tokens = 0

        for ex in examples:
            text = render_training_text(ex)
            body = tokenizer.encode(text)
            budget = max_sequence_length - 2  # room for bos/eos
            if budget < 1:
                raise SmokeDataError(f"max_sequence_length={max_sequence_length} leaves no room for content.")
            body = body[:budget]
            ids = [self.bos_id, *body, self.eos_id]
            n_real = len(ids)
            pad_needed = max_sequence_length - n_real
            input_ids = ids + [self.pad_id] * pad_needed
            attention_mask = [1] * n_real + [0] * pad_needed
            labels = ids + [-100] * pad_needed

            self._input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            self._labels.append(torch.tensor(labels, dtype=torch.long))
            self._attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))
            # Loss is computed on the shifted sequence (T-1 positions); the
            # last real token has no "next token" target of its own, so it
            # never contributes — matches JuniperMathModel.forward's shift.
            self.total_loss_tokens += max(0, n_real - 1)

    def __len__(self) -> int:
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self._input_ids[idx],
            "labels": self._labels[idx],
            "attention_mask": self._attention_mask[idx],
        }


def collate_smoke_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.stack([item[key] for item in items], dim=0)
        for key in ("input_ids", "labels", "attention_mask")
    }


def epoch_order(dataset_size: int, seed: int, epoch: int, shuffle: bool) -> list[int]:
    """Deterministic per-epoch ordering: same (seed, epoch) always yields the same permutation."""
    if not shuffle:
        return list(range(dataset_size))
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
    return torch.randperm(dataset_size, generator=generator).tolist()


__all__ = [
    "SMOKE_MANIFEST_SCHEMA_VERSION",
    "SmokeDataError",
    "SmokeManifest",
    "TokenizedSmokeDataset",
    "build_smoke_manifest",
    "collate_smoke_batch",
    "compute_stride_selection",
    "epoch_order",
    "select_and_record_smoke_subset",
    "select_smoke_examples",
    "write_smoke_manifest",
]
