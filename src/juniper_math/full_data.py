"""Phase 7 full-dataset selection: loads the ENTIRE frozen train/validation
splits (no stratified subsampling — Phase 6's `pilot_data` intentionally
subsamples to a small, fast pilot budget; Phase 7 is full base pretraining
and trains/validates on every frozen example in each split).

Reuses `pilot_data.verify_parent_dataset_shards` (the fail-closed local-shard
hash check added during Phase 6 remediation) and `pilot_data.tokenize_examples`
/ `pack_sequences` / `PackedPilotDataset` unchanged — packing an entire split
is mechanically the same operation as packing a pilot subset, just over more
examples; a second packing implementation would be pure duplication.
Validation reuses `smoke_data.TokenizedSmokeDataset` unchanged, exactly as
Phase 6 does, for the same reason (Sec. "Validation intentionally does NOT
reuse the packed training dataset" in `pilot_pipeline`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from juniper_math.dataset.config import DatasetConfig, load_dataset_config
from juniper_math.dataset.io import example_from_dict
from juniper_math.dataset.schema import Example
from juniper_math.pilot_data import PilotDataError, verify_parent_dataset_shards

FULL_MANIFEST_SCHEMA_VERSION = "1.0.0"


def _split_shard_files(processed_dir: Path, split: str) -> list[Path]:
    files = sorted(processed_dir.glob(f"*.{split}.*.jsonl"))
    if not files:
        raise PilotDataError(f"No shard files found for split {split!r} under {processed_dir}.")
    return files


def load_full_split(dataset_config: DatasetConfig, split: str) -> list[Example]:
    """Reads every record of `split` from its committed shard files, in sorted-filename order.

    Deterministic and total: no sampling, no category floor, no rounding.
    """
    examples: list[Example] = []
    for shard_path in _split_shard_files(dataset_config.output.processed_path, split):
        with shard_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                examples.append(example_from_dict(json.loads(line)))
    if not examples:
        raise PilotDataError(f"Dataset split {split!r} contains zero records.")
    return examples


def _read_dataset_identity(dataset_config: DatasetConfig) -> str:
    path = dataset_config.output.dataset_identity_path
    if not path.is_file():
        raise PilotDataError(f"Dataset identity file not found at {path}. Run `dataset build` first.")
    return path.read_text(encoding="utf-8").split()[0]


def _category_counts(examples: list[Example]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in examples:
        counts[e.category] = counts.get(e.category, 0) + 1
    return counts


def _ids_sha256(examples: list[Example]) -> str:
    ids = sorted(e.example_id for e in examples)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FullManifest:
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


def build_full_manifest(
    dataset_config: DatasetConfig,
    tokenizer_identity: str,
    seed: int,
    max_sequence_length: int,
    pack_sequences_flag: bool,
    selections: dict[str, list[Example]],
) -> FullManifest:
    from juniper_math.hashing import sha256_file

    splits: dict[str, dict[str, Any]] = {}
    for split, examples in selections.items():
        splits[split] = {
            "example_count": len(examples),
            "total_token_count": sum(e.token_count or 0 for e in examples),
            "example_ids_sha256": _ids_sha256(examples),
            "category_counts": _category_counts(examples),
            "tool_required_count": sum(1 for e in examples if e.tool_required),
            "family_count": len({e.family_id for e in examples}),
        }
    return FullManifest(
        schema_version=FULL_MANIFEST_SCHEMA_VERSION,
        parent_dataset_id=dataset_config.dataset_id,
        parent_dataset_identity=_read_dataset_identity(dataset_config),
        parent_manifest_sha256=sha256_file(dataset_config.output.manifest_path),
        tokenizer_identity=tokenizer_identity,
        seed=seed,
        max_sequence_length=max_sequence_length,
        pack_sequences=pack_sequences_flag,
        splits=splits,
    )


def write_full_manifest(manifest: FullManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def select_and_record_full_dataset(
    dataset_id: str,
    tokenizer_identity: str,
    seed: int,
    max_sequence_length: int,
    pack_sequences_flag: bool,
    output_dir: Path,
    dataset_config: DatasetConfig | None = None,
) -> tuple[dict[str, list[Example]], FullManifest]:
    dataset_config = dataset_config or load_dataset_config()
    if dataset_config.dataset_id != dataset_id:
        raise PilotDataError(
            f"full training config dataset_identity {dataset_id!r} does not match config/dataset.yaml "
            f"dataset_id {dataset_config.dataset_id!r}."
        )
    verify_parent_dataset_shards(dataset_config)
    train_examples = load_full_split(dataset_config, "train")
    val_examples = load_full_split(dataset_config, "validation")

    selections = {"train": train_examples, "validation": val_examples}
    manifest = build_full_manifest(
        dataset_config, tokenizer_identity, seed, max_sequence_length, pack_sequences_flag, selections
    )
    write_full_manifest(manifest, output_dir / "full_manifest.json")
    return selections, manifest


__all__ = [
    "FULL_MANIFEST_SCHEMA_VERSION",
    "FullManifest",
    "build_full_manifest",
    "load_full_split",
    "select_and_record_full_dataset",
    "write_full_manifest",
]
