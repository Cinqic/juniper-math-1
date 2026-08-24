"""Phase 7 full-dataset selection tests.

Mirrors tests/test_pilot_data.py's approach (tiny synthetic multi-category
shard sets under tmp_path, never the real 1.6M-example build). Unlike the
pilot subsample, full_data.load_full_split takes EVERY example in a split —
these tests assert totality, determinism, and split isolation rather than
stratified-target behavior (which full_data does not have).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass

import pytest

from juniper_math.dataset.config import OutputConfig, ShardConfig
from juniper_math.dataset.schema import Example
from juniper_math.dataset.shard import write_manifest, write_shards
from juniper_math.full_data import (
    build_full_manifest,
    load_full_split,
    select_and_record_full_dataset,
)
from juniper_math.pilot_data import PilotDataError
from juniper_math.tokenizer import JuniperTokenizer


def _example(i: int, split: str, category: str, token_count: int = 8) -> Example:
    return Example(
        example_id=f"ex{category}{i:05d}",
        generator_id="test",
        generator_version="1.0.0",
        family_id=f"fam-{category}",
        template_id="t0",
        derivation_id=f"d{category}{i}",
        seed=i,
        category=category,
        difficulty="easy",
        synthetic=True,
        split=split,
        prompt=f"What is {i} + 1?",
        expected_behavior="answer",
        expected_answer=str(i + 1),
        tolerance=0,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "deterministic", "expression": {"op": "add", "args": [i, 1]}},
        provenance="test",
        notes="",
        token_count=token_count,
    )


@pytest.fixture
def tiny_dataset_config(tmp_path):
    output = OutputConfig(
        processed_dir=str(tmp_path / "processed"),
        manifest_file=str(tmp_path / "processed" / "shard_manifest.json"),
        stats_file=str(tmp_path / "processed" / "stats.json"),
        dataset_identity_file=str(tmp_path / "processed" / "DATASET_IDENTITY.sha256"),
    )
    shard_config = ShardConfig(
        format="jsonl", records_per_shard=1000, filename_pattern="test.{split}.{shard_index:05d}.jsonl"
    )
    train_examples = (
        [_example(i, "train", "arithmetic") for i in range(200)]
        + [_example(i, "train", "word_problem") for i in range(100)]
        + [_example(i, "train", "tool_error") for i in range(5)]
    )
    val_examples = (
        [_example(i, "validation", "arithmetic") for i in range(40)]
        + [_example(i, "validation", "word_problem") for i in range(20)]
        + [_example(i, "validation", "tool_error") for i in range(3)]
    )
    infos = write_shards({"train": train_examples, "validation": val_examples}, shard_config, output)
    write_manifest(infos, "test-full-dataset-v1", "1.0.0", output)

    @dataclass(frozen=True)
    class _Cfg:
        dataset_id: str
        output: OutputConfig

    return _Cfg(dataset_id="test-full-dataset-v1", output=output)


@pytest.fixture(scope="module")
def tokenizer():
    return JuniperTokenizer.load()


def test_load_full_split_returns_every_example(tiny_dataset_config):
    train = load_full_split(tiny_dataset_config, "train")
    val = load_full_split(tiny_dataset_config, "validation")
    assert len(train) == 200 + 100 + 5
    assert len(val) == 40 + 20 + 3
    assert all(e.split == "train" for e in train)
    assert all(e.split == "validation" for e in val)


def test_load_full_split_includes_rare_category_entirely(tiny_dataset_config):
    train = load_full_split(tiny_dataset_config, "train")
    tool_error = [e for e in train if e.category == "tool_error"]
    assert len(tool_error) == 5  # every rare-category example, no floor/cap needed


def test_load_full_split_is_deterministic(tiny_dataset_config):
    a = load_full_split(tiny_dataset_config, "train")
    b = load_full_split(tiny_dataset_config, "train")
    assert [e.example_id for e in a] == [e.example_id for e in b]


def test_load_full_split_missing_shards_raises(tmp_path):
    output = OutputConfig(
        processed_dir=str(tmp_path / "empty"),
        manifest_file=str(tmp_path / "empty" / "shard_manifest.json"),
        stats_file=str(tmp_path / "empty" / "stats.json"),
        dataset_identity_file=str(tmp_path / "empty" / "DATASET_IDENTITY.sha256"),
    )
    (tmp_path / "empty").mkdir()

    @dataclass(frozen=True)
    class _Cfg:
        dataset_id: str
        output: OutputConfig

    with pytest.raises(PilotDataError):
        load_full_split(_Cfg(dataset_id="x", output=output), "train")


def test_build_full_manifest_counts_match_selection(tiny_dataset_config):
    train = load_full_split(tiny_dataset_config, "train")
    val = load_full_split(tiny_dataset_config, "validation")
    manifest = build_full_manifest(
        tiny_dataset_config,
        tokenizer_identity="juniper-math-tokenizer-v1",
        seed=5004032,
        max_sequence_length=1024,
        pack_sequences_flag=True,
        selections={"train": train, "validation": val},
    )
    assert manifest.splits["train"]["example_count"] == len(train)
    assert manifest.splits["validation"]["example_count"] == len(val)
    assert manifest.splits["train"]["category_counts"]["tool_error"] == 5
    assert manifest.parent_dataset_id == "test-full-dataset-v1"


def test_select_and_record_full_dataset_writes_manifest_and_matches_split_totality(
    tiny_dataset_config, tmp_path
):
    out_dir = tmp_path / "full_out"
    selections, manifest = select_and_record_full_dataset(
        dataset_id="test-full-dataset-v1",
        tokenizer_identity="juniper-math-tokenizer-v1",
        seed=5004032,
        max_sequence_length=64,
        pack_sequences_flag=True,
        output_dir=out_dir,
        dataset_config=tiny_dataset_config,
    )
    assert (out_dir / "full_manifest.json").is_file()
    assert len(selections["train"]) == 305
    assert len(selections["validation"]) == 63
    assert manifest.splits["train"]["example_count"] == 305


def test_select_and_record_full_dataset_rejects_dataset_identity_mismatch(tiny_dataset_config, tmp_path):
    with pytest.raises(PilotDataError):
        select_and_record_full_dataset(
            dataset_id="wrong-dataset-id",
            tokenizer_identity="juniper-math-tokenizer-v1",
            seed=1,
            max_sequence_length=64,
            pack_sequences_flag=True,
            output_dir=tmp_path / "out",
            dataset_config=tiny_dataset_config,
        )


def test_full_selection_rejects_unmanifested_matching_shard(tiny_dataset_config, tmp_path):
    processed = tiny_dataset_config.output.processed_path
    source = next(processed.glob("*.train.*.jsonl"))
    shutil.copyfile(source, processed / "test.train.99999.jsonl")
    with pytest.raises(PilotDataError, match="Unexpected unmanifested"):
        select_and_record_full_dataset(
            dataset_id="test-full-dataset-v1",
            tokenizer_identity="x",
            seed=1,
            max_sequence_length=64,
            pack_sequences_flag=True,
            output_dir=tmp_path / "out",
            dataset_config=tiny_dataset_config,
        )


def test_full_selection_rejects_duplicate_manifest_entry(tiny_dataset_config, tmp_path):
    manifest_path = tiny_dataset_config.output.manifest_path
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"].append(dict(manifest["shards"][0]))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(PilotDataError, match="duplicate filename"):
        select_and_record_full_dataset(
            dataset_id="test-full-dataset-v1",
            tokenizer_identity="x",
            seed=1,
            max_sequence_length=64,
            pack_sequences_flag=True,
            output_dir=tmp_path / "out",
            dataset_config=tiny_dataset_config,
        )


def test_full_selection_rejects_wrong_manifest_split_metadata(tiny_dataset_config, tmp_path):
    manifest_path = tiny_dataset_config.output.manifest_path
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][0]["split"] = "validation"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(PilotDataError, match="No manifest shard files"):
        select_and_record_full_dataset(
            dataset_id="test-full-dataset-v1",
            tokenizer_identity="x",
            seed=1,
            max_sequence_length=64,
            pack_sequences_flag=True,
            output_dir=tmp_path / "out",
            dataset_config=tiny_dataset_config,
        )
