"""Phase 6 pilot-subset selection and packing tests.

Builds tiny synthetic multi-category shard sets under tmp_path (never
touches or requires the real 1.6M-example juniper-math-dataset-v1 build),
mirroring tests/test_smoke_data.py's approach so these tests pass on a
fresh clone before `dataset build` has ever run.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from juniper_math.dataset.config import OutputConfig, ShardConfig
from juniper_math.dataset.schema import Example
from juniper_math.dataset.shard import write_manifest, write_shards
from juniper_math.pilot_data import (
    PackedPilotDataset,
    PilotDataError,
    compute_category_targets,
    count_categories,
    pack_sequences,
    select_and_record_pilot_subset,
    select_pilot_examples,
    tokenize_examples,
)
from juniper_math.smoke_data import compute_stride_selection
from juniper_math.tokenizer import JuniperTokenizer

CATEGORIES = ["arithmetic", "word_problem", "tool_error"]  # includes a deliberately rare one


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

    # arithmetic: 200 examples (common), word_problem: 100 (common),
    # tool_error: 5 (deliberately rare — smaller than any reasonable floor).
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
    write_manifest(infos, "test-pilot-dataset-v1", "1.0.0", output)

    @dataclass(frozen=True)
    class _Cfg:
        dataset_id: str
        output: OutputConfig

    return _Cfg(dataset_id="test-pilot-dataset-v1", output=output)


@pytest.fixture(scope="module")
def tokenizer():
    return JuniperTokenizer.load()


# --------------------------------------------------------------------------
# category counting / target computation
# --------------------------------------------------------------------------


def test_count_categories_exact(tiny_dataset_config):
    counts = count_categories(tiny_dataset_config, "train")
    assert counts.record_count == {"arithmetic": 200, "word_problem": 100, "tool_error": 5}
    assert counts.token_count == {"arithmetic": 1600, "word_problem": 800, "tool_error": 40}


def test_compute_category_targets_floors_rare_category(tiny_dataset_config):
    counts = count_categories(tiny_dataset_config, "train")
    # A small token budget would proportionally round tool_error's target to
    # 0 or 1 without a floor; min_category_examples must guarantee more.
    targets = compute_category_targets(counts, target_total_tokens=200, min_category_examples=5)
    assert targets["tool_error"] == 5  # floor == full availability here
    assert all(targets[cat] <= counts.record_count[cat] for cat in targets)


def test_compute_category_targets_floor_capped_at_availability(tiny_dataset_config):
    counts = count_categories(tiny_dataset_config, "train")
    # Floor requests more than tool_error actually has (5) — must cap, not overrun.
    targets = compute_category_targets(counts, target_total_tokens=100, min_category_examples=50)
    assert targets["tool_error"] == 5


# --------------------------------------------------------------------------
# selection: determinism, rare-category guarantee, no cross-split leakage
# --------------------------------------------------------------------------


def test_select_pilot_examples_is_deterministic(tiny_dataset_config):
    a, _ = select_pilot_examples(tiny_dataset_config, "train", 2000, min_category_examples=5, seed=123)
    b, _ = select_pilot_examples(tiny_dataset_config, "train", 2000, min_category_examples=5, seed=123)
    assert [e.example_id for e in a] == [e.example_id for e in b]


def test_select_pilot_examples_different_seed_can_differ(tiny_dataset_config):
    # A small enough token budget that categories are undersampled (stride > 1),
    # so a different seed's offset actually picks different records.
    a, _ = select_pilot_examples(tiny_dataset_config, "train", 400, min_category_examples=5, seed=1)
    b, _ = select_pilot_examples(tiny_dataset_config, "train", 400, min_category_examples=5, seed=2)
    assert [e.example_id for e in a] != [e.example_id for e in b]


def test_select_pilot_examples_guarantees_rare_category_present(tiny_dataset_config):
    examples, audit = select_pilot_examples(
        tiny_dataset_config, "train", target_total_tokens=200, min_category_examples=5, seed=7
    )
    categories = {e.category for e in examples}
    assert "tool_error" in categories
    assert audit["category_selected_counts"]["tool_error"] == 5


def test_select_pilot_examples_respects_split(tiny_dataset_config):
    examples, _ = select_pilot_examples(
        tiny_dataset_config, "validation", target_total_tokens=200, min_category_examples=3, seed=1
    )
    assert all(e.split == "validation" for e in examples)


def test_select_pilot_examples_uses_same_stride_primitive_as_phase5(tiny_dataset_config):
    """Per-category selection reuses Phase 5's exact compute_stride_selection, not a reimplementation."""
    counts = count_categories(tiny_dataset_config, "train")
    targets = compute_category_targets(counts, target_total_tokens=800, min_category_examples=5)
    examples, audit = select_pilot_examples(
        tiny_dataset_config, "train", target_total_tokens=800, min_category_examples=5, seed=42
    )
    for cat, target in targets.items():
        expected_stride, expected_offset = compute_stride_selection(counts.record_count[cat], target, 42)
        assert audit["category_strides"][cat] == {"stride": expected_stride, "offset": expected_offset}


# --------------------------------------------------------------------------
# tokenization / packing
# --------------------------------------------------------------------------


def test_tokenize_examples_clips_to_budget(tokenizer):
    long_example = _example(0, "train", "arithmetic")
    long_example = long_example.__class__(**{**long_example.__dict__, "prompt": "1 + 1 = " * 200})
    tokenized = tokenize_examples([long_example], tokenizer, max_sequence_length=16)
    assert len(tokenized[0].ids) == 16


def test_pack_sequences_never_splits_an_example(tokenizer):
    examples = [_example(i, "train", "arithmetic") for i in range(10)]
    tokenized = tokenize_examples(examples, tokenizer, max_sequence_length=32)
    bins = pack_sequences(tokenized, max_sequence_length=32)
    # Every tokenized example appears whole in exactly one bin.
    seen_ids = []
    for b in bins:
        total = sum(len(item.ids) for item in b)
        assert total <= 32
        seen_ids.extend(item.example.example_id for item in b)
    assert sorted(seen_ids) == sorted(e.example_id for e in examples)


def test_pack_sequences_reduces_sequence_count_vs_unpacked(tokenizer):
    examples = [_example(i, "train", "arithmetic") for i in range(20)]
    tokenized = tokenize_examples(examples, tokenizer, max_sequence_length=256)
    bins = pack_sequences(tokenized, max_sequence_length=256)
    assert len(bins) < len(examples)  # short examples pack several per 256-token sequence


def test_packed_pilot_dataset_shapes_and_masking(tokenizer):
    examples = [_example(i, "train", "arithmetic") for i in range(6)]
    max_len = 32
    dataset = PackedPilotDataset(examples, tokenizer, max_len)
    assert len(dataset) >= 1
    for i in range(len(dataset)):
        item = dataset[i]
        assert item["input_ids"].shape == (max_len,)
        assert item["labels"].shape == (max_len,)
        assert item["attention_mask"].shape == (max_len,)
        n_real = int(item["attention_mask"].sum().item())
        assert (item["labels"][:n_real] != -100).all()
        if n_real < max_len:
            assert (item["labels"][n_real:] == -100).all()
    assert dataset.total_loss_tokens > 0
    assert 0.0 <= dataset.padding_fraction <= 1.0


def test_packed_pilot_dataset_rejects_empty_input(tokenizer):
    with pytest.raises(PilotDataError):
        PackedPilotDataset([], tokenizer, 32)


# --------------------------------------------------------------------------
# end-to-end selection + manifest
# --------------------------------------------------------------------------


def test_select_and_record_pilot_subset_writes_manifest(tiny_dataset_config, tmp_path):
    out_dir = tmp_path / "pilot_out"
    selections, manifest = select_and_record_pilot_subset(
        dataset_id="test-pilot-dataset-v1",
        tokenizer_identity="juniper-math-tokenizer-v1",
        seed=5,
        target_train_tokens=800,
        validation_examples=20,
        min_category_examples=5,
        min_category_examples_validation=3,
        max_sequence_length=64,
        pack_sequences_flag=True,
        output_dir=out_dir,
        dataset_config=tiny_dataset_config,
    )
    assert (out_dir / "pilot_manifest.json").is_file()
    assert (out_dir / "pilot_selection_audit.json").is_file()
    assert manifest.splits["train"]["example_count"] == len(selections["train"])
    assert manifest.splits["validation"]["example_count"] == len(selections["validation"])
    assert "tool_error" in manifest.splits["train"]["category_counts"]


def test_select_and_record_pilot_subset_rejects_dataset_identity_mismatch(tiny_dataset_config, tmp_path):
    with pytest.raises(PilotDataError):
        select_and_record_pilot_subset(
            dataset_id="wrong-dataset-id",
            tokenizer_identity="juniper-math-tokenizer-v1",
            seed=5,
            target_train_tokens=800,
            validation_examples=20,
            min_category_examples=5,
            min_category_examples_validation=3,
            max_sequence_length=64,
            pack_sequences_flag=True,
            output_dir=tmp_path / "out",
            dataset_config=tiny_dataset_config,
        )
