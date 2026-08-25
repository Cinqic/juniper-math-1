"""Tests for Phase 8 SFT-subset selection (juniper_math.sft_data)."""

from __future__ import annotations

import json

import pytest

from juniper_math.dataset.config import OutputConfig, ShardConfig, load_dataset_config
from juniper_math.dataset.schema import Example
from juniper_math.dataset.shard import write_manifest, write_shards
from juniper_math.sft_data import (
    CategoryCounts,
    SftDataError,
    compute_flattened_targets,
    representation_sha256,
    select_and_record_sft_subset,
    select_sft_examples,
)
from juniper_math.tokenizer import JuniperTokenizer


@pytest.fixture(scope="module")
def tokenizer() -> JuniperTokenizer:
    return JuniperTokenizer.load()


def _example(category: str, idx: int, split: str = "train") -> Example:
    return Example(
        example_id=f"ex_{category}_{idx:05d}",
        generator_id="test",
        generator_version="1.0.0",
        family_id=f"fam_{category}",
        template_id="t",
        derivation_id=f"deriv_{category}_{idx}",
        seed=idx,
        category=category,
        difficulty="easy",
        synthetic=True,
        split=split,
        prompt=f"What is {idx} + 1?",
        expected_behavior="answer",
        expected_answer=str(idx + 1),
        tolerance=0,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "deterministic", "expression": {"op": "add", "args": [idx, 1]}},
        provenance="test",
        notes="",
    )


@pytest.fixture
def tiny_dataset_config(tmp_path, monkeypatch):
    cfg = load_dataset_config()
    processed_dir = tmp_path / "processed"
    # REPO_ROOT / <absolute path> resolves to the absolute path unchanged
    # (pathlib join semantics), so passing tmp_path's absolute string here
    # is safe even though these fields are normally repo-relative.
    output = OutputConfig(
        processed_dir=str(processed_dir),
        manifest_file=str(processed_dir / "shard_manifest.json"),
        stats_file=str(processed_dir / "stats.json"),
        dataset_identity_file=str(processed_dir / "DATASET_IDENTITY.sha256"),
    )
    examples_by_split = {
        "train": [_example("arithmetic", i) for i in range(50)]
        + [_example("tool_use", i, "train") for i in range(20)],
        "validation": [_example("arithmetic", i, "validation") for i in range(20)],
    }
    infos = write_shards(
        examples_by_split,
        ShardConfig(
            format="jsonl", records_per_shard=1000, filename_pattern="{split}.{shard_index:05d}.jsonl"
        ),
        output,
    )
    write_manifest(infos, cfg.dataset_id, "1.0.0", output)
    new_cfg = cfg.__class__(**{**cfg.__dict__, "output": output})
    return new_cfg


def test_count_categories_matches_written_examples(tiny_dataset_config):
    from juniper_math.sft_data import count_categories

    counts = count_categories(tiny_dataset_config, "train")
    assert counts.record_count["arithmetic"] == 50
    assert counts.record_count["tool_use"] == 20


def test_compute_flattened_targets_floors_at_availability():
    counts = CategoryCounts(record_count={"a": 5, "b": 1000})
    targets = compute_flattened_targets(counts, target_per_category=100)
    assert targets["a"] == 5  # capped by availability
    assert targets["b"] == 100


def test_compute_flattened_targets_applies_weight_override():
    counts = CategoryCounts(record_count={"a": 1000})
    targets = compute_flattened_targets(counts, target_per_category=100, category_weight_overrides={"a": 2.0})
    assert targets["a"] == 200


def test_select_sft_examples_respects_targets(tiny_dataset_config, tokenizer):
    outcome = select_sft_examples(
        tiny_dataset_config,
        "train",
        {"arithmetic": 10, "tool_use": 5},
        seed=1,
        tokenizer=tokenizer,
        max_sequence_length=256,
    )
    assert len(outcome.examples) == 15
    counts = {}
    for ex in outcome.examples:
        counts[ex.category] = counts.get(ex.category, 0) + 1
    assert counts == {"arithmetic": 10, "tool_use": 5}


def test_select_sft_examples_deterministic(tiny_dataset_config, tokenizer):
    a = select_sft_examples(
        tiny_dataset_config,
        "train",
        {"arithmetic": 10},
        seed=42,
        tokenizer=tokenizer,
        max_sequence_length=256,
    )
    b = select_sft_examples(
        tiny_dataset_config,
        "train",
        {"arithmetic": 10},
        seed=42,
        tokenizer=tokenizer,
        max_sequence_length=256,
    )
    assert [e.example_id for e in a.examples] == [e.example_id for e in b.examples]


def test_select_sft_examples_raises_on_shortfall(tiny_dataset_config, tokenizer):
    # max_sequence_length=1 rejects every candidate as oversized (BOS alone
    # exceeds it), so no category can reach its target -> shortfall.
    with pytest.raises(SftDataError):
        select_sft_examples(
            tiny_dataset_config,
            "train",
            {"arithmetic": 10},
            seed=1,
            tokenizer=tokenizer,
            max_sequence_length=1,
        )


def test_select_and_record_sft_subset_writes_manifest(tiny_dataset_config, tokenizer, tmp_path):
    out_dir = tmp_path / "sft_out"
    selections, manifest = select_and_record_sft_subset(
        tokenizer_identity="juniper-math-tokenizer-v1",
        seed=7,
        train_target_per_category=10,
        validation_target_per_category=5,
        max_sequence_length=256,
        output_dir=out_dir,
        tokenizer=tokenizer,
        dataset_config=tiny_dataset_config,
    )
    assert (out_dir / "sft_manifest.json").is_file()
    assert (out_dir / "sft_selection_audit.json").is_file()
    payload = json.loads((out_dir / "sft_manifest.json").read_text())
    assert payload["sft_identity"] == manifest.sft_identity
    assert payload["splits"]["train"]["example_count"] == len(selections["train"])


def test_sft_identity_is_stable_and_order_independent():
    from juniper_math.sft_data import SftManifest

    m1 = SftManifest(
        schema_version="1.0.0",
        sft_dataset_id="x",
        parent_dataset_id="p",
        parent_dataset_identity="pi",
        parent_manifest_sha256="pm",
        tokenizer_identity="t",
        seed=1,
        max_sequence_length=256,
        splits={"train": {"example_ids_sha256": "aaa"}, "validation": {"example_ids_sha256": "bbb"}},
    )
    m2 = SftManifest(
        schema_version="1.0.0",
        sft_dataset_id="x",
        parent_dataset_id="p",
        parent_dataset_identity="pi",
        parent_manifest_sha256="pm",
        tokenizer_identity="t",
        seed=1,
        max_sequence_length=256,
        splits={"validation": {"example_ids_sha256": "bbb"}, "train": {"example_ids_sha256": "aaa"}},
    )
    assert m1.sft_identity == m2.sft_identity


def test_representation_identity_changes_when_labels_change(tokenizer):
    first = _example("arithmetic", 1)
    changed_label = Example(**{**first.__dict__, "expected_answer": "999"})
    assert representation_sha256([first], tokenizer, 256) != representation_sha256(
        [changed_label], tokenizer, 256
    )
