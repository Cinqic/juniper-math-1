"""Phase 5 smoke-subset selection and tokenized-dataset tests.

Builds a tiny synthetic shard set under tmp_path (never touches or requires
the real 1.6M-example juniper-math-dataset-v1 build) so these tests pass on
a fresh clone before `dataset build` has ever been run.
"""

from __future__ import annotations

import pytest

from juniper_math.dataset.config import OutputConfig, ShardConfig
from juniper_math.dataset.schema import Example
from juniper_math.dataset.shard import write_manifest, write_shards
from juniper_math.smoke_data import (
    SmokeDataError,
    TokenizedSmokeDataset,
    collate_smoke_batch,
    compute_stride_selection,
    epoch_order,
    select_smoke_examples,
)
from juniper_math.tokenizer import JuniperTokenizer


def _example(i: int, split: str) -> Example:
    return Example(
        example_id=f"ex{i:04d}",
        generator_id="test",
        generator_version="1.0.0",
        family_id="f",
        template_id="t0",
        derivation_id=f"d{i}",
        seed=i,
        category="arithmetic",
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
        token_count=8,
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

    examples_by_split = {
        "train": [_example(i, "train") for i in range(40)],
        "validation": [_example(i, "validation") for i in range(20)],
    }
    infos = write_shards(examples_by_split, shard_config, output)
    write_manifest(infos, "test-dataset-v1", "1.0.0", output)

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Cfg:
        dataset_id: str
        output: OutputConfig

    return _Cfg(dataset_id="test-dataset-v1", output=output)


def test_compute_stride_selection_basic():
    stride, offset = compute_stride_selection(100, 10, seed=5)
    assert stride == 10
    assert offset == 5


def test_compute_stride_selection_rejects_over_target():
    with pytest.raises(SmokeDataError):
        compute_stride_selection(10, 20, seed=0)


def test_select_smoke_examples_is_deterministic(tiny_dataset_config):
    first = select_smoke_examples(tiny_dataset_config, "train", 5, seed=123)
    second = select_smoke_examples(tiny_dataset_config, "train", 5, seed=123)
    assert [e.example_id for e in first] == [e.example_id for e in second]
    assert len(first) == 5


def test_select_smoke_examples_different_seed_differs(tiny_dataset_config):
    a = select_smoke_examples(tiny_dataset_config, "train", 5, seed=1)
    b = select_smoke_examples(tiny_dataset_config, "train", 5, seed=2)
    assert [e.example_id for e in a] != [e.example_id for e in b]


def test_select_smoke_examples_respects_split(tiny_dataset_config):
    selected = select_smoke_examples(tiny_dataset_config, "validation", 5, seed=1)
    assert all(e.split == "validation" for e in selected)


@pytest.fixture(scope="module")
def tokenizer():
    return JuniperTokenizer.load()


def test_tokenized_smoke_dataset_shapes_and_masking(tokenizer):
    examples = [_example(i, "train") for i in range(3)]
    max_len = 32
    dataset = TokenizedSmokeDataset(examples, tokenizer, max_len)
    assert len(dataset) == 3
    for i in range(3):
        item = dataset[i]
        assert item["input_ids"].shape == (max_len,)
        assert item["labels"].shape == (max_len,)
        assert item["attention_mask"].shape == (max_len,)
        n_real = int(item["attention_mask"].sum().item())
        assert (item["labels"][:n_real] != -100).all()
        if n_real < max_len:
            assert (item["labels"][n_real:] == -100).all()
            assert (item["attention_mask"][n_real:] == 0).all()
    assert dataset.total_loss_tokens > 0


def test_tokenized_smoke_dataset_truncates_long_examples(tokenizer):
    long_example = _example(0, "train")
    long_example = long_example.__class__(**{**long_example.__dict__, "prompt": "1 + 1 = " * 200})
    dataset = TokenizedSmokeDataset([long_example], tokenizer, max_sequence_length=16)
    item = dataset[0]
    assert item["input_ids"].shape == (16,)


def test_collate_smoke_batch(tokenizer):
    examples = [_example(i, "train") for i in range(4)]
    dataset = TokenizedSmokeDataset(examples, tokenizer, 16)
    batch = collate_smoke_batch([dataset[i] for i in range(4)])
    assert batch["input_ids"].shape == (4, 16)


def test_epoch_order_deterministic_and_shuffle_flag():
    ordered = epoch_order(10, seed=42, epoch=0, shuffle=False)
    assert ordered == list(range(10))

    a = epoch_order(10, seed=42, epoch=1, shuffle=True)
    b = epoch_order(10, seed=42, epoch=1, shuffle=True)
    assert a == b
    assert sorted(a) == list(range(10))

    c = epoch_order(10, seed=42, epoch=2, shuffle=True)
    assert c != a
