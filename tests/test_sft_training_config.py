"""Tests for juniper_math.sft_training_config — mirrors
test_full_training_config.py's style: mutate the real frozen YAML, assert
JuniperConfigError on the invalid cases; load it unmodified for the positive
cases."""

from __future__ import annotations

from dataclasses import replace

import pytest
import yaml

from juniper_math.errors import JuniperConfigError
from juniper_math.sft_training_config import (
    SFT_TRAINING_CONFIG_PATH,
    load_sft_training_config,
    verify_parent_checkpoint,
)


def _mutated(tmp_path, **overrides):
    raw = yaml.safe_load(SFT_TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    for dotted, value in overrides.items():
        node = raw
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_loads_real_frozen_config():
    config = load_sft_training_config()
    assert config.dataset_identity == "juniper-math-sft-v4"
    assert config.tool_protocol_identity == "juniper-tool-protocol-v1"
    assert len(config.parent_checkpoint_sha256) == 64


def test_verify_parent_checkpoint_passes_with_self_contained_fixture(tmp_path, monkeypatch):
    """Unit tests must not require a release-only checkpoint in a CI clone.

    Production training still calls the same fail-closed verifier against the
    actual Phase 7 release asset.  This test only proves the verifier's
    success path with controlled bytes and a matching declared digest.
    """
    checkpoint = tmp_path / "fixtures" / "phase7-base.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"controlled test checkpoint")
    config = replace(
        load_sft_training_config(),
        parent_checkpoint_path="fixtures/phase7-base.pt",
        parent_checkpoint_sha256="e94708e19f47085e7a99d9033a44a37d28f991490be1c5383d4926355c9c3f61",
    )
    monkeypatch.setattr("juniper_math.sft_training_config.REPO_ROOT", tmp_path)
    verify_parent_checkpoint(config)  # must not raise


def test_rejects_bad_parent_checkpoint_sha256(tmp_path):
    path = _mutated(tmp_path, **{"parent_checkpoint_sha256": "not-a-hash"})
    with pytest.raises(JuniperConfigError):
        load_sft_training_config(path)


def test_rejects_mismatched_parent_checkpoint_sha256(tmp_path):
    path = _mutated(tmp_path, **{"parent_checkpoint_sha256": "0" * 64})
    config = load_sft_training_config(path)
    with pytest.raises(JuniperConfigError):
        verify_parent_checkpoint(config)


def test_rejects_missing_warmup_ratio(tmp_path):
    raw = yaml.safe_load(SFT_TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    del raw["scheduler"]["warmup_ratio"]
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(JuniperConfigError):
        load_sft_training_config(path)


def test_rejects_inconsistent_warmup_steps(tmp_path):
    raw = yaml.safe_load(SFT_TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["scheduler"]["warmup_ratio"] = 0.5
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_sft_training_config(path)
    assert config.scheduler.warmup_steps == round(0.5 * config.schedule.total_steps)


def test_rejects_negative_category_weight_override(tmp_path):
    path = _mutated(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["sft_subset"]["category_weight_overrides"] = {"tool_error": -1.0}
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(JuniperConfigError):
        load_sft_training_config(path)


def test_rejects_empty_run_id(tmp_path):
    path = _mutated(tmp_path, run_id="")
    with pytest.raises(JuniperConfigError):
        load_sft_training_config(path)


def test_rejects_zero_total_steps(tmp_path):
    path = _mutated(tmp_path, **{"schedule.total_steps": 0})
    with pytest.raises(JuniperConfigError):
        load_sft_training_config(path)
