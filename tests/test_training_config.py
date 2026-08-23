"""Phase 5 training configuration loading tests."""

from __future__ import annotations

import pytest

from juniper_math.architecture import load_architecture_config
from juniper_math.errors import JuniperConfigError
from juniper_math.training_config import load_training_config


def test_training_config_loads():
    cfg = load_training_config()
    assert cfg.run_id
    assert cfg.schedule.total_steps > 0
    assert cfg.data.effective_batch_size == cfg.data.micro_batch_size * cfg.data.gradient_accumulation_steps


def test_training_config_matches_frozen_architecture_identity():
    cfg = load_training_config()
    arch = load_architecture_config()
    assert cfg.architecture_identity == arch.architecture_version


def test_resume_test_interrupt_step_is_within_total_steps():
    cfg = load_training_config()
    assert 0 < cfg.resume_test.interrupt_step < cfg.schedule.total_steps


def test_missing_config_raises(tmp_path):
    with pytest.raises(JuniperConfigError):
        load_training_config(tmp_path / "missing.yaml")


def test_malformed_config_raises(tmp_path):
    bad = tmp_path / "training.yaml"
    bad.write_text("run_id: x\n", encoding="utf-8")
    with pytest.raises(JuniperConfigError):
        load_training_config(bad)
