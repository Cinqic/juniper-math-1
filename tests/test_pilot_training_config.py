"""Phase 6 pilot training configuration loading tests."""

from __future__ import annotations

import pytest
import yaml

from juniper_math.architecture import load_architecture_config
from juniper_math.errors import JuniperConfigError
from juniper_math.pilot_training_config import PILOT_TRAINING_CONFIG_PATH, load_pilot_training_config


def test_pilot_training_config_loads():
    cfg = load_pilot_training_config()
    assert cfg.run_id
    assert cfg.schedule.total_steps > 0
    assert cfg.data.effective_batch_size == cfg.data.micro_batch_size * cfg.data.gradient_accumulation_steps


def test_pilot_training_config_matches_frozen_architecture_identity():
    cfg = load_pilot_training_config()
    arch = load_architecture_config()
    assert cfg.architecture_identity == arch.architecture_version


def test_pilot_token_budget_within_envelope():
    cfg = load_pilot_training_config()
    assert 3_000_000 <= cfg.pilot_subset.target_train_tokens <= 10_000_000


def test_pilot_resume_test_interrupt_step_is_within_total_steps():
    cfg = load_pilot_training_config()
    assert 0 < cfg.resume_test.interrupt_step < cfg.schedule.total_steps


def test_pilot_milestone_fractions_span_zero_to_one():
    cfg = load_pilot_training_config()
    assert cfg.milestone_fractions[0] == 0.0
    assert cfg.milestone_fractions[-1] == 1.0
    assert cfg.milestone_fractions == sorted(cfg.milestone_fractions)


def test_pilot_missing_config_raises(tmp_path):
    with pytest.raises(JuniperConfigError):
        load_pilot_training_config(tmp_path / "missing.yaml")


def test_pilot_malformed_config_raises(tmp_path):
    bad = tmp_path / "training_phase6_pilot.yaml"
    bad.write_text("run_id: x\n", encoding="utf-8")
    with pytest.raises(JuniperConfigError):
        load_pilot_training_config(bad)


def test_pilot_config_rejects_token_budget_outside_envelope(tmp_path):
    raw = yaml.safe_load(PILOT_TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["pilot_subset"]["target_train_tokens"] = 50_000  # far below the 3-10M envelope
    bad = tmp_path / "training_phase6_pilot.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(JuniperConfigError, match="envelope"):
        load_pilot_training_config(bad)


def test_pilot_config_rejects_non_fp32_precision(tmp_path):
    raw = yaml.safe_load(PILOT_TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["precision"] = "fp16"
    bad = tmp_path / "training_phase6_pilot.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(JuniperConfigError):
        load_pilot_training_config(bad)


def test_pilot_config_rejects_malformed_generation_prompts(tmp_path):
    raw = yaml.safe_load(PILOT_TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["fixed_generation_prompts"] = ["not a dict"]
    bad = tmp_path / "training_phase6_pilot.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(JuniperConfigError):
        load_pilot_training_config(bad)


def test_pilot_config_rejects_milestone_fractions_not_spanning_zero_to_one(tmp_path):
    raw = yaml.safe_load(PILOT_TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["milestone_fractions"] = [0.1, 0.5, 0.9]
    bad = tmp_path / "training_phase6_pilot.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(JuniperConfigError):
        load_pilot_training_config(bad)


def test_pilot_does_not_mutate_phase5_smoke_config():
    """Loading the pilot config must never touch Phase 5's own config/loader."""
    from juniper_math.training_config import load_training_config

    smoke_cfg = load_training_config()
    assert smoke_cfg.run_id == "phase5-smoke-v1"
    assert smoke_cfg.smoke_subset.max_sequence_length == 256
