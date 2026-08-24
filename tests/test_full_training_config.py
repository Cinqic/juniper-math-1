"""Phase 7 full training configuration loading tests."""

from __future__ import annotations

import pytest
import yaml

from juniper_math.architecture import load_architecture_config
from juniper_math.errors import JuniperConfigError
from juniper_math.full_training_config import FULL_TRAINING_CONFIG_PATH, load_full_training_config


def test_full_training_config_loads():
    cfg = load_full_training_config()
    assert cfg.run_id
    assert cfg.schedule.total_steps > 0
    assert cfg.data.effective_batch_size == cfg.data.micro_batch_size * cfg.data.gradient_accumulation_steps


def test_full_training_config_matches_frozen_architecture_identity():
    cfg = load_full_training_config()
    arch = load_architecture_config()
    assert cfg.architecture_identity == arch.architecture_version


def test_full_warmup_steps_computed_from_ratio():
    cfg = load_full_training_config()
    assert cfg.scheduler.warmup_ratio is not None
    assert cfg.scheduler.warmup_steps == round(cfg.scheduler.warmup_ratio * cfg.schedule.total_steps)


def test_full_resume_test_interrupt_step_is_within_total_steps():
    cfg = load_full_training_config()
    assert 0 < cfg.resume_test.interrupt_step < cfg.schedule.total_steps


def test_full_milestone_fractions_span_zero_to_one():
    cfg = load_full_training_config()
    assert cfg.milestone_fractions[0] == 0.0
    assert cfg.milestone_fractions[-1] == 1.0
    assert cfg.milestone_fractions == sorted(cfg.milestone_fractions)


def test_full_uses_fresh_seed_matching_project_default():
    from juniper_math.seed import DEFAULT_PROJECT_SEED

    cfg = load_full_training_config()
    assert cfg.seed == DEFAULT_PROJECT_SEED


def test_full_missing_config_raises(tmp_path):
    with pytest.raises(JuniperConfigError):
        load_full_training_config(tmp_path / "missing.yaml")


def test_full_malformed_config_raises(tmp_path):
    bad = tmp_path / "training_phase7_full.yaml"
    bad.write_text("run_id: x\n", encoding="utf-8")
    with pytest.raises(JuniperConfigError):
        load_full_training_config(bad)


def test_full_config_requires_warmup_ratio(tmp_path):
    raw = yaml.safe_load(FULL_TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    del raw["scheduler"]["warmup_ratio"]
    bad = tmp_path / "training_phase7_full.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(JuniperConfigError):
        load_full_training_config(bad)


def test_full_config_rejects_warmup_ratio_out_of_range(tmp_path):
    raw = yaml.safe_load(FULL_TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["scheduler"]["warmup_ratio"] = 1.5
    bad = tmp_path / "training_phase7_full.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(JuniperConfigError):
        load_full_training_config(bad)


def test_full_config_rejects_non_fp32_precision(tmp_path):
    raw = yaml.safe_load(FULL_TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["precision"] = "fp16"
    bad = tmp_path / "training_phase7_full.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(JuniperConfigError):
        load_full_training_config(bad)


def test_full_config_rejects_malformed_generation_prompts(tmp_path):
    raw = yaml.safe_load(FULL_TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["fixed_generation_prompts"] = ["not a dict"]
    bad = tmp_path / "training_phase7_full.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(JuniperConfigError):
        load_full_training_config(bad)


def test_full_config_rejects_milestone_fractions_not_spanning_zero_to_one(tmp_path):
    raw = yaml.safe_load(FULL_TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["milestone_fractions"] = [0.1, 0.5, 0.9]
    bad = tmp_path / "training_phase7_full.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(JuniperConfigError):
        load_full_training_config(bad)


def test_full_does_not_mutate_phase5_or_phase6_configs():
    """Loading the Phase 7 config must never touch Phase 5/6's own configs/loaders."""
    from juniper_math.pilot_training_config import load_pilot_training_config
    from juniper_math.training_config import load_training_config

    smoke_cfg = load_training_config()
    pilot_cfg = load_pilot_training_config()
    assert smoke_cfg.run_id == "phase5-smoke-v1"
    assert pilot_cfg.run_id == "phase6-pilot-v1"
    assert smoke_cfg.scheduler.warmup_ratio is None
    assert pilot_cfg.scheduler.warmup_ratio is None
