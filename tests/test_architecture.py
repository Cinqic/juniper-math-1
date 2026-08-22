from __future__ import annotations

import pytest
import yaml

from juniper_math.architecture import ARCHITECTURE_CONFIG_PATH, load_architecture_config
from juniper_math.errors import JuniperConfigError


def test_loads_frozen_spec_exactly():
    config = load_architecture_config()
    assert config.d_model == 256
    assert config.n_layers == 5
    assert config.n_query_heads == 4
    assert config.n_kv_heads == 4
    assert config.head_dim == 64
    assert config.d_ff == 688
    assert config.vocab_size == 4096
    assert config.max_context_length == 1024
    assert config.rope_theta == 10000
    assert config.weight_tying is True
    assert config.biases is False
    assert config.dropout == 0.0
    assert config.parameter_target == 5004032


def test_estimated_parameter_count_matches_target():
    config = load_architecture_config()
    assert config.estimated_parameter_count() == config.parameter_target


def test_missing_file_raises(tmp_path):
    with pytest.raises(JuniperConfigError, match="not found"):
        load_architecture_config(tmp_path / "nonexistent.yaml")


def test_missing_field_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"d_model": 256}), encoding="utf-8")
    with pytest.raises(JuniperConfigError, match="missing required field"):
        load_architecture_config(bad)


def test_wrong_type_rejected(tmp_path):
    valid = yaml.safe_load(ARCHITECTURE_CONFIG_PATH.read_text(encoding="utf-8"))
    valid["d_model"] = "256"  # should be int, not str
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(valid), encoding="utf-8")
    with pytest.raises(JuniperConfigError, match="must be of type int"):
        load_architecture_config(bad)


def test_inconsistent_head_dims_rejected(tmp_path):
    valid = yaml.safe_load(ARCHITECTURE_CONFIG_PATH.read_text(encoding="utf-8"))
    valid["head_dim"] = 999  # 4 * 999 != d_model (256)
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(valid), encoding="utf-8")
    with pytest.raises(JuniperConfigError, match="must equal d_model"):
        load_architecture_config(bad)


def test_negative_dropout_rejected(tmp_path):
    valid = yaml.safe_load(ARCHITECTURE_CONFIG_PATH.read_text(encoding="utf-8"))
    valid["dropout"] = 1.5
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(valid), encoding="utf-8")
    with pytest.raises(JuniperConfigError, match="dropout must be in"):
        load_architecture_config(bad)


def test_malformed_yaml_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("d_model: [unclosed", encoding="utf-8")
    with pytest.raises(JuniperConfigError, match="invalid YAML"):
        load_architecture_config(bad)
