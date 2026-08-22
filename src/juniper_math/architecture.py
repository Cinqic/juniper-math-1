"""Frozen Phase 0 architecture specification.

This module only represents and validates the architecture configuration
declared in config/architecture.yaml. It does not implement any model code —
that is Phase 1 work. Programmatic parameter-count verification against a
real nn.Module also belongs to Phase 1; the `estimated_parameter_count`
helper here is a documented arithmetic estimate only, used to catch gross
config typos early.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from juniper_math.errors import JuniperConfigError
from juniper_math.paths import CONFIG_DIR

ARCHITECTURE_CONFIG_PATH = CONFIG_DIR / "architecture.yaml"

_REQUIRED_FIELDS = {
    "architecture_version": str,
    "architecture_class": str,
    "parameter_target": int,
    "parameter_class_approx": str,
    "d_model": int,
    "n_layers": int,
    "n_query_heads": int,
    "n_kv_heads": int,
    "head_dim": int,
    "attention_type": str,
    "ffn_type": str,
    "d_ff": int,
    "normalization": str,
    "normalization_layout": str,
    "position_encoding": str,
    "rope_theta": int,
    "biases": bool,
    "vocab_size": int,
    "weight_tying": bool,
    "max_context_length": int,
    "dropout": float,
}


@dataclass(frozen=True)
class ArchitectureConfig:
    architecture_version: str
    architecture_class: str
    parameter_target: int
    parameter_class_approx: str
    d_model: int
    n_layers: int
    n_query_heads: int
    n_kv_heads: int
    head_dim: int
    attention_type: str
    ffn_type: str
    d_ff: int
    normalization: str
    normalization_layout: str
    position_encoding: str
    rope_theta: int
    biases: bool
    vocab_size: int
    weight_tying: bool
    max_context_length: int
    dropout: float

    def estimated_parameter_count(self) -> int:
        """Rough arithmetic estimate. Not authoritative — see module docstring."""
        embedding = self.vocab_size * self.d_model
        attn_per_layer = 4 * (self.d_model * self.d_model)  # q, k, v, o
        ffn_per_layer = 3 * (self.d_model * self.d_ff)  # gate, up, down (SwiGLU)
        norms_per_layer = 2 * self.d_model  # pre-attn + pre-ffn RMSNorm
        per_layer = attn_per_layer + ffn_per_layer + norms_per_layer
        final_norm = self.d_model
        total = embedding + self.n_layers * per_layer + final_norm
        if self.weight_tying:
            pass  # output projection reuses the embedding matrix; already counted once
        return total


def _validate_raw(raw: dict[str, Any], source: Path) -> None:
    if not isinstance(raw, dict):
        raise JuniperConfigError(f"{source}: top-level YAML content must be a mapping.")

    missing = [key for key in _REQUIRED_FIELDS if key not in raw]
    if missing:
        raise JuniperConfigError(f"{source}: missing required field(s): {', '.join(sorted(missing))}")

    for key, expected_type in _REQUIRED_FIELDS.items():
        value = raw[key]
        # bool is a subclass of int in Python; guard against accidental type confusion.
        if expected_type is int and isinstance(value, bool):
            raise JuniperConfigError(f"{source}: field '{key}' must be an int, got bool.")
        if expected_type is float and isinstance(value, int) and not isinstance(value, bool):
            continue  # allow integral floats like dropout: 0
        if not isinstance(value, expected_type):
            raise JuniperConfigError(
                f"{source}: field '{key}' must be of type {expected_type.__name__}, "
                f"got {type(value).__name__}."
            )

    if raw["d_model"] <= 0:
        raise JuniperConfigError(f"{source}: d_model must be positive.")
    if raw["n_layers"] <= 0:
        raise JuniperConfigError(f"{source}: n_layers must be positive.")
    if raw["n_query_heads"] <= 0 or raw["n_kv_heads"] <= 0:
        raise JuniperConfigError(f"{source}: head counts must be positive.")
    if raw["n_query_heads"] % raw["n_kv_heads"] != 0:
        raise JuniperConfigError(f"{source}: n_query_heads must be a multiple of n_kv_heads.")
    if raw["n_query_heads"] * raw["head_dim"] != raw["d_model"]:
        raise JuniperConfigError(
            f"{source}: n_query_heads * head_dim ({raw['n_query_heads']}*{raw['head_dim']}) "
            f"must equal d_model ({raw['d_model']})."
        )
    if raw["vocab_size"] <= 0:
        raise JuniperConfigError(f"{source}: vocab_size must be positive.")
    if raw["max_context_length"] <= 0:
        raise JuniperConfigError(f"{source}: max_context_length must be positive.")
    if not (0.0 <= float(raw["dropout"]) < 1.0):
        raise JuniperConfigError(f"{source}: dropout must be in [0.0, 1.0).")
    if raw["parameter_target"] <= 0:
        raise JuniperConfigError(f"{source}: parameter_target must be positive.")


def load_architecture_config(path: Path | None = None) -> ArchitectureConfig:
    source = path or ARCHITECTURE_CONFIG_PATH
    if not source.is_file():
        raise JuniperConfigError(f"Architecture config not found at {source}.")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise JuniperConfigError(f"{source}: invalid YAML ({exc}).") from exc

    _validate_raw(raw, source)
    return ArchitectureConfig(**{key: raw[key] for key in _REQUIRED_FIELDS})
