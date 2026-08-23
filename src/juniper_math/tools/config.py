"""Loader and validator for the frozen Phase 3 tool configuration
(config/tools.yaml) — the single source of truth for protocol identity,
approved tools, statuses, error codes, and every runtime limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from juniper_math.errors import JuniperConfigError
from juniper_math.paths import CONFIG_DIR, REPO_ROOT

TOOLS_CONFIG_PATH = CONFIG_DIR / "tools.yaml"


@dataclass(frozen=True)
class Limits:
    max_call_bytes: int
    max_expression_length: int
    max_string_argument_length: int
    max_json_depth: int
    max_json_members: int
    max_ast_nodes: int
    max_ast_depth: int
    max_numeric_literal_digits: int
    max_exponent_magnitude: int
    max_pow_result_bits: int
    max_factorial_n: int


@dataclass(frozen=True)
class ToolsConfig:
    protocol_id: str
    protocol_version: str
    tools: tuple[str, ...]
    result_statuses: tuple[str, ...]
    error_codes: tuple[str, ...]
    limits: Limits
    evaluate: dict[str, Any]
    convert: dict[str, Any]
    finance: dict[str, Any]
    upstream: dict[str, Any]
    schemas: dict[str, str]

    def schema_path(self, key: str) -> Path:
        return REPO_ROOT / self.schemas[key]


_REQUIRED_LIMIT_KEYS = {
    "max_call_bytes",
    "max_expression_length",
    "max_string_argument_length",
    "max_json_depth",
    "max_json_members",
    "max_ast_nodes",
    "max_ast_depth",
    "max_numeric_literal_digits",
    "max_exponent_magnitude",
    "max_pow_result_bits",
    "max_factorial_n",
}


def load_tools_config(path: Path | None = None) -> ToolsConfig:
    source = path or TOOLS_CONFIG_PATH
    if not source.is_file():
        raise JuniperConfigError(f"Tool configuration not found at {source}.")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise JuniperConfigError(f"{source}: invalid YAML ({exc}).") from exc
    if not isinstance(raw, dict):
        raise JuniperConfigError(f"{source}: expected a top-level mapping.")

    required_top_level = (
        "protocol",
        "tools",
        "result_statuses",
        "error_codes",
        "limits",
        "evaluate",
        "convert",
        "finance",
        "upstream",
        "schemas",
    )
    for key in required_top_level:
        if key not in raw:
            raise JuniperConfigError(f"{source}: missing top-level key {key!r}.")

    protocol = raw["protocol"]
    if not isinstance(protocol, dict) or "protocol_id" not in protocol or "protocol_version" not in protocol:
        raise JuniperConfigError(f"{source}: 'protocol' must declare protocol_id and protocol_version.")

    limits_raw = raw["limits"]
    if not isinstance(limits_raw, dict):
        raise JuniperConfigError(f"{source}: 'limits' must be a mapping.")
    missing_limits = _REQUIRED_LIMIT_KEYS - set(limits_raw)
    if missing_limits:
        raise JuniperConfigError(f"{source}: 'limits' missing key(s): {sorted(missing_limits)}")
    extra_limits = set(limits_raw) - _REQUIRED_LIMIT_KEYS
    if extra_limits:
        raise JuniperConfigError(f"{source}: 'limits' has unknown key(s): {sorted(extra_limits)}")
    for key, value in limits_raw.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise JuniperConfigError(f"{source}: limits.{key} must be a positive integer.")

    tools = tuple(raw["tools"])
    if len(tools) != len(set(tools)):
        raise JuniperConfigError(f"{source}: 'tools' contains duplicate entries.")

    return ToolsConfig(
        protocol_id=protocol["protocol_id"],
        protocol_version=protocol["protocol_version"],
        tools=tools,
        result_statuses=tuple(raw["result_statuses"]),
        error_codes=tuple(raw["error_codes"]),
        limits=Limits(**limits_raw),
        evaluate=raw["evaluate"],
        convert=raw["convert"],
        finance=raw["finance"],
        upstream=raw["upstream"],
        schemas=raw["schemas"],
    )
