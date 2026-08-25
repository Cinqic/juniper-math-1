"""Canonical project metadata / phase-status loading.

Reads config/project.yaml, the single source of truth for the project's
current phase and review status. Human-facing docs must stay consistent
with this file; nothing should hardcode phase status elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from juniper_math.errors import JuniperConfigError
from juniper_math.paths import CONFIG_DIR

PROJECT_CONFIG_PATH = CONFIG_DIR / "project.yaml"

_VALID_PHASE_STATUSES = {
    "NOT_STARTED",
    "IN_PROGRESS",
    "AWAITING_OPUS_5_REVIEW",
    "AWAITING_GPT_5_6_TERRA_REVIEW",
    "AWAITING_HUMAN_REVIEW",
    "APPROVED",
    "COMPLETE",
    "CONCLUDED — MODEL CHECKPOINT NOT APPROVED",
}


@dataclass(frozen=True)
class ProjectMetadata:
    project_name: str
    project_family: str
    schema_version: str
    config_version: str
    research_question: str
    repository_principle: str
    parameter_target: int
    architecture_version: str
    architecture_config: str
    current_phase: int
    phase_name: str
    phase_status: str
    primary_hardware: dict[str, Any]
    review_chain: list[dict[str, Any]]
    phase_approval: dict[str, Any]
    next_phase: dict[str, Any]


def _require(raw: dict, key: str, expected_type: type) -> Any:
    if key not in raw:
        raise JuniperConfigError(f"{PROJECT_CONFIG_PATH}: missing required field '{key}'.")
    value = raw[key]
    if not isinstance(value, expected_type):
        raise JuniperConfigError(
            f"{PROJECT_CONFIG_PATH}: field '{key}' must be {expected_type.__name__}, "
            f"got {type(value).__name__}."
        )
    return value


def load_project_metadata(path: Path | None = None) -> ProjectMetadata:
    source = path or PROJECT_CONFIG_PATH
    if not source.is_file():
        raise JuniperConfigError(f"Project metadata not found at {source}.")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise JuniperConfigError(f"{source}: invalid YAML ({exc}).") from exc
    if not isinstance(raw, dict):
        raise JuniperConfigError(f"{source}: top-level YAML content must be a mapping.")

    phase_status = _require(raw, "phase_status", str)
    if phase_status not in _VALID_PHASE_STATUSES:
        raise JuniperConfigError(
            f"{source}: phase_status '{phase_status}' is not one of {sorted(_VALID_PHASE_STATUSES)}."
        )

    return ProjectMetadata(
        project_name=_require(raw, "project_name", str),
        project_family=_require(raw, "project_family", str),
        schema_version=_require(raw, "schema_version", str),
        config_version=_require(raw, "config_version", str),
        research_question=_require(raw, "research_question", str),
        repository_principle=_require(raw, "repository_principle", str),
        parameter_target=_require(raw, "parameter_target", int),
        architecture_version=_require(raw, "architecture_version", str),
        architecture_config=_require(raw, "architecture_config", str),
        current_phase=_require(raw, "current_phase", int),
        phase_name=_require(raw, "phase_name", str),
        phase_status=phase_status,
        primary_hardware=_require(raw, "primary_hardware", dict),
        review_chain=_require(raw, "review_chain", list),
        phase_approval=_require(raw, "phase_approval", dict),
        next_phase=_require(raw, "next_phase", dict),
    )
