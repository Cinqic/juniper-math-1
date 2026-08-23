from __future__ import annotations

import pytest
import yaml

from juniper_math.errors import JuniperConfigError
from juniper_math.metadata import load_project_metadata


def test_loads_current_status():
    meta = load_project_metadata()
    assert meta.project_name == "Juniper Math 1"
    assert meta.current_phase == 4
    assert meta.phase_status == "AWAITING_GPT_5_6_TERRA_REVIEW"
    assert "five-million-parameter" in meta.research_question
    assert meta.parameter_target == 5004032


def test_invalid_phase_status_rejected(tmp_path):
    from juniper_math.metadata import PROJECT_CONFIG_PATH

    valid = yaml.safe_load(PROJECT_CONFIG_PATH.read_text(encoding="utf-8"))
    valid["phase_status"] = "TOTALLY_DONE"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(valid), encoding="utf-8")
    with pytest.raises(JuniperConfigError, match="not one of"):
        load_project_metadata(bad)


def test_missing_file_raises(tmp_path):
    with pytest.raises(JuniperConfigError, match="not found"):
        load_project_metadata(tmp_path / "missing.yaml")


def test_phase_approval_record_present():
    """Phase 4 records the full approval chain (in progress), not just a
    status string. The bare `phase_approval` key always describes whichever
    phase is currently in progress; Phase 3's now-superseded record lives
    at `phase_3_approval`."""
    meta = load_project_metadata()
    assert meta.phase_status == "AWAITING_GPT_5_6_TERRA_REVIEW"
    approval = meta.phase_approval
    assert approval["sonnet_5_implementation"] == "complete"
    assert approval["terra_independent_review"] == "pending"
    assert approval["starting_foundation_tag"] == "phase-3-tools"


def test_phase_5_is_not_authorized():
    meta = load_project_metadata()
    assert meta.next_phase["number"] == 5
    assert meta.next_phase["status"] == "NOT_AUTHORIZED"
    assert meta.next_phase["started"] is False
