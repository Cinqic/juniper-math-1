from __future__ import annotations

import pytest
import yaml

from juniper_math.errors import JuniperConfigError
from juniper_math.metadata import load_project_metadata


def test_loads_current_status():
    meta = load_project_metadata()
    assert meta.project_name == "Juniper Math 1"
    assert meta.current_phase == 6
    assert meta.phase_status == "COMPLETE"
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
    """Phase 4 records its completed independent approval chain."""
    meta = load_project_metadata()
    assert meta.phase_status == "COMPLETE"
    approval = meta.phase_approval
    assert approval["sonnet_5_implementation"] == "complete"
    assert approval["terra_independent_review"] == "approved"
    assert approval["terra_final_approval"] == "approved"
    assert approval["starting_foundation_tag"] == "phase-3-tools"


def test_phase_6_approved_phase_7_authorized_not_started():
    """Phase 6 is the current independently approved phase; Phase 7 has
    authorization only and must not have started."""
    meta = load_project_metadata()
    assert meta.current_phase == 6
    assert meta.next_phase["number"] == 7
    assert meta.next_phase["name"] == "Full Base Pretraining"
    assert meta.next_phase["started"] is False
    assert meta.next_phase["status"] == "AUTHORIZED — NOT STARTED"

    from juniper_math.metadata import PROJECT_CONFIG_PATH

    raw = yaml.safe_load(PROJECT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert raw["phase_5_engineering"]["sonnet_5_implementation"] == "complete"
    assert raw["phase_5_engineering"]["terra_independent_review"] == "complete"
    assert raw["phase_5_engineering"]["terra_final_approval"] == "approved"
    assert raw["phase_6_engineering"]["sonnet_5_implementation"] == "complete"
    assert raw["phase_6_engineering"]["terra_independent_review"] == "approved_with_remediation"
    assert raw["phase_6_engineering"]["terra_final_approval"] == "approved"
