"""Executes the frozen Phase 3 tool conformance/security suite
(evals/phase3_tools_v1.json) against the real runtime. This is the suite
referenced by reports/PHASE3_TOOL_VALIDATION.md and the Terra handoff.
"""

from __future__ import annotations

import json

import pytest

from juniper_math.paths import EVALS_DIR
from juniper_math.tools.protocol import ToolCall
from juniper_math.tools.runtime import ToolRuntime

SUITE_PATH = EVALS_DIR / "phase3_tools_v1.json"


def _load_suite():
    return json.loads(SUITE_PATH.read_text(encoding="utf-8"))


SUITE = _load_suite()


def test_suite_has_required_top_level_fields():
    assert SUITE["suite_id"] == "phase3-tools-v1"
    assert "cases" in SUITE and len(SUITE["cases"]) > 0


def test_case_ids_are_unique():
    ids = [c["id"] for c in SUITE["cases"]]
    assert len(ids) == len(set(ids))


def _run_single(runtime, case):
    if "call" in case:
        result = runtime.execute_call(ToolCall(**case["call"]))
    else:
        result = runtime.execute_text(case["call_text"])
    assert result.status == case["expected_status"], f"{case['id']}: status mismatch"
    if case["expected_result"] is not None:
        assert dict(result.result) == case["expected_result"], f"{case['id']}: result mismatch"
    if case["expected_error_code"] is not None:
        assert result.error is not None and result.error.code == case["expected_error_code"], (
            f"{case['id']}: error code mismatch"
        )


@pytest.mark.parametrize("case", [c for c in SUITE["cases"] if "steps" not in c], ids=lambda c: c["id"])
def test_single_step_case(case):
    runtime = ToolRuntime()
    _run_single(runtime, case)


@pytest.mark.parametrize("case", [c for c in SUITE["cases"] if "steps" in c], ids=lambda c: c["id"])
def test_multi_step_case(case):
    runtime = ToolRuntime()
    for step in case["steps"]:
        result = runtime.execute_call(ToolCall(**step["call"]))
        assert result.status == step["expected_status"], f"{case['id']}: step status mismatch"
        assert dict(result.result) == step["expected_result"], f"{case['id']}: step result mismatch"


def test_every_required_category_is_covered():
    required = {
        "correct_call",
        "incorrect_tool_name",
        "missing_argument",
        "extra_argument",
        "invalid_argument_type",
        "malformed_json",
        "duplicate_json_key",
        "unsupported_protocol_version",
        "division_by_zero",
        "domain_error",
        "unsupported_unit",
        "resource_limit",
        "fabricated_tool_result",
        "multi_step",
        "tool_unsupported",
    }
    present = {c["category"] for c in SUITE["cases"]}
    missing = required - present
    assert not missing, f"Missing required eval-suite categories: {sorted(missing)}"
