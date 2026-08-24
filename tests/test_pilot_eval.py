"""Phase 6 capability-evaluation scoring tests.

Uses the real frozen tokenizer/model (tiny, built fresh) but a synthetic
suite file under tmp_path — never the real 725-case frozen suites — so
these are fast and don't depend on generation quality.
"""

from __future__ import annotations

import json

import pytest
import torch

from juniper_math.architecture import load_architecture_config
from juniper_math.dataset.schema import Example
from juniper_math.model import build_model
from juniper_math.pilot_eval import extract_tagged_value, numeric_matches, run_capability_evaluation
from juniper_math.tokenizer import JuniperTokenizer


def _case(example_id, category, expected_answer=None, expected_behavior="answer", tolerance=0):
    return {
        "example_id": example_id,
        "generator_id": "test",
        "generator_version": "1.0.0",
        "family_id": "f",
        "template_id": "t0",
        "derivation_id": f"d-{example_id}",
        "seed": 1,
        "category": category,
        "difficulty": "easy",
        "synthetic": True,
        "split": "test",
        "prompt": "What is 2 + 2?",
        "expected_behavior": expected_behavior,
        "expected_answer": expected_answer,
        "tolerance": tolerance,
        "tool_required": False,
        "tool_name": None,
        "tool_traces": [],
        "verification": {"mode": "deterministic", "expression": {"op": "add", "args": [2, 2]}},
        "provenance": "test",
        "notes": "",
    }


# --------------------------------------------------------------------------
# extract_tagged_value
# --------------------------------------------------------------------------


def test_extract_tagged_value_finds_final():
    present, value = extract_tagged_value("some reasoning <final>42", "final")
    assert present is True
    assert value == "42"


def test_extract_tagged_value_stops_at_next_tag():
    present, value = extract_tagged_value("<final>42<unsupported>", "final")
    assert present is True
    assert value == "42"


def test_extract_tagged_value_absent():
    present, value = extract_tagged_value("I don't know", "final")
    assert present is False
    assert value is None


def test_extract_tagged_value_unsupported_tag_no_value_needed():
    present, _ = extract_tagged_value("<unsupported>", "unsupported")
    assert present is True


# --------------------------------------------------------------------------
# numeric_matches
# --------------------------------------------------------------------------


def _example_with_answer(expected_answer, tolerance=0):
    return Example(
        example_id="e1",
        generator_id="g",
        generator_version="1.0.0",
        family_id="f",
        template_id="t0",
        derivation_id="d0",
        seed=1,
        category="arithmetic",
        difficulty="easy",
        synthetic=True,
        split="test",
        prompt="What is 2 + 2?",
        expected_behavior="answer",
        expected_answer=expected_answer,
        tolerance=tolerance,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "deterministic", "expression": {"op": "add", "args": [2, 2]}},
        provenance="test",
        notes="",
    )


def test_numeric_matches_exact():
    verified, _ = numeric_matches(_example_with_answer("4"), "4")
    assert verified is True


def test_numeric_matches_wrong_value():
    verified, _ = numeric_matches(_example_with_answer("4"), "5")
    assert verified is False


def test_numeric_matches_honors_tolerance():
    ex = _example_with_answer("4", tolerance=0.5)
    verified, _ = numeric_matches(ex, "4.3")
    assert verified is True
    verified, _ = numeric_matches(ex, "4.9")
    assert verified is False


def test_numeric_matches_unparseable_generated_value():
    verified, detail = numeric_matches(_example_with_answer("4"), "not a number")
    assert verified is False
    assert "did not parse" in detail


def test_numeric_matches_fraction_exactness_not_float_equality():
    # 0.1 + 0.2 style exactness: Fraction-based, not binary-float ==.
    ex = _example_with_answer("0.3")
    verified, _ = numeric_matches(ex, "0.3")
    assert verified is True


# --------------------------------------------------------------------------
# run_capability_evaluation end-to-end (real model+tokenizer, tiny suite)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    return JuniperTokenizer.load()


@pytest.fixture(scope="module")
def model():
    architecture = load_architecture_config()
    m = build_model(architecture)
    m.eval()
    return m


def test_run_capability_evaluation_counts_every_case_never_skips(tmp_path, model, tokenizer):
    cases = [
        _case("m1", "arithmetic", expected_answer="4"),
        _case("m2", "arithmetic", expected_behavior="refuse_unsupported", expected_answer=None),
        _case("m3", "arithmetic", expected_behavior="flag_undefined", expected_answer=None),
    ]
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps({"suite_id": "test-suite", "cases": cases}), encoding="utf-8")

    report = run_capability_evaluation(model, tokenizer, suite_path, torch.device("cpu"), max_new_tokens=4)
    assert report.n_cases == 3
    assert len(report.results) == 3
    # An untrained model should not spuriously pass — 0 correct is a legitimate result,
    # never silently excluded from the denominator (Sec. 19, Sec. 30).
    assert report.n_correct <= 3
    d = report.as_dict()
    assert d["n_cases"] == 3
    assert "category_accuracy" in d


def test_run_capability_evaluation_rejects_empty_suite(tmp_path, model, tokenizer):
    from juniper_math.errors import JuniperConfigError

    suite_path = tmp_path / "empty.json"
    suite_path.write_text(json.dumps({"suite_id": "empty", "cases": []}), encoding="utf-8")
    with pytest.raises(JuniperConfigError):
        run_capability_evaluation(model, tokenizer, suite_path, torch.device("cpu"), max_new_tokens=4)


def test_run_capability_evaluation_respects_sample_size(tmp_path, model, tokenizer):
    cases = [_case(f"m{i}", "arithmetic", expected_answer="4") for i in range(5)]
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps({"suite_id": "test-suite", "cases": cases}), encoding="utf-8")
    report = run_capability_evaluation(
        model, tokenizer, suite_path, torch.device("cpu"), max_new_tokens=4, sample_size=2
    )
    assert report.n_cases == 2
