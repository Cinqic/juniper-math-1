"""Tests for deterministic evaluation ground-truth verification.

The headline test here is `test_original_tool_001_defect_is_now_caught`: it
re-injects the exact wrong value that shipped in the Phase 0 review candidate
and asserts the verifier rejects it. That defect must never be able to survive
validation again. See reports/OPUS5_PHASE0_REVIEW.md (F-01, F-02).
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from juniper_math.errors import JuniperConfigError
from juniper_math.evals import load_eval_suite, verify_suite_ground_truth
from juniper_math.verification import (
    ALLOWED_OPERATIONS,
    MODE_DETERMINISTIC,
    evaluate_expression,
    verify_case,
)


def _case(suite, case_id):
    return next(c for c in suite.cases if c.id == case_id)


# --- The regression that matters -------------------------------------------


def test_original_tool_001_defect_is_now_caught():
    """The exact wrong answer from candidate b39e600 must fail verification."""
    suite = load_eval_suite()
    tool_case = _case(suite, "tool-001")

    # Sanity: the corrected value verifies.
    assert verify_case(tool_case).verified is True
    assert tool_case.expected_answer == 836930542

    # Re-inject the original defect: 84317 * 9926 recorded as 837042742.
    defective = replace(tool_case, expected_answer=837042742)
    result = verify_case(defective)
    assert result.verified is False
    assert "836930542" in result.detail
    assert "837042742" in result.detail


def test_whole_frozen_suite_ground_truth_verifies():
    """Every deterministic case in the shipped suite recomputes correctly."""
    suite = load_eval_suite()
    results = verify_suite_ground_truth(suite)
    failures = [r for r in results if not r.verified]
    assert failures == [], f"ground-truth failures: {[(r.case_id, r.detail) for r in failures]}"
    assert len(results) == 22


def test_suite_has_expected_verification_mode_split():
    suite = load_eval_suite()
    results = verify_suite_ground_truth(suite)
    deterministic = [r for r in results if r.mode == MODE_DETERMINISTIC]
    semantic = [r for r in results if r.mode == "semantic"]
    assert len(deterministic) == 18
    assert len(semantic) == 4
    assert {r.case_id for r in semantic} == {"amb-001", "miss-001", "undef-001", "unsup-001"}


@pytest.mark.parametrize(
    ("case_id", "expected"),
    [
        ("arith-001", 19),
        ("arith-002", 11),
        ("neg-001", 7),
        ("pct-001", 36),
        ("ratio-001", 6),
        ("prop-001", 8),
        ("alg-001", 6),
        ("units-001", 3500),
        ("sci-001", 64000000),
        ("word-001", 21),
        ("tool-001", 836930542),
        ("direct-001", 19),
        ("err-001", 5),
    ],
)
def test_deterministic_cases_recompute_to_known_values(case_id, expected):
    """Independently assert the recomputed value, not just self-consistency."""
    suite = load_eval_suite()
    computed = evaluate_expression(_case(suite, case_id).verification["expression"])
    assert computed == Fraction(expected)


def test_exact_decimal_and_fraction_arithmetic():
    """Decimals and fractions are exact, not binary-float approximations."""
    suite = load_eval_suite()
    assert evaluate_expression(_case(suite, "dec-001").verification["expression"]) == Fraction(25, 4)
    assert evaluate_expression(_case(suite, "frac-001").verification["expression"]) == Fraction(5, 6)
    assert evaluate_expression(_case(suite, "cur-001").verification["expression"]) == Fraction(51, 4)
    # The classic float trap must not appear.
    assert evaluate_expression({"op": "add", "args": [0.1, 0.2]}) == Fraction(3, 10)


def test_estimation_case_uses_tolerance_band():
    """est-001 records 4000 while the exact product is 3992; tolerance covers it."""
    suite = load_eval_suite()
    case = _case(suite, "est-001")
    assert evaluate_expression(case.verification["expression"]) == Fraction(3992)
    assert case.expected_answer == 4000
    assert verify_case(case).verified is True
    # Outside the band it must fail.
    assert verify_case(replace(case, tolerance=1)).verified is False


def test_boolean_case_verifies():
    suite = load_eval_suite()
    case = _case(suite, "wrong-001")
    assert verify_case(case).verified is True
    assert verify_case(replace(case, expected_answer=True)).verified is False


# --- Safety: no code execution ---------------------------------------------


def test_verifier_rejects_unknown_operations():
    with pytest.raises(JuniperConfigError, match="unknown operation"):
        evaluate_expression({"op": "__import__", "args": ["os"]})
    with pytest.raises(JuniperConfigError, match="unknown operation"):
        evaluate_expression({"op": "system", "args": ["echo pwned"]})


def test_operation_allowlist_is_closed():
    assert ALLOWED_OPERATIONS == {
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "pow",
        "percent_of",
        "equals",
    }


def test_verifier_rejects_malformed_expressions():
    with pytest.raises(JuniperConfigError, match="missing required key 'op'"):
        evaluate_expression({"args": [1, 2]})
    with pytest.raises(JuniperConfigError, match="non-empty 'args'"):
        evaluate_expression({"op": "add", "args": []})
    with pytest.raises(JuniperConfigError, match="expects 2 argument"):
        evaluate_expression({"op": "div", "args": [1, 2, 3]})
    with pytest.raises(JuniperConfigError, match="unexpected key"):
        evaluate_expression({"op": "add", "args": [1, 2], "shell": True})


def test_verifier_rejects_division_by_zero_rather_than_producing_a_number():
    with pytest.raises(JuniperConfigError, match="division by zero"):
        evaluate_expression({"op": "div", "args": [5, 0]})


def test_verifier_rejects_fractional_exponent():
    with pytest.raises(JuniperConfigError, match="must be an integer"):
        evaluate_expression({"op": "pow", "args": [4, 0.5]})


# --- Mode/schema coherence --------------------------------------------------


def test_semantic_mode_requires_null_expected_answer():
    suite = load_eval_suite()
    case = _case(suite, "amb-001")
    assert verify_case(case).verified is True
    with pytest.raises(JuniperConfigError, match="requires expected_answer to be null"):
        verify_case(replace(case, expected_answer=42))


def test_deterministic_mode_requires_non_null_expected_answer():
    suite = load_eval_suite()
    case = _case(suite, "arith-001")
    with pytest.raises(JuniperConfigError, match="requires a non-null expected_answer"):
        verify_case(replace(case, expected_answer=None))


def test_unknown_verification_mode_rejected():
    suite = load_eval_suite()
    case = _case(suite, "arith-001")
    with pytest.raises(JuniperConfigError, match="mode must be one of"):
        verify_case(replace(case, verification={"mode": "trust_me", "expression": None}))


def test_negative_tolerance_rejected():
    suite = load_eval_suite()
    case = _case(suite, "arith-001")
    with pytest.raises(JuniperConfigError, match="tolerance must be non-negative"):
        verify_case(replace(case, tolerance=-1))
