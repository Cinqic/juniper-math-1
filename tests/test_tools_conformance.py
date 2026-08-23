"""Conformance suite against the pinned Cinqic Calculator upstream commit
(8024cf107d6240386fa42b6c5193dd8b34848032). Expected values here are taken
from independently re-deriving what upstream's
evaluator.py/conversions.py/financial.py compute for each case (not by
importing or cloning the upstream package at test time — the upstream
commit is pinned and vendored, not a moving dependency; see
manifests/sources.yaml). Where Juniper intentionally diverges, the test
documents why instead of silently asserting upstream's number.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from juniper_math.tools.calculator_backend import compute_finance, convert_value, evaluate_expression
from juniper_math.tools.config import load_tools_config

LIMITS = load_tools_config().limits


def d(value):
    return Decimal(str(value))


# ---------------------------------------------------------------------------
# calculator.evaluate: same behavior as upstream for pure arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("84317 * 9926", 836930542),
        ("100 / 4", 25.0),
        ("2 ** 8", 256),
        ("7 // 2", 3),
    ],
)
def test_evaluate_matches_upstream_for_pure_arithmetic(expression, expected):
    assert evaluate_expression(expression, LIMITS).value == expected


def test_evaluate_modulo_intentionally_diverges_from_upstream_fmod():
    # Upstream uses math.fmod(-7, 3) == -1.0 (sign follows dividend, always
    # float). Juniper uses Python's % (sign follows divisor, preserves int
    # exactness) — see config/tools.yaml evaluate.modulo_semantics and
    # docs/TOOLS.md for the documented rationale.
    import math

    upstream_equivalent = math.fmod(-7, 3)
    juniper_result = evaluate_expression("-7 % 3", LIMITS).value
    assert upstream_equivalent == -1.0
    assert juniper_result == 2
    assert juniper_result != upstream_equivalent


def test_evaluate_preserves_large_integer_precision_unlike_upstream_float_cast():
    # Upstream's evaluate() always returns float(result), which for a very
    # large product silently rounds via IEEE-754. Juniper preserves exact
    # Python int precision instead — an intentional hardening, not a bug.
    expression = "123456789123456789 * 987654321987654321"
    outcome = evaluate_expression(expression, LIMITS)
    exact_expected = 123456789123456789 * 987654321987654321
    assert outcome.value == exact_expected
    assert outcome.exact is True
    assert float(exact_expected) != exact_expected  # demonstrates the float cast would have lost precision


# ---------------------------------------------------------------------------
# calculator.convert: byte-identical to upstream (same conversion tables)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category,from_unit,to_unit,value,expected",
    [
        ("length", "mile", "meter", 1, Decimal("1609.344")),
        ("mass", "ounce", "gram", 1, Decimal("28.349523125")),
        ("data_storage", "mebibyte", "byte", 1, Decimal(1024) ** 2),
        ("temperature", "fahrenheit", "celsius", 32, Decimal(0)),
    ],
)
def test_convert_matches_upstream(category, from_unit, to_unit, value, expected):
    assert convert_value(category, from_unit, to_unit, d(value)) == expected


# ---------------------------------------------------------------------------
# calculator.finance: same rounding policy (ROUND_HALF_UP), same formulas
# ---------------------------------------------------------------------------


def test_finance_tip_matches_upstream():
    assert compute_finance("tip", {"bill_total": d("42.50"), "tip_percent": d(20)}) == Decimal("8.50")


def test_finance_compound_interest_matches_upstream():
    result = compute_finance(
        "compound_interest",
        {"principal": d(1000), "annual_rate_percent": d(5), "years": d(10), "compounds_per_year": d(12)},
    )
    # Independent oracle via the standard compound interest formula.
    from decimal import ROUND_HALF_UP

    rate = Decimal(5) / Decimal(100)
    n = Decimal(12)
    growth = (Decimal(1) + rate / n) ** (n * Decimal(10))
    expected = (Decimal(1000) * growth).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    assert result == expected
