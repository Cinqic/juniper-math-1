from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest

from juniper_math.tools.calculator_backend import evaluate_expression
from juniper_math.tools.config import load_tools_config
from juniper_math.tools.errors import ToolProtocolError

LIMITS = load_tools_config().limits


def ev(expression: str):
    return evaluate_expression(expression, LIMITS)


# ---------------------------------------------------------------------------
# Known-answer / independent-oracle tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2+2", 4),
        ("84317 * 9926", 84317 * 9926),
        ("10 - 3", 7),
        ("2 ** 10", 1024),
        ("7 // 2", 3),
        ("-5", -5),
        ("+5", 5),
        ("abs(-7)", 7),
        ("factorial(10)", 3628800),
    ],
)
def test_exact_integer_results(expression, expected):
    outcome = ev(expression)
    assert outcome.value == expected
    assert outcome.exact is True
    assert isinstance(outcome.value, int)


def test_division_is_never_marked_exact():
    outcome = ev("10 / 2")
    assert outcome.value == 5.0
    assert outcome.exact is False


def test_modulo_uses_python_sign_of_divisor_semantics():
    # -7 % 3 == 2 in Python (sign follows divisor), independent of this module.
    outcome = ev("-7 % 3")
    assert outcome.value == (-7) % 3 == 2
    assert outcome.exact is True


def test_sqrt_matches_independent_oracle():
    outcome = ev("sqrt(2)")
    assert outcome.value == pytest.approx(math.sqrt(2))
    assert outcome.exact is False


def test_ln_and_log_use_documented_bases():
    assert ev("log(100)").value == pytest.approx(2.0)  # base 10
    assert ev("ln(1)").value == pytest.approx(0.0)  # natural log


def test_trig_uses_radians():
    assert ev("sin(0)").value == pytest.approx(0.0)
    assert ev("cos(0)").value == pytest.approx(1.0)


def test_constants_match_math_module():
    assert ev("pi").value == pytest.approx(math.pi)
    assert ev("e").value == pytest.approx(math.e)


def test_cbrt_matches_independent_oracle():
    assert ev("cbrt(27)").value == pytest.approx(3.0)
    assert ev("cbrt(-27)").value == pytest.approx(-3.0)


def test_reciprocal_matches_fraction_oracle():
    assert ev("reciprocal(4)").value == pytest.approx(float(Fraction(1, 4)))


# ---------------------------------------------------------------------------
# Errors / domain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("expression", ["1/0", "5 % 0", "5 // 0"])
def test_division_by_zero_variants(expression):
    with pytest.raises(ToolProtocolError) as exc:
        ev(expression)
    assert exc.value.code == "DIVISION_BY_ZERO"


@pytest.mark.parametrize(
    "expression", ["sqrt(-1)", "log(0)", "log(-5)", "ln(0)", "factorial(-1)", "factorial(1.5)"]
)
def test_domain_errors(expression):
    with pytest.raises(ToolProtocolError) as exc:
        ev(expression)
    assert exc.value.code == "DOMAIN_ERROR"


def test_empty_expression_is_invalid():
    with pytest.raises(ToolProtocolError) as exc:
        ev("   ")
    assert exc.value.code == "INVALID_ARGUMENT_VALUE"


def test_syntax_error_is_invalid():
    with pytest.raises(ToolProtocolError) as exc:
        ev("2 +* 2")
    assert exc.value.code == "INVALID_ARGUMENT_VALUE"


# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------


def test_deep_nested_binops_hit_ast_limit():
    expr = "1"
    for _ in range(500):
        expr = f"({expr}+1)"
    with pytest.raises(ToolProtocolError) as exc:
        ev(expr)
    assert exc.value.code == "RESOURCE_LIMIT"


def test_huge_exponent_rejected_without_computing():
    with pytest.raises(ToolProtocolError) as exc:
        ev("2 ** 1000000000")
    assert exc.value.code == "RESOURCE_LIMIT"


def test_huge_base_and_exponent_rejected():
    with pytest.raises(ToolProtocolError) as exc:
        ev("999999 ** 999999")
    assert exc.value.code == "RESOURCE_LIMIT"


def test_huge_factorial_rejected():
    with pytest.raises(ToolProtocolError) as exc:
        ev("factorial(100000000)")
    assert exc.value.code == "RESOURCE_LIMIT"


def test_oversized_numeric_literal_rejected():
    with pytest.raises(ToolProtocolError) as exc:
        ev("9" * 100)
    assert exc.value.code == "RESOURCE_LIMIT"


def test_expression_length_limit():
    with pytest.raises(ToolProtocolError) as exc:
        ev("1+" * 10000 + "1")
    assert exc.value.code == "RESOURCE_LIMIT"


def test_within_limit_exponent_still_works():
    outcome = ev("2 ** 100")
    assert outcome.value == 2**100
    assert outcome.exact is True


def test_factorial_at_limit_boundary_works():
    outcome = ev("factorial(500)")
    assert outcome.value == math.factorial(500)


# ---------------------------------------------------------------------------
# Property test: random safe integer arithmetic vs. Python int oracle
# ---------------------------------------------------------------------------


def test_random_integer_arithmetic_matches_python_oracle():
    rng = random.Random(20260822)
    ops = ["+", "-", "*"]
    for _ in range(200):
        a = rng.randint(-10_000, 10_000)
        b = rng.randint(-10_000, 10_000)
        op = rng.choice(ops)
        expr = f"{a} {op} {b}"
        expected = {"+": a + b, "-": a - b, "*": a * b}[op]
        assert ev(expr).value == expected
