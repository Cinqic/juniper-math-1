from __future__ import annotations

import random
from decimal import ROUND_HALF_UP, Decimal, getcontext

import pytest

from juniper_math.tools.calculator_backend import compute_finance
from juniper_math.tools.errors import ToolProtocolError


def d(value):
    return Decimal(str(value))


def test_tip_known_answer():
    assert compute_finance("tip", {"bill_total": d("42.50"), "tip_percent": d(20)}) == Decimal("8.50")


def test_sales_tax_known_answer():
    assert compute_finance("sales_tax", {"price": d(100), "tax_rate_percent": d("8.25")}) == Decimal("8.25")


def test_discount_known_answer():
    assert compute_finance("discount", {"price": d(100), "percent": d(25)}) == Decimal("75.00")


def test_final_price_known_answer():
    result = compute_finance(
        "final_price", {"price": d(100), "discount_percent": d(10), "tax_rate_percent": d(5)}
    )
    # Independent oracle: 100 * 0.9 = 90; 90 * 1.05 = 94.5
    assert result == Decimal("94.50")


def test_final_price_uses_defaults_when_optional_args_omitted():
    result = compute_finance(
        "final_price", {"price": d(50), "discount_percent": d(0), "tax_rate_percent": d(0)}
    )
    assert result == Decimal("50.00")


def test_split_bill_known_answer():
    result = compute_finance("split_bill", {"bill_total": d(100), "num_people": d(4), "tip_percent": d(0)})
    assert result == Decimal("25.00")


def test_split_bill_rejects_zero_people():
    with pytest.raises(ToolProtocolError) as exc:
        compute_finance("split_bill", {"bill_total": d(100), "num_people": d(0), "tip_percent": d(0)})
    assert exc.value.code == "INVALID_ARGUMENT_VALUE"


def test_simple_interest_known_answer():
    # Independent oracle: I = P * r * t = 1000 * 0.05 * 2 = 100
    result = compute_finance(
        "simple_interest", {"principal": d(1000), "annual_rate_percent": d(5), "years": d(2)}
    )
    assert result == Decimal("100.00")


def test_compound_interest_known_answer():
    # Independent oracle: A = P(1 + r/n)^(nt); P=1000, r=0.05, n=1, t=1 -> 1050.00
    result = compute_finance(
        "compound_interest",
        {"principal": d(1000), "annual_rate_percent": d(5), "years": d(1), "compounds_per_year": d(1)},
    )
    assert result == Decimal("1050.00")


def test_compound_interest_rejects_zero_compounds():
    with pytest.raises(ToolProtocolError) as exc:
        compute_finance(
            "compound_interest",
            {"principal": d(1000), "annual_rate_percent": d(5), "years": d(1), "compounds_per_year": d(0)},
        )
    assert exc.value.code == "INVALID_ARGUMENT_VALUE"


def test_percentage_difference_rejects_zero_old_value():
    with pytest.raises(ToolProtocolError) as exc:
        compute_finance("percentage_difference", {"old_value": d(0), "new_value": d(10)})
    assert exc.value.code == "DIVISION_BY_ZERO"


def test_percentage_of_known_answer():
    assert compute_finance("percentage_of", {"number": d(200), "percent": d(15)}) == Decimal("30.000000")


def test_extremely_large_percentage_of_input_is_resource_limit_not_internal_error():
    # decimal.InvalidOperation (default 28-sig-digit context precision
    # exceeded during quantize) must surface as RESOURCE_LIMIT, never a
    # generic INTERNAL_ERROR — see reports/PHASE3_SELF_REVIEW.md.
    with pytest.raises(ToolProtocolError) as exc:
        compute_finance("percentage_of", {"number": d("1e30"), "percent": d(50)})
    assert exc.value.code == "RESOURCE_LIMIT"


def test_extreme_compound_interest_inputs_are_overflow_not_internal_error():
    with pytest.raises(ToolProtocolError) as exc:
        compute_finance(
            "compound_interest",
            {
                "principal": d(1000),
                "annual_rate_percent": d(99999),
                "years": d(9999),
                "compounds_per_year": d(365),
            },
        )
    assert exc.value.code == "OVERFLOW"


def test_unsupported_operation():
    with pytest.raises(ToolProtocolError) as exc:
        compute_finance("solve_equation", {})
    assert exc.value.code == "UNSUPPORTED_OPERATION"


# ---------------------------------------------------------------------------
# ROUND_HALF_UP boundary tests — the classic x.xx5 case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "price,expected",
    [
        # 100 * 0.125 = 12.5 -> not a boundary case by itself; construct exact halves below.
        (Decimal("0.125"), Decimal("0.13")),
        (Decimal("0.135"), Decimal("0.14")),
        (Decimal("0.145"), Decimal("0.15")),
    ],
)
def test_round_half_up_boundary(price, expected):
    # sales_tax with tax_rate_percent=100 returns price unchanged through the
    # Decimal quantize path, isolating the rounding behavior under test.
    result = compute_finance("sales_tax", {"price": price, "tax_rate_percent": d(100)})
    assert result == expected
    # Cross-check against a hand-rolled Decimal oracle, not the production function twice.
    assert result == price.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Property test: random normal-range finance values vs. independent Decimal oracle
# ---------------------------------------------------------------------------


def test_tip_matches_independent_decimal_oracle():
    rng = random.Random(20260822)
    for _ in range(200):
        bill = d(round(rng.uniform(1, 500), 2))
        percent = d(round(rng.uniform(0, 50), 2))
        expected = (bill * percent / Decimal(100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        assert compute_finance("tip", {"bill_total": bill, "tip_percent": percent}) == expected


def test_simple_interest_matches_independent_decimal_oracle():
    rng = random.Random(20260822 + 1)
    for _ in range(200):
        principal = d(round(rng.uniform(1, 100000), 2))
        rate = d(round(rng.uniform(0, 20), 3))
        years = d(round(rng.uniform(0, 30), 2))
        expected = (principal * rate / Decimal(100) * years).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        actual = compute_finance(
            "simple_interest", {"principal": principal, "annual_rate_percent": rate, "years": years}
        )
        assert actual == expected


def test_finance_is_independent_of_process_global_decimal_context():
    previous_precision = getcontext().prec
    args = {"number": Decimal(1), "percent": Decimal(1)}
    try:
        getcontext().prec = 8
        low_precision = compute_finance("percentage_of", args)
        getcontext().prec = 100
        high_precision = compute_finance("percentage_of", args)
    finally:
        getcontext().prec = previous_precision
    assert low_precision == high_precision == Decimal("0.010000")
