from __future__ import annotations

import random
from decimal import Decimal

import pytest

from juniper_math.tools.calculator_backend import convert_value
from juniper_math.tools.errors import ToolProtocolError


def d(value):
    return Decimal(str(value))


# ---------------------------------------------------------------------------
# Known-answer tests — independently derived, not round-tripped through the
# same table twice.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category,from_unit,to_unit,value,expected",
    [
        ("length", "inch", "meter", 1, Decimal("0.0254")),
        ("length", "foot", "meter", 1, Decimal("0.3048")),
        ("length", "mile", "meter", 1, Decimal("1609.344")),
        ("mass", "pound", "gram", 1, Decimal("453.59237")),
        ("data_storage", "kibibyte", "byte", 1, Decimal(1024)),
        ("data_storage", "kilobyte", "byte", 1, Decimal(1000)),
    ],
)
def test_known_conversions(category, from_unit, to_unit, value, expected):
    assert convert_value(category, from_unit, to_unit, d(value)) == expected


def test_temperature_celsius_to_fahrenheit():
    assert convert_value("temperature", "celsius", "fahrenheit", d(0)) == 32


def test_temperature_kelvin_to_celsius():
    assert convert_value("temperature", "kelvin", "celsius", d("273.15")) == 0


def test_decimal_and_binary_data_units_are_distinct():
    kb = convert_value("data_storage", "kilobyte", "byte", d(1))
    kib = convert_value("data_storage", "kibibyte", "byte", d(1))
    assert kb == 1000
    assert kib == 1024
    assert kb != kib


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_unknown_category():
    with pytest.raises(ToolProtocolError) as exc:
        convert_value("volume_of_regret", "liter", "liter", d(1))
    assert exc.value.code == "INVALID_ARGUMENT_VALUE"


def test_unknown_unit_in_known_category():
    with pytest.raises(ToolProtocolError) as exc:
        convert_value("length", "smoot", "meter", d(1))
    assert exc.value.code == "UNSUPPORTED_UNIT"


def test_unknown_temperature_unit():
    with pytest.raises(ToolProtocolError) as exc:
        convert_value("temperature", "rankine", "celsius", d(1))
    assert exc.value.code == "UNSUPPORTED_UNIT"


def test_below_absolute_zero_converts_without_error():
    # Documented policy (config/tools.yaml convert.temperature_policy): this
    # is a mathematical converter, not a physical-plausibility validator.
    result = convert_value("temperature", "celsius", "kelvin", d(-300))
    assert result == Decimal("-26.85")


# ---------------------------------------------------------------------------
# Round-trip property test with a fixed seed
# ---------------------------------------------------------------------------

_CATEGORY_UNITS = {
    "length": ["millimeter", "centimeter", "meter", "kilometer", "inch", "foot", "yard", "mile"],
    "mass": ["milligram", "gram", "kilogram", "ounce", "pound"],
    "area": ["square_meter", "square_kilometer", "square_foot", "acre", "hectare"],
    "volume": ["milliliter", "liter", "cubic_meter", "gallon_us", "quart_us", "cup_us"],
    "speed": ["meters_per_second", "kilometers_per_hour", "miles_per_hour"],
    "time": ["second", "minute", "hour", "day", "week"],
    "data_storage": ["byte", "kilobyte", "megabyte", "gigabyte", "kibibyte", "mebibyte", "gibibyte"],
    "temperature": ["celsius", "fahrenheit", "kelvin"],
}


def test_round_trip_deterministic_random_cases():
    rng = random.Random(20260822)
    for _ in range(300):
        category = rng.choice(list(_CATEGORY_UNITS))
        unit_a, unit_b = rng.sample(_CATEGORY_UNITS[category], 2)
        original = d(rng.uniform(-1000, 1000) if category == "temperature" else rng.uniform(0.001, 100000))
        forward = convert_value(category, unit_a, unit_b, original)
        back = convert_value(category, unit_b, unit_a, forward)
        # Decimal division introduces bounded rounding; tolerance is generous
        # relative to the 28-digit default Decimal context precision.
        assert abs(back - original) <= abs(original) * Decimal("1e-20") + Decimal("1e-15")
