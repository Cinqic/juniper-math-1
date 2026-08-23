"""Closed-allowlist ground-truth verification for Phase 4 dataset generators.

Deliberately separate from ``juniper_math.verification`` (the frozen Phase 0
evaluation-suite ground-truth checker): that module's allowlist was scoped
narrowly to the 22-case Phase 0 suite and is frozen infrastructure this phase
must not silently expand. This module serves the dataset's much broader
generator surface (estimation/rounding, comparisons, sqrt) with its own
closed allowlist, following the identical safety pattern: no ``eval``,
``exec``, or ``compile`` — only explicit JSON-shaped nodes dispatched through
a fixed operation table. Unknown operations raise rather than being ignored.

Every deterministic example's ``expected_answer`` MUST be produced by calling
:func:`evaluate_expression` on its own ``verification.expression`` — a
generator that hardcodes an answer without recomputing it here reproduces
exactly the class of defect Opus 5 found in Phase 0 case ``tool-001``
(reports/OPUS5_PHASE0_REVIEW.md, F-01).
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

from juniper_math.errors import JuniperConfigError

Numeric = Fraction | bool


def to_exact(value: Any, context: str) -> Fraction:
    if isinstance(value, bool):
        raise JuniperConfigError(f"{context}: boolean is not a numeric value.")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(str(value))
    if isinstance(value, str):
        try:
            return Fraction(value)
        except (ValueError, ZeroDivisionError) as exc:
            raise JuniperConfigError(f"{context}: cannot interpret {value!r} as an exact number.") from exc
    raise JuniperConfigError(f"{context}: unsupported numeric type {type(value).__name__}.")


def _add(*args: Fraction) -> Fraction:
    total = Fraction(0)
    for value in args:
        total += value
    return total


def _sub(left: Fraction, right: Fraction) -> Fraction:
    return left - right


def _mul(*args: Fraction) -> Fraction:
    product = Fraction(1)
    for value in args:
        product *= value
    return product


def _div(numerator: Fraction, denominator: Fraction) -> Fraction:
    if denominator == 0:
        raise JuniperConfigError(
            "verify: division by zero. An example whose mathematics is undefined must use "
            "mode 'semantic', not a deterministic expression."
        )
    return numerator / denominator


def _neg(value: Fraction) -> Fraction:
    return -value


def _abs(value: Fraction) -> Fraction:
    return abs(value)


def _pow(base: Fraction, exponent: Fraction) -> Fraction:
    if exponent.denominator != 1:
        raise JuniperConfigError(f"verify: 'pow' exponent must be an integer, got {exponent}.")
    exp = int(exponent)
    if base == 0 and exp < 0:
        raise JuniperConfigError("verify: 0 raised to a negative power is undefined; use mode 'semantic'.")
    return base**exp


def _percent_of(percent: Fraction, whole: Fraction) -> Fraction:
    return (percent / 100) * whole


def _max(*args: Fraction) -> Fraction:
    return max(args)


def _min(*args: Fraction) -> Fraction:
    return min(args)


def _ratio(a: Fraction, b: Fraction) -> Fraction:
    if b == 0:
        raise JuniperConfigError("verify: ratio denominator is zero; use mode 'semantic'.")
    return a / b


def _round_half_up(value: Fraction, places: Fraction) -> Fraction:
    if places.denominator != 1 or places < 0:
        raise JuniperConfigError("verify: 'round' places must be a non-negative integer.")
    quantum = Decimal(1).scaleb(-int(places))
    try:
        rounded = Decimal(value.numerator) / Decimal(value.denominator)
        rounded = rounded.quantize(quantum, rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:  # pragma: no cover - defensive
        raise JuniperConfigError(f"verify: could not round {value}") from exc
    return Fraction(rounded)


def _sqrt(value: Fraction, precision_places: Fraction) -> Fraction:
    if value < 0:
        raise JuniperConfigError("verify: sqrt of a negative number is undefined; use mode 'semantic'.")
    if precision_places.denominator != 1 or precision_places < 0:
        raise JuniperConfigError("verify: 'sqrt' precision_places must be a non-negative integer.")
    quantum = Decimal(1).scaleb(-int(precision_places))
    d = Decimal(value.numerator) / Decimal(value.denominator)
    rounded = d.sqrt().quantize(quantum, rounding=ROUND_HALF_UP)
    return Fraction(rounded)


def _compare(op: str):  # noqa: ANN201
    ops = {
        "lt": lambda a, b: a < b,
        "le": lambda a, b: a <= b,
        "gt": lambda a, b: a > b,
        "ge": lambda a, b: a >= b,
        "eq": lambda a, b: a == b,
    }
    return ops[op]


def _lt(a: Fraction, b: Fraction) -> bool:
    return _compare("lt")(a, b)


def _le(a: Fraction, b: Fraction) -> bool:
    return _compare("le")(a, b)


def _gt(a: Fraction, b: Fraction) -> bool:
    return _compare("gt")(a, b)


def _ge(a: Fraction, b: Fraction) -> bool:
    return _compare("ge")(a, b)


def _equals(left: Fraction, right: Fraction) -> bool:
    return left == right


_OPERATIONS: dict[str, tuple[Any, int | None]] = {
    "add": (_add, None),
    "sub": (_sub, 2),
    "mul": (_mul, None),
    "div": (_div, 2),
    "neg": (_neg, 1),
    "abs": (_abs, 1),
    "pow": (_pow, 2),
    "percent_of": (_percent_of, 2),
    "max": (_max, None),
    "min": (_min, None),
    "ratio": (_ratio, 2),
    "round": (_round_half_up, 2),
    "sqrt": (_sqrt, 2),
    "lt": (_lt, 2),
    "le": (_le, 2),
    "gt": (_gt, 2),
    "ge": (_ge, 2),
    "equals": (_equals, 2),
}

ALLOWED_OPERATIONS = frozenset(_OPERATIONS)


def evaluate_expression(node: Any, context: str = "verify") -> Fraction | bool:
    if not isinstance(node, dict):
        return to_exact(node, context)

    if "op" not in node:
        raise JuniperConfigError(f"{context}: expression object is missing required key 'op'.")
    op_name = node["op"]
    if not isinstance(op_name, str) or op_name not in _OPERATIONS:
        raise JuniperConfigError(
            f"{context}: unknown operation {op_name!r}. Allowed operations: {sorted(ALLOWED_OPERATIONS)}."
        )
    unexpected = set(node) - {"op", "args"}
    if unexpected:
        raise JuniperConfigError(f"{context}: unexpected key(s) in expression: {sorted(unexpected)}.")

    args = node.get("args")
    if not isinstance(args, list) or not args:
        raise JuniperConfigError(f"{context}: operation {op_name!r} requires a non-empty 'args' list.")

    func, arity = _OPERATIONS[op_name]
    if arity is not None and len(args) != arity:
        raise JuniperConfigError(
            f"{context}: operation {op_name!r} expects {arity} argument(s), got {len(args)}."
        )

    evaluated: list[Fraction] = []
    for index, arg in enumerate(args):
        value = evaluate_expression(arg, f"{context}.{op_name}[{index}]")
        if isinstance(value, bool):
            raise JuniperConfigError(f"{context}: operation {op_name!r} cannot consume a boolean operand.")
        evaluated.append(value)

    return func(*evaluated)  # type: ignore[no-any-return]


def verify_deterministic(
    expression: Any, expected_answer: Any, tolerance: Any, context: str
) -> tuple[bool, str]:
    """Recompute ``expression`` and compare against ``expected_answer`` within ``tolerance``."""
    computed = evaluate_expression(expression, context)

    if isinstance(computed, bool) or isinstance(expected_answer, bool):
        if not isinstance(computed, bool) or not isinstance(expected_answer, bool):
            return (
                False,
                f"boolean/numeric type mismatch: computed {computed!r} vs expected {expected_answer!r}",
            )
        ok = computed == expected_answer
        return ok, f"computed {computed} {'==' if ok else '!='} expected {expected_answer}"

    expected = to_exact(expected_answer, f"{context} expected_answer")
    tol = Fraction(0) if tolerance is None else to_exact(tolerance, f"{context} tolerance")
    if tol < 0:
        raise JuniperConfigError(f"{context}: tolerance must be non-negative, got {tolerance!r}.")
    diff = abs(computed - expected)
    ok = diff <= tol
    detail = f"computed {computed} vs expected {expected} (|Δ|={diff}, tolerance={tol})"
    return ok, detail
