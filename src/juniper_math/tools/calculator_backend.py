"""Deterministic calculator backend: evaluate / convert / finance.

Provenance: this module is a narrow, security-hardened adaptation of the
platform-independent deterministic core of Cinqic Calculator
(https://github.com/Cinqic/Cinqic-Calculator, commit
8024cf107d6240386fa42b6c5193dd8b34848032, MIT licensed by BlessomYT) —
specifically ``src/cinqic_calculator/evaluator.py``,
``src/cinqic_calculator/conversions.py``, and
``src/cinqic_calculator/financial.py``. No GUI, app, or platform-specific
code from that repository is used. See manifests/sources.yaml and
manifests/licenses.yaml for the full provenance record, and docs/TOOLS.md
"Calculator backend relationship" for what Juniper preserved vs. changed.

Security model (evaluate): expressions are parsed with Python's ``ast``
module and walked through an explicit node-type allowlist. There is no
``eval``, ``exec``, or dynamic-import path anywhere in this module or its
callers. Every numeric literal, the overall AST node count and depth, and
every exponent/factorial operand is bounded by config/tools.yaml limits
before any computation happens, so a syntactically legal expression cannot
become a CPU/memory denial-of-service payload. See reports/PHASE3_SECURITY.md.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, DecimalException, InvalidOperation
from typing import Any, NoReturn

from juniper_math.tools.config import Limits
from juniper_math.tools.errors import ToolProtocolError

# ---------------------------------------------------------------------------
# calculator.evaluate
# ---------------------------------------------------------------------------

_UNARY_OPS = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}
_ALLOWED_AST_TYPES = (
    ast.Expression,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Call,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.FloorDiv,
    ast.UAdd,
    ast.USub,
    ast.Load,
)

_ALLOWED_FUNCTION_NAMES = frozenset(
    {"sqrt", "cbrt", "abs", "log", "ln", "sin", "cos", "tan", "factorial", "reciprocal"}
)
_ALLOWED_NAME_CONSTANTS = {"pi": math.pi, "e": math.e}


@dataclass(frozen=True)
class EvalOutcome:
    value: int | float
    exact: bool


def _domain_error(message: str) -> NoReturn:
    raise ToolProtocolError("DOMAIN_ERROR", message)


def _walk_limits(tree: ast.AST, limits: Limits) -> None:
    node_count = 0
    max_depth_seen = 0
    stack: list[tuple[ast.AST, int]] = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        node_count += 1
        max_depth_seen = max(max_depth_seen, depth)
        if node_count > limits.max_ast_nodes:
            raise ToolProtocolError(
                "RESOURCE_LIMIT", "Expression AST node count exceeds the configured limit"
            )
        if depth > limits.max_ast_depth:
            raise ToolProtocolError("RESOURCE_LIMIT", "Expression AST depth exceeds the configured limit")
        if not isinstance(node, _ALLOWED_AST_TYPES):
            raise ToolProtocolError("INVALID_ARGUMENT_VALUE", f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ToolProtocolError("INVALID_ARGUMENT_VALUE", "Only numeric literals are allowed")
            digits = len(str(abs(node.value)).replace(".", "").lstrip("0") or "0")
            if digits > limits.max_numeric_literal_digits:
                raise ToolProtocolError(
                    "RESOURCE_LIMIT", "Numeric literal exceeds the configured digit limit"
                )
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAME_CONSTANTS:
            raise ToolProtocolError("INVALID_ARGUMENT_VALUE", f"Unknown identifier: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTION_NAMES:
                raise ToolProtocolError("INVALID_ARGUMENT_VALUE", "Unsupported or disallowed function call")
            if node.keywords:
                raise ToolProtocolError("INVALID_ARGUMENT_VALUE", "Keyword arguments are not allowed")
            # node.func (the function-name Name node) is validated above by
            # membership in _ALLOWED_FUNCTION_NAMES, not the constants-only
            # Name check below — only walk the call arguments as children.
            for call_arg in node.args:
                stack.append((call_arg, depth + 1))
            continue
        for descendant in ast.iter_child_nodes(node):
            stack.append((descendant, depth + 1))


def _safe_pow(base: int | float, exponent: int | float, limits: Limits) -> tuple[int | float, bool]:
    if isinstance(exponent, float) and not exponent.is_integer():
        try:
            result = base**exponent
        except (OverflowError, ValueError) as exc:
            raise ToolProtocolError("OVERFLOW", "Result too large") from exc
        if isinstance(result, complex):
            _domain_error("Result is not a real number (negative base with fractional exponent)")
        return result, False

    exponent_int = int(exponent)
    abs_base = abs(base)
    if abs_base > 1 and abs(exponent_int) > limits.max_exponent_magnitude:
        raise ToolProtocolError("RESOURCE_LIMIT", "Exponent magnitude exceeds the configured limit")
    if abs_base > 1 and exponent_int > 0:
        estimated_bits = exponent_int * math.log2(abs_base)
        if estimated_bits > limits.max_pow_result_bits:
            raise ToolProtocolError(
                "RESOURCE_LIMIT", "Exponentiation result would exceed the configured bit-length limit"
            )

    exact = isinstance(base, int) and isinstance(exponent, int) and exponent_int >= 0
    try:
        result = base**exponent_int if isinstance(base, int) and exponent_int < 0 else base**exponent
    except OverflowError as exc:
        raise ToolProtocolError("OVERFLOW", "Result too large") from exc
    except ZeroDivisionError as exc:
        raise ToolProtocolError("DIVISION_BY_ZERO", "Zero cannot be raised to a negative power") from exc
    if isinstance(result, complex):
        _domain_error("Result is not a real number")
    if isinstance(base, int) and exponent_int < 0:
        exact = False
    return result, exact


_BIN_OP_HANDLERS: dict[type, str] = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mult",
    ast.Div: "div",
    ast.Mod: "mod",
    ast.Pow: "pow",
    ast.FloorDiv: "floordiv",
}


def _eval_binop(op_type: type, left: EvalOutcome, right: EvalOutcome, limits: Limits) -> EvalOutcome:
    kind = _BIN_OP_HANDLERS[op_type]
    a, b = left.value, right.value
    exact = left.exact and right.exact

    if kind == "add":
        return EvalOutcome(a + b, exact)
    if kind == "sub":
        return EvalOutcome(a - b, exact)
    if kind == "mult":
        return EvalOutcome(a * b, exact)
    if kind == "div":
        if b == 0:
            raise ToolProtocolError("DIVISION_BY_ZERO", "Division by zero")
        return EvalOutcome(a / b, False)
    if kind == "mod":
        if b == 0:
            raise ToolProtocolError("DIVISION_BY_ZERO", "Modulo by zero")
        return EvalOutcome(a % b, exact)
    if kind == "floordiv":
        if b == 0:
            raise ToolProtocolError("DIVISION_BY_ZERO", "Floor division by zero")
        return EvalOutcome(a // b, exact)
    if kind == "pow":
        value, pow_exact = _safe_pow(a, b, limits)
        return EvalOutcome(value, exact and pow_exact)
    raise AssertionError(f"Unhandled binary operator kind: {kind}")  # pragma: no cover


def _eval_node(node: ast.AST, limits: Limits) -> EvalOutcome:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolProtocolError("INVALID_ARGUMENT_VALUE", "Only numeric literals are allowed")
        return EvalOutcome(value, isinstance(value, int))

    if isinstance(node, ast.Name):
        return EvalOutcome(_ALLOWED_NAME_CONSTANTS[node.id], False)

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, limits)
        unary_op_type = type(node.op)
        if unary_op_type not in _UNARY_OPS:
            raise ToolProtocolError("INVALID_ARGUMENT_VALUE", "Operator not allowed")
        return EvalOutcome(_UNARY_OPS[unary_op_type](operand.value), operand.exact)

    if isinstance(node, ast.BinOp):
        binary_op_type = type(node.op)
        if binary_op_type not in _BIN_OP_HANDLERS:
            raise ToolProtocolError("INVALID_ARGUMENT_VALUE", "Operator not allowed")
        left = _eval_node(node.left, limits)
        right = _eval_node(node.right, limits)
        return _eval_binop(binary_op_type, left, right, limits)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ToolProtocolError("INVALID_ARGUMENT_VALUE", "Unsupported call target")
        name = node.func.id
        args = [_eval_node(a, limits) for a in node.args]
        return _call_function(name, args, limits)

    raise ToolProtocolError("INVALID_ARGUMENT_VALUE", f"Disallowed syntax: {type(node).__name__}")


def _require_arity(name: str, args: list[EvalOutcome], n: int) -> None:
    if len(args) != n:
        raise ToolProtocolError("INVALID_ARGUMENT_VALUE", f"{name}() takes exactly {n} argument(s)")


def _call_function(name: str, args: list[EvalOutcome], limits: Limits) -> EvalOutcome:
    if name == "factorial":
        _require_arity(name, args, 1)
        x = args[0].value
        if not isinstance(x, int) or x < 0:
            _domain_error("factorial() requires a non-negative integer")
        if x > limits.max_factorial_n:
            raise ToolProtocolError("RESOURCE_LIMIT", "factorial() argument exceeds the configured limit")
        return EvalOutcome(math.factorial(x), True)

    if name == "abs":
        _require_arity(name, args, 1)
        return EvalOutcome(abs(args[0].value), args[0].exact)

    if name == "reciprocal":
        _require_arity(name, args, 1)
        x = args[0].value
        if x == 0:
            raise ToolProtocolError("DIVISION_BY_ZERO", "Reciprocal of zero is undefined")
        return EvalOutcome(1.0 / x, False)

    if name == "sqrt":
        _require_arity(name, args, 1)
        x = args[0].value
        if x < 0:
            _domain_error("sqrt() of a negative number is not a real number")
        try:
            return EvalOutcome(math.sqrt(x), False)
        except (OverflowError, ValueError) as exc:
            raise ToolProtocolError("OVERFLOW", "Result too large") from exc

    if name == "cbrt":
        _require_arity(name, args, 1)
        x = args[0].value
        try:
            return EvalOutcome(math.copysign(abs(x) ** (1 / 3), x), False)
        except OverflowError as exc:
            raise ToolProtocolError("OVERFLOW", "Result too large") from exc

    if name in ("log", "ln"):
        _require_arity(name, args, 1)
        x = args[0].value
        if x <= 0:
            _domain_error(f"{name}() requires a positive argument")
        func = math.log10 if name == "log" else math.log
        try:
            return EvalOutcome(func(x), False)
        except (OverflowError, ValueError) as exc:
            raise ToolProtocolError("OVERFLOW", "Result too large") from exc

    if name in ("sin", "cos", "tan"):
        _require_arity(name, args, 1)
        x = args[0].value
        try:
            value = {"sin": math.sin, "cos": math.cos, "tan": math.tan}[name](x)
        except (OverflowError, ValueError) as exc:
            raise ToolProtocolError("OVERFLOW", "Result too large") from exc
        return EvalOutcome(value, False)

    raise ToolProtocolError("INVALID_ARGUMENT_VALUE", f"Function not allowed: {name}")


def evaluate_expression(expression: str, limits: Limits) -> EvalOutcome:
    """Evaluate a whitelisted arithmetic/scientific expression.

    Never uses eval/exec/compile-and-execute. See the module docstring.
    """
    if len(expression) > limits.max_expression_length:
        raise ToolProtocolError("RESOURCE_LIMIT", "Expression exceeds the configured length limit")
    if not expression.strip():
        raise ToolProtocolError("INVALID_ARGUMENT_VALUE", "Empty expression")

    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ToolProtocolError("INVALID_ARGUMENT_VALUE", "Invalid expression syntax") from exc

    _walk_limits(tree, limits)

    outcome = _eval_node(tree.body, limits)
    value = outcome.value
    if isinstance(value, complex):
        _domain_error("Result is not a real number")
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ToolProtocolError("NON_FINITE_RESULT", "Result is not finite")
    return outcome


# ---------------------------------------------------------------------------
# calculator.convert
# ---------------------------------------------------------------------------

_LENGTH = {
    "millimeter": Decimal("0.001"),
    "centimeter": Decimal("0.01"),
    "meter": Decimal("1"),
    "kilometer": Decimal("1000"),
    "inch": Decimal("0.0254"),
    "foot": Decimal("0.3048"),
    "yard": Decimal("0.9144"),
    "mile": Decimal("1609.344"),
}
_MASS = {
    "milligram": Decimal("0.001"),
    "gram": Decimal("1"),
    "kilogram": Decimal("1000"),
    "ounce": Decimal("28.349523125"),
    "pound": Decimal("453.59237"),
}
_AREA = {
    "square_meter": Decimal("1"),
    "square_kilometer": Decimal("1000000"),
    "square_foot": Decimal("0.09290304"),
    "acre": Decimal("4046.8564224"),
    "hectare": Decimal("10000"),
}
_VOLUME = {
    "milliliter": Decimal("0.001"),
    "liter": Decimal("1"),
    "cubic_meter": Decimal("1000"),
    "gallon_us": Decimal("3.785411784"),
    "quart_us": Decimal("0.946352946"),
    "cup_us": Decimal("0.2365882365"),
}
_SPEED = {
    "meters_per_second": Decimal("1"),
    "kilometers_per_hour": Decimal("1000") / Decimal("3600"),
    "miles_per_hour": Decimal("1609.344") / Decimal("3600"),
}
_TIME = {
    "second": Decimal("1"),
    "minute": Decimal("60"),
    "hour": Decimal("3600"),
    "day": Decimal("86400"),
    "week": Decimal("604800"),
}
_DATA_STORAGE = {
    "byte": Decimal("1"),
    "kilobyte": Decimal("1000"),
    "megabyte": Decimal("1000000"),
    "gigabyte": Decimal("1000000000"),
    "terabyte": Decimal("1000000000000"),
    "kibibyte": Decimal(1024),
    "mebibyte": Decimal(1024) ** 2,
    "gibibyte": Decimal(1024) ** 3,
    "tebibyte": Decimal(1024) ** 4,
}

CONVERSION_CATEGORIES: dict[str, dict[str, Decimal]] = {
    "length": _LENGTH,
    "mass": _MASS,
    "area": _AREA,
    "volume": _VOLUME,
    "speed": _SPEED,
    "time": _TIME,
    "data_storage": _DATA_STORAGE,
}
_TEMPERATURE_UNITS = frozenset({"celsius", "fahrenheit", "kelvin"})
ALL_CONVERT_CATEGORIES = frozenset(set(CONVERSION_CATEGORIES) | {"temperature"})


def _to_celsius(unit: str, value: Decimal) -> Decimal:
    if unit == "celsius":
        return value
    if unit == "fahrenheit":
        return (value - 32) * Decimal(5) / Decimal(9)
    return value - Decimal("273.15")  # kelvin


def _from_celsius(unit: str, celsius: Decimal) -> Decimal:
    if unit == "celsius":
        return celsius
    if unit == "fahrenheit":
        return celsius * Decimal(9) / Decimal(5) + 32
    return celsius + Decimal("273.15")  # kelvin


def convert_value(category: str, from_unit: str, to_unit: str, value: Decimal) -> Decimal:
    if category == "temperature":
        if from_unit not in _TEMPERATURE_UNITS:
            raise ToolProtocolError("UNSUPPORTED_UNIT", f"Unknown temperature unit: {from_unit}")
        if to_unit not in _TEMPERATURE_UNITS:
            raise ToolProtocolError("UNSUPPORTED_UNIT", f"Unknown temperature unit: {to_unit}")
        return _from_celsius(to_unit, _to_celsius(from_unit, value))

    table = CONVERSION_CATEGORIES.get(category)
    if table is None:
        raise ToolProtocolError("INVALID_ARGUMENT_VALUE", f"Unknown category: {category}")
    if from_unit not in table:
        raise ToolProtocolError("UNSUPPORTED_UNIT", f"Unknown unit {from_unit!r} in category {category!r}")
    if to_unit not in table:
        raise ToolProtocolError("UNSUPPORTED_UNIT", f"Unknown unit {to_unit!r} in category {category!r}")

    base_value = value * table[from_unit]
    return base_value / table[to_unit]


# ---------------------------------------------------------------------------
# calculator.finance
# ---------------------------------------------------------------------------

_TWO_PLACES = Decimal("0.01")


def _round_currency(value: Decimal) -> Decimal:
    return value.quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class FinanceOperation:
    required: tuple[str, ...]
    optional: dict[str, Decimal]


FINANCE_OPERATIONS: dict[str, FinanceOperation] = {
    "percentage_of": FinanceOperation(("number", "percent"), {}),
    "percentage_increase": FinanceOperation(("number", "percent"), {}),
    "percentage_decrease": FinanceOperation(("number", "percent"), {}),
    "percentage_difference": FinanceOperation(("old_value", "new_value"), {}),
    "discount": FinanceOperation(("price", "percent"), {}),
    "sales_tax": FinanceOperation(("price", "tax_rate_percent"), {}),
    "final_price": FinanceOperation(
        ("price",), {"discount_percent": Decimal(0), "tax_rate_percent": Decimal(0)}
    ),
    "tip": FinanceOperation(("bill_total", "tip_percent"), {}),
    "split_bill": FinanceOperation(("bill_total", "num_people"), {"tip_percent": Decimal(0)}),
    "simple_interest": FinanceOperation(("principal", "annual_rate_percent", "years"), {}),
    "compound_interest": FinanceOperation(
        ("principal", "annual_rate_percent", "years"), {"compounds_per_year": Decimal(1)}
    ),
}


def compute_finance(operation: str, args: dict[str, Decimal]) -> Decimal:
    """All currency-oriented arithmetic uses Decimal end-to-end (no float
    round-trip) so that ROUND_HALF_UP is applied to the true computed value.

    Wraps the whole dispatch in a Decimal-precision guard: an input like
    ``percentage_of(number=1e30, percent=50)`` can exceed the default
    28-significant-digit Decimal context precision during quantize(), which
    otherwise raises decimal.InvalidOperation (an ArithmeticError) — that
    must never surface as a generic INTERNAL_ERROR when it is really an
    over-large *input*, not an internal bug. See reports/PHASE3_SELF_REVIEW.md.
    """
    try:
        return _compute_finance_dispatch(operation, args)
    except DecimalException as exc:
        raise ToolProtocolError(
            "RESOURCE_LIMIT", "Finance result exceeds representable decimal precision"
        ) from exc


def _compute_finance_dispatch(operation: str, args: dict[str, Decimal]) -> Decimal:
    if operation == "percentage_of":
        return float_safe_round(args["number"] * args["percent"] / Decimal(100), currency=False)
    if operation == "percentage_increase":
        factor = Decimal(1) + args["percent"] / Decimal(100)
        return float_safe_round(args["number"] * factor, currency=False)
    if operation == "percentage_decrease":
        factor = Decimal(1) - args["percent"] / Decimal(100)
        return float_safe_round(args["number"] * factor, currency=False)
    if operation == "percentage_difference":
        old = args["old_value"]
        if old == 0:
            raise ToolProtocolError("DIVISION_BY_ZERO", "Cannot compute percentage difference from zero")
        return float_safe_round((args["new_value"] - old) / abs(old) * Decimal(100), currency=False)
    if operation == "discount":
        return _round_currency(args["price"] * (Decimal(1) - args["percent"] / Decimal(100)))
    if operation == "sales_tax":
        return _round_currency(args["price"] * args["tax_rate_percent"] / Decimal(100))
    if operation == "final_price":
        discounted = args["price"] * (Decimal(1) - args["discount_percent"] / Decimal(100))
        taxed = discounted * (Decimal(1) + args["tax_rate_percent"] / Decimal(100))
        return _round_currency(taxed)
    if operation == "tip":
        return _round_currency(args["bill_total"] * args["tip_percent"] / Decimal(100))
    if operation == "split_bill":
        num_people = args["num_people"]
        if num_people <= 0:
            raise ToolProtocolError("INVALID_ARGUMENT_VALUE", "num_people must be positive")
        total_with_tip = args["bill_total"] * (Decimal(1) + args["tip_percent"] / Decimal(100))
        return _round_currency(total_with_tip / num_people)
    if operation == "simple_interest":
        interest = args["principal"] * args["annual_rate_percent"] / Decimal(100) * args["years"]
        return _round_currency(interest)
    if operation == "compound_interest":
        compounds_per_year = args["compounds_per_year"]
        if compounds_per_year <= 0:
            raise ToolProtocolError("INVALID_ARGUMENT_VALUE", "compounds_per_year must be positive")
        rate = args["annual_rate_percent"] / Decimal(100)
        growth_base = Decimal(1) + rate / compounds_per_year
        exponent = compounds_per_year * args["years"]
        try:
            growth = growth_base**exponent
        except (DecimalException, OverflowError) as exc:
            raise ToolProtocolError("OVERFLOW", "Compound interest growth factor is too large") from exc
        return _round_currency(args["principal"] * growth)
    raise ToolProtocolError("UNSUPPORTED_OPERATION", f"Unsupported finance operation: {operation}")


def float_safe_round(value: Decimal, *, currency: bool) -> Decimal:
    if currency:
        return _round_currency(value)
    return value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


def to_decimal(value: Any, field_name: str) -> Decimal:
    """Convert a validated JSON number (int or float, never bool/NaN/Inf) to Decimal.

    Uses str(value) as the intermediary (never Decimal(float) directly) so
    the Decimal reflects the JSON-visible digits rather than float's binary
    approximation artifacts.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolProtocolError("INVALID_ARGUMENT_TYPE", f"{field_name} must be a number")
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ToolProtocolError("INVALID_ARGUMENT_VALUE", f"{field_name} must be finite")
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ToolProtocolError("INVALID_ARGUMENT_VALUE", f"{field_name} is not a valid number") from exc
