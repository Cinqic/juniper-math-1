"""Deterministic ground-truth verification for the frozen evaluation suite.

Why this module exists
----------------------
The Phase 0 candidate reviewed at commit ``b39e600`` declared that every
deterministically-checkable evaluation answer had been "hand-verified".
That claim was false: case ``tool-001`` recorded ``84317 * 9926`` as
``837042742`` when the correct product is ``836930542``. Nothing in the
repository was capable of detecting it, because ``evals validate`` only
checked *schema*. See ``reports/OPUS5_PHASE0_REVIEW.md`` (F-01, F-02).

This module closes that gap. Each evaluation case now carries an explicit,
structured ``verification`` block, and this evaluator recomputes the answer
from that structure and compares it against the recorded
``expected_answer``.

Safety
------
Evaluation prompts are natural-language text and are NEVER executed. This
evaluator does not use ``eval``, ``exec``, ``compile``, ``pickle``, or any
form of expression parsing. It walks an explicit JSON structure and
dispatches on a closed allowlist of arithmetic operations (``_OPERATIONS``).
Any unknown operation, malformed node, or unexpected type raises
``JuniperConfigError`` rather than being coerced or ignored.

Exactness
---------
All arithmetic is performed with ``fractions.Fraction``, so decimals,
fractions, and percentages are exact — ``0.1 + 0.2`` is exactly ``0.3``
here, not ``0.30000000000000004``. Floats appearing in the suite are
converted via their decimal string representation to preserve authored
intent.

Scope
-----
This is evaluation-integrity infrastructure, deliberately minimal. It is NOT
the Phase 3 deterministic tool runtime ("Cinqic Calculator") and must not
grow into a general-purpose symbolic mathematics engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from juniper_math.errors import JuniperConfigError

# Verification modes.
MODE_DETERMINISTIC = "deterministic"
MODE_SEMANTIC = "semantic"
VALID_VERIFICATION_MODES = {MODE_DETERMINISTIC, MODE_SEMANTIC}


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
            "verification: division by zero. A case whose mathematics is undefined must use "
            f"mode {MODE_SEMANTIC!r}, not a deterministic expression."
        )
    return numerator / denominator


def _neg(value: Fraction) -> Fraction:
    return -value


def _pow(base: Fraction, exponent: Fraction) -> Fraction:
    if exponent.denominator != 1:
        raise JuniperConfigError(
            f"verification: 'pow' exponent must be an integer, got {exponent}. "
            "Fractional powers are outside the Phase 0 deterministic allowlist."
        )
    return base ** int(exponent)


def _percent_of(percent: Fraction, whole: Fraction) -> Fraction:
    """``percent_of(15, 240)`` -> 15% of 240 -> 36."""
    return (percent / 100) * whole


def _equals(left: Fraction, right: Fraction) -> bool:
    return left == right


# Closed allowlist. Adding an entry here is a deliberate schema decision.
# Each value is (callable, expected_arity or None for variadic).
_OPERATIONS: dict[str, tuple[Any, int | None]] = {
    "add": (_add, None),
    "sub": (_sub, 2),
    "mul": (_mul, None),
    "div": (_div, 2),
    "neg": (_neg, 1),
    "pow": (_pow, 2),
    "percent_of": (_percent_of, 2),
    "equals": (_equals, 2),
}

ALLOWED_OPERATIONS = frozenset(_OPERATIONS)


def to_exact(value: Any, context: str) -> Fraction:
    """Convert a JSON scalar to an exact Fraction.

    Floats are routed through their decimal string form so that a suite
    authored as ``2.5`` verifies as exactly 5/2 rather than a binary
    approximation. Strings may express fractions such as ``"5/6"``.
    """
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


def evaluate_expression(node: Any, context: str = "verification") -> Fraction | bool:
    """Recursively evaluate a structured expression node.

    A node is either a numeric literal (int/float/str) or a mapping of the
    form ``{"op": <allowlisted name>, "args": [<node>, ...]}``.
    """
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

    result = func(*evaluated)
    return result  # type: ignore[no-any-return]


@dataclass(frozen=True)
class VerificationResult:
    case_id: str
    mode: str
    verified: bool
    detail: str


def _format(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else str(value)


def verify_case(case: Any) -> VerificationResult:
    """Recompute and check one evaluation case's recorded ground truth.

    ``case`` is any object exposing ``id``, ``expected_answer``,
    ``tolerance``, and ``verification`` attributes (i.e. an
    ``EvalCase``). Semantic cases are reported as not deterministically
    checkable rather than silently passing.
    """
    verification = case.verification
    mode = verification.get("mode")
    context = f"case {case.id!r} verification"

    if mode == MODE_SEMANTIC:
        if case.expected_answer is not None:
            raise JuniperConfigError(
                f"{context}: mode 'semantic' requires expected_answer to be null, "
                f"got {case.expected_answer!r}."
            )
        return VerificationResult(
            case_id=case.id,
            mode=MODE_SEMANTIC,
            verified=True,
            detail="semantic expectation — no deterministic answer to recompute",
        )

    if mode != MODE_DETERMINISTIC:
        raise JuniperConfigError(
            f"{context}: mode must be one of {sorted(VALID_VERIFICATION_MODES)}, got {mode!r}."
        )

    if case.expected_answer is None:
        raise JuniperConfigError(f"{context}: mode 'deterministic' requires a non-null expected_answer.")

    computed = evaluate_expression(verification.get("expression"), context)

    # Boolean-valued cases (e.g. "Is it true that 7 * 8 = 54?").
    if isinstance(computed, bool) or isinstance(case.expected_answer, bool):
        if not isinstance(computed, bool) or not isinstance(case.expected_answer, bool):
            raise JuniperConfigError(
                f"{context}: boolean/numeric mismatch — computed {computed!r} vs "
                f"expected_answer {case.expected_answer!r}."
            )
        verified = computed == case.expected_answer
        return VerificationResult(
            case_id=case.id,
            mode=MODE_DETERMINISTIC,
            verified=verified,
            detail=(
                f"computed {computed} == expected {case.expected_answer}"
                if verified
                else f"computed {computed} != recorded expected_answer {case.expected_answer}"
            ),
        )

    expected = to_exact(case.expected_answer, f"{context} expected_answer")
    tolerance = Fraction(0) if case.tolerance is None else to_exact(case.tolerance, f"{context} tolerance")
    if tolerance < 0:
        raise JuniperConfigError(f"{context}: tolerance must be non-negative, got {case.tolerance!r}.")

    difference = abs(computed - expected)
    verified = difference <= tolerance
    if verified:
        detail = f"computed {_format(computed)} matches expected {_format(expected)}"
        if tolerance > 0:
            detail += f" (|Δ| {_format(difference)} <= tolerance {_format(tolerance)})"
    else:
        detail = (
            f"computed {_format(computed)} does NOT match recorded expected_answer "
            f"{_format(expected)} (|Δ| {_format(difference)} > tolerance {_format(tolerance)})"
        )
    return VerificationResult(case_id=case.id, mode=MODE_DETERMINISTIC, verified=verified, detail=detail)


def verify_suite(suite: Any) -> list[VerificationResult]:
    """Verify every case in a suite. Raises on malformed verification data."""
    return [verify_case(case) for case in suite.cases]
