"""Pure-arithmetic generator families: no tool involvement, ground truth
established by evaluating an explicit verification.expression tree via
juniper_math.dataset.verify (the same closed-allowlist pattern the frozen
Phase 0 evaluator uses, extended for this dataset's broader operation set).
"""

from __future__ import annotations

from fractions import Fraction

from juniper_math.dataset.generators.common import (
    GENERATOR_VERSION,
    choose_template,
    difficulty_for,
    family_rng,
    fmt_frac,
    make_example,
    rand_decimal_str,
    rand_int,
    rand_signed_int,
)
from juniper_math.dataset.idgen import derive_id
from juniper_math.dataset.schema import Example
from juniper_math.dataset.verify import evaluate_expression

GENERATOR_ID = "arithmetic_core"

# --------------------------------------------------------------------------
# arithmetic
# --------------------------------------------------------------------------

_ARITH_TEMPLATES = [
    "What is {a} {opword} {b}?",
    "Compute {a} {op} {b}.",
    "Work out the result of {a} {op} {b}.",
    "Find the value of {a} {op} {b}.",
]
_OPS = {
    "+": ("add", "plus"),
    "-": ("sub", "minus"),
    "*": ("mul", "times"),
    "/": ("div", "divided by"),
}


def make_arithmetic(index: int, master_seed: int, _runtime: object) -> Example:
    family_id = "arith_two_operand"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    scale = {"trivial": 12, "easy": 100, "medium": 10_000, "hard": 1_000_000}[difficulty]
    op = rng.choice(list(_OPS))
    a = rand_signed_int(rng, 1, scale)
    b = rand_signed_int(rng, 1, scale, allow_negative=(op != "/"))
    if op == "/":
        # keep division exact and well-defined
        b = rand_int(rng, 1, max(2, scale // 4), nonzero=True)
        a = b * rand_int(rng, -scale, scale, nonzero=False)
        if a == 0:
            a = b
    expr_op, opword = _OPS[op]
    t_idx, template = choose_template(rng, _ARITH_TEMPLATES)
    prompt = template.format(a=a, b=b, op=op, opword=opword)
    expression = {"op": expr_op, "args": [a, b]}
    answer = evaluate_expression(expression)
    assert not isinstance(answer, bool)
    # Large-magnitude arithmetic is deliberately left to the "tool_use"
    # category (juniper_math.dataset.generators.tools.make_tool_use), which
    # actually executes calculator.evaluate — this family never marks
    # tool_required without a recorded ToolTrace to back it (Sec. 9).
    return make_example(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}",
        derivation_id=derive_id(family_id, a, b, op),
        seed=seed,
        category="arithmetic",
        difficulty=difficulty,
        prompt=prompt,
        expected_behavior="answer",
        expected_answer=fmt_frac(answer),
        tolerance=0,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "deterministic", "expression": expression},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes="",
    )


# --------------------------------------------------------------------------
# operator_precedence
# --------------------------------------------------------------------------

_PREC_TEMPLATES = [
    "Evaluate {expr}, applying standard order of operations.",
    "What is the value of {expr}?",
    "Simplify {expr} using the correct order of operations.",
]


def make_operator_precedence(index: int, master_seed: int, _runtime: object) -> Example:
    family_id = "precedence_three_term"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    scale = {"trivial": 9, "easy": 20, "medium": 50, "hard": 200}[difficulty]
    a, b, c = (rand_int(rng, 1, scale) for _ in range(3))
    shape = rng.choice(["add_mul", "sub_mul", "mul_add", "paren"])
    if shape == "add_mul":
        expr_text, tree = f"{a} + {b} * {c}", {"op": "add", "args": [a, {"op": "mul", "args": [b, c]}]}
    elif shape == "sub_mul":
        expr_text, tree = f"{a} - {b} * {c}", {"op": "sub", "args": [a, {"op": "mul", "args": [b, c]}]}
    elif shape == "mul_add":
        expr_text, tree = f"{a} * {b} + {c}", {"op": "add", "args": [{"op": "mul", "args": [a, b]}, c]}
    else:
        expr_text, tree = f"({a} + {b}) * {c}", {"op": "mul", "args": [{"op": "add", "args": [a, b]}, c]}
    t_idx, template = choose_template(rng, _PREC_TEMPLATES)
    prompt = template.format(expr=expr_text)
    answer = evaluate_expression(tree)
    assert not isinstance(answer, bool)
    return make_example(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}_{shape}",
        derivation_id=derive_id(family_id, a, b, c, shape),
        seed=seed,
        category="operator_precedence",
        difficulty=difficulty,
        prompt=prompt,
        expected_behavior="answer",
        expected_answer=fmt_frac(answer),
        tolerance=0,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "deterministic", "expression": tree},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes="",
    )


# --------------------------------------------------------------------------
# negative_values
# --------------------------------------------------------------------------

_NEG_TEMPLATES = [
    "What is {a} + ({b})?",
    "Subtract {b_pos} from {a}: what is {a} - {b_pos}?",
    "What is the sum of {a} and {b}?",
    "If the temperature is {a} degrees and drops by {b_pos} degrees, what is the new temperature?",
]


def make_negative_values(index: int, master_seed: int, _runtime: object) -> Example:
    family_id = "negative_signed_sum"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    scale = {"trivial": 15, "easy": 50, "medium": 500, "hard": 5000}[difficulty]
    a = rand_signed_int(rng, 1, scale)
    b = rand_signed_int(rng, 1, scale)
    t_idx, template = choose_template(rng, _NEG_TEMPLATES)
    if t_idx == 1:
        tree = {"op": "sub", "args": [a, abs(b)]}
        prompt = template.format(a=a, b_pos=abs(b))
    elif t_idx == 3:
        tree = {"op": "sub", "args": [a, abs(b)]}
        prompt = template.format(a=a, b_pos=abs(b))
    else:
        tree = {"op": "add", "args": [a, b]}
        prompt = template.format(a=a, b=b)
    answer = evaluate_expression(tree)
    assert not isinstance(answer, bool)
    return make_example(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}",
        derivation_id=derive_id(family_id, a, b, t_idx),
        seed=seed,
        category="negative_values",
        difficulty=difficulty,
        prompt=prompt,
        expected_behavior="answer",
        expected_answer=fmt_frac(answer),
        tolerance=0,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "deterministic", "expression": tree},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes="",
    )


# --------------------------------------------------------------------------
# decimals
# --------------------------------------------------------------------------

_DECIMAL_TEMPLATES = [
    "What is {a} + {b}?",
    "Compute {a} - {b}.",
    "What is the sum {a} + {b}, expressed as a decimal?",
]


def make_decimals(index: int, master_seed: int, _runtime: object) -> Example:
    family_id = "decimal_add_sub"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    places = {"trivial": 1, "easy": 2, "medium": 3, "hard": 4}[difficulty]
    a_str = rand_decimal_str(rng, 1, 500, places)
    b_str = rand_decimal_str(rng, 1, 500, places)
    op = rng.choice(["add", "sub"])
    t_idx, template = choose_template(rng, _DECIMAL_TEMPLATES)
    prompt = template.format(a=a_str, b=b_str)
    tree = {"op": op, "args": [a_str, b_str]}
    answer = evaluate_expression(tree)
    assert not isinstance(answer, bool)
    return make_example(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}_{op}",
        derivation_id=derive_id(family_id, a_str, b_str, op),
        seed=seed,
        category="decimals",
        difficulty=difficulty,
        prompt=prompt,
        expected_behavior="answer",
        expected_answer=fmt_frac(answer),
        tolerance=0,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "deterministic", "expression": tree},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes="",
    )


# --------------------------------------------------------------------------
# fractions
# --------------------------------------------------------------------------

_FRACTION_TEMPLATES = [
    "What is {a}/{b} + {c}/{d}?",
    "Add the fractions {a}/{b} and {c}/{d}.",
    "What is {a}/{b} multiplied by {c}/{d}?",
]


def make_fractions(index: int, master_seed: int, _runtime: object) -> Example:
    family_id = "fraction_arith"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    bound = {"trivial": 6, "easy": 12, "medium": 20, "hard": 40}[difficulty]
    a, b = rand_int(rng, 1, bound), rand_int(rng, 2, bound)
    c, d = rand_int(rng, 1, bound), rand_int(rng, 2, bound)
    op = rng.choice(["add", "mul"])
    t_idx = 0 if op == "add" else 2
    template = _FRACTION_TEMPLATES[t_idx if op == "mul" else rng.choice([0, 1])]
    prompt = template.format(a=a, b=b, c=c, d=d)
    frac_a = {"op": "ratio", "args": [a, b]}
    frac_c = {"op": "ratio", "args": [c, d]}
    tree = {"op": op, "args": [frac_a, frac_c]}
    answer = evaluate_expression(tree)
    assert not isinstance(answer, bool)
    return make_example(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}_{op}",
        derivation_id=derive_id(family_id, a, b, c, d, op),
        seed=seed,
        category="fractions",
        difficulty=difficulty,
        prompt=prompt,
        expected_behavior="answer",
        expected_answer=fmt_frac(answer),
        tolerance=0,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "deterministic", "expression": tree},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes="",
    )


# --------------------------------------------------------------------------
# scientific_notation
# --------------------------------------------------------------------------

_SCI_TEMPLATES = [
    "When {value} is written in scientific notation as m x 10^n (with 1 <= m < 10), what is n?",
    "What power of 10 (the exponent n) is used when {value} is expressed in scientific notation m x 10^n, "
    "1 <= m < 10?",
]


def make_scientific_notation(index: int, master_seed: int, _runtime: object) -> Example:
    family_id = "sci_notation_exponent"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    exponent = rng.randint(3, 9) if difficulty in ("medium", "hard") else rng.randint(2, 5)
    mantissa_int = rand_int(rng, 1, 9)
    mantissa_frac = rng.randint(0, 99)
    value_str = f"{mantissa_int}.{mantissa_frac:02d}"
    full_value = Fraction(value_str) * (Fraction(10) ** exponent)
    t_idx, template = choose_template(rng, _SCI_TEMPLATES)
    prompt = template.format(value=fmt_frac(full_value))

    # Ground truth: n is the unique integer such that 1 <= value / 10^n < 10.
    # Checked via the closed allowlist (ge/lt on the rescaled value), not
    # asserted from generator-internal state alone.
    rescaled = {"op": "ratio", "args": [fmt_frac(full_value), {"op": "pow", "args": [10, exponent]}]}
    lower_ok = evaluate_expression({"op": "ge", "args": [rescaled, 1]})
    upper_ok = evaluate_expression({"op": "lt", "args": [rescaled, 10]})
    assert lower_ok is True and upper_ok is True
    return make_example(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}",
        derivation_id=derive_id(family_id, value_str, exponent),
        seed=seed,
        category="scientific_notation",
        difficulty=difficulty,
        prompt=prompt,
        expected_behavior="answer",
        expected_answer=exponent,
        tolerance=0,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "deterministic", "expression": {"op": "add", "args": [exponent, 0]}},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes="expected_answer is the scientific-notation exponent n; a build-time invariant "
        "(checked here at generation time, not just recorded) confirms 1 <= value/10^n < 10 "
        "via the closed ge/lt allowlist before the example is emitted",
    )


# --------------------------------------------------------------------------
# estimation
# --------------------------------------------------------------------------

_ESTIMATE_TEMPLATES = [
    "Estimate {a} * {b} by rounding each number to the nearest ten, then multiplying.",
    "Round {a} and {b} to the nearest ten and estimate their product.",
]


def make_estimation(index: int, master_seed: int, _runtime: object) -> Example:
    family_id = "estimate_round_multiply"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    scale = {"trivial": 50, "easy": 200, "medium": 2000, "hard": 20000}[difficulty]
    a = rand_int(rng, 10, scale)
    b = rand_int(rng, 10, scale)
    t_idx, template = choose_template(rng, _ESTIMATE_TEMPLATES)
    prompt = template.format(a=a, b=b)
    round_a = {"op": "round", "args": [{"op": "ratio", "args": [a, 10]}, 0]}
    round_b = {"op": "round", "args": [{"op": "ratio", "args": [b, 10]}, 0]}
    scaled_a = {"op": "mul", "args": [round_a, 10]}
    scaled_b = {"op": "mul", "args": [round_b, 10]}
    tree = {"op": "mul", "args": [scaled_a, scaled_b]}
    answer = evaluate_expression(tree)
    assert not isinstance(answer, bool)
    tolerance = max(10, int(answer) // 20)
    return make_example(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}",
        derivation_id=derive_id(family_id, a, b),
        seed=seed,
        category="estimation",
        difficulty=difficulty,
        prompt=prompt,
        expected_behavior="answer",
        expected_answer=fmt_frac(answer),
        tolerance=tolerance,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "deterministic", "expression": tree},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes="nonzero tolerance: estimation has a range of acceptable answers, not one exact value",
    )


# --------------------------------------------------------------------------
# numerical_comparison
# --------------------------------------------------------------------------

_COMPARE_TEMPLATES = [
    "Is {a} greater than {b}?",
    "Is it true that {a} < {b}?",
    "Which is larger, {a} or {b}?",
]


def make_numerical_comparison(index: int, master_seed: int, _runtime: object) -> Example:
    family_id = "compare_two_values"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    scale = {"trivial": 20, "easy": 200, "medium": 5000, "hard": 100000}[difficulty]
    a = rand_signed_int(rng, 1, scale)
    b = rand_signed_int(rng, 1, scale)
    if a == b:
        b += 1
    t_idx, template = choose_template(rng, _COMPARE_TEMPLATES)
    prompt = template.format(a=a, b=b)

    if t_idx == 2:
        # "Which is larger" — the answer is the winning numeric value itself.
        tree = {"op": "max", "args": [a, b]}
        computed = evaluate_expression(tree)
        assert not isinstance(computed, bool)
        answer_val: object = fmt_frac(computed)
        notes = "expected_answer is the larger of the two operands (a != b by construction)"
    else:
        op_name = "gt" if t_idx == 0 else "lt"
        tree = {"op": op_name, "args": [a, b]}
        computed = evaluate_expression(tree)
        assert isinstance(computed, bool)
        answer_val = computed
        notes = ""

    return make_example(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}",
        derivation_id=derive_id(family_id, a, b, t_idx),
        seed=seed,
        category="numerical_comparison",
        difficulty=difficulty,
        prompt=prompt,
        expected_behavior="answer",
        expected_answer=answer_val,
        tolerance=None,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "deterministic", "expression": tree},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes=notes,
    )


# --------------------------------------------------------------------------
# expression_translation
# --------------------------------------------------------------------------

_TRANSLATE_TEMPLATES = [
    "Translate 'the sum of {a} and {b}' into a mathematical expression and evaluate it.",
    "Translate '{a} more than {b}' into a mathematical expression and evaluate it.",
    "Translate 'twice {a}, then add {b}' into a mathematical expression and evaluate it.",
]


def make_expression_translation(index: int, master_seed: int, _runtime: object) -> Example:
    family_id = "translate_phrase"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    scale = {"trivial": 15, "easy": 60, "medium": 500, "hard": 5000}[difficulty]
    a = rand_int(rng, 1, scale)
    b = rand_int(rng, 1, scale)
    t_idx, template = choose_template(rng, _TRANSLATE_TEMPLATES)
    prompt = template.format(a=a, b=b)
    if t_idx == 2:
        tree = {"op": "add", "args": [{"op": "mul", "args": [2, a]}, b]}
    else:
        tree = {"op": "add", "args": [a, b]}
    answer = evaluate_expression(tree)
    assert not isinstance(answer, bool)
    return make_example(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}",
        derivation_id=derive_id(family_id, a, b, t_idx),
        seed=seed,
        category="expression_translation",
        difficulty=difficulty,
        prompt=prompt,
        expected_behavior="answer",
        expected_answer=fmt_frac(answer),
        tolerance=0,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "deterministic", "expression": tree},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes="",
    )


# --------------------------------------------------------------------------
# basic_algebra
# --------------------------------------------------------------------------

_ALGEBRA_TEMPLATES = [
    "Solve for x: x + {b} = {c}",
    "Solve for x: {a}x = {c}",
    "Solve for x: {a}x + {b} = {c}",
]


def make_basic_algebra(index: int, master_seed: int, _runtime: object) -> Example:
    family_id = "linear_equation_one_var"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    scale = {"trivial": 10, "easy": 30, "medium": 100, "hard": 1000}[difficulty]
    x = rand_signed_int(rng, 1, scale)
    shape = rng.choice(["add", "mul", "mul_add"])
    if shape == "add":
        b = rand_int(rng, 1, scale)
        c = x + b
        prompt = _ALGEBRA_TEMPLATES[0].format(b=b, c=c)
        tree = {"op": "sub", "args": [c, b]}
    elif shape == "mul":
        a = rand_int(rng, 2, 12)
        c = a * x
        prompt = _ALGEBRA_TEMPLATES[1].format(a=a, c=c)
        tree = {"op": "ratio", "args": [c, a]}
    else:
        a = rand_int(rng, 2, 12)
        b = rand_int(rng, 1, scale)
        c = a * x + b
        prompt = _ALGEBRA_TEMPLATES[2].format(a=a, b=b, c=c)
        tree = {"op": "ratio", "args": [{"op": "sub", "args": [c, b]}, a]}
    answer = evaluate_expression(tree)
    assert not isinstance(answer, bool)
    assert answer == x
    return make_example(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=shape,
        derivation_id=derive_id(family_id, shape, prompt),
        seed=seed,
        category="basic_algebra",
        difficulty=difficulty,
        prompt=prompt,
        expected_behavior="answer",
        expected_answer=fmt_frac(answer),
        tolerance=0,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "deterministic", "expression": tree},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes="",
    )


FAMILIES = [
    ("arithmetic", make_arithmetic),
    ("operator_precedence", make_operator_precedence),
    ("negative_values", make_negative_values),
    ("decimals", make_decimals),
    ("fractions", make_fractions),
    ("scientific_notation", make_scientific_notation),
    ("estimation", make_estimation),
    ("numerical_comparison", make_numerical_comparison),
    ("expression_translation", make_expression_translation),
    ("basic_algebra", make_basic_algebra),
]
