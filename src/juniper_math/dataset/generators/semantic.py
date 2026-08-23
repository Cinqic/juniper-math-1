"""Semantic/behavioral generator families: ambiguity, missing_information,
undefined_operation, unsupported_capability (Sec. 10).

These four categories are not "no verifiable answer" as an afterthought —
they are constructed so that classification is correct BY CONSTRUCTION (the
template itself encodes why the case is ambiguous/missing/undefined/
unsupported), and each carries an explicit reason in ``notes``. Every case
uses ``verification.mode = "semantic"`` and ``expected_answer = None``,
matching the frozen Phase 0 evaluator's semantic-case convention exactly
(juniper_math.verification / juniper_math.evals).
"""

from __future__ import annotations

from juniper_math.dataset.generators.common import (
    GENERATOR_VERSION,
    difficulty_for,
    family_rng,
    make_example,
    rand_int,
)
from juniper_math.dataset.idgen import derive_id
from juniper_math.dataset.schema import Example

GENERATOR_ID = "semantic_classification_core"

_NAMES = ["Maria", "James", "Priya", "Wei", "Fatima", "Diego", "Aisha", "Noah"]

# --------------------------------------------------------------------------
# ambiguity — the request itself has more than one valid interpretation, or
# is underdetermined (fewer equations than unknowns).
# --------------------------------------------------------------------------

_AMBIGUITY_TEMPLATES = [
    (
        "A rectangle has an area of {a} square meters. What is its length?",
        "underdetermined: area alone does not fix a unique length without the width",
    ),
    (
        "{name} is {ratio} times as old as {name2}. How old is {name}?",
        "underdetermined: a ratio between two unknown ages does not fix either age",
    ),
    (
        "x + y = {a}. What is x?",
        "underdetermined: one linear equation in two unknowns has infinitely many solutions",
    ),
    ("What is 'it' divided by {a}?", "ambiguous referent: 'it' does not identify a specific number"),
]


def make_ambiguity(index: int, master_seed: int, _runtime: object) -> Example:
    family_id = "underdetermined_or_ambiguous_referent"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    scale = {"trivial": 10, "easy": 50, "medium": 500, "hard": 5000}[difficulty]
    t_idx = rng.randrange(len(_AMBIGUITY_TEMPLATES))
    template, reason = _AMBIGUITY_TEMPLATES[t_idx]
    name, name2 = rng.sample(_NAMES, 2)
    a = rand_int(rng, 1, scale)
    ratio = rng.choice([2, 3, 4, "one and a half"])
    prompt = template.format(a=a, name=name, name2=name2, ratio=ratio)
    return make_example(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}",
        derivation_id=derive_id(family_id, t_idx, a, name, name2, str(ratio)),
        seed=seed,
        category="ambiguity",
        difficulty=difficulty,
        prompt=prompt,
        expected_behavior="request_clarification",
        expected_answer=None,
        tolerance=None,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "semantic", "expression": None},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes=reason,
    )


# --------------------------------------------------------------------------
# missing_information — a necessary operand was simply never supplied.
# --------------------------------------------------------------------------

_MISSING_TEMPLATES = [
    (
        "{name} bought some {item} at ${price} each. How much did {name} spend in total?",
        "quantity purchased is never stated",
    ),
    ("What is the sales tax on a ${price} purchase?", "the tax rate is never stated"),
    ("A car travels for {hours} hours. How far does it go?", "the car's speed is never stated"),
    (
        "{name} invests some money at {rate}% annual interest for {years} years. "
        "How much interest is earned?",
        "the principal amount invested is never stated",
    ),
]
_ITEMS = ["notebooks", "apples", "tickets", "chairs"]


def make_missing_information(index: int, master_seed: int, _runtime: object) -> Example:
    family_id = "unsupplied_required_operand"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    t_idx = rng.randrange(len(_MISSING_TEMPLATES))
    template, reason = _MISSING_TEMPLATES[t_idx]
    name = rng.choice(_NAMES)
    item = rng.choice(_ITEMS)
    price = rand_int(rng, 1, 100)
    hours = rand_int(rng, 1, 20)
    rate = rng.choice([2, 3, 4, 5, 6])
    years = rand_int(rng, 1, 10)
    prompt = template.format(name=name, item=item, price=price, hours=hours, rate=rate, years=years)
    return make_example(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}",
        derivation_id=derive_id(family_id, t_idx, name, item, price, hours, rate, years),
        seed=seed,
        category="missing_information",
        difficulty=difficulty,
        prompt=prompt,
        expected_behavior="flag_missing_information",
        expected_answer=None,
        tolerance=None,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "semantic", "expression": None},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes=reason,
    )


# --------------------------------------------------------------------------
# undefined_operation — the mathematics itself is undefined, not merely
# hard or unsupported.
# --------------------------------------------------------------------------

_UNDEFINED_TEMPLATES = [
    ("What is {a} divided by 0?", "division by zero is undefined"),
    ("What is the square root of -{a}?", "the square root of a negative number is undefined over the reals"),
    ("What is 0 to the power of -{a}?", "zero raised to a negative power is undefined"),
    ("What is the value of 0/0?", "0/0 is indeterminate, not a defined numeric value"),
]


def make_undefined_operation(index: int, master_seed: int, _runtime: object) -> Example:
    family_id = "mathematically_undefined"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    scale = {"trivial": 10, "easy": 50, "medium": 500, "hard": 5000}[difficulty]
    t_idx = rng.randrange(len(_UNDEFINED_TEMPLATES))
    template, reason = _UNDEFINED_TEMPLATES[t_idx]
    a = rand_int(rng, 1, scale)
    prompt = template.format(a=a)
    return make_example(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}",
        derivation_id=derive_id(family_id, t_idx, a),
        seed=seed,
        category="undefined_operation",
        difficulty=difficulty,
        prompt=prompt,
        expected_behavior="flag_undefined",
        expected_answer=None,
        tolerance=None,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "semantic", "expression": None},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes=reason,
    )


# --------------------------------------------------------------------------
# unsupported_capability — outside Juniper Math 1's intended scope, per
# config/tools.yaml (radians-only trig, no symbolic algebra/calculus/
# matrices in Phase 3).
# --------------------------------------------------------------------------

_UNSUPPORTED_TEMPLATES = [
    (
        "Solve the differential equation dy/dx = {a}x for y.",
        "symbolic calculus is outside the Phase 3 tool runtime's scope",
    ),
    (
        "What is the determinant of the matrix [[1, {a}], [{b}, 4]]?",
        "matrix operations are outside the Phase 3 tool runtime's scope",
    ),
    (
        "Factor the polynomial x^2 - {a}.",
        "symbolic polynomial factoring is outside the Phase 3 tool runtime's scope",
    ),
    (
        "What is sin({a} degrees) using degree mode?",
        "the Phase 3 tool runtime's trig functions are radians-only (config/tools.yaml); "
        "degree mode is unsupported",
    ),
    (
        "Prove that the square root of {a} is irrational.",
        "formal mathematical proof is outside this project's intended scope",
    ),
]


def make_unsupported_capability(index: int, master_seed: int, _runtime: object) -> Example:
    family_id = "outside_intended_scope"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    scale = {"trivial": 10, "easy": 50, "medium": 200, "hard": 1000}[difficulty]
    t_idx = rng.randrange(len(_UNSUPPORTED_TEMPLATES))
    template, reason = _UNSUPPORTED_TEMPLATES[t_idx]
    a = rand_int(rng, 2, scale)
    b = rand_int(rng, 2, scale)
    prompt = template.format(a=a, b=b)
    return make_example(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}",
        derivation_id=derive_id(family_id, t_idx, a, b),
        seed=seed,
        category="unsupported_capability",
        difficulty=difficulty,
        prompt=prompt,
        expected_behavior="refuse_unsupported",
        expected_answer=None,
        tolerance=None,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "semantic", "expression": None},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes=reason,
    )


FAMILIES = [
    ("ambiguity", make_ambiguity),
    ("missing_information", make_missing_information),
    ("undefined_operation", make_undefined_operation),
    ("unsupported_capability", make_unsupported_capability),
]
