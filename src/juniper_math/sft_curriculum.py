"""Independent direct-mathematics curriculum for Phase 8 recovery.

These are derived SFT records, not edits to the frozen Phase 4 corpus. Each
family uses new prompt structures and independently verified arithmetic so a
recovery run can test whether broader supervision improves linguistic
generalization rather than merely repeating parent phrasing.
"""

# ruff: noqa: E501 -- long prompt literals are deliberately kept intact for readability.

from __future__ import annotations

import hashlib
import random
from fractions import Fraction

from juniper_math.dataset.generators.common import fmt_frac
from juniper_math.dataset.schema import Example
from juniper_math.dataset.verify import evaluate_expression

CURRICULUM_SCHEMA_VERSION = "1.0.0"
GENERATOR_ID = "phase8_independent_direct_curriculum"

DIRECT_CATEGORIES = (
    "arithmetic",
    "operator_precedence",
    "negative_values",
    "decimals",
    "fractions",
    "percentages",
    "ratios_proportions",
    "scientific_notation",
    "basic_algebra",
    "expression_translation",
    "word_problem",
    "multi_step",
)

SAFETY_CATEGORIES = (
    ("ambiguity", "request_clarification"),
    ("missing_information", "flag_missing_information"),
    ("undefined_operation", "flag_undefined"),
    ("unsupported_capability", "refuse_unsupported"),
)


def _rng(category: str, split: str, index: int, seed: int) -> random.Random:
    digest = hashlib.sha256(
        f"{CURRICULUM_SCHEMA_VERSION}:{category}:{split}:{index}:{seed}".encode()
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _example(
    *,
    category: str,
    split: str,
    index: int,
    seed: int,
    template_id: str,
    prompt: str,
    expression: dict,
    difficulty: str = "medium",
) -> Example:
    answer = evaluate_expression(expression)
    assert not isinstance(answer, bool)
    key = f"{CURRICULUM_SCHEMA_VERSION}:{category}:{split}:{index}:{template_id}:{prompt}"
    return Example(
        example_id=hashlib.sha256(key.encode()).hexdigest()[:24],
        generator_id=GENERATOR_ID,
        generator_version=CURRICULUM_SCHEMA_VERSION,
        family_id=f"independent_{category}",
        template_id=template_id,
        derivation_id=hashlib.sha256(key.encode()).hexdigest()[:16],
        seed=seed,
        category=category,
        difficulty=difficulty,
        synthetic=True,
        split=split,
        prompt=prompt,
        expected_behavior="answer",
        expected_answer=fmt_frac(answer),
        tolerance=0,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "deterministic", "expression": expression},
        provenance=f"{GENERATOR_ID} v{CURRICULUM_SCHEMA_VERSION}",
        notes="Independent Phase 8 direct-math curriculum record.",
    )


def _make(category: str, split: str, index: int, seed: int) -> Example:
    rng = _rng(category, split, index, seed)
    a, b, c = (rng.randint(2, 120) for _ in range(3))
    if category == "arithmetic":
        op = rng.choice(("add", "sub", "mul"))
        if op == "add":
            prompt = rng.choice(
                (f"Combine {a} and {b}. What total do you get?", f"Starting from {a}, increase it by {b}.")
            )
            tree = {"op": "add", "args": [a, b]}
        elif op == "sub":
            b = min(a, b)
            prompt = rng.choice(
                (
                    f"A quantity of {a} is reduced by {b}. What remains?",
                    f"Find the difference between {a} and {b}.",
                )
            )
            tree = {"op": "sub", "args": [a, b]}
        else:
            prompt = rng.choice(
                (
                    f"There are {a} groups containing {b} each. How many are there altogether?",
                    f"Calculate {a} lots of {b}.",
                )
            )
            tree = {"op": "mul", "args": [a, b]}
        return _example(
            category=category,
            split=split,
            index=index,
            seed=seed,
            template_id=op,
            prompt=prompt,
            expression=tree,
        )
    if category == "operator_precedence":
        if index % 2:
            prompt = (
                f"A worksheet says: begin with {a} plus {b} times {c}. Evaluate it using normal precedence."
            )
            tree = {"op": "add", "args": [a, {"op": "mul", "args": [b, c]}]}
            template = "verbal_add_then_multiply"
        else:
            prompt = f"First add {a} and {b}; then multiply that result by {c}. What is the result?"
            tree = {"op": "mul", "args": [{"op": "add", "args": [a, b]}, c]}
            template = "ordered_parentheses"
        return _example(
            category=category,
            split=split,
            index=index,
            seed=seed,
            template_id=template,
            prompt=prompt,
            expression=tree,
        )
    if category == "negative_values":
        below = rng.randint(1, 90)
        move = rng.randint(1, 60)
        prompt = rng.choice(
            (
                f"A diver begins at {below} meters below sea level and descends another {move} meters. Give the signed position.",
                f"The temperature is −{below}° and then falls {move}°. What temperature is recorded?",
            )
        )
        return _example(
            category=category,
            split=split,
            index=index,
            seed=seed,
            template_id="contextual_negative",
            prompt=prompt,
            expression={"op": "sub", "args": [-below, move]},
        )
    if category == "decimals":
        left = Fraction(rng.randint(10, 900), 20)
        right = Fraction(rng.randint(10, 900), 20)
        prompt = rng.choice(
            (
                f"A scale reads {float(left):.2f} g. Adding {float(right):.2f} g gives what total?",
                f"A bill of {float(left):.2f} dollars is combined with {float(right):.2f} dollars. Find the sum.",
            )
        )
        return _example(
            category=category,
            split=split,
            index=index,
            seed=seed,
            template_id="measurement_add",
            prompt=prompt,
            expression={"op": "add", "args": [str(left), str(right)]},
        )
    if category == "fractions":
        n1, d1, n2, d2 = rng.randint(1, 9), rng.randint(2, 12), rng.randint(1, 9), rng.randint(2, 12)
        if index % 2:
            prompt = f"A recipe uses {n1}/{d1} cup in the morning and {n2}/{d2} cup later. How many cups were used?"
            tree = {
                "op": "add",
                "args": [{"op": "ratio", "args": [n1, d1]}, {"op": "ratio", "args": [n2, d2]}],
            }
            template = "recipe_sum"
        else:
            prompt = f"Find {n1}/{d1} of {n2}/{d2}."
            tree = {
                "op": "mul",
                "args": [{"op": "ratio", "args": [n1, d1]}, {"op": "ratio", "args": [n2, d2]}],
            }
            template = "fraction_of_fraction"
        return _example(
            category=category,
            split=split,
            index=index,
            seed=seed,
            template_id=template,
            prompt=prompt,
            expression=tree,
        )
    if category == "percentages":
        percent = rng.choice((5, 10, 12, 15, 20, 25, 40, 60, 75))
        whole = rng.randint(20, 900)
        prompt = rng.choice(
            (
                f"Out of {whole} survey responses, {percent}% satisfy a condition. How many responses is that?",
                f"A {percent}% discount is applied to {whole}. What is the amount of the discount?",
            )
        )
        return _example(
            category=category,
            split=split,
            index=index,
            seed=seed,
            template_id="contextual_percent_of",
            prompt=prompt,
            expression={"op": "percent_of", "args": [percent, whole]},
        )
    if category == "ratios_proportions":
        first, second, known = rng.randint(2, 12), rng.randint(2, 12), rng.randint(2, 30)
        prompt = rng.choice(
            (
                f"A dye mixture has components in the ratio {first}:{second}. If the first amount is {first * known}, what is the second amount?",
                f"The ratio of red to blue tickets is {first}:{second}. With {first * known} red tickets, how many blue tickets are needed?",
            )
        )
        return _example(
            category=category,
            split=split,
            index=index,
            seed=seed,
            template_id="scaled_ratio",
            prompt=prompt,
            expression={"op": "mul", "args": [second, known]},
        )
    if category == "scientific_notation":
        exponent = rng.randint(2, 7)
        mantissa = rng.randint(11, 99)
        value = Fraction(mantissa, 10) * (10**exponent)
        prompt = rng.choice(
            (
                f"Write {fmt_frac(value)} in normalized scientific notation. What exponent of 10 is used?",
                f"In the form m × 10^n with 1 ≤ m < 10, what is n for {fmt_frac(value)}?",
            )
        )
        return _example(
            category=category,
            split=split,
            index=index,
            seed=seed,
            template_id="normalized_exponent",
            prompt=prompt,
            expression={"op": "add", "args": [exponent, 0]},
        )
    if category == "basic_algebra":
        x = rng.randint(-40, 80)
        coefficient, offset = rng.randint(2, 12), rng.randint(-30, 30)
        total = coefficient * x + offset
        prompt = rng.choice(
            (
                f"A linear rule gives {coefficient} times a number plus {offset} as {total}. What is the number?",
                f"Find the value of n if {coefficient}n {offset:+d} = {total}.",
            )
        )
        tree = {"op": "ratio", "args": [{"op": "sub", "args": [total, offset]}, coefficient]}
        return _example(
            category=category,
            split=split,
            index=index,
            seed=seed,
            template_id="linear_rule",
            prompt=prompt,
            expression=tree,
        )
    if category == "expression_translation":
        if index % 2:
            prompt = (
                f"Take {a}, double it, and then decrease the result by {b}. What number does this describe?"
            )
            factor = 2
            template = "double_then_decrease"
        else:
            prompt = f"What is the value of the phrase: {b} less than three times {a}?"
            factor = 3
            template = "three_times_less"
        tree = {"op": "sub", "args": [{"op": "mul", "args": [factor, a]}, b]}
        return _example(
            category=category,
            split=split,
            index=index,
            seed=seed,
            template_id=template,
            prompt=prompt,
            expression=tree,
        )
    if category == "word_problem":
        start, change = rng.randint(10, 250), rng.randint(1, 100)
        prompt = rng.choice(
            (
                f"A library had {start} new books. After donating {change} books, how many new books remained?",
                f"{start} tickets were printed and {change} more were added. How many tickets are available?",
            )
        )
        tree = (
            {"op": "sub", "args": [start, change]}
            if "donating" in prompt
            else {"op": "add", "args": [start, change]}
        )
        return _example(
            category=category,
            split=split,
            index=index,
            seed=seed,
            template_id="inventory_change",
            prompt=prompt,
            expression=tree,
        )
    if category == "multi_step":
        rate1, time1, rate2, time2 = (
            rng.randint(10, 90),
            rng.randint(1, 8),
            rng.randint(10, 90),
            rng.randint(1, 8),
        )
        prompt = rng.choice(
            (
                f"A courier travels {time1} hours at {rate1} km/h, then {time2} hours at {rate2} km/h. What total distance is traveled?",
                f"A tank holds {rate1} L, receives {time1} L, then loses {time2} L. How many liters are left?",
            )
        )
        if "courier" in prompt:
            tree = {
                "op": "add",
                "args": [{"op": "mul", "args": [rate1, time1]}, {"op": "mul", "args": [rate2, time2]}],
            }
            template = "two_leg_distance"
        else:
            tree = {"op": "sub", "args": [{"op": "add", "args": [rate1, time1]}, time2]}
            template = "tank_change"
        return _example(
            category=category,
            split=split,
            index=index,
            seed=seed,
            template_id=template,
            prompt=prompt,
            expression=tree,
        )
    raise ValueError(f"Unsupported direct curriculum category {category!r}")


def build_independent_direct_examples(split: str, examples_per_category: int, seed: int) -> list[Example]:
    if split not in {"train", "validation"}:
        raise ValueError("Independent direct curriculum supports train or validation only.")
    if examples_per_category < 1:
        raise ValueError("examples_per_category must be positive.")
    return [
        _make(category, split, index, seed)
        for category in DIRECT_CATEGORIES
        for index in range(examples_per_category)
    ]


def build_independent_safety_examples(split: str, examples_per_category: int, seed: int) -> list[Example]:
    """Build diverse answerless honesty cases with canonical terminal tags.

    These records deliberately contain no tool trace or result. They teach
    refusal/clarification decisions, never a model-authored trusted result.
    """
    if split not in {"train", "validation"}:
        raise ValueError("Independent safety curriculum supports train or validation only.")
    if examples_per_category < 1:
        raise ValueError("examples_per_category must be positive.")
    templates = {
        "ambiguity": (
            "What is the answer?",
            "Can you calculate it for me?",
            "Please solve this without any numbers being provided.",
            "Which value should I use?",
        ),
        "missing_information": (
            "A jacket is discounted. What is the sale price?",
            "If a trip takes some time at some speed, how far is it?",
            "What is the total after an unknown tax rate is applied?",
            "A rectangle has one side length. What is its area?",
        ),
        "undefined_operation": (
            "Compute 17 divided by 0.",
            "What number results from 0/0?",
            "Evaluate the expression 45 ÷ (9 - 9).",
            "Find 8 divided by zero.",
        ),
        "unsupported_capability": (
            "Prove the Riemann hypothesis.",
            "Give a rigorous proof of the Goldbach conjecture.",
            "Solve this arbitrary nonlinear differential equation symbolically: y' = y^2 + sin(x).",
            "Derive a closed-form solution for every polynomial of degree five.",
        ),
    }
    out: list[Example] = []
    for category, behavior in SAFETY_CATEGORIES:
        for index in range(examples_per_category):
            prompt = templates[category][index % len(templates[category])]
            key = f"{CURRICULUM_SCHEMA_VERSION}:safety:{category}:{split}:{index}:{seed}:{prompt}"
            out.append(
                Example(
                    example_id=hashlib.sha256(key.encode()).hexdigest()[:24],
                    generator_id=GENERATOR_ID,
                    generator_version=CURRICULUM_SCHEMA_VERSION,
                    family_id=f"independent_safety_{category}",
                    template_id=f"safety_{index % len(templates[category])}",
                    derivation_id=hashlib.sha256(key.encode()).hexdigest()[:16],
                    seed=seed,
                    category=category,
                    difficulty="medium",
                    synthetic=True,
                    split=split,
                    prompt=prompt,
                    expected_behavior=behavior,
                    expected_answer=None,
                    tolerance=None,
                    tool_required=False,
                    tool_name=None,
                    tool_traces=(),
                    verification={"mode": "semantic", "reason": "independent safety curriculum"},
                    provenance=f"{GENERATOR_ID} v{CURRICULUM_SCHEMA_VERSION}",
                    notes="Independent Phase 8 safety curriculum record.",
                )
            )
    return out


__all__ = [
    "CURRICULUM_SCHEMA_VERSION",
    "DIRECT_CATEGORIES",
    "SAFETY_CATEGORIES",
    "build_independent_direct_examples",
    "build_independent_safety_examples",
]
