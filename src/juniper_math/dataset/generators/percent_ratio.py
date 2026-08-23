"""percentages and ratios_proportions generator families."""

from __future__ import annotations

from juniper_math.dataset.generators.common import (
    GENERATOR_VERSION,
    choose_template,
    difficulty_for,
    family_rng,
    fmt_frac,
    make_example,
    rand_int,
)
from juniper_math.dataset.idgen import derive_id
from juniper_math.dataset.schema import Example
from juniper_math.dataset.verify import evaluate_expression

GENERATOR_ID = "percent_ratio_core"

_PERCENT_TEMPLATES = [
    "What is {p}% of {whole}?",
    "Find {p} percent of {whole}.",
    "A store has {whole} items in stock. What is {p}% of that amount?",
]


def make_percentages(index: int, master_seed: int, _runtime: object) -> Example:
    family_id = "percent_of_whole"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    whole = rand_int(rng, 10, {"trivial": 100, "easy": 500, "medium": 5000, "hard": 50000}[difficulty])
    p = rng.choice([5, 10, 15, 20, 25, 33, 40, 50, 60, 75, 90])
    t_idx, template = choose_template(rng, _PERCENT_TEMPLATES)
    prompt = template.format(p=p, whole=whole)
    tree = {"op": "percent_of", "args": [p, whole]}
    answer = evaluate_expression(tree)
    assert not isinstance(answer, bool)
    return make_example(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}",
        derivation_id=derive_id(family_id, p, whole),
        seed=seed,
        category="percentages",
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


_RATIO_TEMPLATES = [
    "A recipe uses {a} cups of flour for every {b} cups of sugar. If you use {c} cups of sugar, "
    "how many cups of flour do you need?",
    "The ratio of cats to dogs at a shelter is {a}:{b}. If there are {c} dogs, how many cats are there?",
    "Two numbers are in the ratio {a}:{b}. If the smaller share of a total of {total} is split "
    "proportionally, what is the {a}-part share?",
]


def make_ratios_proportions(index: int, master_seed: int, _runtime: object) -> Example:
    family_id = "ratio_proportional_scale"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    bound = {"trivial": 12, "easy": 30, "medium": 100, "hard": 500}[difficulty]
    a = rand_int(rng, 2, bound)
    b = rand_int(rng, 2, bound)
    t_idx, template = choose_template(rng, _RATIO_TEMPLATES)
    tree: dict[str, object]
    if t_idx == 2:
        total = (a + b) * rand_int(rng, 1, 20)
        prompt = template.format(a=a, b=b, total=total)
        tree = {"op": "ratio", "args": [{"op": "mul", "args": [a, total]}, {"op": "add", "args": [a, b]}]}
    else:
        c = rand_int(rng, 1, bound)
        prompt = template.format(a=a, b=b, c=c)
        tree = {"op": "ratio", "args": [{"op": "mul", "args": [a, c]}, b]}
    answer = evaluate_expression(tree)
    assert not isinstance(answer, bool)
    return make_example(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}",
        derivation_id=derive_id(family_id, a, b, t_idx),
        seed=seed,
        category="ratios_proportions",
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
    ("percentages", make_percentages),
    ("ratios_proportions", make_ratios_proportions),
]
