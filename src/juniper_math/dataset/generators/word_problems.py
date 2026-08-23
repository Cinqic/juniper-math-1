"""word_problem and multi_step generator families (pure arithmetic, no tool)."""

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

GENERATOR_ID = "word_problem_core"

_NAMES = ["Maria", "James", "Priya", "Wei", "Fatima", "Diego", "Aisha", "Noah", "Elena", "Kofi"]
_ITEMS = ["apples", "notebooks", "marbles", "stickers", "coins", "pencils", "books", "tickets"]

_WORD_TEMPLATES = [
    "{name} has {a} {item}. {name2} gives {name} {b} more {item}. How many {item} does {name} have now?",
    "{name} had {a} {item} and gave away {b}. How many {item} does {name} have left?",
    "A box contains {a} {item}. {b} identical boxes are combined. How many {item} are there in total?",
]


def make_word_problem(index: int, master_seed: int, _runtime: object) -> Example:
    family_id = "one_step_narrative"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    scale = {"trivial": 20, "easy": 100, "medium": 1000, "hard": 10000}[difficulty]
    name, name2 = rng.sample(_NAMES, 2)
    item = rng.choice(_ITEMS)
    a = rand_int(rng, 1, scale)
    b = rand_int(rng, 1, scale)
    t_idx, template = choose_template(rng, _WORD_TEMPLATES)
    if t_idx == 0:
        prompt = template.format(name=name, name2=name2, a=a, b=b, item=item)
        tree = {"op": "add", "args": [a, b]}
    elif t_idx == 1:
        b = min(b, a)
        prompt = template.format(name=name, a=a, b=b, item=item)
        tree = {"op": "sub", "args": [a, b]}
    else:
        n_boxes = rand_int(rng, 2, 8)
        prompt = template.format(a=a, b=n_boxes, item=item)
        tree = {"op": "mul", "args": [a, n_boxes]}
    answer = evaluate_expression(tree)
    assert not isinstance(answer, bool)
    return make_example(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}",
        derivation_id=derive_id(family_id, t_idx, a, b, item),
        seed=seed,
        category="word_problem",
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


_MULTI_STEP_TEMPLATES = [
    "{name} buys {a} {item} at ${price} each, then sells {b} of them at ${price2} each. "
    "How much profit or loss does {name} make on the {b} sold, compared to what they paid for those {b}?",
    "A tank starts with {a} liters of water. {b} liters are added, then {c} liters are drained. "
    "How many liters remain?",
    "{name} earns ${a} per hour and works {b} hours, then receives a ${c} bonus. What is the total earned?",
]


def make_multi_step(index: int, master_seed: int, _runtime: object) -> Example:
    family_id = "two_step_narrative"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    scale = {"trivial": 10, "easy": 40, "medium": 200, "hard": 2000}[difficulty]
    name = rng.choice(_NAMES)
    item = rng.choice(_ITEMS)
    t_idx, template = choose_template(rng, _MULTI_STEP_TEMPLATES)

    tree: dict[str, object]
    if t_idx == 0:
        a = rand_int(rng, 5, scale)
        b = rand_int(rng, 1, a)
        price = rand_int(rng, 1, 50)
        price2 = rand_int(rng, 1, 50)
        prompt = template.format(name=name, a=a, b=b, item=item, price=price, price2=price2)
        tree = {"op": "sub", "args": [{"op": "mul", "args": [b, price2]}, {"op": "mul", "args": [b, price]}]}
    elif t_idx == 1:
        a = rand_int(rng, 10, scale)
        b = rand_int(rng, 1, scale)
        c = rand_int(rng, 1, a + b)
        prompt = template.format(a=a, b=b, c=c)
        tree = {"op": "sub", "args": [{"op": "add", "args": [a, b]}, c]}
    else:
        a = rand_int(rng, 5, 100)
        b = rand_int(rng, 1, scale)
        c = rand_int(rng, 1, scale)
        prompt = template.format(name=name, a=a, b=b, c=c)
        tree = {"op": "add", "args": [{"op": "mul", "args": [a, b]}, c]}

    answer = evaluate_expression(tree)
    assert not isinstance(answer, bool)
    return make_example(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}",
        derivation_id=derive_id(family_id, t_idx, str(tree)),
        seed=seed,
        category="multi_step",
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
    ("word_problem", make_word_problem),
    ("multi_step", make_multi_step),
]
