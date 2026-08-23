"""incorrect_supplied_answer generator family: a wrong answer is presented
alongside a computation; the model must flag it and supply the correct one.
"""

from __future__ import annotations

from juniper_math.dataset.generators.common import (
    GENERATOR_VERSION,
    difficulty_for,
    family_rng,
    fmt_frac,
    make_example,
    rand_int,
)
from juniper_math.dataset.idgen import derive_id
from juniper_math.dataset.schema import Example
from juniper_math.dataset.verify import evaluate_expression

GENERATOR_ID = "error_detection_core"

_TEMPLATES = [
    "Someone claims that {a} * {b} = {wrong}. Is that correct? If not, what is the right answer?",
    "A student computed {a} + {b} and got {wrong}. Check their work — what should the answer be?",
    "Is it true that {a} - {b} = {wrong}?",
]


def make_incorrect_supplied_answer(index: int, master_seed: int, _runtime: object) -> Example:
    family_id = "wrong_claimed_answer"
    rng, seed = family_rng(GENERATOR_ID, family_id, index, master_seed)
    difficulty = difficulty_for(rng)
    scale = {"trivial": 20, "easy": 100, "medium": 1000, "hard": 20000}[difficulty]
    a = rand_int(rng, 1, scale)
    b = rand_int(rng, 1, scale)
    t_idx = rng.randrange(len(_TEMPLATES))
    op = {"0": "mul", "1": "add", "2": "sub"}[str(t_idx)]
    tree = {"op": op, "args": [a, b]}
    correct = evaluate_expression(tree)
    assert not isinstance(correct, bool)

    # Perturb by a small, deterministic, always-wrong offset — never
    # accidentally equal to the correct value.
    offset = rand_int(rng, 1, max(2, scale // 10), nonzero=True)
    wrong = int(correct) + offset
    if wrong == correct:
        wrong += 1

    prompt = _TEMPLATES[t_idx].format(a=a, b=b, wrong=wrong)
    return make_example(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        family_id=family_id,
        template_id=f"t{t_idx}",
        derivation_id=derive_id(family_id, t_idx, a, b, wrong),
        seed=seed,
        category="incorrect_supplied_answer",
        difficulty=difficulty,
        prompt=prompt,
        expected_behavior="flag_incorrect_answer",
        expected_answer=fmt_frac(correct),
        tolerance=0,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "deterministic", "expression": tree},
        provenance=f"{GENERATOR_ID}/{family_id} v{GENERATOR_VERSION}",
        notes=f"the prompt's claimed answer ({wrong}) is deliberately wrong by construction",
    )


FAMILIES = [
    ("incorrect_supplied_answer", make_incorrect_supplied_answer),
]
