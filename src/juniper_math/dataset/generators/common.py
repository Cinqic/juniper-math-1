"""Shared helpers for Phase 4 dataset generators."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

from juniper_math.dataset.idgen import derive_id, derive_seed
from juniper_math.dataset.schema import Example, ToolTrace

GENERATOR_VERSION = "1.0.0"


def fmt_frac(value: Fraction) -> str:
    """Render a Fraction as the shortest exact decimal/integer string, or a/b."""
    if value.denominator == 1:
        return str(value.numerator)
    # Try an exact decimal representation (denominator a product of 2s and 5s).
    denom = value.denominator
    for p in (2, 5):
        while denom % p == 0:
            denom //= p
    if denom == 1:
        from decimal import Decimal

        return format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    return f"{value.numerator}/{value.denominator}"


def rand_int(rng: random.Random, lo: int, hi: int, *, nonzero: bool = False) -> int:
    while True:
        v = rng.randint(lo, hi)
        if not nonzero or v != 0:
            return v


def rand_signed_int(rng: random.Random, lo: int, hi: int, *, allow_negative: bool = True) -> int:
    v = rand_int(rng, lo, hi, nonzero=True)
    if allow_negative and rng.random() < 0.5:
        v = -v
    return v


def rand_decimal_str(rng: random.Random, lo: int, hi: int, places: int) -> str:
    whole = rand_int(rng, lo, hi)
    frac_digits = "".join(str(rng.randint(0, 9)) for _ in range(places))
    return f"{whole}.{frac_digits}"


def difficulty_for(rng: random.Random) -> str:
    return rng.choices(
        ["trivial", "easy", "medium", "hard"],
        weights=[0.15, 0.35, 0.35, 0.15],
        k=1,
    )[0]


def make_example(
    *,
    generator_id: str,
    generator_version: str,
    family_id: str,
    template_id: str,
    derivation_id: str,
    seed: int,
    category: str,
    difficulty: str,
    prompt: str,
    expected_behavior: str,
    expected_answer: Any,
    tolerance: float | None,
    tool_required: bool,
    tool_name: str | None,
    tool_traces: tuple[ToolTrace, ...],
    verification: dict[str, Any],
    provenance: str,
    notes: str = "",
) -> Example:
    example_id = derive_id("example", generator_id, family_id, template_id, derivation_id, seed, length=24)
    return Example(
        example_id=example_id,
        generator_id=generator_id,
        generator_version=generator_version,
        family_id=family_id,
        template_id=template_id,
        derivation_id=derivation_id,
        seed=seed,
        category=category,
        difficulty=difficulty,
        synthetic=True,
        split="train",  # reassigned deterministically by dataset.split
        prompt=prompt,
        expected_behavior=expected_behavior,
        expected_answer=expected_answer,
        tolerance=tolerance,
        tool_required=tool_required,
        tool_name=tool_name,
        tool_traces=tool_traces,
        verification=verification,
        provenance=provenance,
        notes=notes,
    )


def family_rng(generator_id: str, family_id: str, index: int, master_seed: int) -> tuple[random.Random, int]:
    seed = derive_seed(master_seed, generator_id, family_id, index)
    return random.Random(seed), seed


def choose_template(rng: random.Random, templates: list[str]) -> tuple[int, str]:
    idx = rng.randrange(len(templates))
    return idx, templates[idx]
