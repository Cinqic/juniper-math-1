"""Central registry mapping category -> ordered list of (generator_id,
family maker) callables. A category with more than one family round-robins
between them, which is what keeps a single cheap template from dominating
the category (Sec. 17).
"""

from __future__ import annotations

from collections.abc import Callable

from juniper_math.dataset.generators import arithmetic, errors, percent_ratio, semantic, tools, word_problems
from juniper_math.dataset.schema import VALID_CATEGORIES, Example

MakerFn = Callable[[int, int, object], Example]

_MODULES = [arithmetic, percent_ratio, tools, word_problems, semantic, errors]


def build_registry() -> dict[str, list[tuple[str, MakerFn]]]:
    registry: dict[str, list[tuple[str, MakerFn]]] = {}
    for module in _MODULES:
        generator_id = module.GENERATOR_ID
        for category, maker in module.FAMILIES:
            registry.setdefault(category, []).append((generator_id, maker))
    missing = VALID_CATEGORIES - set(registry)
    if missing:
        raise RuntimeError(f"No generator registered for category(ies): {sorted(missing)}")
    return registry


__all__ = ["build_registry"]
