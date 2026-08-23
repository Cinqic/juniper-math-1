"""Deterministic, family-aware train/validation/test split assignment (Sec. 14).

Grouping key is ``(generator_id, family_id, derivation_id)`` — every example
sharing a derivation_id is the same underlying problem instance (same
operands/structure, possibly different surface phrasing) and MUST land in
the same split, or a model could see the test-split answer to a
near-identical train-split question. Assignment is a deterministic hash of
the grouping key, never Python's ``random`` module and never row order, so
the same corpus config always reproduces the same split.
"""

from __future__ import annotations

from juniper_math.dataset.config import SplitConfig
from juniper_math.dataset.idgen import derive_seed


def assign_split(
    generator_id: str, family_id: str, derivation_id: str, master_seed: int, config: SplitConfig
) -> str:
    seed = derive_seed(master_seed, "split", generator_id, family_id, derivation_id)
    # 3-decimal-digit resolution is far finer than the split proportions
    # this project uses; a plain modulo keeps assignment simple and exactly
    # reproducible without pulling in a full RNG per lookup.
    bucket = seed % 100_000
    train_cut = int(config.train * 100_000)
    val_cut = train_cut + int(config.validation * 100_000)
    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "validation"
    return "test"
