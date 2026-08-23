"""Deterministic identifier and seed derivation.

Every seed and every stable ID in the dataset pipeline is derived from a
handful of strings via SHA-256 — never from Python's ``hash()`` (which is
PYTHONHASHSEED-randomized per-process and therefore not reproducible across
runs, see Sec. 29 of the Phase 4 instructions) and never from
``random.random()``/``time``-seeded state. The same inputs always produce the
same outputs on any machine, in any process, in any hash-seed configuration.
"""

from __future__ import annotations

import hashlib
import random

_SEED_MASK = (1 << 63) - 1


def derive_seed(*parts: str | int) -> int:
    """Derive a stable, non-negative 63-bit integer seed from ``parts``.

    Used to build a ``random.Random`` instance per example so that
    generation is reproducible independent of iteration order, thread count,
    or process — the seed depends only on the logical identity of the thing
    being generated (generator id, family id, index), never on when or where
    it runs.
    """
    joined = "\x1f".join(str(p) for p in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & _SEED_MASK


def derive_rng(*parts: str | int) -> random.Random:
    return random.Random(derive_seed(*parts))


def derive_id(*parts: str | int, length: int = 16) -> str:
    """Derive a short stable hex identifier from ``parts``."""
    joined = "\x1f".join(str(p) for p in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return digest[:length]
