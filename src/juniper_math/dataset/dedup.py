"""Exact and near deduplication (Sec. 16).

Exact dedup: a cryptographic hash of the canonical (normalized prompt,
normalized expected_answer) pair, checked against every example seen so far
in the whole corpus.

Near dedup: a bounded, family-scoped shingled-Jaccard check. Comparing every
example against every other example in a multi-million-row corpus is
quadratic and impractical; instead each family keeps a bounded recent-window
buffer (Sec. "no silent caps" — this bound is a documented engineering
trade-off, not a hidden one) and a new example is compared against that
window only. This catches the common failure mode this control targets —
a template re-firing with only its numeric operands changed — without
requiring full corpus-wide pairwise comparison.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict, deque

_NEAR_DUP_WINDOW = 200

# This deliberately operates on the prompt *shape*, not its wording alone.
# Numeric substitution was the blind spot in the original 5-word Jaccard
# implementation: "What is 12 plus 13?" and "What is 87 plus 642?" share
# almost no literal shingles yet are the same training template.
_NUMBER = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:/\d+)?(?:e[-+]?\d+)?", re.IGNORECASE)
_CURRENCY = re.compile(r"\$\s*<NUM>")


def structural_normalize(text: str) -> str:
    """Return a stable prompt-shape representation for duplicate analysis.

    It is intentionally conservative: only value-like numeric literals are
    abstracted. Mathematical operators, units, nouns, and word order remain,
    so distinct tasks such as addition and conversion do not collapse.
    """
    normalized = _NUMBER.sub("<NUM>", text.lower())
    normalized = _CURRENCY.sub("$<NUM>", normalized)
    return " ".join(normalized.split())


def exact_key(normalized_prompt: str, expected_answer: object) -> str:
    joined = f"{normalized_prompt}\x1f{expected_answer!r}"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def shingles(text: str, size: int) -> set[str]:
    tokens = text.lower().split()
    if len(tokens) < size:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class ExactDeduplicator:
    def __init__(self) -> None:
        self._seen: set[str] = set()
        self.removed = 0

    def is_duplicate(self, key: str) -> bool:
        if key in self._seen:
            self.removed += 1
            return True
        self._seen.add(key)
        return False

    def seed(self, key: str) -> None:
        """Register a key as already-seen without counting it as a removal.

        Used to reserve evaluation-suite content so the corpus build's own
        dedup pass naturally excludes anything that exactly matches a frozen
        eval case (Sec. 13 contamination isolation), without a second,
        separate exclusion mechanism.
        """
        self._seen.add(key)


class NearDeduplicator:
    def __init__(
        self,
        shingle_size: int,
        threshold: float,
        *,
        max_structural_repeats: int = 1,
        window: int = _NEAR_DUP_WINDOW,
    ) -> None:
        self.shingle_size = shingle_size
        self.threshold = threshold
        self.max_structural_repeats = max_structural_repeats
        self.window = window
        self._recent: dict[str, deque[tuple[set[str], str]]] = defaultdict(lambda: deque(maxlen=window))
        self._structural_counts: dict[tuple[str, str], int] = defaultdict(int)
        self.removed = 0

    def is_near_duplicate(self, family_key: str, text: str) -> bool:
        shape = structural_normalize(text)
        candidate = shingles(shape, self.shingle_size)
        bucket = self._recent[family_key]
        count_key = (family_key, shape)
        # Numeric/template repeats are overwhelmingly the common case.  Make
        # that check O(1); the bounded Jaccard scan remains for close but not
        # identical normalized shapes.
        if self._structural_counts[count_key]:
            if self._structural_counts[count_key] >= self.max_structural_repeats:
                self.removed += 1
                return True
            bucket.append((candidate, shape))
            self._structural_counts[count_key] += 1
            return False
        for prior, prior_shape in bucket:
            # Exact structural matches catch operand/template substitution.
            if shape == prior_shape or jaccard(candidate, prior) >= self.threshold:
                if self._structural_counts[count_key] >= self.max_structural_repeats:
                    self.removed += 1
                    return True
                break
        bucket.append((candidate, shape))
        self._structural_counts[(family_key, shape)] += 1
        return False

    def seed(self, family_key: str, text: str) -> None:
        """Reserve ``text`` for ``family_key`` without treating it as a
        removal — see :meth:`ExactDeduplicator.seed`."""
        shape = structural_normalize(text)
        self._recent[family_key].append((shingles(shape, self.shingle_size), shape))
        self._structural_counts[(family_key, shape)] += 1
