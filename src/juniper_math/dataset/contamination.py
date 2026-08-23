"""Evaluation-contamination checks (Sec. 13).

Defense in depth: split assignment (dataset.split) already makes leakage by
derivation_id structurally impossible, and the eval-suite seed offset
(config/dataset.yaml eval_suite_seed_offset) puts eval-suite generation in a
disjoint seed namespace from train/validation/test. This module verifies
both of those invariants actually held for a built corpus, plus checks for
accidental exact/near duplication across splits that a derivation_id
grouping bug would not catch by itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from juniper_math.dataset.dedup import jaccard, shingles
from juniper_math.dataset.schema import Example


@dataclass(frozen=True)
class ContaminationReport:
    derivation_id_split_violations: list[str]
    exact_cross_split_duplicates: list[str]
    near_duplicate_eval_train_pairs: list[tuple[str, str]]

    @property
    def clean(self) -> bool:
        return not (
            self.derivation_id_split_violations
            or self.exact_cross_split_duplicates
            or self.near_duplicate_eval_train_pairs
        )


def check_derivation_id_isolation(examples: list[Example]) -> list[str]:
    """Every derivation_id must map to exactly one split."""
    seen: dict[str, str] = {}
    violations: list[str] = []
    for ex in examples:
        key = f"{ex.generator_id}/{ex.family_id}/{ex.derivation_id}"
        if key in seen and seen[key] != ex.split:
            violations.append(f"{key}: appears in both {seen[key]!r} and {ex.split!r}")
        else:
            seen[key] = ex.split
    return violations


def check_exact_cross_split_duplicates(examples: list[Example]) -> list[str]:
    from juniper_math.dataset.dedup import exact_key

    seen: dict[str, str] = {}
    violations: list[str] = []
    for ex in examples:
        key = exact_key(ex.prompt, ex.expected_answer)
        if key in seen and seen[key] != ex.split:
            violations.append(f"{ex.example_id}: exact duplicate of an example in split {seen[key]!r}")
        else:
            seen[key] = ex.split
    return violations


def check_near_duplicate_eval_vs_train(
    eval_prompts: list[str],
    train_examples: list[Example],
    shingle_size: int,
    threshold: float,
) -> list[tuple[str, str]]:
    """Inverted-index candidate filtering, not a naive O(eval * train) scan.

    A first version compared every train example against every eval prompt
    directly — fine at the small scale it was tested against, but at
    ~1.65M train examples and ~700 eval prompts that is well over a billion
    set operations and takes tens of minutes. Instead, build a shingle ->
    eval-prompt-index inverted index once (eval suites are small — hundreds
    of prompts), then for each train example only compute exact Jaccard
    against the handful of eval prompts that share at least one shingle
    with it. A true near-duplicate always shares shingles, so this changes
    nothing about what gets caught — only how many candidates are actually
    scored.
    """
    eval_shingle_sets = [shingles(p, shingle_size) for p in eval_prompts]
    inverted_index: dict[str, list[int]] = {}
    for idx, shingle_set in enumerate(eval_shingle_sets):
        for shingle in shingle_set:
            inverted_index.setdefault(shingle, []).append(idx)

    violations: list[tuple[str, str]] = []
    for ex in train_examples:
        train_shingles = shingles(ex.prompt, shingle_size)
        candidate_indices: set[int] = set()
        for shingle in train_shingles:
            candidate_indices.update(inverted_index.get(shingle, ()))
        for idx in candidate_indices:
            if jaccard(eval_shingle_sets[idx], train_shingles) >= threshold:
                violations.append((eval_prompts[idx], ex.prompt))
    return violations


def build_contamination_report(
    examples: list[Example],
    eval_prompts: list[str],
    shingle_size: int,
    threshold: float,
) -> ContaminationReport:
    train_examples = [e for e in examples if e.split == "train"]
    return ContaminationReport(
        derivation_id_split_violations=check_derivation_id_isolation(examples),
        exact_cross_split_duplicates=check_exact_cross_split_duplicates(examples),
        near_duplicate_eval_train_pairs=check_near_duplicate_eval_vs_train(
            eval_prompts, train_examples, shingle_size, threshold
        ),
    )
