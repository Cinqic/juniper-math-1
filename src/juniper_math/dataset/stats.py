"""Dataset statistics computation (Sec. 23)."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from juniper_math.dataset.schema import Example


def compute_stats(
    examples: list[Example], max_context_length: int, build_counters: dict[str, int]
) -> dict[str, Any]:
    token_counts = [e.token_count or 0 for e in examples]
    by_split: dict[str, list[Example]] = {}
    for e in examples:
        by_split.setdefault(e.split, []).append(e)

    def _pct(sorted_vals: list[int], p: float) -> int:
        if not sorted_vals:
            return 0
        idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p))
        return sorted_vals[idx]

    sorted_tokens = sorted(token_counts)

    return {
        "total_examples": len(examples),
        "total_tokens": sum(token_counts),
        "total_bytes": sum(len(e.prompt.encode("utf-8")) for e in examples),
        "split_composition": {
            split: {
                "examples": len(exs),
                "tokens": sum(e.token_count or 0 for e in exs),
            }
            for split, exs in sorted(by_split.items())
        },
        "category_distribution": dict(sorted(Counter(e.category for e in examples).items())),
        "difficulty_distribution": dict(sorted(Counter(e.difficulty for e in examples).items())),
        "tool_required_distribution": dict(
            Counter("tool_required" if e.tool_required else "no_tool" for e in examples)
        ),
        "synthetic_vs_external": dict(Counter("synthetic" if e.synthetic else "external" for e in examples)),
        "generator_family_distribution": dict(
            sorted(Counter(f"{e.generator_id}/{e.family_id}" for e in examples).items())
        ),
        "tool_name_distribution": dict(Counter(e.tool_name for e in examples if e.tool_name)),
        "expected_behavior_distribution": dict(
            sorted(Counter(e.expected_behavior for e in examples).items())
        ),
        "average_tokens_per_example": (sum(token_counts) / len(token_counts)) if token_counts else 0,
        "median_tokens_per_example": statistics.median(token_counts) if token_counts else 0,
        "token_length_percentiles": {
            "p50": _pct(sorted_tokens, 0.50),
            "p90": _pct(sorted_tokens, 0.90),
            "p99": _pct(sorted_tokens, 0.99),
        },
        "max_example_tokens": max(token_counts) if token_counts else 0,
        "fraction_exceeding_context": (
            sum(1 for t in token_counts if t > max_context_length) / len(token_counts) if token_counts else 0
        ),
        "build_counters": build_counters,
    }
