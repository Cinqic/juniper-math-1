"""Stable internal record schema for Phase 4 dataset examples (Sec. 6).

Every field answers one of the provenance questions Sec. 6 requires: where
an example came from, how its ground truth was established, which tool (if
any) actually ran, and which split it may enter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from juniper_math.errors import JuniperConfigError

# Reuses the exact expected_behavior vocabulary already frozen in
# juniper_math.evals — one behavior vocabulary for the whole project, not a
# second, subtly different one invented for the training corpus.
VALID_CATEGORIES = frozenset(
    {
        "arithmetic",
        "operator_precedence",
        "negative_values",
        "decimals",
        "fractions",
        "percentages",
        "ratios_proportions",
        "scientific_notation",
        "unit_conversion",
        "financial_math",
        "basic_algebra",
        "expression_translation",
        "word_problem",
        "estimation",
        "numerical_comparison",
        "multi_step",
        "tool_use",
        "incorrect_supplied_answer",
        "incorrect_tool_call",
        "ambiguity",
        "missing_information",
        "undefined_operation",
        "unsupported_capability",
        "tool_error",
    }
)

VALID_DIFFICULTIES = frozenset({"trivial", "easy", "medium", "hard"})

VALID_EXPECTED_BEHAVIORS = frozenset(
    {
        "answer",
        "request_clarification",
        "refuse_unsupported",
        "flag_missing_information",
        "flag_undefined",
        "flag_incorrect_answer",
        "invoke_tool",
    }
)

VALID_VERIFICATION_MODES = frozenset({"deterministic", "semantic", "tool"})

VALID_SPLITS = frozenset({"train", "validation", "test"})


@dataclass(frozen=True)
class ToolTrace:
    call: dict[str, Any]
    result: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"call": self.call, "result": self.result}


@dataclass(frozen=True)
class Example:
    example_id: str
    generator_id: str
    generator_version: str
    family_id: str
    template_id: str
    derivation_id: str
    seed: int
    category: str
    difficulty: str
    synthetic: bool
    split: str
    prompt: str
    expected_behavior: str
    expected_answer: Any
    tolerance: float | None
    tool_required: bool
    tool_name: str | None
    tool_traces: tuple[ToolTrace, ...]
    verification: dict[str, Any]
    provenance: str
    notes: str
    token_count: int | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tool_traces"] = [t.to_dict() for t in self.tool_traces]
        return d

    def with_token_count(self, token_count: int) -> Example:
        return Example(
            **{**asdict(self), "tool_traces": self.tool_traces, "token_count": token_count},
        )

    def with_split(self, split: str) -> Example:
        data = asdict(self)
        data["tool_traces"] = self.tool_traces
        data["split"] = split
        return Example(**data)


def validate_example(ex: Example, *, context: str = "example") -> None:
    """Raise JuniperConfigError on any structural/invariant violation."""
    if not ex.example_id:
        raise JuniperConfigError(f"{context}: example_id must be non-empty")
    if ex.category not in VALID_CATEGORIES:
        raise JuniperConfigError(f"{context} {ex.example_id!r}: unknown category {ex.category!r}")
    if ex.difficulty not in VALID_DIFFICULTIES:
        raise JuniperConfigError(f"{context} {ex.example_id!r}: unknown difficulty {ex.difficulty!r}")
    if ex.expected_behavior not in VALID_EXPECTED_BEHAVIORS:
        raise JuniperConfigError(
            f"{context} {ex.example_id!r}: unknown expected_behavior {ex.expected_behavior!r}"
        )
    if ex.split not in VALID_SPLITS:
        raise JuniperConfigError(f"{context} {ex.example_id!r}: unknown split {ex.split!r}")
    if not ex.prompt or not ex.prompt.strip():
        raise JuniperConfigError(f"{context} {ex.example_id!r}: prompt must be non-empty")
    mode = ex.verification.get("mode")
    if mode not in VALID_VERIFICATION_MODES:
        raise JuniperConfigError(f"{context} {ex.example_id!r}: invalid verification mode {mode!r}")
    if mode == "semantic" and ex.expected_answer is not None:
        raise JuniperConfigError(
            f"{context} {ex.example_id!r}: semantic verification requires expected_answer=null"
        )
    if ex.tool_required and not ex.tool_traces:
        raise JuniperConfigError(
            f"{context} {ex.example_id!r}: tool_required=True but no tool_traces recorded "
            "(ground truth for a tool example must come from executing the real runtime)"
        )
    if ex.tool_required and ex.tool_name is None:
        raise JuniperConfigError(f"{context} {ex.example_id!r}: tool_required=True but tool_name is null")
    if not ex.tool_required and ex.tool_traces:
        raise JuniperConfigError(f"{context} {ex.example_id!r}: tool_traces recorded but tool_required=False")
    if ex.tolerance is not None and ex.tolerance < 0:
        raise JuniperConfigError(f"{context} {ex.example_id!r}: tolerance must be non-negative")
