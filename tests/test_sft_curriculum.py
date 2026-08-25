"""Independent direct-math curriculum invariants."""

from __future__ import annotations

from juniper_math.dataset.schema import validate_example
from juniper_math.dataset.verify import verify_deterministic
from juniper_math.sft_curriculum import (
    DIRECT_CATEGORIES,
    SAFETY_CATEGORIES,
    TOOL_BUILDERS,
    build_independent_direct_examples,
    build_independent_safety_examples,
    build_independent_tool_examples,
)


def test_independent_curriculum_is_deterministic_and_verifiable():
    first = build_independent_direct_examples("train", 3, 5004032)
    second = build_independent_direct_examples("train", 3, 5004032)
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert len(first) == len(DIRECT_CATEGORIES) * 3
    assert {item.category for item in first} == set(DIRECT_CATEGORIES)
    for item in first:
        validate_example(item)
        valid, detail = verify_deterministic(
            item.verification["expression"], item.expected_answer, item.tolerance, item.example_id
        )
        assert valid, detail
        assert item.tool_required is False
        assert item.tool_traces == ()


def test_independent_curriculum_changes_with_split():
    train = build_independent_direct_examples("train", 1, 7)
    validation = build_independent_direct_examples("validation", 1, 7)
    assert {item.example_id for item in train}.isdisjoint(item.example_id for item in validation)


def test_independent_safety_curriculum_is_deterministic_and_answerless():
    first = build_independent_safety_examples("train", 3, 5004032)
    second = build_independent_safety_examples("train", 3, 5004032)
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert len(first) == len(SAFETY_CATEGORIES) * 3
    for item in first:
        validate_example(item)
        assert item.expected_answer is None
        assert item.tool_required is False
        assert item.tool_traces == ()


def test_independent_tool_curriculum_is_deterministic_and_runtime_backed():
    first = build_independent_tool_examples("train", 2, 5004032)
    second = build_independent_tool_examples("train", 2, 5004032)
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert len(first) == len(TOOL_BUILDERS) * 2
    for item in first:
        validate_example(item)
        assert item.tool_required is True
        assert len(item.tool_traces) == 1
        assert item.tool_traces[0].result["status"] == "success"
