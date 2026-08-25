"""Sec. 11's required loss-masking tests, plus the segment/tokenization
correctness tests reports/PHASE8_PLAN.md Sec. 5 commits to running before any
training."""

from __future__ import annotations

import pytest

from juniper_math.dataset.schema import Example, ToolTrace
from juniper_math.sft_rendering import SftRenderingError, render_segments, rendered_text, tokenize_and_mask
from juniper_math.tokenizer import JuniperTokenizer


@pytest.fixture(scope="module")
def tokenizer() -> JuniperTokenizer:
    return JuniperTokenizer.load()


def _direct_example(expected_answer: str = "42") -> Example:
    return Example(
        example_id="test_direct_0001",
        generator_id="test",
        generator_version="1.0.0",
        family_id="test_family",
        template_id="test_template",
        derivation_id="test_derivation",
        seed=1,
        category="arithmetic",
        difficulty="easy",
        synthetic=True,
        split="train",
        prompt="What is 20 + 22?",
        expected_behavior="answer",
        expected_answer=expected_answer,
        tolerance=0,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "deterministic", "expression": {"op": "add", "args": [20, 22]}},
        provenance="test",
        notes="",
    )


def _tool_example() -> Example:
    call = {"protocol_version": "1.0.0", "tool": "calculator.evaluate", "arguments": {"expression": "2 + 2"}}
    result = {
        "protocol_version": "1.0.0",
        "tool": "calculator.evaluate",
        "status": "success",
        "result": {"value": "4", "exact": True},
        "error": None,
    }
    return Example(
        example_id="test_tool_0001",
        generator_id="test",
        generator_version="1.0.0",
        family_id="test_family",
        template_id="test_template",
        derivation_id="test_derivation",
        seed=2,
        category="tool_use",
        difficulty="easy",
        synthetic=True,
        split="train",
        prompt="Use the calculator to evaluate 2 + 2.",
        expected_behavior="answer",
        expected_answer="4",
        tolerance=0,
        tool_required=True,
        tool_name="calculator.evaluate",
        tool_traces=(ToolTrace(call=call, result=result),),
        verification={"mode": "tool", "expression": None},
        provenance="test",
        notes="",
    )


def _clarification_example() -> Example:
    return Example(
        example_id="test_clarify_0001",
        generator_id="test",
        generator_version="1.0.0",
        family_id="test_family",
        template_id="test_template",
        derivation_id="test_derivation",
        seed=3,
        category="ambiguity",
        difficulty="easy",
        synthetic=True,
        split="train",
        prompt="How much is it?",
        expected_behavior="request_clarification",
        expected_answer=None,
        tolerance=None,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "semantic", "expression": None},
        provenance="test",
        notes="",
    )


def _padded(mt, max_len, pad_id):
    ids = mt.ids + [pad_id] * (max_len - len(mt.ids))
    labels = mt.labels + [-100] * (max_len - len(mt.labels))
    return ids, labels


# --- Sec. 11 required tests ---


def test_user_prompt_tokens_are_masked(tokenizer):
    ex = _direct_example()
    mt = tokenize_and_mask(ex, tokenizer, 64)
    prompt_len = len(tokenizer.encode(ex.prompt))
    # BOS (index 0) + every prompt token must be -100.
    assert mt.labels[0] == -100
    for i in range(1, 1 + prompt_len):
        assert mt.labels[i] == -100, f"prompt token at index {i} is not masked"


def test_tool_result_tokens_are_masked(tokenizer):
    ex = _tool_example()
    segments = render_segments(ex)
    mt = tokenize_and_mask(ex, tokenizer, 256)
    # Recompute the tool_result segment's token span and assert every label there is -100.
    cursor = 1  # after BOS
    for seg in segments:
        seg_ids = tokenizer.encode(seg.text)
        span = mt.labels[cursor : cursor + len(seg_ids)]
        if seg.role == "context" and "<tool_result>" in seg.text:
            assert all(x == -100 for x in span), "tool_result segment has a supervised label"
        cursor += len(seg_ids)


def test_tool_call_tokens_are_supervised(tokenizer):
    ex = _tool_example()
    segments = render_segments(ex)
    mt = tokenize_and_mask(ex, tokenizer, 256)
    cursor = 1
    found_supervised_call = False
    for seg in segments:
        seg_ids = tokenizer.encode(seg.text)
        span = mt.labels[cursor : cursor + len(seg_ids)]
        if seg.role == "supervised" and "<tool_call>" in seg.text:
            assert all(x != -100 for x in span), "tool_call segment has a masked label"
            assert span == seg_ids
            found_supervised_call = True
        cursor += len(seg_ids)
    assert found_supervised_call


def test_final_answer_tokens_are_supervised(tokenizer):
    ex = _direct_example()
    segments = render_segments(ex)
    mt = tokenize_and_mask(ex, tokenizer, 64)
    cursor = 1
    found = False
    for seg in segments:
        seg_ids = tokenizer.encode(seg.text)
        span = mt.labels[cursor : cursor + len(seg_ids)]
        if seg.role == "supervised" and "<final>" in seg.text:
            assert all(x != -100 for x in span)
            found = True
        cursor += len(seg_ids)
    assert found
    # EOS itself is also supervised (it is the correct stop signal).
    assert mt.labels[-1] == mt.ids[-1] != -100


def test_padding_is_masked(tokenizer):
    ex = _direct_example()
    mt = tokenize_and_mask(ex, tokenizer, 64)
    pad_id = tokenizer.token_to_id("<pad>")
    ids, labels = _padded(mt, 64, pad_id)
    for i in range(len(mt.ids), 64):
        assert ids[i] == pad_id
        assert labels[i] == -100


def test_causal_shift_preserves_intended_target(tokenizer):
    """The model shifts labels by one internally (JuniperMathModel.forward):
    logits[:, t] predicts labels[:, t+1]. This test asserts the label at
    position t+1 is exactly the token id at position t+1 whenever that
    position is supervised — i.e. shifting never accidentally aligns a
    supervised label with the WRONG input token."""
    ex = _tool_example()
    mt = tokenize_and_mask(ex, tokenizer, 256)
    for t in range(len(mt.ids) - 1):
        if mt.labels[t + 1] != -100:
            assert mt.labels[t + 1] == mt.ids[t + 1]


def test_no_target_shifted_onto_context_only_token(tokenizer):
    """No supervised label may coincide with a position whose role is
    context-only (prompt or tool_result) — cross-checks the per-segment
    role assignment directly against the final label array."""
    ex = _tool_example()
    segments = render_segments(ex)
    mt = tokenize_and_mask(ex, tokenizer, 256)
    cursor = 1
    for seg in segments:
        seg_ids = tokenizer.encode(seg.text)
        span = mt.labels[cursor : cursor + len(seg_ids)]
        if seg.role == "context":
            assert all(x == -100 for x in span), f"context segment {seg.text!r} leaked a supervised label"
        cursor += len(seg_ids)


@pytest.mark.parametrize(
    "example_factory",
    [_direct_example, _tool_example, _clarification_example],
    ids=["direct", "tool", "clarify"],
)
def test_different_trajectory_types_produce_correct_masks(tokenizer, example_factory):
    ex = example_factory()
    mt = tokenize_and_mask(ex, tokenizer, 256)
    assert len(mt.ids) == len(mt.labels)
    assert mt.labels[0] == -100  # BOS
    assert any(x != -100 for x in mt.labels), "every trajectory type must have SOME supervised token"
    assert mt.labels[-1] == mt.ids[-1]  # EOS supervised


def test_truncation_rejects_rather_than_corrupts(tokenizer):
    ex = _tool_example()
    full = tokenize_and_mask(ex, tokenizer, 256)
    with pytest.raises(SftRenderingError):
        tokenize_and_mask(ex, tokenizer, len(full.ids) - 1)


# --- Rendering/segmentation correctness ---


def test_render_segments_concatenation_matches_dataset_shard_render(tokenizer):
    from juniper_math.dataset.shard import render_training_text

    for ex in (_direct_example(), _tool_example(), _clarification_example()):
        assert rendered_text(ex) == render_training_text(ex)


def test_segment_wise_tokenization_matches_joint_tokenization(tokenizer):
    from juniper_math.dataset.shard import render_training_text

    for ex in (_direct_example(), _tool_example(), _clarification_example()):
        joint_text = render_training_text(ex)
        joint_ids = (
            [tokenizer.token_to_id("<s>")] + tokenizer.encode(joint_text) + [tokenizer.token_to_id("</s>")]
        )
        mt = tokenize_and_mask(ex, tokenizer, 256)
        assert mt.ids == joint_ids


def test_tool_error_gets_derived_supervised_error_completion(tokenizer):
    """The derived Phase 8 representation must train the next assistant
    turn after a trusted runtime error rather than EOS after context."""
    call = {"protocol_version": "1.0.0", "tool": "calculator.evaluate", "arguments": {"expression": "2 + 2"}}
    result = {
        "protocol_version": "1.0.0",
        "tool": "calculator.evaluate",
        "status": "error",
        "result": None,
        "error": {"code": "DIVISION_BY_ZERO", "message": "Division by zero"},
    }
    ex = Example(
        example_id="test_invoke_only_0001",
        generator_id="test",
        generator_version="1.0.0",
        family_id="f",
        template_id="t",
        derivation_id="d",
        seed=4,
        category="tool_error",
        difficulty="easy",
        synthetic=True,
        split="train",
        prompt="Use the calculator to evaluate 2 + 2.",
        expected_behavior="invoke_tool",
        expected_answer=None,
        tolerance=None,
        tool_required=True,
        tool_name="calculator.evaluate",
        tool_traces=(ToolTrace(call=call, result=result),),
        verification={"mode": "tool", "expression": None},
        provenance="test",
        notes="",
    )
    segments = render_segments(ex)
    assert segments[-1].role == "supervised"
    assert segments[-1].text == "\n<error>DIVISION_BY_ZERO: Division by zero"
    mt = tokenize_and_mask(ex, tokenizer, 256)
    assert mt.labels[-1] == mt.ids[-1]


def test_answerless_non_error_invoke_tool_is_rejected():
    ex = _tool_example()
    ex = Example(**{**ex.__dict__, "expected_answer": None, "expected_behavior": "invoke_tool"})
    with pytest.raises(SftRenderingError, match="no terminal assistant completion"):
        render_segments(ex)


def test_unknown_answerless_behavior_raises():
    ex = _direct_example()
    bad = Example(**{**ex.__dict__, "expected_answer": None, "expected_behavior": "flag_incorrect_answer"})
    with pytest.raises(SftRenderingError):
        render_segments(bad)
