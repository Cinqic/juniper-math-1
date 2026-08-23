"""Phase 5 tool-format evaluation infrastructure tests.

Focus: the evaluator must never crash on malformed model output and must
correctly extract/parse well-formed `<tool_call>` blocks, per Sec. 19 ("the
smoke model is allowed to perform badly; the evaluator is not allowed to
break").
"""

from __future__ import annotations

from juniper_math.tool_format_eval import extract_tool_call_text
from juniper_math.tools.protocol import parse_tool_call
from juniper_math.tools.runtime import ToolRuntime


def _limits():
    return ToolRuntime().limits


def test_extract_tool_call_text_finds_well_formed_block():
    text = '<tool_call>{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{"expression":"2+2"}}'  # noqa: E501
    extracted = extract_tool_call_text(text)
    assert extracted is not None
    call = parse_tool_call(extracted, _limits())
    assert call.tool == "calculator.evaluate"


def test_extract_tool_call_text_stops_at_next_tag():
    text = (
        '<tool_call>{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{"expression":"2+2"}}'
        '<tool_result>{"status":"success"}'
    )
    extracted = extract_tool_call_text(text)
    assert extracted is not None
    assert "<tool_result>" not in extracted
    call = parse_tool_call(extracted, _limits())
    assert call.tool == "calculator.evaluate"


def test_extract_tool_call_text_returns_none_when_absent():
    assert extract_tool_call_text("just some rambling generated text with no tags") is None


def test_extract_tool_call_text_handles_garbage_without_crashing():
    garbage = "<tool_call>not json at all {{{"
    extracted = extract_tool_call_text(garbage)
    assert extracted == "not json at all {{{"
    try:
        parse_tool_call(extracted, _limits())
        raised = False
    except Exception:  # noqa: BLE001 - asserting *some* ToolProtocolError-shaped failure, not a crash
        raised = True
    assert raised


def test_extract_tool_call_text_finds_call_after_final_tag_garbage():
    text = '<final>garbage<tool_call>{"protocol_version":"1.0.0","tool":"calculator.convert","arguments":{}}'
    extracted = extract_tool_call_text(text)
    assert extracted is not None
    assert extracted.startswith("{")
