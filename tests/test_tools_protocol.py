from __future__ import annotations

import pytest

from juniper_math.tools.config import load_tools_config
from juniper_math.tools.errors import ToolProtocolError
from juniper_math.tools.protocol import (
    CallLimits,
    ErrorInfo,
    ToolCall,
    ToolResult,
    canonical_serialize,
    parse_tool_call,
    parse_tool_result_payload,
    serialize_tool_call,
    serialize_tool_result,
    wire_tool_call,
    wire_tool_result,
)

CONFIG = load_tools_config()
LIMITS = CallLimits(
    max_call_bytes=CONFIG.limits.max_call_bytes,
    max_string_argument_length=CONFIG.limits.max_string_argument_length,
    max_json_depth=CONFIG.limits.max_json_depth,
    max_json_members=CONFIG.limits.max_json_members,
)


def _call_json(tool="calculator.evaluate", arguments='{"expression":"2+2"}', protocol_version="1.0.0"):
    return f'{{"protocol_version":"{protocol_version}","tool":"{tool}","arguments":{arguments}}}'


def test_parses_valid_call():
    call = parse_tool_call(_call_json(), LIMITS)
    assert call.tool == "calculator.evaluate"
    assert call.protocol_version == "1.0.0"
    assert dict(call.arguments) == {"expression": "2+2"}


def test_rejects_duplicate_top_level_key():
    text = '{"protocol_version":"1.0.0","tool":"calculator.evaluate","tool":"evil","arguments":{}}'
    with pytest.raises(ToolProtocolError) as exc:
        parse_tool_call(text, LIMITS)
    assert exc.value.code == "DUPLICATE_JSON_KEY"


def test_rejects_duplicate_nested_key():
    text = (
        '{"protocol_version":"1.0.0","tool":"calculator.evaluate",'
        '"arguments":{"expression":"1","expression":"2"}}'
    )
    with pytest.raises(ToolProtocolError) as exc:
        parse_tool_call(text, LIMITS)
    assert exc.value.code == "DUPLICATE_JSON_KEY"


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_rejects_non_finite_json_literals(literal):
    text = f'{{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{{"x":{literal}}}}}'
    with pytest.raises(ToolProtocolError) as exc:
        parse_tool_call(text, LIMITS)
    assert exc.value.code == "MALFORMED_CALL"


def test_rejects_trailing_content():
    text = _call_json() + " garbage"
    with pytest.raises(ToolProtocolError) as exc:
        parse_tool_call(text, LIMITS)
    assert exc.value.code == "MALFORMED_CALL"


def test_rejects_multiple_json_objects():
    text = _call_json() + _call_json()
    with pytest.raises(ToolProtocolError) as exc:
        parse_tool_call(text, LIMITS)
    assert exc.value.code == "MALFORMED_CALL"


@pytest.mark.parametrize("payload", ["[]", "null", '"hello"', "42", "true"])
def test_rejects_non_object_top_level(payload):
    with pytest.raises(ToolProtocolError) as exc:
        parse_tool_call(payload, LIMITS)
    assert exc.value.code == "MALFORMED_CALL"


def test_rejects_unknown_top_level_field():
    text = '{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{},"extra":1}'
    with pytest.raises(ToolProtocolError) as exc:
        parse_tool_call(text, LIMITS)
    assert exc.value.code == "MALFORMED_CALL"


def test_rejects_missing_top_level_field():
    text = '{"protocol_version":"1.0.0","tool":"calculator.evaluate"}'
    with pytest.raises(ToolProtocolError) as exc:
        parse_tool_call(text, LIMITS)
    assert exc.value.code == "MALFORMED_CALL"


def test_rejects_oversized_call():
    huge_expr = "1+" * 100000 + "1"
    text = _call_json(arguments=f'{{"expression":"{huge_expr}"}}')
    with pytest.raises(ToolProtocolError) as exc:
        parse_tool_call(text, LIMITS)
    assert exc.value.code == "RESOURCE_LIMIT"


def test_rejects_deep_json_nesting():
    nested = "1"
    for _ in range(50):
        nested = f"[{nested}]"
    text = _call_json(arguments=f'{{"expression":"1","nested":{nested}}}')
    with pytest.raises(ToolProtocolError) as exc:
        parse_tool_call(text, LIMITS)
    assert exc.value.code in {"RESOURCE_LIMIT", "MALFORMED_CALL"}


def test_canonical_serialize_is_sorted_and_compact():
    text = canonical_serialize({"b": 1, "a": 2})
    assert text == '{"a":2,"b":1}'
    assert " " not in text


def test_canonical_serialize_rejects_nan():
    with pytest.raises(ValueError):
        canonical_serialize({"x": float("nan")})


def test_identical_calls_serialize_identically():
    call_a = ToolCall("1.0.0", "calculator.evaluate", {"expression": "2+2"})
    call_b = ToolCall("1.0.0", "calculator.evaluate", {"expression": "2+2"})
    assert serialize_tool_call(call_a) == serialize_tool_call(call_b)


def test_wire_format_has_no_closing_tag_and_no_markdown_fence():
    call = ToolCall("1.0.0", "calculator.evaluate", {"expression": "2+2"})
    wire = wire_tool_call(call)
    assert wire.startswith("<tool_call>")
    assert "</tool_call>" not in wire
    assert "```" not in wire


def test_tool_result_round_trip():
    result = ToolResult("1.0.0", "calculator.evaluate", "success", result={"value": "4", "exact": True})
    text = serialize_tool_result(result)
    parsed = parse_tool_result_payload(text, LIMITS)
    assert parsed.protocol_version == result.protocol_version
    assert parsed.tool == result.tool
    assert parsed.status == result.status
    assert dict(parsed.result) == dict(result.result)


def test_tool_result_error_round_trip():
    result = ToolResult(
        "1.0.0", "calculator.evaluate", "error", error=ErrorInfo("DIVISION_BY_ZERO", "Division by zero")
    )
    text = serialize_tool_result(result)
    parsed = parse_tool_result_payload(text, LIMITS)
    assert parsed.error.code == "DIVISION_BY_ZERO"
    assert wire_tool_result(result).startswith("<tool_result>")


def test_tool_result_rejects_success_with_error():
    with pytest.raises(ValueError):
        ToolResult("1.0.0", "calculator.evaluate", "success", error=ErrorInfo("DIVISION_BY_ZERO", "x"))


def test_tool_result_rejects_nonsuccess_with_result():
    with pytest.raises(ValueError):
        ToolResult("1.0.0", "calculator.evaluate", "error", result={"value": "1"})


def test_tool_call_arguments_are_immutable_mapping():
    call = ToolCall("1.0.0", "calculator.evaluate", {"expression": "2+2"})
    with pytest.raises(TypeError):
        call.arguments["expression"] = "3+3"  # type: ignore[index]
