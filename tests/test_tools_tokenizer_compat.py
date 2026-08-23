"""Phase 2 tokenizer / Phase 1 model compatibility for the Phase 3 wire
protocol. This is a mechanical compatibility check only — it does not claim
the untrained model knows when or how to use a tool. See config/tools.yaml
and docs/TOOLS.md "Tokenizer representation".
"""

from __future__ import annotations

import pytest

from juniper_math.tokenizer import JuniperTokenizer
from juniper_math.tools.protocol import ToolCall, ToolResult, wire_tool_call, wire_tool_result
from juniper_math.tools.runtime import ToolRuntime

_UNK_ID = 0


@pytest.fixture(scope="module")
def tokenizer():
    return JuniperTokenizer.load()


@pytest.fixture(scope="module")
def runtime():
    return ToolRuntime()


def _sample_calls():
    return [
        ToolCall("1.0.0", "calculator.evaluate", {"expression": "84317 * 9926"}),
        ToolCall(
            "1.0.0",
            "calculator.convert",
            {"category": "length", "from_unit": "mile", "to_unit": "meter", "value": 1},
        ),
        ToolCall("1.0.0", "calculator.finance", {"operation": "tip", "bill_total": 42.5, "tip_percent": 20}),
    ]


def test_tool_call_wire_strings_have_no_unk_and_round_trip(tokenizer):
    for call in _sample_calls():
        text = wire_tool_call(call)
        ids = tokenizer.encode(text)
        assert _UNK_ID not in ids
        assert all(0 <= i <= 4095 for i in ids)
        assert tokenizer.decode(ids) == text


def test_tool_result_wire_strings_have_no_unk_and_round_trip(tokenizer, runtime):
    for call in _sample_calls():
        result = runtime.execute_call(call)
        text = wire_tool_result(result)
        ids = tokenizer.encode(text)
        assert _UNK_ID not in ids
        assert all(0 <= i <= 4095 for i in ids)
        assert tokenizer.decode(ids) == text


def test_wire_strings_begin_with_the_frozen_control_token(tokenizer):
    call = ToolCall("1.0.0", "calculator.evaluate", {"expression": "2+2"})
    call_ids = tokenizer.encode(wire_tool_call(call))
    assert call_ids[0] == 4  # <tool_call>, frozen Phase 2 ID

    result = ToolResult("1.0.0", "calculator.evaluate", "success", result={"value": "4", "exact": True})
    result_ids = tokenizer.encode(wire_tool_result(result))
    assert result_ids[0] == 5  # <tool_result>, frozen Phase 2 ID


def test_model_embedding_accepts_tool_protocol_ids(tokenizer, runtime):
    torch = pytest.importorskip("torch")
    from juniper_math.model import build_model

    model = build_model()
    call = ToolCall("1.0.0", "calculator.evaluate", {"expression": "2+2"})
    ids = tokenizer.encode(wire_tool_call(call))
    with torch.no_grad():
        output = model(torch.tensor([ids]))
    assert output is not None
