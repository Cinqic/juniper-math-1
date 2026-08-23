from __future__ import annotations

import pytest

from juniper_math.tools.config import load_tools_config
from juniper_math.tools.protocol import ToolCall
from juniper_math.tools.registry import ToolRegistry
from juniper_math.tools.runtime import ToolRuntime

CONFIG = load_tools_config()


def test_registry_rejects_registering_unapproved_tool():
    registry = ToolRegistry(CONFIG)
    with pytest.raises(ValueError):
        registry.register("calculator.python", lambda call: {})


def test_registry_reports_unknown_tools():
    registry = ToolRegistry(CONFIG)
    assert registry.is_known("calculator.evaluate") is True
    assert registry.is_known("shell.exec") is False


def test_unavailable_tool_reports_tool_unavailable_not_traceback():
    runtime = ToolRuntime()
    runtime.registry.set_available("calculator.convert", False)
    result = runtime.execute_call(ToolCall("1.0.0", "calculator.convert", {}))
    assert result.status == "unavailable"
    assert result.error.code == "TOOL_UNAVAILABLE"


def test_disabling_one_tool_does_not_affect_others():
    runtime = ToolRuntime()
    runtime.registry.set_available("calculator.convert", False)
    result = runtime.execute_call(ToolCall("1.0.0", "calculator.evaluate", {"expression": "2+2"}))
    assert result.status == "success"


def test_set_available_unknown_tool_raises():
    registry = ToolRegistry(CONFIG)
    with pytest.raises(KeyError):
        registry.set_available("calculator.evaluate", False)
