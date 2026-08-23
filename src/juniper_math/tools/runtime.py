"""Trusted dispatch pipeline: untrusted serialized call -> parser -> schema
validator -> typed trusted call -> registry -> calculator backend -> typed
result -> canonical serializer.

Every ToolResult returned by this module was built by this module. A
model-generated string is never treated as a ToolResult merely because it
parses as one — see protocol.py's module docstring and
docs/TOOLS.md "Fabricated result protection". The runtime is stateless: a
result depends only on (tool version, call arguments, fixed configuration),
never on any previous call — see docs/TOOLS.md "Statelessness".
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from juniper_math.tools.calculator_backend import compute_finance, convert_value, evaluate_expression
from juniper_math.tools.config import ToolsConfig, load_tools_config
from juniper_math.tools.errors import ToolProtocolError
from juniper_math.tools.protocol import CallLimits, ErrorInfo, ToolCall, ToolResult, parse_tool_call
from juniper_math.tools.registry import ToolRegistry
from juniper_math.tools.schemas import (
    validate_convert_arguments,
    validate_evaluate_arguments,
    validate_finance_arguments,
)


def _format_decimal(value: Decimal) -> str:
    return format(value, "f")


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return repr(value)


def _handle_evaluate(call: ToolCall, config: ToolsConfig) -> dict[str, Any]:
    expression = validate_evaluate_arguments(dict(call.arguments))
    outcome = evaluate_expression(expression, config.limits)
    return {"value": _format_number(outcome.value), "exact": outcome.exact}


def _handle_convert(call: ToolCall, config: ToolsConfig) -> dict[str, Any]:
    category, from_unit, to_unit, value = validate_convert_arguments(dict(call.arguments))
    result = convert_value(category, from_unit, to_unit, value)
    return {"value": _format_decimal(result)}


def _handle_finance(call: ToolCall, config: ToolsConfig) -> dict[str, Any]:
    operation, args = validate_finance_arguments(dict(call.arguments))
    result = compute_finance(operation, args)
    return {"value": _format_decimal(result)}


class ToolRuntime:
    def __init__(self, config: ToolsConfig | None = None) -> None:
        self.config = config or load_tools_config()
        self.limits = CallLimits(
            max_call_bytes=self.config.limits.max_call_bytes,
            max_string_argument_length=self.config.limits.max_string_argument_length,
            max_json_depth=self.config.limits.max_json_depth,
            max_json_members=self.config.limits.max_json_members,
        )
        self.registry = ToolRegistry(self.config)
        self.registry.register("calculator.evaluate", lambda call: _handle_evaluate(call, self.config))
        self.registry.register("calculator.convert", lambda call: _handle_convert(call, self.config))
        self.registry.register("calculator.finance", lambda call: _handle_finance(call, self.config))

    def _error_result(self, tool: str, code: str, message: str) -> ToolResult:
        # UNKNOWN_TOOL is always "unsupported"; a known tool with an
        # unsupported *operation* argument is still an "error" (the tool
        # itself is supported — see docs/TOOLS.md "Execution states").
        if code == "UNKNOWN_TOOL":
            status = "unsupported"
        elif code == "TOOL_UNAVAILABLE":
            status = "unavailable"
        else:
            status = "error"
        return ToolResult(
            protocol_version=self.config.protocol_version,
            tool=tool,
            status=status,
            error=ErrorInfo(code=code, message=message),
        )

    def execute_call(self, call: ToolCall) -> ToolResult:
        if call.protocol_version != self.config.protocol_version:
            message = (
                f"Unsupported protocol_version: {call.protocol_version!r} "
                f"(expected {self.config.protocol_version!r})"
            )
            return self._error_result(call.tool, "UNSUPPORTED_PROTOCOL_VERSION", message)

        if not self.registry.is_known(call.tool):
            return self._error_result(call.tool, "UNKNOWN_TOOL", f"Unknown tool: {call.tool!r}")

        registration = self.registry.get(call.tool)
        if registration is None or not registration.available:
            message = f"Tool currently unavailable: {call.tool!r}"
            return self._error_result(call.tool, "TOOL_UNAVAILABLE", message)

        try:
            result_payload = registration.handler(call)
        except ToolProtocolError as exc:
            return self._error_result(call.tool, exc.code, exc.message)
        except (ArithmeticError, ValueError, OverflowError):  # defensive backstop, never a raw traceback
            return self._error_result(call.tool, "INTERNAL_ERROR", "Internal error during tool execution")

        return ToolResult(
            protocol_version=self.config.protocol_version,
            tool=call.tool,
            status="success",
            result=result_payload,
        )

    def execute_text(self, text: str) -> ToolResult:
        try:
            call = parse_tool_call(text, self.limits)
        except ToolProtocolError as exc:
            return ToolResult(
                protocol_version=self.config.protocol_version,
                tool="",
                status="error",
                error=ErrorInfo(code=exc.code, message=exc.message),
            )
        return self.execute_call(call)
