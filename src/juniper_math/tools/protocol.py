"""Canonical tool-call / tool-result envelopes: strict parsing, typed
representation, and deterministic serialization.

Trust boundary (see docs/TOOLS.md): everything entering :func:`parse_tool_call`
is untrusted, model-generated text. Nothing here executes anything — this
module only turns bytes into a validated, typed :class:`ToolCall`, or raises
a :class:`~juniper_math.tools.errors.ToolProtocolError`. A :class:`ToolResult`
is only ever constructed by the trusted host runtime (see runtime.py); a
model-generated string that merely *looks like* a ToolResult (even one that
round-trips through :func:`parse_tool_result`) is never treated as evidence
that a tool actually ran — see docs/TOOLS.md "Fabricated result protection".
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from juniper_math.tools.errors import ToolProtocolError

CANONICAL_SEPARATORS = (",", ":")


@dataclass(frozen=True)
class CallLimits:
    max_call_bytes: int
    max_string_argument_length: int
    max_json_depth: int
    max_json_members: int


@dataclass(frozen=True)
class ToolCall:
    protocol_version: str
    tool: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))


@dataclass(frozen=True)
class ErrorInfo:
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class ToolResult:
    protocol_version: str
    tool: str
    status: str
    result: Mapping[str, Any] | None = None
    error: ErrorInfo | None = None

    def __post_init__(self) -> None:
        if self.status not in {"success", "error", "unavailable", "unsupported"}:
            raise ValueError(f"Invalid ToolResult.status: {self.status!r}")
        if self.status == "success" and self.error is not None:
            raise ValueError("A success result must not carry an error")
        if self.status != "success" and self.result is not None:
            raise ValueError("A non-success result must not carry a result payload")
        if self.result is not None:
            object.__setattr__(self, "result", MappingProxyType(dict(self.result)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "tool": self.tool,
            "status": self.status,
            "result": dict(self.result) if self.result is not None else None,
            "error": self.error.to_dict() if self.error is not None else None,
        }


def _reject_constant(text: str) -> Any:
    raise ToolProtocolError("MALFORMED_CALL", f"Non-finite numeric literal not allowed: {text}")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ToolProtocolError("DUPLICATE_JSON_KEY", f"Duplicate JSON key: {key!r}")
        seen[key] = value
    return seen


def _check_depth_and_members(value: Any, limits: CallLimits, depth: int = 0) -> None:
    if depth > limits.max_json_depth:
        raise ToolProtocolError("RESOURCE_LIMIT", "JSON nesting depth exceeds the configured limit")
    if isinstance(value, dict):
        if len(value) > limits.max_json_members:
            raise ToolProtocolError("RESOURCE_LIMIT", "JSON object has too many members")
        for v in value.values():
            _check_depth_and_members(v, limits, depth + 1)
    elif isinstance(value, list):
        if len(value) > limits.max_json_members:
            raise ToolProtocolError("RESOURCE_LIMIT", "JSON array has too many elements")
        for v in value:
            _check_depth_and_members(v, limits, depth + 1)
    elif isinstance(value, str):
        if len(value) > limits.max_string_argument_length:
            raise ToolProtocolError("RESOURCE_LIMIT", "JSON string value exceeds the configured length limit")


_CALL_TOP_LEVEL_FIELDS = {"protocol_version", "tool", "arguments"}


def parse_tool_call(text: str, limits: CallLimits) -> ToolCall:
    """Parse untrusted model-generated text into a validated ToolCall.

    Raises ToolProtocolError (never a bare Python exception) for anything
    that is not a strictly well-formed, in-limits, correctly-shaped call.
    """
    try:
        raw_bytes = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ToolProtocolError("MALFORMED_CALL", "Call text is not valid Unicode") from exc

    if len(raw_bytes) > limits.max_call_bytes:
        raise ToolProtocolError("RESOURCE_LIMIT", "Serialized tool call exceeds the configured size limit")

    try:
        payload = json.loads(
            text,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ToolProtocolError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        # ValueError (not just JSONDecodeError) is caught deliberately: on
        # Python 3.11+, json.loads raises a bare ValueError — not
        # JSONDecodeError — when a single integer literal exceeds the
        # interpreter's int-to-string conversion digit limit
        # (sys.get_int_max_str_digits(), default 4300). An oversized
        # integer literal is untrusted-input noise, not a crash — see
        # reports/PHASE3_SELF_REVIEW.md.
        raise ToolProtocolError("MALFORMED_CALL", f"Invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ToolProtocolError("MALFORMED_CALL", "Top-level tool call must be a JSON object")

    _check_depth_and_members(payload, limits)

    unknown = set(payload) - _CALL_TOP_LEVEL_FIELDS
    if unknown:
        raise ToolProtocolError("MALFORMED_CALL", f"Unknown top-level field(s): {sorted(unknown)}")
    missing = _CALL_TOP_LEVEL_FIELDS - set(payload)
    if missing:
        raise ToolProtocolError("MALFORMED_CALL", f"Missing top-level field(s): {sorted(missing)}")

    protocol_version = payload["protocol_version"]
    tool = payload["tool"]
    arguments = payload["arguments"]

    if not isinstance(protocol_version, str):
        raise ToolProtocolError("MALFORMED_CALL", "protocol_version must be a string")
    if not isinstance(tool, str):
        raise ToolProtocolError("MALFORMED_CALL", "tool must be a string")
    if not isinstance(arguments, dict):
        raise ToolProtocolError("MALFORMED_CALL", "arguments must be a JSON object")

    return ToolCall(protocol_version=protocol_version, tool=tool, arguments=arguments)


def canonical_serialize(payload: Mapping[str, Any]) -> str:
    """Deterministic canonical JSON: sorted keys, compact separators, UTF-8 text.

    Never emits NaN/Infinity (allow_nan=False raises TypeError if a caller
    ever tries to serialize one — non-finite results must be converted to an
    error before reaching this function).
    """
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=CANONICAL_SEPARATORS,
        ensure_ascii=False,
        allow_nan=False,
    )


def serialize_tool_call(call: ToolCall) -> str:
    return canonical_serialize(
        {"protocol_version": call.protocol_version, "tool": call.tool, "arguments": dict(call.arguments)}
    )


def serialize_tool_result(result: ToolResult) -> str:
    return canonical_serialize(result.to_dict())


def wire_tool_call(call: ToolCall) -> str:
    """Canonical model-facing wire representation: ``<tool_call>{...}`` (no closing tag)."""
    return f"<tool_call>{serialize_tool_call(call)}"


def wire_tool_result(result: ToolResult) -> str:
    """Canonical model-facing wire representation: ``<tool_result>{...}`` (no closing tag)."""
    return f"<tool_result>{serialize_tool_result(result)}"


def parse_tool_result_payload(text: str, limits: CallLimits) -> ToolResult:
    """Parse a canonical serialized ToolResult (internal round-trip use only).

    This function exists for protocol round-trip testing. It must NEVER be
    used to accept a model-generated ``<tool_result>...`` string as a trusted
    execution outcome — see the module docstring and docs/TOOLS.md.
    """
    try:
        raw_bytes = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ToolProtocolError("MALFORMED_CALL", "Result text is not valid Unicode") from exc
    if len(raw_bytes) > limits.max_call_bytes:
        raise ToolProtocolError("RESOURCE_LIMIT", "Serialized tool result exceeds the configured size limit")

    try:
        payload = json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant)
    except ToolProtocolError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        # ValueError (not just JSONDecodeError) is caught deliberately: on
        # Python 3.11+, json.loads raises a bare ValueError — not
        # JSONDecodeError — when a single integer literal exceeds the
        # interpreter's int-to-string conversion digit limit
        # (sys.get_int_max_str_digits(), default 4300). An oversized
        # integer literal is untrusted-input noise, not a crash — see
        # reports/PHASE3_SELF_REVIEW.md.
        raise ToolProtocolError("MALFORMED_CALL", f"Invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ToolProtocolError("MALFORMED_CALL", "Top-level tool result must be a JSON object")
    _check_depth_and_members(payload, limits)

    required = {"protocol_version", "tool", "status", "result", "error"}
    if set(payload) != required:
        raise ToolProtocolError(
            "MALFORMED_CALL", f"Tool result must have exactly the fields {sorted(required)}"
        )

    error_payload = payload["error"]
    error_info = None
    if error_payload is not None:
        if not isinstance(error_payload, dict) or set(error_payload) != {"code", "message"}:
            raise ToolProtocolError("MALFORMED_CALL", "error must be null or {code, message}")
        error_info = ErrorInfo(code=error_payload["code"], message=error_payload["message"])

    result_payload = payload["result"]
    if result_payload is not None and not isinstance(result_payload, dict):
        raise ToolProtocolError("MALFORMED_CALL", "result must be null or a JSON object")

    return ToolResult(
        protocol_version=payload["protocol_version"],
        tool=payload["tool"],
        status=payload["status"],
        result=result_payload,
        error=error_info,
    )
