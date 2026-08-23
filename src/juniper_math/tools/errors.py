"""Stable machine-readable error codes for the Phase 3 tool runtime.

Codes are the stable contract; messages are human-readable and may change.
Never serialize a raw Python traceback, exception repr, or internal path
into a ToolResult — see docs/TOOLS.md "Trust boundary and error truthfulness".
"""

from __future__ import annotations

ERROR_CODES: frozenset[str] = frozenset(
    {
        "MALFORMED_CALL",
        "DUPLICATE_JSON_KEY",
        "UNSUPPORTED_PROTOCOL_VERSION",
        "UNKNOWN_TOOL",
        "TOOL_UNAVAILABLE",
        "MISSING_ARGUMENT",
        "UNKNOWN_ARGUMENT",
        "INVALID_ARGUMENT_TYPE",
        "INVALID_ARGUMENT_VALUE",
        "UNSUPPORTED_OPERATION",
        "UNSUPPORTED_UNIT",
        "DIVISION_BY_ZERO",
        "DOMAIN_ERROR",
        "OVERFLOW",
        "RESOURCE_LIMIT",
        "NON_FINITE_RESULT",
        "INTERNAL_ERROR",
    }
)


class ToolProtocolError(Exception):
    """Base class for all Phase 3 tool-runtime errors.

    Every instance carries a stable ``code`` (one of ERROR_CODES) and a
    human-readable ``message``. Runtime code catches this exception type at
    the dispatch boundary and turns it into a ToolResult — it is never
    allowed to propagate as a raw traceback into a tool result.
    """

    def __init__(self, code: str, message: str) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"Unknown error code: {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message
