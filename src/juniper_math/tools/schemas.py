"""Argument validation for each canonical tool, plus deterministic JSON
Schema generation from the same Python definitions that runtime validation
uses.

Single source of truth (see docs/TOOLS.md "Schema single source of truth"):
the argument specs below (``_EVALUATE_ARGS``, ``FINANCE_OPERATIONS`` in
calculator_backend, the category/unit tables in calculator_backend) are the
only place these facts are declared. The JSON Schema files under
tools/schemas/ are *generated* from them by :func:`generate_all_schemas` and
a test asserts the checked-in files match current generation output byte for
byte — schema drift is therefore a test failure, not a silent possibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from juniper_math.tools.calculator_backend import ALL_CONVERT_CATEGORIES, FINANCE_OPERATIONS
from juniper_math.tools.calculator_backend import to_decimal as _to_decimal
from juniper_math.tools.config import ToolsConfig
from juniper_math.tools.errors import ToolProtocolError

# ---------------------------------------------------------------------------
# calculator.evaluate
# ---------------------------------------------------------------------------


def validate_evaluate_arguments(arguments: dict[str, Any]) -> str:
    unknown = set(arguments) - {"expression"}
    if unknown:
        raise ToolProtocolError("UNKNOWN_ARGUMENT", f"Unknown argument(s): {sorted(unknown)}")
    if "expression" not in arguments:
        raise ToolProtocolError("MISSING_ARGUMENT", "Missing required argument: expression")
    expression = arguments["expression"]
    if not isinstance(expression, str):
        raise ToolProtocolError("INVALID_ARGUMENT_TYPE", "expression must be a string")
    return expression


# ---------------------------------------------------------------------------
# calculator.convert
# ---------------------------------------------------------------------------

_CONVERT_REQUIRED = {"category", "from_unit", "to_unit", "value"}


def validate_convert_arguments(arguments: dict[str, Any]) -> tuple[str, str, str, Any]:
    unknown = set(arguments) - _CONVERT_REQUIRED
    if unknown:
        raise ToolProtocolError("UNKNOWN_ARGUMENT", f"Unknown argument(s): {sorted(unknown)}")
    missing = _CONVERT_REQUIRED - set(arguments)
    if missing:
        raise ToolProtocolError("MISSING_ARGUMENT", f"Missing required argument(s): {sorted(missing)}")

    category, from_unit, to_unit = arguments["category"], arguments["from_unit"], arguments["to_unit"]
    for name, value in (("category", category), ("from_unit", from_unit), ("to_unit", to_unit)):
        if not isinstance(value, str):
            raise ToolProtocolError("INVALID_ARGUMENT_TYPE", f"{name} must be a string")
    if category not in ALL_CONVERT_CATEGORIES:
        raise ToolProtocolError("INVALID_ARGUMENT_VALUE", f"Unknown category: {category}")

    value = arguments["value"]
    decimal_value = _to_decimal(value, "value")
    return category, from_unit, to_unit, decimal_value


# ---------------------------------------------------------------------------
# calculator.finance
# ---------------------------------------------------------------------------


def validate_finance_arguments(arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "operation" not in arguments:
        raise ToolProtocolError("MISSING_ARGUMENT", "Missing required argument: operation")
    operation = arguments["operation"]
    if not isinstance(operation, str):
        raise ToolProtocolError("INVALID_ARGUMENT_TYPE", "operation must be a string")
    spec = FINANCE_OPERATIONS.get(operation)
    if spec is None:
        raise ToolProtocolError("UNSUPPORTED_OPERATION", f"Unsupported finance operation: {operation}")

    allowed = {"operation", *spec.required, *spec.optional}
    unknown = set(arguments) - allowed
    if unknown:
        raise ToolProtocolError("UNKNOWN_ARGUMENT", f"Unknown argument(s) for {operation}: {sorted(unknown)}")
    missing = set(spec.required) - set(arguments)
    if missing:
        raise ToolProtocolError(
            "MISSING_ARGUMENT", f"Missing required argument(s) for {operation}: {sorted(missing)}"
        )

    resolved: dict[str, Any] = {}
    for field_name in spec.required:
        resolved[field_name] = _to_decimal(arguments[field_name], field_name)
    for field_name, default in spec.optional.items():
        if field_name in arguments:
            resolved[field_name] = _to_decimal(arguments[field_name], field_name)
        else:
            resolved[field_name] = default
    return operation, resolved


# ---------------------------------------------------------------------------
# JSON Schema generation (documentation / frozen artifacts, see docs/TOOLS.md)
# ---------------------------------------------------------------------------


def _schema(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)


def build_call_schema(config: ToolsConfig) -> dict[str, Any]:
    return _schema(
        **{"$schema": "https://json-schema.org/draft/2020-12/schema"},
        title="Juniper Math 1 tool call",
        type="object",
        additionalProperties=False,
        required=["protocol_version", "tool", "arguments"],
        properties={
            "protocol_version": {"type": "string", "const": config.protocol_version},
            "tool": {"type": "string", "enum": list(config.tools)},
            "arguments": {"type": "object"},
        },
    )


def build_result_schema(config: ToolsConfig) -> dict[str, Any]:
    return _schema(
        **{"$schema": "https://json-schema.org/draft/2020-12/schema"},
        title="Juniper Math 1 tool result",
        type="object",
        additionalProperties=False,
        required=["protocol_version", "tool", "status", "result", "error"],
        properties={
            "protocol_version": {"type": "string", "const": config.protocol_version},
            "tool": {
                "type": "string",
                "description": "May be empty or unrecognized for malformed/unknown-tool calls.",
            },
            "status": {"type": "string", "enum": list(config.result_statuses)},
            "result": {"type": ["object", "null"]},
            "error": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "required": ["code", "message"],
                "properties": {
                    "code": {"type": "string", "enum": list(config.error_codes)},
                    "message": {"type": "string"},
                },
            },
        },
    )


def build_evaluate_schema(config: ToolsConfig) -> dict[str, Any]:
    return _schema(
        **{"$schema": "https://json-schema.org/draft/2020-12/schema"},
        title="calculator.evaluate arguments",
        type="object",
        additionalProperties=False,
        required=["expression"],
        properties={
            "expression": {
                "type": "string",
                "maxLength": config.limits.max_expression_length,
                "description": (
                    f"Operators: {config.evaluate['operators']}. "
                    f"Unary: {config.evaluate['unary_operators']}. "
                    f"Functions: {config.evaluate['functions']}. "
                    f"Constants: {config.evaluate['constants']}. "
                    f"Angle mode: {config.evaluate['angle_mode']}."
                ),
            }
        },
    )


def build_convert_schema(config: ToolsConfig) -> dict[str, Any]:
    return _schema(
        **{"$schema": "https://json-schema.org/draft/2020-12/schema"},
        title="calculator.convert arguments",
        type="object",
        additionalProperties=False,
        required=["category", "from_unit", "to_unit", "value"],
        properties={
            "category": {"type": "string", "enum": sorted(ALL_CONVERT_CATEGORIES)},
            "from_unit": {"type": "string"},
            "to_unit": {"type": "string"},
            "value": {"type": "number"},
        },
    )


def build_finance_schema(config: ToolsConfig) -> dict[str, Any]:
    one_of = []
    for operation, spec in sorted(FINANCE_OPERATIONS.items()):
        properties = {"operation": {"const": operation}}
        for field_name in spec.required:
            properties[field_name] = {"type": "number"}
        for field_name in spec.optional:
            properties[field_name] = {"type": "number"}
        one_of.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["operation", *spec.required],
                "properties": properties,
            }
        )
    return _schema(
        **{"$schema": "https://json-schema.org/draft/2020-12/schema"},
        title="calculator.finance arguments",
        oneOf=one_of,
    )


def generate_all_schemas(config: ToolsConfig) -> dict[str, dict[str, Any]]:
    return {
        "call": build_call_schema(config),
        "result": build_result_schema(config),
        "evaluate_arguments": build_evaluate_schema(config),
        "convert_arguments": build_convert_schema(config),
        "finance_arguments": build_finance_schema(config),
    }


def render_schema_file(schema: dict[str, Any]) -> str:
    """Deterministic on-disk rendering shared by generation and drift tests."""
    return json.dumps(schema, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def write_schema_files(config: ToolsConfig, repo_root: Path) -> None:
    schemas = generate_all_schemas(config)
    for key, schema in schemas.items():
        path = repo_root / config.schemas[key]
        path.write_text(render_schema_file(schema), encoding="utf-8")
