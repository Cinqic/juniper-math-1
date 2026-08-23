"""Schema single-source-of-truth tests: the checked-in JSON Schema files must
match current generation output byte for byte (no drift), and the generic
protocol schemas must agree with the runtime's own top-level field set.
"""

from __future__ import annotations

import json

from juniper_math.paths import REPO_ROOT
from juniper_math.tools.config import load_tools_config
from juniper_math.tools.protocol import _CALL_TOP_LEVEL_FIELDS
from juniper_math.tools.schemas import generate_all_schemas, render_schema_file

CONFIG = load_tools_config()


def test_checked_in_schema_files_match_current_generation():
    schemas = generate_all_schemas(CONFIG)
    for key, schema in schemas.items():
        path = REPO_ROOT / CONFIG.schemas[key]
        on_disk = path.read_text(encoding="utf-8")
        expected = render_schema_file(schema)
        assert on_disk == expected, f"{path} is stale — regenerate via write_schema_files()"


def test_call_schema_required_fields_match_protocol_parser():
    schema = generate_all_schemas(CONFIG)["call"]
    assert set(schema["required"]) == _CALL_TOP_LEVEL_FIELDS
    assert set(schema["properties"]) == _CALL_TOP_LEVEL_FIELDS
    assert schema["additionalProperties"] is False


def test_call_schema_tool_enum_matches_config_tools():
    schema = generate_all_schemas(CONFIG)["call"]
    assert set(schema["properties"]["tool"]["enum"]) == set(CONFIG.tools)


def test_result_schema_status_enum_matches_config():
    schema = generate_all_schemas(CONFIG)["result"]
    assert set(schema["properties"]["status"]["enum"]) == set(CONFIG.result_statuses)


def test_result_schema_error_code_enum_matches_config():
    schema = generate_all_schemas(CONFIG)["result"]
    codes = schema["properties"]["error"]["properties"]["code"]["enum"]
    assert set(codes) == set(CONFIG.error_codes)


def test_finance_schema_operations_match_backend_registry():
    from juniper_math.tools.calculator_backend import FINANCE_OPERATIONS

    schema = generate_all_schemas(CONFIG)["finance_arguments"]
    operations_in_schema = {branch["properties"]["operation"]["const"] for branch in schema["oneOf"]}
    assert operations_in_schema == set(FINANCE_OPERATIONS)


def test_schema_files_are_valid_json():
    for rel_path in CONFIG.schemas.values():
        path = REPO_ROOT / rel_path
        json.loads(path.read_text(encoding="utf-8"))  # must not raise
