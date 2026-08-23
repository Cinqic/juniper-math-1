from __future__ import annotations

import json

from juniper_math.cli import main


def test_tools_list_exits_zero(capsys):
    assert main(["tools", "list"]) == 0
    out = capsys.readouterr().out
    assert "calculator.evaluate" in out
    assert "calculator.convert" in out
    assert "calculator.finance" in out


def test_tools_schemas_exits_zero(capsys):
    assert main(["tools", "schemas"]) == 0
    out = capsys.readouterr().out
    assert "calculator_evaluate.schema.json" in out or "evaluate_arguments" in out


def test_tools_validate_valid_call_exits_zero(capsys):
    call = '{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{"expression":"2+2"}}'
    assert main(["tools", "validate", call]) == 0
    out = capsys.readouterr().out
    assert "VALID" in out


def test_tools_validate_malformed_call_exits_nonzero(capsys):
    assert main(["tools", "validate", "{not json"]) == 1


def test_tools_call_success_exits_zero_and_prints_canonical_json(capsys):
    call = '{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{"expression":"2+2"}}'
    assert main(["tools", "call", call]) == 0
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["status"] == "success"
    assert payload["result"]["value"] == "4"


def test_tools_call_error_exits_nonzero(capsys):
    call = '{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{"expression":"1/0"}}'
    assert main(["tools", "call", call]) == 1
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "DIVISION_BY_ZERO"


def test_tools_call_reads_from_file(capsys, tmp_path):
    call_file = tmp_path / "call.json"
    call_file.write_text(
        '{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{"expression":"3*3"}}',
        encoding="utf-8",
    )
    assert main(["tools", "call", "--file", str(call_file)]) == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["result"]["value"] == "9"


def test_tools_call_no_input_exits_two(capsys, monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert main(["tools", "call"]) == 2


def test_tools_self_test_exits_zero(capsys):
    assert main(["tools", "self-test"]) == 0
    out = capsys.readouterr().out
    assert "PASS: all" in out


def test_cli_does_not_shell_out_for_tool_calls():
    # The CLI parses argv itself; tool execution never touches subprocess or
    # os.system regardless of what the "call" argument contains.
    import subprocess
    import sys

    from juniper_math.paths import REPO_ROOT

    proc = subprocess.run(
        [sys.executable, "-m", "juniper_math", "tools", "call", "'; echo pwned #"],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=REPO_ROOT,
    )
    assert "pwned" not in proc.stdout
