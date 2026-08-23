from __future__ import annotations

import subprocess
import sys

from juniper_math.paths import REPO_ROOT
from juniper_math.tools.protocol import serialize_tool_result
from juniper_math.tools.runtime import ToolRuntime

_CASES = [
    '{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{"expression":"84317 * 9926"}}',
    '{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{"expression":"sqrt(2)"}}',
    '{"protocol_version":"1.0.0","tool":"calculator.convert","arguments":{"category":"length","from_unit":"mile","to_unit":"meter","value":1}}',
    '{"protocol_version":"1.0.0","tool":"calculator.finance","arguments":{"operation":"tip","bill_total":42.5,"tip_percent":20}}',
    '{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{"expression":"1/0"}}',
]


def test_identical_calls_produce_identical_canonical_bytes():
    runtime_a = ToolRuntime()
    runtime_b = ToolRuntime()
    for text in _CASES:
        result_a = serialize_tool_result(runtime_a.execute_text(text))
        result_b = serialize_tool_result(runtime_b.execute_text(text))
        assert result_a == result_b


def test_repeated_execution_in_same_process_is_stable():
    runtime = ToolRuntime()
    text = _CASES[0]
    outputs = {serialize_tool_result(runtime.execute_text(text)) for _ in range(50)}
    assert len(outputs) == 1


def test_cross_process_determinism():
    script = (
        "import sys; sys.path.insert(0, 'src'); "
        "from juniper_math.tools.runtime import ToolRuntime; "
        "from juniper_math.tools.protocol import serialize_tool_result; "
        "rt = ToolRuntime(); "
        "print(serialize_tool_result(rt.execute_text(" + repr(_CASES[0]) + ")))"
    )
    outputs = set()
    for _ in range(3):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            cwd=REPO_ROOT,
        )
        outputs.add(proc.stdout.strip())
    assert len(outputs) == 1


def test_no_hidden_time_or_random_state_in_results():
    runtime = ToolRuntime()
    text = _CASES[0]
    first = runtime.execute_text(text).to_dict()
    for key in ("value",):
        assert key in first["result"]
    # No timestamp-like or random-id-like keys leak into the result payload.
    forbidden_keys = {"timestamp", "time", "id", "uuid", "random_seed", "date"}
    assert forbidden_keys.isdisjoint(first["result"].keys())
