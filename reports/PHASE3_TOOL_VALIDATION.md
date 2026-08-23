# Phase 3 Tool Validation

## Tool counts

3 canonical tools: `calculator.evaluate`, `calculator.convert`,
`calculator.finance`. Registered in a closed `ToolRegistry`
(`src/juniper_math/tools/registry.py`); no others recognized.

## Schemas

5 generated JSON Schema files under `tools/schemas/`, generated
deterministically from Python definitions in `src/juniper_math/tools/schemas.py`
and `calculator_backend.py` (single source of truth — see `docs/TOOLS.md`
§18). `tests/test_tools_schemas.py` asserts the checked-in files match
current generation output byte for byte; any drift is a test failure.

| Schema | File |
|---|---|
| Tool call envelope | `tools/schemas/tool_call.schema.json` |
| Tool result envelope | `tools/schemas/tool_result.schema.json` |
| `calculator.evaluate` arguments | `tools/schemas/calculator_evaluate.schema.json` |
| `calculator.convert` arguments | `tools/schemas/calculator_convert.schema.json` |
| `calculator.finance` arguments | `tools/schemas/calculator_finance.schema.json` (`oneOf`, one branch per operation) |

## Known-answer tests (independent oracles)

- **Evaluate:** `84317 * 9926 == 836930542`, `2**10 == 1024`, `sqrt(2)` vs.
  `math.sqrt`, `log`/`ln` base semantics, trig at 0, `cbrt(±27)` vs. cube
  root, `reciprocal(4)` vs. `fractions.Fraction(1,4)`.
  (`tests/test_tools_evaluate.py`)
- **Convert:** `1 inch = 0.0254 m`, `1 foot = 0.3048 m`, `1 mile = 1609.344
  m`, `1 pound = 453.59237 g`, `1 KiB = 1024 B`, `1 KB = 1000 B`, `0°C =
  32°F`, `273.15 K = 0°C` — all hand-derived, none call the production
  function as its own oracle. (`tests/test_tools_convert.py`)
- **Finance:** tip, sales tax, discount, final price, split bill, simple
  interest, and compound interest each checked against an independently
  hand-written `Decimal` formula, not the production function twice.
  (`tests/test_tools_finance.py`)

## Randomized / property tests (fixed seed `20260822`)

- 200 random safe integer arithmetic expressions vs. Python's own `int`
  arithmetic (`test_tools_evaluate.py::test_random_integer_arithmetic_matches_python_oracle`).
- 300 random round-trip conversions (`A → B → A`) across all 8 categories,
  bounded numeric tolerance (`test_tools_convert.py::test_round_trip_deterministic_random_cases`).
- 200 random `tip` and 200 random `simple_interest` calls vs. independent
  `Decimal` oracles, including `ROUND_HALF_UP` boundary cases at `x.xx5`
  (`test_tools_finance.py`).

## Upstream conformance

`tests/test_tools_conformance.py` checks representative
evaluate/convert/finance cases against independently re-derived expected
values for the pinned Cinqic Calculator commit
(`8024cf107d6240386fa42b6c5193dd8b34848032`), and explicitly documents the
two points of intentional divergence:

1. **Modulo semantics** — Juniper uses Python's `%` (sign of divisor,
   preserves integer exactness); upstream uses `math.fmod` (sign of
   dividend, always float). See `docs/TOOLS.md` §11.
2. **Integer-exactness preservation** — Juniper's `evaluate` keeps large
   products as exact Python `int`; upstream's `evaluate()` unconditionally
   casts to `float`, which silently loses precision for large values (e.g.
   `123456789123456789 * 987654321987654321`).

Conversion tables and the `ROUND_HALF_UP` currency rounding policy are
byte-identical to upstream.

## Error / unavailable / unsupported handling

- Every documented error code in `config/tools.yaml` has at least one
  triggering test case across `test_tools_evaluate.py`,
  `test_tools_convert.py`, `test_tools_finance.py`,
  `test_tools_protocol.py`, and `test_tools_security.py`.
- `ToolRegistry.set_available(name, False)` produces `status: unavailable,
  code: TOOL_UNAVAILABLE` for that tool only, leaving the other two
  unaffected (`test_tools_registry.py`).
- Unrecognized tool names produce `status: unsupported, code: UNKNOWN_TOOL`
  (`test_tools_registry.py`, `test_tools_security.py`).

## Determinism

- Identical calls produce byte-identical canonical JSON, verified in the
  same process (50 repetitions), across two independent `ToolRuntime`
  instances, and across three separate OS subprocesses
  (`tests/test_tools_determinism.py`).
- Repeated invalid calls return byte-identical error output
  (`tests/test_tools_security.py::test_repeated_invalid_calls_are_deterministic`).

## Frozen conformance/security suite

`evals/phase3_tools_v1.json` (`phase3-tools-v1`, `suite_version: 1.0.0`) —
26 cases (24 single-step + 2 multi-step) covering every required category
from Sec. 89 of the Phase 3 engineering instructions: correct calls,
incorrect tool names, missing/extra arguments, invalid types, malformed
JSON, duplicate JSON keys, unsupported protocol version, division by zero,
domain errors, unsupported units, resource limits, fabricated tool
results, multi-step calculations, and unsupported tools. Executed against
the real runtime by `tests/test_tools_eval_suite.py`; hashed in
`manifests/artifacts.yaml` as `phase3_tools_eval_suite`.

## Tokenizer / model compatibility

`tests/test_tools_tokenizer_compat.py` confirms representative
`<tool_call>...`/`<tool_result>...` wire strings for all three tools
tokenize with zero `<unk>`, all IDs in `[0, 4095]`, exact round-trip
through `JuniperTokenizer`, and begin with the frozen Phase 2 control-token
IDs (`<tool_call>` = 4, `<tool_result>` = 5). A mechanical forward pass
through the frozen Phase 1 architecture on these token IDs succeeds
(`test_model_embedding_accepts_tool_protocol_ids`) — a compatibility check
only, not a claim about learned tool-use behavior.

## Test totals

504 tests pass overall (210 in the Phase 3 tools package specifically) —
see `reports/PHASE3_REPORT.md` for the exact `pytest` invocation and
output, and `reports/PHASE3_SELF_REVIEW.md` for the two defects found and
fixed during adversarial review before these numbers were recorded.
