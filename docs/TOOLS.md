# Juniper Math 1 — Phase 3 Tool Protocol (`juniper-tool-protocol-v1`)

Status: implemented and historically independently approved in Phase 3 (see
`reports/PHASE3_TERRA_HANDOFF.md`). This document describes the deterministic
tool runtime as built, not aspirational future behavior.

Phase 3 does **not** claim the untrained Phase 1 model knows when or how to
call a tool. It proves the runtime a future trained model will call into is
correct, deterministic, and safe against hostile input.

## 1. Trust boundary

```
model-generated text  (UNTRUSTED)
        |
        v
  parser (protocol.py)         -- strict JSON, size/depth limits
        |
        v
  schema validator (schemas.py) -- per-tool argument shape/type checks
        |
        v
  typed ToolCall                -- trusted from here on
        |
        v
  registry (registry.py)        -- closed set of 3 known tools, no dynamic import
        |
        v
  calculator backend            -- AST-sandboxed evaluator / Decimal convert / Decimal finance
        |
        v
  typed ToolResult               -- built ONLY by the host runtime
        |
        v
  canonical serializer (protocol.py) -- sorted keys, compact, deterministic
```

A model can **request** a tool. A model cannot **author** a tool result. A
string that merely looks like `<tool_result>{...}` in model output is never
treated as evidence a tool ran — see §9.

## 2. Protocol identity

```
protocol_id:      juniper-tool-protocol-v1
protocol_version: 1.0.0
```

Both are frozen in `config/tools.yaml` after Phase 3 approval. A breaking
protocol change requires a new `protocol_version` and re-hashing every
frozen artifact in `manifests/artifacts.yaml`.

## 3. Canonical tools

Exactly three tools are approved in Phase 3:

- `calculator.evaluate`
- `calculator.convert`
- `calculator.finance`

The registry (`src/juniper_math/tools/registry.py`) only recognizes these
names. An unrecognized tool name never reaches Python import machinery —
there is no `importlib.import_module(model_supplied_name)` anywhere in this
codebase.

## 4. Call and result envelopes

Call:

```json
{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{"expression":"2+2"}}
```

Success result:

```json
{"protocol_version":"1.0.0","tool":"calculator.evaluate","status":"success","result":{"exact":true,"value":"4"},"error":null}
```

Error result:

```json
{"protocol_version":"1.0.0","tool":"calculator.evaluate","status":"error","result":null,"error":{"code":"DIVISION_BY_ZERO","message":"Division by zero"}}
```

`additionalProperties: false` at every level — see
`tools/schemas/tool_call.schema.json` and `tools/schemas/tool_result.schema.json`.

## 5. Execution states

| status        | meaning                                                                 |
|---------------|--------------------------------------------------------------------------|
| `success`     | valid call, computed a real result                                      |
| `error`       | valid call to a known, available tool, but computation failed           |
| `unavailable` | recognized tool whose backend is currently disabled                     |
| `unsupported` | the tool name itself is not recognized                                  |

A recognized tool given an unsupported *operation* (e.g.
`calculator.finance` with `operation: "solve_equation"`) is `error` with
code `UNSUPPORTED_OPERATION` — the tool is supported, the operation is not.
"Tool requested" (a syntactically valid call awaiting execution) is a
transient/logical state, never a serialized `status` value.

## 6. Stable error codes

`MALFORMED_CALL`, `DUPLICATE_JSON_KEY`, `UNSUPPORTED_PROTOCOL_VERSION`,
`UNKNOWN_TOOL`, `TOOL_UNAVAILABLE`, `MISSING_ARGUMENT`, `UNKNOWN_ARGUMENT`,
`INVALID_ARGUMENT_TYPE`, `INVALID_ARGUMENT_VALUE`, `UNSUPPORTED_OPERATION`,
`UNSUPPORTED_UNIT`, `DIVISION_BY_ZERO`, `DOMAIN_ERROR`, `OVERFLOW`,
`RESOURCE_LIMIT`, `NON_FINITE_RESULT`, `INTERNAL_ERROR`.

Codes are the stable contract; `message` text may change between versions.
No raw Python traceback, internal file path, or exception repr is ever
serialized into a result — see `reports/PHASE3_SECURITY.md`.

## 7. Deterministic serialization

`json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False,
allow_nan=False)`. No NaN/Infinity, no trailing whitespace, no trailing
prose. Two logically identical calls/results always produce byte-identical
canonical text (verified by `tests/test_tools_determinism.py`, including a
cross-process check).

Strict parsing (`protocol.py:parse_tool_call`) explicitly rejects: duplicate
JSON keys, `NaN`/`Infinity`/`-Infinity` literals, trailing content after the
JSON value, non-object top-level values, unknown top-level fields, and
anything over the configured size/depth/member-count limits
(`config/tools.yaml` → `limits`).

## 8. Tokenizer representation

Canonical model-facing wire format uses the frozen Phase 2 control tokens
with no closing tag and no Markdown fences:

```
<tool_call>{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{"expression":"2+2"}}
<tool_result>{"protocol_version":"1.0.0","tool":"calculator.evaluate","status":"success","result":{"exact":true,"value":"4"},"error":null}
```

`<tool_call>` = ID 4, `<tool_result>` = ID 5 (frozen at Phase 2 tokenizer
training; unchanged by Phase 3). Every representative call/result string
round-trips through `JuniperTokenizer` with zero `<unk>` tokens and all IDs
in `[0, 4095]` (`tests/test_tools_tokenizer_compat.py`), and the resulting
IDs feed cleanly into the Phase 1 model's embedding layer — a mechanical
compatibility check only, not a claim about learned tool-use behavior.

## 9. Fabricated-result protection

The only way to construct a `ToolResult` is:

1. `ToolRuntime.execute_call` / `execute_text`, which always performs real
   computation through the calculator backend, or
2. direct trusted construction in Python (tests, internal tooling).

A model-generated string beginning with `<tool_result>...` is never fed
into anything that treats it as an execution outcome. `execute_text` only
ever parses **calls**; feeding it a fabricated result string fails to parse
as a call and returns a `MALFORMED_CALL`/`error` result — the forged value
never appears anywhere in the output. See
`tests/test_tools_security.py::test_fabricated_tool_result_is_never_trusted`.

## 10. Security model

- **No `eval`/`exec`/compile-and-execute** anywhere in the tools package.
- `calculator.evaluate` parses expressions with Python's `ast` module and
  walks an explicit node-type allowlist (`ast.Constant`, `ast.BinOp`,
  `ast.UnaryOp`, `ast.Name` restricted to `pi`/`e`, `ast.Call` restricted to
  10 named functions). Attribute access, subscripts, comprehensions,
  lambdas, string/bytes literals, collections, and arbitrary names are all
  rejected before evaluation.
- **No dynamic tool dispatch** — the registry is a closed, statically wired
  set; unknown tool names never reach `importlib`.
- **No filesystem, network, or subprocess access** anywhere in
  `src/juniper_math/tools/` (enforced by
  `tests/test_tools_security.py::test_tools_package_imports_no_filesystem_network_or_subprocess_modules`,
  which statically scans every module's imports).
- **Resource limits**, all configured in `config/tools.yaml` and enforced
  *before* computation: call byte size, expression length, JSON depth/member
  count, AST node count/depth, numeric literal digit count, exponent
  magnitude, estimated `**` result bit-length, `factorial()` argument, and
  canonical serialized result size. A syntactically legal expression cannot
  become a CPU/memory DoS payload.
- **Errors are truthful** — every failure mode gets a real code (not a
  generic `error: true`), and identical invalid calls return identical
  errors every time (`tests/test_tools_security.py::test_repeated_invalid_calls_are_deterministic`).

Full narrative: `reports/PHASE3_SECURITY.md`.

## 11. `calculator.evaluate`

Operators: `+ - * / % // **` (binary), `+ -` (unary).
Functions: `sqrt cbrt abs log ln sin cos tan factorial reciprocal`.
Constants: `pi e`. Angle mode: **radians only** (no degree mode in Phase 3).
`log` is base 10; `ln` is natural log.

**Modulo semantics (intentional divergence from upstream):** `%` uses
Python's modulo operator (result takes the sign of the divisor) for both int
and float operands. Upstream Cinqic Calculator uses `math.fmod` (result
takes the sign of the dividend, always returns float). Juniper uses one
consistent semantic across operand types and preserves integer exactness
where fmod would force a float — see `config/tools.yaml` →
`evaluate.modulo_semantics` and `tests/test_tools_conformance.py::test_evaluate_modulo_intentionally_diverges_from_upstream_fmod`.

**Exactness policy:** a result is `exact: true` only when every operand is
an exact Python `int` and the operation chain used only `+ - * // % **`
(non-negative exponent), `abs`, unary `+/-`, or `factorial`. True division
(`/`) and every transcendental/rational function always produce
`exact: false`. This lets Juniper preserve full integer precision for large
products (e.g. `84317 * 9926`) instead of upstream's unconditional
`float(result)` cast, which silently rounds via IEEE-754 for large values —
see `tests/test_tools_conformance.py::test_evaluate_preserves_large_integer_precision_unlike_upstream_float_cast`.

**Resource limits** (`config/tools.yaml` → `limits`): expression length
512 chars, AST node count 200, AST depth 40, numeric literal 40 digits,
`**` exponent magnitude 10,000, estimated `**` result bit-length 4096,
`factorial()` argument 5,000, and complete serialized result size 8,192 UTF-8
bytes. The generic JSON-string bound is 512 characters, matching the
expression limit; unit/category/operation names are additionally constrained
by their closed runtime allowlists. Violations return `RESOURCE_LIMIT` before
unbounded output formatting or computation is attempted.

## 12. `calculator.convert`

Categories: `length mass temperature area volume speed time data_storage`.
Canonical unit identifiers only (e.g. `kilometer`, `gallon_us`, `kibibyte`)
— no fuzzy free-text unit parsing; that is a future model-learned mapping,
not a runtime responsibility.

**Decimal vs. binary data units are always distinct**: `kilobyte` = 1000
bytes (SI/decimal), `kibibyte` = 1024 bytes (IEC/binary). They are never
treated as equivalent.

**Temperature policy (explicit, not accidental):** values below absolute
zero (e.g. `-300 celsius`) convert numerically without error.
`calculator.convert` is a mathematical unit-conversion utility, not a
physical-plausibility validator. This matches upstream Cinqic Calculator
behavior and is a deliberate choice, not an oversight — see
`config/tools.yaml` → `convert.temperature_policy`.

All conversion arithmetic uses `decimal.Decimal` end-to-end (constructed via
`Decimal(str(value))`, never `Decimal(float)` directly) to avoid binary
float rounding artifacts feeding into the conversion tables. It runs in an
explicit fixed Decimal context (precision 28, `ROUND_HALF_UP`), so mutations
to process-global `decimal.getcontext()` cannot alter results.

## 13. `calculator.finance`

One tool, one `operation` enum with per-operation argument schemas:
`percentage_of percentage_increase percentage_decrease
percentage_difference discount sales_tax final_price tip split_bill
simple_interest compound_interest`.

Currency-oriented operations (discount, sales_tax, final_price, tip,
split_bill, simple_interest, compound_interest) round to 2 decimal places
with `ROUND_HALF_UP`, matching upstream Cinqic Calculator's rounding policy.
Percentage-math operations that are not currency (percentage_of,
percentage_increase, percentage_decrease, percentage_difference) round to 6
decimal places with the same rounding mode, for deterministic serialization
without the false precision of raw float division.

**All calculator.finance results are deterministic mathematical estimates.
They are not tax, accounting, or investment advice, and no return or
outcome is guaranteed** (`config/tools.yaml` → `finance.disclaimer`).

Domain validation: `split_bill` requires `num_people > 0`;
`compound_interest` requires `compounds_per_year > 0`;
`percentage_difference` rejects `old_value == 0`. Negative interest rates,
negative prices, and negative tips are accepted (no artificial floor) since
they are mathematically well-defined for this tool's formulas; this is a
deliberate choice, matching upstream, not an unspecified fallthrough.

## 14. Calculator backend relationship

`src/juniper_math/tools/calculator_backend.py` is a narrow,
security-hardened adaptation of the platform-independent core of
**Cinqic Calculator** (https://github.com/Cinqic/Cinqic-Calculator, commit
`8024cf107d6240386fa42b6c5193dd8b34848032`, MIT licensed) — specifically
`evaluator.py`, `conversions.py`, and `financial.py`. No GUI, Android/Kivy,
Windows/Tkinter, history, settings, or user-storage code was integrated;
Juniper Math 1's tool backend runs cleanly headless on Linux with no
platform-specific dependency. **Juniper Math 1 uses a deterministic tool
backend derived from Cinqic Calculator's mathematical core — the standalone
Calculator application does not contain or depend on Juniper.**

Same behavior as upstream: conversion tables, arithmetic operators (except
`%`, see §11), scientific functions, and the `ROUND_HALF_UP` currency
rounding policy.

Juniper hardening beyond upstream: AST node/depth/exponent/factorial
resource limits, strict Decimal-based (not float-based) conversion and
finance math, integer-exactness preservation in `evaluate`, structured
error codes instead of bare `ValueError`, and a versioned, schema-validated
protocol envelope. Full provenance record: `manifests/sources.yaml`
(`phase3-cinqic-calculator-core-v1`) and `manifests/licenses.yaml`
(`cinqic-calculator-upstream`). Conformance evidence:
`tests/test_tools_conformance.py` and `reports/PHASE3_TOOL_VALIDATION.md`.

## 15. Statelessness and no dynamic dispatch

The runtime holds no global calculator memory, no call history, and no
implicit shared angle mode — every call carries all information needed to
produce its result deterministically from `(protocol version, arguments,
fixed configuration)` alone. No tool is ever dispatched via
`importlib.import_module` or reflection on a model-supplied string; the
registry (`registry.py`) is a closed, explicitly wired mapping.

## 16. No dynamic dispatch / no recursive tool calls

`calculator.finance` and `calculator.convert` treat any `{"tool": ...}`-
shaped value inside their own arguments as inert JSON data, never as a
nested tool invocation — there is no code path in `runtime.py` that
re-enters `execute_call`/`execute_text` from inside a handler.

## 17. CLI

```
python -m juniper_math tools list                 # tool names + availability
python -m juniper_math tools schemas               # print generated JSON Schemas
python -m juniper_math tools validate '<call json>'   # parse + schema check only, no execution
python -m juniper_math tools call '<call json>'       # execute, print canonical result
python -m juniper_math tools self-test             # fast in-process happy-path + security battery
```

`validate`/`call` also accept `--file <path>` or `-` (stdin) for complex
JSON. The CLI never shells out for tool execution — argv is parsed by
`argparse`, and the call text is handed directly to the in-process Python
runtime.

## 18. Schema single source of truth

`src/juniper_math/tools/schemas.py` and `calculator_backend.py`'s
`FINANCE_OPERATIONS` / conversion-category tables are the only place these
facts are declared in Python. The JSON Schema files under `tools/schemas/`
are *generated* from them (`generate_all_schemas` /
`write_schema_files`), and `tests/test_tools_schemas.py` asserts the
checked-in files match current generation output byte for byte — schema
drift is a test failure, not a silent possibility.
