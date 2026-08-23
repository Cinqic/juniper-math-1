# Phase 3 Security Review

Scope: `src/juniper_math/tools/` — the Phase 3 deterministic tool runtime.
This document is the narrative backing the security battery in
`tests/test_tools_security.py` and the eval-suite security cases in
`evals/phase3_tools_v1.json`. See `reports/PHASE3_SELF_REVIEW.md` for the
two defects found and fixed during adversarial self-review before this
report was finalized.

## 1. Untrusted-input boundary

Everything reaching `juniper_math.tools.protocol.parse_tool_call` is
treated as untrusted, model-generated text — see the trust-boundary
diagram in `docs/TOOLS.md` §1. Nothing downstream of that function is
executed until parsing, size/depth checks, and schema validation all pass.

## 2. Parser limits (enforced before any computation)

| Limit | Value | Enforced by |
|---|---|---|
| Serialized call size | 8,192 bytes | `parse_tool_call` (byte-length check before `json.loads`) |
| JSON nesting depth | 16 | `_check_depth_and_members` |
| JSON object/array member count | 64 | `_check_depth_and_members` |
| Any JSON string value length | 256 chars | `_check_depth_and_members` |
| `calculator.evaluate` expression length | 512 chars | `evaluate_expression` (in practice bounded tighter by the 256-char string limit above) |

Strict JSON parsing rejects, before any of the above limits are even
consulted: duplicate top-level or nested keys (`object_pairs_hook`),
`NaN`/`Infinity`/`-Infinity` literals (`parse_constant`), trailing content
after the JSON value (Python's `json.loads` already rejects this by
default), non-object top-level values, and unknown top-level fields
(`additionalProperties: false` semantics, enforced in code, mirrored in
`tools/schemas/tool_call.schema.json`).

**Fixed during self-review:** a bare JSON integer literal with more digits
than CPython's `sys.get_int_max_str_digits()` limit (default 4,300)
previously crashed `json.loads` with an uncaught `ValueError`. `parse_tool_call`
and `parse_tool_result_payload` now catch this alongside
`json.JSONDecodeError` and convert it to `MALFORMED_CALL`. See
`reports/PHASE3_SELF_REVIEW.md` F-01.

## 3. AST allowlist (`calculator.evaluate`)

`_eval_node`/`_walk_limits` in `calculator_backend.py` parse expressions
with Python's `ast` module in `eval` mode and walk an explicit allowlist of
node types: `Expression`, `Constant` (numeric, non-bool only), `BinOp`,
`UnaryOp`, `Name` (restricted to `pi`/`e`), `Call` (restricted to 10 named
functions, no `**kwargs`), and the operator node types themselves. Every
other node type — `Attribute`, `Subscript`, `List`, `Dict`, `Set`,
`SetComp`/`ListComp`/`DictComp`/`GeneratorExp`, `Lambda`, `Str`/`Bytes`
literals (via the `Constant` type check), `Compare`, `BoolOp`, arbitrary
`Name`s — is rejected with `INVALID_ARGUMENT_VALUE` before it is ever
evaluated. There is no `eval`, `exec`, `compile`, or dynamic-import call
anywhere in `src/juniper_math/tools/`.

Verified against the full hostile-expression list (Sec. 66-70 of the Phase
3 engineering instructions):
`__import__('os').system('id')`, `open('/etc/passwd').read()`,
`eval('2+2')`, `exec(...)`, `globals()`, `locals()`, `getattr(...)`,
`(1).__class__`, `().__class__.__mro__`, `pi.__class__`, `[1,2,3]`,
`{"x":1}`, `{x for x in [1]}`, `[x for x in range(100)]`, `"hello"`,
`b'abc'`, `lambda: 1`, `True + True`, `pi[0]` — all rejected without
executing, in `tests/test_tools_security.py::test_malicious_expressions_are_rejected_not_executed`.

## 4. Resource limits (bounded before computation, not after)

| Limit | Value | Prevents |
|---|---|---|
| AST node count | 200 | expression-tree DoS |
| AST depth | 40 | deep-recursion DoS |
| Numeric literal digit count | 40 | oversized-literal parsing cost |
| `**` exponent magnitude (when `|base| > 1`) | 10,000 | `2 ** 1000000000`-style attacks |
| Estimated `**` result bit-length | 4,096 | `999999 ** 999999`-style attacks (checked via `math.log2` *before* computing) |
| `factorial()` argument | 5,000 | `factorial(100000000)`-style attacks |

All six limits are checked before the corresponding Python arithmetic
executes — the runtime never "tries and times out"; it estimates cost from
the untrusted operands and refuses up front, returning `RESOURCE_LIMIT`.

Finance arithmetic uses `decimal.Decimal` with the interpreter's default 28
significant-digit context precision. **Fixed during self-review:** inputs
large enough to exceed that precision during `quantize()` or Decimal power
now return `RESOURCE_LIMIT`/`OVERFLOW` instead of a generic
`INTERNAL_ERROR` — see `reports/PHASE3_SELF_REVIEW.md` F-02.

## 5. No dynamic tool dispatch

`registry.py`'s `ToolRegistry` is a closed, explicitly-populated mapping
(`{"calculator.evaluate": ..., "calculator.convert": ..., "calculator.finance":
...}`, wired once in `ToolRuntime.__init__`). An unrecognized tool name is
compared against `config.tools` (a tuple of exactly 3 strings) and, if not
found, returns `status: unsupported, code: UNKNOWN_TOOL` — it never reaches
`importlib`, `getattr` on a module, or any other reflective mechanism.
Verified against `calculator.solve_equation`, `calculator.python`,
`shell.exec`, `filesystem.read`, `os.system`
(`tests/test_tools_security.py::test_unsupported_tool_names_are_unsupported`).

## 6. No filesystem / network / subprocess surface

`tests/test_tools_security.py::test_tools_package_imports_no_filesystem_network_or_subprocess_modules`
statically parses every `.py` file in `src/juniper_math/tools/` with `ast`
and asserts none imports `os`, `subprocess`, `socket`, `shutil`, `urllib`,
`http`, `requests`, or `ftplib` (directly or via `from X import Y`). The
package's only third-party import is `yaml` (`config.py`, for reading the
frozen `config/tools.yaml`), which does not perform I/O beyond the file the
caller explicitly hands it.

## 7. Fabricated-result protection

See `docs/TOOLS.md` §9 for the full trust-boundary argument. In short: the
only code path that constructs a `ToolResult` with `status: success` is
`ToolRuntime.execute_call`, which always calls into the real calculator
backend. `execute_text` only parses **calls** — a model-generated string
that begins with `<tool_result>...` fails to parse as a call
(`MALFORMED_CALL`) and the forged value inside it never reaches any output.
Verified in
`tests/test_tools_security.py::test_fabricated_tool_result_is_never_trusted`
and eval-suite case `fabricated-tool-result-001`.

## 8. Error truthfulness / determinism

Every `except` clause in `runtime.py` and `calculator_backend.py` converts
its exception into one of the 17 stable `config/tools.yaml` error codes
(never a raw exception repr, traceback, or internal file path). Identical
invalid calls return byte-identical error output on repeated invocation —
`tests/test_tools_security.py::test_repeated_invalid_calls_are_deterministic`
— and across separate OS processes —
`tests/test_tools_determinism.py::test_cross_process_determinism`.

## 9. Remaining limitations (honest disclosure)

- The Phase 3 evaluator supports only the operator/function surface listed
  in `docs/TOOLS.md` §11 — it is not a general calculator; anything outside
  that list is rejected, which is by design, not a gap.
- `calculator.convert`/`calculator.finance` numeric arguments are bounded
  in aggregate by the 8,192-byte call-size limit and, for finance, by
  Decimal's 28-digit context precision (now cleanly surfaced as
  `RESOURCE_LIMIT`); there is no separate explicit per-field digit-count
  limit for these two tools the way `calculator.evaluate` has one for
  expression literals. This is judged acceptable because the failure mode
  is now a clean, fast, deterministic error rather than a crash or
  excessive computation — but a future phase could add an explicit digit
  cap for defense in depth if Terra's review finds it warranted.
- This review covers the tool runtime only. It makes no claim about the
  security properties of a future trained model's tool-selection behavior
  — that is out of scope for Phase 3 by design (Sec. 62 of the Phase 3
  engineering instructions).
