# Phase 3 Self-Review

Author: Claude Sonnet 5 (primary Phase 3 engineer). Adversarial self-review
performed after the implementation appeared feature-complete, before
candidate handoff. Two BLOCKER-severity defects were found and fixed; both
are documented below with root cause, fix, and regression coverage. Nothing
found during this review was hidden or downplayed.

## Method

Read every module in `src/juniper_math/tools/` as if reviewing someone
else's code, then actively tried to make the runtime do something it should
never do: escape the arithmetic sandbox, exhaust CPU/memory on a
syntactically legal expression, crash on adversarial JSON, silently trust a
fabricated tool result, or produce a non-deterministic/imprecise result.
Every finding below was reproduced with a concrete failing command before
being fixed, and a regression test now exists for each.

## Findings

### F-01 (BLOCKER, fixed): oversized JSON integer literal crashes the parser

**Symptom:** `calculator.convert` (or any tool) called with a bare JSON
numeric argument containing thousands of digits — e.g.
`{"value": 999...9}` (7,000 nines) — raised an unhandled Python
`ValueError` out of `parse_tool_call`, not a clean `ToolProtocolError`.

**Root cause:** CPython 3.11+ enforces a global integer-to-string
conversion digit limit (`sys.get_int_max_str_digits()`, default 4,300)
inside `int()`. `json.loads` calls `int()` internally when it encounters an
integer token, and on this limit it raises a *bare* `ValueError` — not
`json.JSONDecodeError`. `protocol.py:parse_tool_call` only caught
`json.JSONDecodeError`, so this specific failure mode escaped uncaught. The
protocol's own size limits (`max_call_bytes`, `max_string_argument_length`)
did not protect against this because they bound JSON *string* length, not
the digit count of a bare (unquoted) JSON *number* literal.

**Fix:** `protocol.py:parse_tool_call` and `parse_tool_result_payload` now
catch `(json.JSONDecodeError, ValueError)` around `json.loads` and convert
either into a clean `MALFORMED_CALL` `ToolProtocolError`.

**Regression test:**
`tests/test_tools_security.py::test_huge_bare_json_integer_literal_does_not_crash`.

**Why `calculator.evaluate` was already safe from this specific vector:**
the generic per-string-argument length limit (256 chars,
`max_string_argument_length`) applies to the `expression` field before
`evaluate_expression`'s own AST walk runs, and 256 characters cannot
contain a >4,300-digit literal — so this crash class could only reach
`calculator.convert`/`calculator.finance`'s bare-number arguments, which
have no per-field string-length gate. The fix at the JSON-parsing layer
closes the vector for all three tools and any future one, rather than
patching each call site separately.

### F-02 (BLOCKER, fixed): large-but-plausible finance inputs reported as INTERNAL_ERROR

**Symptom:** `calculator.finance percentage_of(number=1e30, percent=50)`
and `calculator.finance compound_interest(principal=1000,
annual_rate_percent=99999, years=9999, compounds_per_year=365)` — inputs a
model could plausibly generate, not deliberately pathological — both
returned `status: error, code: INTERNAL_ERROR`, which is supposed to be
reserved for genuine internal bugs, not oversized-but-well-formed input.

**Root cause:** Python's `decimal` module defaults to 28 significant digits
of context precision. `Decimal.quantize()` on a value whose significant
digits exceed that precision raises `decimal.InvalidOperation`; a `Decimal`
power operation that would produce an out-of-range exponent raises
`decimal.Overflow`. Both are subclasses of `decimal.DecimalException`,
which is a subclass of the builtin `ArithmeticError` — so both were being
caught by `runtime.py`'s generic backstop
(`except (ArithmeticError, ValueError, OverflowError)`) and collapsed into
the catch-all `INTERNAL_ERROR`, discarding the more specific and more
truthful classification.

**Fix:** `compute_finance` in `calculator_backend.py` now wraps its entire
dispatch in a `try/except decimal.DecimalException` that raises
`ToolProtocolError("RESOURCE_LIMIT", ...)`, and the existing
`compound_interest`-specific power computation was corrected to catch
`decimal.DecimalException` (its prior `except (InvalidOperation,
OverflowError)` did not actually catch `decimal.Overflow`, since that
class does not subclass either of those two) so it keeps returning the
more specific `OVERFLOW` code for that case.

**Regression tests:**
`tests/test_tools_finance.py::test_extremely_large_percentage_of_input_is_resource_limit_not_internal_error`,
`tests/test_tools_finance.py::test_extreme_compound_interest_inputs_are_overflow_not_internal_error`.

## Areas specifically challenged and found sound (no fix needed)

- **AST sandbox escape:** every item in the Sec. 66-70 hostile-expression
  list (`__import__`, `open`, `eval`, `exec`, `globals`, `locals`,
  `getattr`, attribute access, `.__class__`/`.__mro__` chains, list/dict/set
  comprehensions, lambdas, string/bytes literals, `True + True`,
  subscripting) is rejected before evaluation —
  `tests/test_tools_security.py::test_malicious_expressions_are_rejected_not_executed`.
- **Resource exhaustion:** deep nesting, huge exponents (`2**1000000000`,
  `999999**999999`), huge factorials, and oversized calls all return
  `RESOURCE_LIMIT` without attempting the computation — verified by
  inspection of `_walk_limits`/`_safe_pow` (limits checked *before* any
  Python-level arithmetic) and by `tests/test_tools_evaluate.py` and
  `tests/test_tools_security.py`.
- **Schema strictness / weird JSON:** duplicate keys, `NaN`/`Infinity`,
  trailing content, non-object top-level values, and unknown fields are all
  rejected — `tests/test_tools_protocol.py`, `tests/test_tools_security.py`.
- **Result authority / fabricated results:** confirmed by code inspection
  that `ToolResult` is only ever constructed by `ToolRuntime` (real
  computation) or trusted internal code; a model-generated
  `<tool_result>...` string is never fed into anything that treats it as an
  execution outcome — `tests/test_tools_security.py::test_fabricated_tool_result_is_never_trusted`.
- **Determinism:** identical calls produce byte-identical canonical output,
  including across separate processes —
  `tests/test_tools_determinism.py`.
- **Numeric fidelity:** large exact-integer arithmetic (e.g.
  `84317 * 9926`, or an 18-digit × 18-digit product) is preserved as an
  exact Python `int` rather than silently rounded through `float()` the way
  upstream Cinqic Calculator's evaluator does —
  `tests/test_tools_conformance.py::test_evaluate_preserves_large_integer_precision_unlike_upstream_float_cast`.
- **Error truthfulness:** every failure path in this codebase now resolves
  to one of the 17 stable codes in `config/tools.yaml`; no bare Python
  exception message or traceback reaches a `ToolResult` (verified by
  reading every `except` clause in `runtime.py` and
  `calculator_backend.py`, and by
  `tests/test_tools_security.py::test_no_raw_traceback_ever_serialized`).
- **Upstream drift:** the integration is pinned to a single upstream commit
  (`8024cf107d6240386fa42b6c5193dd8b34848032`), recorded in
  `manifests/sources.yaml`, `manifests/licenses.yaml`, and
  `config/tools.yaml`; nothing in the runtime clones or imports the
  upstream repository at build or run time.

## Outcome

Both BLOCKER findings were fixed and covered with regression tests before
this report was written. `ruff check`, `ruff format --check`, `mypy`, and
the complete test suite (504 tests, tools package: 210) all pass with the
fixes applied — see `reports/PHASE3_REPORT.md` for exact counts. No
unresolved BLOCKER, HIGH, or material MEDIUM findings remain.
