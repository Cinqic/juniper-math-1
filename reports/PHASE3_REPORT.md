# Phase 3 Engineering Report — Cinqic Calculator Tool Runtime

## Identification

| | |
|---|---|
| Starting main SHA | `8f81a3b967b3fd58f3ab0b71c115e85e5af36611` |
| Frozen Phase 2 tag/SHA | `phase-2-tokenizer` → `eaf8fd33837b7bb73c41f2f21bc81386d09dc516` |
| Cinqic Calculator upstream repository | `https://github.com/Cinqic/Cinqic-Calculator` |
| Cinqic Calculator upstream commit | `8024cf107d6240386fa42b6c5193dd8b34848032` |
| Engineer | Claude Sonnet 5 |
| Environment | Linux, Python 3.12.3, PyTorch 2.13.0+cu130, CUDA available (RTX 2060) |

## Protocol

- Identity: `juniper-tool-protocol-v1`, `protocol_version: "1.0.0"` — frozen in `config/tools.yaml`.
- Serialization: `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)`.
- Schemas: 5 generated JSON Schema files under `tools/schemas/`, generated from Python (single source of truth — `docs/TOOLS.md` §18).
- Statuses: `success`, `error`, `unavailable`, `unsupported`.
- Error codes: 17 (see `docs/TOOLS.md` §6 / `config/tools.yaml` → `error_codes`).

## Tools

### `calculator.evaluate`

Operators `+ - * / % // **` (binary), `+ -` (unary); functions `sqrt cbrt
abs log ln sin cos tan factorial reciprocal`; constants `pi e`; radians
only. Resource limits: 512-char expression (in practice bounded tighter by
the 256-char generic string limit), 200 AST nodes, 40 AST depth, 40-digit
numeric literals, 10,000 exponent magnitude, 4,096-bit estimated power
result, 5,000 max factorial argument. Preserves exact integer precision
where upstream would silently cast to `float`.

### `calculator.convert`

Categories: `length mass temperature area volume speed time data_storage`.
Canonical unit identifiers only. Decimal (`kilobyte`) and binary
(`kibibyte`) data-storage units always distinct. All conversion arithmetic
in `Decimal`. Below-absolute-zero temperatures convert numerically without
error (documented policy, `docs/TOOLS.md` §12).

### `calculator.finance`

One tool, `operation` enum with 11 operations, per-operation argument
schemas. Currency operations round to 2 places with `ROUND_HALF_UP`;
non-currency percentage operations round to 6 places, same rounding mode.
Explicit domain validation (`num_people > 0`, `compounds_per_year > 0`,
`old_value != 0` for percentage_difference).

## Security

No `eval`/`exec`/dynamic import anywhere in the tools package. AST
allowlist rejects the full Sec. 66-70 hostile-expression battery. Six
resource limits checked before computation. No filesystem/network/
subprocess imports (statically verified by test). Closed tool registry, no
reflective dispatch. Fabricated `<tool_result>...` strings never trusted.
Full detail: `reports/PHASE3_SECURITY.md`. Two BLOCKER defects were found
during self-review and fixed before this report was written —
`reports/PHASE3_SELF_REVIEW.md`.

## Tests

```
$ pytest -v          # (summarized; full output in CI log)
504 passed, 2 warnings in 17.34s
```

Tools package specifically: `pytest tests/test_tools_*.py -q` → **210
passed**. No regressions in the existing 294-test Phase 0/1/2 baseline (in
fact the full suite includes 2 pre-existing tests updated to reflect the
Phase 2 → Phase 3 status transition, see the diff on `tests/test_cli.py`
and `tests/test_metadata.py`).

```
$ ruff check .
All checks passed!
$ ruff format --check .
120 files already formatted
$ mypy
Success: no issues found in 28 source files
```

## Determinism

Identical calls produce byte-identical canonical output: same-process
(50×), separate `ToolRuntime` instances, and 3 separate OS subprocesses
(`tests/test_tools_determinism.py`). Repeated invalid calls return
byte-identical errors (`tests/test_tools_security.py`).

## Upstream integration

Vendored-and-adapted, not a runtime dependency: `calculator_backend.py`
reimplements the algorithms in upstream's `evaluator.py`, `conversions.py`,
and `financial.py` at the pinned commit
`8024cf107d6240386fa42b6c5193dd8b34848032`, with Juniper-specific
hardening layered on top (AST resource limits, Decimal-based conversion
and finance math, integer-exactness preservation, structured error codes).
No GUI/Android/Windows/history/settings code was integrated. Full
provenance: `manifests/sources.yaml` (`phase3-cinqic-calculator-core-v1`),
`manifests/licenses.yaml` (`cinqic-calculator-upstream`). Conformance
evidence: `tests/test_tools_conformance.py`,
`reports/PHASE3_TOOL_VALIDATION.md`.

## Artifacts (paths and hashes)

| Artifact | Path | SHA-256 |
|---|---|---|
| Tool protocol config | `config/tools.yaml` | `0f41577398dda611dc6a6e37281c803f3354bd6877a02dcb15232bc3c231cdb7` |
| Tool call schema | `tools/schemas/tool_call.schema.json` | `db9de20dadacb9419f9a4f395eb9454979278119876ec36c98610772cf06aa41` |
| Tool result schema | `tools/schemas/tool_result.schema.json` | `5da4eb098b0cfc0eb14f83bf5b11ebf859dbb72f41383e12a94b3ad966c647c4` |
| `calculator.evaluate` schema | `tools/schemas/calculator_evaluate.schema.json` | `53d143d33b728b7df8ee9e86eb02645f04d44c1ed4a1699d420589c15ba2b22f` |
| `calculator.convert` schema | `tools/schemas/calculator_convert.schema.json` | `db32d68a16edfb2d3a10cc185287d1c350425b6721ee094418586c61c68c9fb1` |
| `calculator.finance` schema | `tools/schemas/calculator_finance.schema.json` | `a15bbf0fae868bacfbf6c343b7a7dff672eb3906591ecf8e2d5aeb4845587073` |
| Phase 3 eval suite | `evals/phase3_tools_v1.json` | `0342752eccba0276e20b33b341da8fec55fdb08daff2693ee84b9455a221ca6d` |

All hashes independently verified with both `sha256sum` and
`python -m juniper_math hash verify` — see Recovery below.

## Recovery

```bash
git clone https://github.com/Cinqic/juniper-math-1.git /tmp/juniper_recovery_check_p3
cd /tmp/juniper_recovery_check_p3
git checkout phase-3-review-candidate
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e . --no-deps
python -m juniper_math validate-env
python -m juniper_math validate-config
python -m juniper_math hash verify
python -m juniper_math evals validate
python -m juniper_math manifests-validate
python -m juniper_math deps-check
python -m juniper_math model --device cpu
python -m juniper_math tokenizer validate
python -m juniper_math tools self-test
pytest -v
ruff check . && ruff format --check . && mypy
```

This exact procedure was run against the pushed candidate before handoff —
see `reports/PHASE3_TERRA_HANDOFF.md`. No external Calculator repository
checkout is required for this recovery; the tool backend is fully vendored
in this repository.

## Known limitations

- No degree-mode trig — radians only, by design for Phase 3.
- No fuzzy unit-name parsing (`"kg"` → `kilogram`) — deterministic runtime
  requires canonical unit identifiers; free-text mapping is future
  model-learned behavior, not a runtime responsibility.
- `calculator.convert`/`calculator.finance` numeric arguments have no
  explicit per-field digit cap beyond the 8,192-byte overall call limit and
  Decimal's 28-digit context precision (now cleanly surfaced as
  `RESOURCE_LIMIT` rather than crashing or hanging) — see
  `reports/PHASE3_SECURITY.md` §9.
- This phase proves the tool runtime is correct and secure; it makes no
  claim about whether the untrained Phase 1 model knows when or how to
  invoke a tool — that is Phase 4+ work.

## Status

```
AWAITING_GPT_5_6_TERRA_REVIEW
```
