# Phase 3 → GPT-5.6 Terra Handoff Package

**Purpose:** self-contained handoff. Terra should not need any memory of
the conversation that produced this candidate — everything needed to
independently review, audit, attack, remediate if necessary, and approve
Phase 3 is either in this document or linked from it.

## Candidate identity

| | |
|---|---|
| Repository | `https://github.com/Cinqic/juniper-math-1.git` |
| Branch | `main` |
| Phase 3 review candidate commit | resolve via `git rev-parse phase-3-review-candidate^{commit}` on the cloned repo — the tag is the authoritative pointer; do not trust a hardcoded SHA in this doc, which would go stale on any follow-up commit |
| Candidate tag | `phase-3-review-candidate` (non-final — points at the Sonnet 5 candidate; Terra owns `phase-3-tools`, the final tag) |
| Starting foundation | tag `phase-2-tokenizer`, commit `eaf8fd33837b7bb73c41f2f21bc81386d09dc516` |
| Cinqic Calculator upstream | `https://github.com/Cinqic/Cinqic-Calculator`, commit `8024cf107d6240386fa42b6c5193dd8b34848032` (MIT, BlessomYT) |

## Reports (read in this order)

1. [`reports/PHASE3_REPORT.md`](PHASE3_REPORT.md) — overall Phase 3 summary
2. [`docs/TOOLS.md`](../docs/TOOLS.md) — full protocol, trust boundary, security model
3. [`reports/PHASE3_SECURITY.md`](PHASE3_SECURITY.md) — security review narrative
4. [`reports/PHASE3_TOOL_VALIDATION.md`](PHASE3_TOOL_VALIDATION.md) — full validation-battery evidence
5. [`reports/PHASE3_SELF_REVIEW.md`](PHASE3_SELF_REVIEW.md) — 2 defects found and fixed during development

## Protocol identity

```
Protocol ID:      juniper-tool-protocol-v1
Protocol version: 1.0.0
Tools:            calculator.evaluate, calculator.convert, calculator.finance
```

Reproduce: `python -m juniper_math tools list` / `tools schemas`.

## Tool runtime source layout

| File | Purpose |
|---|---|
| `config/tools.yaml` | Canonical frozen protocol/limits config — single source of truth |
| `src/juniper_math/tools/errors.py` | Stable error codes, `ToolProtocolError` |
| `src/juniper_math/tools/protocol.py` | Strict JSON parsing, `ToolCall`/`ToolResult`, canonical serialization |
| `src/juniper_math/tools/config.py` | `config/tools.yaml` loader/validator |
| `src/juniper_math/tools/schemas.py` | Per-tool argument validation + JSON Schema generation |
| `src/juniper_math/tools/calculator_backend.py` | AST-sandboxed evaluator, Decimal convert/finance (upstream-adapted) |
| `src/juniper_math/tools/registry.py` | Closed tool registry, no dynamic dispatch |
| `src/juniper_math/tools/runtime.py` | Trusted dispatch pipeline (`ToolRuntime`) |
| `tools/schemas/*.json` | Generated, frozen JSON Schema artifacts |
| `evals/phase3_tools_v1.json` | Frozen conformance/security suite (26 cases) |
| `manifests/sources.yaml` | Cinqic Calculator upstream provenance record (`phase3-cinqic-calculator-core-v1`) |
| `manifests/licenses.yaml` | Upstream license record (`cinqic-calculator-upstream`, MIT) |

## Test commands (run all of these; all must pass)

```bash
python -m juniper_math validate-env
python -m juniper_math validate-config
python -m juniper_math hash verify
python -m juniper_math evals validate
python -m juniper_math evals verify
python -m juniper_math manifests-validate
python -m juniper_math deps-check
python -m juniper_math model --device cpu
python -m juniper_math tokenizer validate
python -m juniper_math tools list
python -m juniper_math tools schemas
python -m juniper_math tools self-test
pytest -v                    # 504 tests total, includes all Phase 0/1/2 gates (no regressions)
ruff check .
ruff format --check .
mypy
```

## Security commands (adversarial battery)

```bash
pytest tests/test_tools_security.py -v
pytest tests/test_tools_eval_suite.py -v   # 26-case frozen conformance/security suite
python -m juniper_math tools call '{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{"expression":"__import__(\"os\").system(\"id\")"}}'
# must return status: error, code: INVALID_ARGUMENT_VALUE — never execute
```

## Recovery procedure (fresh clone)

```bash
git clone https://github.com/Cinqic/juniper-math-1.git /tmp/juniper_recovery_check_p3
cd /tmp/juniper_recovery_check_p3
git checkout phase-3-review-candidate   # or the exact SHA resolved above
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e . --no-deps
python -m juniper_math validate-env
pytest -v
python -m juniper_math tools self-test
python -m juniper_math hash verify
```

This exact procedure was run by Claude Sonnet 5 against the pushed
candidate before handoff — see the Recovery section of `PHASE3_REPORT.md`.
Terra should re-run it independently rather than trust that account. No
external Cinqic Calculator repository checkout is required — the pinned
integration is fully vendored in this repository (Sec. 117 of the Phase 3
engineering instructions: "If FLOWBOX is wiped immediately after Sonnet
hands off Phase 3, what critical tool-runtime state disappears?" — **None.**).

## Model/tokenizer compatibility

```bash
pytest tests/test_tools_tokenizer_compat.py -v
```

Confirms `<tool_call>...`/`<tool_result>...` wire strings for all three
tools round-trip through the frozen `juniper-math-tokenizer-v1` with zero
`<unk>` and all IDs in `[0, 4095]`, begin with the frozen control-token IDs
(`<tool_call>`=4, `<tool_result>`=5), and feed cleanly into the frozen
Phase 1 model's embedding layer.

## Known limitations

- Radians-only trig, no degree mode, in Phase 3.
- No fuzzy free-text unit parsing — canonical unit identifiers only.
- `calculator.convert`/`calculator.finance` numeric arguments rely on the
  overall 8,192-byte call-size limit and Decimal's 28-digit context
  precision (now surfaced as clean `RESOURCE_LIMIT` errors, not crashes)
  rather than a dedicated per-field digit cap — see
  `reports/PHASE3_SECURITY.md` §9 for the full reasoning; Terra may add an
  explicit cap for defense in depth if warranted.
- Proves the runtime is correct/secure; makes no claim about the untrained
  model's tool-selection behavior.

## Terra's authority

Per the Phase 3 engineering instructions governing this project, GPT-5.6
Terra is authorized to: audit the full Phase 3 diff; inspect the pinned
Cinqic Calculator source at the recorded commit; challenge the integration
strategy; independently verify calculator math; attack AST security, JSON
parsing, schemas, resource limits, fabricated-result handling, numeric
formatting, conversions, and finance; add fuzz/property tests; **fix
defects directly**; modify the protocol candidate before freeze if
required; regenerate schemas/hashes; update documentation; push
remediation; rerun CI; repeat recovery; approve Phase 3; create the final
`phase-3-tools` tag; and authorize Phase 4.

Terra does not need to return routine defects to Sonnet 5.

## What Terra must not change

Terra must not silently alter the frozen Phase 1 architecture, the frozen
Phase 2 tokenizer, or the frozen special-token IDs (`<tool_call>`=4,
`<tool_result>`=5, `<final>`=6, `<unsupported>`=7, `<error>`=8; vocab
size 4096; IDs 0-4095). If Phase 3 cannot work without changing either,
Terra must stop approval and escalate to Cinqic rather than change frozen
state unilaterally.

## Status

```
AWAITING_GPT_5_6_TERRA_REVIEW
```

Phase 4 (Dataset and Evaluation Freeze) remains `NOT_AUTHORIZED` until
Terra's final approval of Phase 3.
