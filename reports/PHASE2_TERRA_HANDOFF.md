# Phase 2 → GPT-5.6 Terra Handoff Package

**Purpose:** self-contained handoff. Terra should not need any memory of the
conversation that produced this candidate — everything needed to
independently review, audit, retrain if necessary, remediate, and approve
Phase 2 is either in this document or linked from it.

## Candidate identity

| | |
|---|---|
| Repository | `https://github.com/Cinqic/juniper-math-1` |
| Branch | `main` |
| Phase 2 review candidate commit | resolve via `git rev-parse phase-2-review-candidate^{commit}` on the cloned repo — the tag is the authoritative pointer; do not trust a hardcoded SHA in this doc, which would go stale on any follow-up commit |
| Candidate tag | `phase-2-review-candidate` (non-final — points at the Sonnet 5 candidate; Terra owns `phase-2-tokenizer`, the final tag) |
| Starting foundation | tag `phase-1-architecture`, commit `83d6106bae2a465a1c166c3137b10b87eda22f91` |

## Reports (read in this order)

1. [`reports/PHASE2_REPORT.md`](PHASE2_REPORT.md) — overall Phase 2 summary
2. [`reports/PHASE2_TOKENIZER_VALIDATION.md`](PHASE2_TOKENIZER_VALIDATION.md) — full validation battery evidence
3. [`reports/PHASE2_TOKENIZER_INSPECTION.md`](PHASE2_TOKENIZER_INSPECTION.md) — curated manual inspection of real encode/decode examples
4. [`reports/PHASE2_TOKENIZER_BENCHMARKS.md`](PHASE2_TOKENIZER_BENCHMARKS.md) — per-category token efficiency + baseline comparison
5. [`reports/PHASE2_SELF_REVIEW.md`](PHASE2_SELF_REVIEW.md) — defects found and fixed during development

## Tokenizer artifacts and configs

| File | Purpose |
|---|---|
| `config/tokenizer.yaml` | Canonical training configuration — single source of truth for tokenizer identity |
| `releases/tokenizer/juniper-math-tokenizer-v1.model` | Frozen SentencePiece BPE model (4096 pieces) |
| `releases/tokenizer/juniper-math-tokenizer-v1.vocab` | Human-readable vocabulary listing |
| `releases/tokenizer/juniper-math-tokenizer-v1.special_tokens.json` | Machine-readable special/control/digit token ID map |
| `releases/tokenizer/juniper-math-tokenizer-v1.config_snapshot.json` | JSON snapshot of the exact config used to train this artifact |
| `manifests/sources.yaml` | Corpus provenance (100% synthetic, project-generated) |
| `manifests/artifacts.yaml` | SHA-256 hashes for all 5 tokenizer files above plus every other frozen project artifact |
| `src/juniper_math/tokenizer_corpus.py` | Deterministic synthetic corpus generator (12 categories, seed `20260201`) |
| `src/juniper_math/tokenizer.py` | Training entry point, project-owned encode/decode/audit wrapper |
| `src/juniper_math/tokenizer_validation.py` | The 11-check validation battery |
| `src/juniper_math/tokenizer_benchmark.py` | Per-category efficiency + baseline comparison |

## Exact identity

```
Tokenizer version: juniper-math-tokenizer-v1
Vocabulary size:   4096 (exact)
Library:           sentencepiece 0.2.2
Model type:        BPE, byte_fallback=true, split_digits=true
```

Reproduce: `python -m juniper_math tokenizer inspect`

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
python -m juniper_math tokenizer inspect
python -m juniper_math tokenizer benchmark
pytest -v                    # 289 tests, includes all Phase 0/1 gates (no regressions)
ruff check .
ruff format --check .
mypy
```

## Rebuild command (deterministic reconstruction)

```bash
python -m juniper_math tokenizer train --overwrite
python -m juniper_math hash file releases/tokenizer/juniper-math-tokenizer-v1.model
# must equal a855b35bffbbd4b934c25f82afc7cad733f83a66f614e4fea40c112f3072fd97
```

`train_tokenizer` stages the corpus and output at a fixed-name temp path
before writing the final artifact, specifically so this rebuild is
byte-identical regardless of where the repository is checked out — see
`reports/PHASE2_SELF_REVIEW.md` F-01 for why this matters and how it was
verified.

## Recovery procedure (fresh clone)

```bash
git clone https://github.com/Cinqic/juniper-math-1.git /tmp/juniper_recovery_check_p2
cd /tmp/juniper_recovery_check_p2
git checkout phase-2-review-candidate   # or the exact SHA resolved above
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e . --no-deps
python -m juniper_math validate-env
pytest -v
python -m juniper_math tokenizer validate
python -m juniper_math hash verify
python -m juniper_math tokenizer train --overwrite   # deterministic rebuild check
python -m juniper_math hash file releases/tokenizer/juniper-math-tokenizer-v1.model
```

This exact procedure was run by Claude Sonnet 5 against the pushed candidate
before handoff — see the Recovery section of `PHASE2_REPORT.md`. Terra
should re-run it independently rather than trust that account.

## Model/tokenizer compatibility

```bash
python -m juniper_math model --device cpu
python -m juniper_math tokenizer validate
```

Both must report `vocab_size=4096`. `tokenizer_validation.check_id_range`
additionally confirms every ID produced by the tokenizer on a broad text
sample is in `[0, 4096)`, matching the frozen Phase 1 embedding matrix.

## Known limitations

- Numeric/symbolic categories (scientific notation, ratios, units) are less
  token-efficient than the informational gpt2 baseline — an intended
  consequence of digit atomicity (Sec. 57 of the Phase 2 instructions), not
  a defect to "fix" by relaxing the numeric policy.
- The tokenizer-training corpus is 100% synthetic and template-based; it
  teaches BPE segmentation, not model capability, and is explicitly not the
  Phase 4 model-training corpus.
- The deterministic-reconstruction fix relies on a fixed staging path under
  the OS temp directory (`$TMPDIR/juniper_tokenizer_build/`), which assumes
  single-process, non-concurrent tokenizer training.
- `tiktoken` (the baseline comparison library) is a dev-only dependency;
  `tokenizer benchmark` degrades gracefully (reports "baseline unavailable")
  if it is not installed, verified by explicit uninstall/reinstall testing
  during self-review.

## Terra's authority

Per the Phase 2 instructions governing this project, GPT-5.6 Terra is
authorized to: review the full Phase 2 diff; audit the tokenizer corpus,
its provenance, and licenses; retrain the tokenizer independently; inspect
all 4096 vocabulary entries; challenge digit atomicity and the
multi-digit-token policy; test Unicode/byte fallback; verify special-token
IDs; rerun the baseline comparison; test round trips; inspect
normalization; find defects; **directly fix defects**; retrain the
tokenizer and create a corrected version if required; update hashes,
docs, and tests; commit and push remediation; rerun recovery; issue final
Phase 2 approval; create the final `phase-2-tokenizer` tag; and authorize
Phase 3.

Terra must preserve the frozen Phase 1 architecture — the model's
`vocab_size` remains 4096. If the tokenizer cannot satisfy that constraint,
fix the tokenizer design, not the architecture. Terra does not need to
return defects to Sonnet 5 unless Terra chooses to.

## Status

```
AWAITING_GPT_5_6_TERRA_REVIEW
```
