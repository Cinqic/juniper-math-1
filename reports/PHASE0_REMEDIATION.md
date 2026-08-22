# Phase 0 Remediation Report

**Remediator:** Claude Opus 5 (the same agent that performed the independent
review; engineer/reviewer separation was explicitly superseded by Cinqic for
this final pass).
**Date:** 2026-08-22
**Original review candidate:** `b39e60031e85822a293ca60b5676ce5b7286a66f`
(tag `phase-0-review-candidate`) — **did not pass** independent review.
**Original verdict:** CHANGES REQUIRED — 0 BLOCKER, 1 HIGH, 4 MEDIUM, 6 LOW.

Every finding from `reports/OPUS5_PHASE0_REVIEW.md` is listed below, including
all six LOW findings and all five non-blocking notes. Nothing was silently
dropped.

---

## Findings

| Finding | Severity | Problem | Fix | Tests / Verification | Final Status |
|---|---|---|---|---|---|
| **F-01** | HIGH | `evals/phase0_v1.json` case `tool-001` recorded `84317 * 9926` as `837042742`; correct product is `836930542` (error 112,200). Frozen, hashed artifact with `tolerance: 0`. | Corrected the answer. Bumped `suite_version` `0.1.0` → `0.1.1` per the suite's own immutability policy. Regenerated SHA-256 and updated `manifests/artifacts.yaml`. Documented the correction in the suite `description`, the case `notes`, and a version-history table in `docs/EVALUATIONS.md`. | `test_verification.py::test_original_tool_001_defect_is_now_caught` re-injects `837042742` and asserts rejection. `test_evals.py::test_tool_001_ground_truth_is_correct` asserts `== 84317 * 9926`. Hash cross-checked with `sha256sum`. | **RESOLVED** |
| **F-02** | MEDIUM | `evals validate` and all 8 eval tests checked schema only. `evals.py` docstring claimed "deterministic-reference validation" it did not perform. This is why F-01 survived self-review, 63 tests, and CI. | Added `src/juniper_math/verification.py`: a safe structured-expression evaluator over `fractions.Fraction` with a closed 8-operation allowlist. Added a `verification` block to all 22 cases (18 `deterministic`, 4 `semantic`). `evals validate` now runs schema **and** ground-truth checks; new `evals verify` runs ground truth alone. Corrected the docstring to describe both modes accurately. | `tests/test_verification.py` — 25 tests covering recomputation, exact decimal/fraction arithmetic, tolerance bands, boolean cases, mode coherence, and rejection of unknown operations / malformed expressions / division by zero / fractional exponents. | **RESOLVED** |
| **F-03** | MEDIUM | No lock, constraints, or requirements file. Only bounded ranges (`ruff>=0.6,<1.0`, `mypy>=1.10,<2.0`, `torch>=2.2,<3.0`). The validated environment could not be reconstructed, and ruff/mypy drift can break `ruff format --check` / `mypy` on unchanged code. | Added `requirements-lock.txt` pinning all 42 packages (direct + transitive) from the validated environment. Documented install order in `README.md`, `docs/ENVIRONMENT.md`, `docs/RECOVERY.md`. `scripts/bootstrap.sh` and CI now install from the lock by default. Compatibility ranges retained in `pyproject.toml` as metadata. Header explicitly states the lock does **not** pin kernel, NVIDIA driver, CUDA stack, or firmware. | `tests/test_bootstrap_and_lock.py` — lock exists, every entry exactly `==` pinned with no range operators, covers every declared direct dependency, pins the quality-gate tools, and documents its system-layer limits. Fresh-clone install from the lock re-run end to end. | **RESOLVED** |
| **F-04** | MEDIUM | `docs/RECOVERY.md` step 3 (`python3 -m venv .venv`) fails on the documented target platform: on Debian/Ubuntu/Linux Mint, `python3` ships without `ensurepip`/`pip`/`venv`. Reproduced on the actual dev host (Linux Mint 22.3). The candidate's recovery report misattributed this to a "test sandbox quirk". | Added `python3-venv`/`python3-pip` to prerequisites in `docs/RECOVERY.md`, `docs/ENVIRONMENT.md`, `README.md`, and `reports/HANDOFF.md`, with a verification one-liner and the version-specific fallback package name. `scripts/bootstrap.sh` now runs an `ensurepip` preflight **before** `venv` and prints an actionable message naming the package; it never runs `sudo` or a package manager itself. Corrected the mischaracterization in `reports/RECOVERY_TEST_REPORT.md`. | `tests/test_bootstrap_and_lock.py` — preflight ordering, package naming, `bash -n` syntax validity, no `sudo` in command position, no package-manager invocation, and both docs mention `python3-venv`. | **RESOLVED** |
| **F-05** | MEDIUM | NumPy declared as a runtime dependency in `pyproject.toml` but absent from `manifests/licenses.yaml`, which claims to track dependencies. `manifests-validate` passed anyway — nothing cross-checked. | Rewrote `manifests/licenses.yaml` with all 7 direct dependencies plus project code (3 → 8 entries), each carrying a `package` field and an SPDX expression **read from installed package metadata**, not guessed. Added `check_dependency_licenses()` + `deps-check` CLI command; `manifests-validate` now includes it. Manifest header states its scope explicitly. | `tests/test_dependency_licenses.py` — 9 tests including removing NumPy from a manifest copy (must fail), adding an undeclared dependency (must fail), stale entries, scope mismatch, and PEP 503 name normalization. | **RESOLVED** |
| **F-06** | LOW | `reports/RECOVERY_TEST_REPORT.md` recorded commit `1308128` (58 tests), not the candidate `b39e600` (63 tests). No recovery evidence covered the submitted state. | Annotated the historical report with the correction. Final recovery test re-run against the approved commit from a fresh clone of the canonical remote, with results recorded in `reports/PHASE0_FINAL_APPROVAL.md`. | Fresh-clone recovery re-executed; see final approval report. | **RESOLVED** |
| **F-07** | LOW | `_cmd_status` treated empty `git status --porcelain` output as a clean tree — indistinguishable from a failed invocation. Outside a repository it printed `Git tree state: clean`. | Added `describe_git_state()` returning explicit `clean` / `dirty` / `unknown`, checking `returncode` rather than empty stdout, and catching `OSError`/`SubprocessError`. Failure now reports `unknown`, never `clean`. | `tests/test_git_state.py` — 8 tests: clean repo, modified file, untracked file, non-git directory, missing git executable, git timeout, and end-to-end `status` output asserting `"clean" not in output`. | **RESOLVED** |
| **F-08** | LOW | `undef-001` ("5 divided by 0", category `undefined_operation`) used `expected_behavior: flag_missing_information` — semantically wrong. `refuse_ambiguous` was declared but unused. | Added `flag_undefined` to the behavior vocabulary and applied it to `undef-001`. Removed the unused `refuse_ambiguous` (`request_clarification` already covers it). Documented the full vocabulary in `docs/EVALUATIONS.md`. Both changes are part of suite `0.1.1`. | `test_evals.py::test_undefined_operation_is_not_labelled_missing_information` and `::test_removed_behavior_is_rejected`. | **RESOLVED** |
| **F-09** | LOW | `tolerance` semantics (absolute vs relative) undefined in a frozen schema. `sci-001` used `0.001` against `64000000`; `est-001` used `200` against `4000`. | Documented in `docs/EVALUATIONS.md`: tolerance is **absolute**, pass condition `abs(computed - expected) <= tolerance`, `0`/`null` both mean exact, all comparisons use `Fraction`. Implemented exactly that in `verification.py`. | `test_verification.py::test_estimation_case_uses_tolerance_band` (passes at 200, fails at 1) and `::test_exact_decimal_and_fraction_arithmetic`. | **RESOLVED** |
| **F-10** | LOW | `.gitignore` had `!data/*/.gitkeep` negations but no `.gitkeep` files were tracked, so a fresh clone lacked the `data/` subdirectory skeleton `data/README.md` describes. | Created and tracked `.gitkeep` in `data/raw`, `interim`, `processed`, `external`, `cache`. Verified real data files are still ignored. | `git check-ignore` probes confirm `.gitkeep` tracked while `data/raw/corpus.txt` and `data/cache/x.bin` remain ignored; directories now present in a fresh clone. | **RESOLVED** |
| **F-11** | LOW | `reports/HANDOFF.md` identified the candidate as "the `HEAD` of `main` at the time this file was pushed" — a moving reference. | Pinned the exact SHA `b39e600…`, added the tag name, and stated plainly that this candidate did **not** pass review. | Verified by inspection; the pinned SHA matches `phase-0-review-candidate`. | **RESOLVED** |

---

## Non-blocking notes

| Note | Disposition |
|---|---|
| **NOTE-A** — Python 3.10/3.11 declared but never tested; CI ran 3.11 only. | **RESOLVED.** Declared support narrowed to `>=3.12,<3.13` in `pyproject.toml`, `environment.py`, and the docs, because the validated dependency set genuinely requires it (`numpy` 2.5.2 needs >= 3.12) — installing on 3.10/3.11 would silently resolve a different environment. CI pinned to 3.12. Declared support now equals tested support; widening requires adding the interpreter to CI first. A side effect worth noting: retargeting ruff to 3.12 surfaced a real modernization (`enum.StrEnum`), now applied. |
| **NOTE-B** — `mypy` scoped to `src/` only. | **ACCEPTED, documented.** Proportionate for Phase 0. `mypy` passes on 14 source files. Revisit if `tests/` grows type-sensitive helpers. |
| **NOTE-C** — No `gpu`-marked test exists, so the CPU-skip path is unexercised. | **DEFERRED to Phase 1** (genuinely later-phase). No GPU-bound code exists in Phase 0, so a GPU-marked test would test nothing. The skip mechanism in `tests/conftest.py` remains in place for Phase 1. |
| **NOTE-D** — Non-editable `pip install .` would raise at import because `paths.py` resolves the repo root at import time. | **RESOLVED (documented).** `docs/RECOVERY.md` now has a "Note on installation mode" stating editable install is required and why. All documented install paths use `-e`. |
| **NOTE-E** — `manifests/artifacts.yaml` did not track reproducibility-critical files. | **PARTIALLY RESOLVED.** `requirements-lock.txt` added as a tracked, hashed artifact (`dependency_lock`), since it now defines the reproducible environment. `pyproject.toml`, `.gitignore`, and the CI workflow remain untracked by the manifest — they are already integrity-protected by Git history, and hashing them would force a manifest update on every routine edit. Documented as a deliberate scope choice. |

---

## What changed, by area

**New source modules**
- `src/juniper_math/verification.py` — safe deterministic ground-truth evaluator.

**New tests** (63 → 128)
- `tests/test_verification.py` (25), `tests/test_dependency_licenses.py` (9),
  `tests/test_git_state.py` (8), `tests/test_bootstrap_and_lock.py` (11),
  plus 6 added to `tests/test_evals.py`.

**New repository files**
- `requirements-lock.txt`, `data/{raw,interim,processed,external,cache}/.gitkeep`,
  `reports/PHASE0_REMEDIATION.md`, `reports/PHASE0_FINAL_APPROVAL.md`,
  `reports/OPUS5_PHASE0_REVIEW.md`.

**Changed frozen artifacts**
| Artifact | Before | After |
|---|---|---|
| `evals/phase0_v1.json` | `9f181afb16b0730e4c7432cd287f5ebc12556ee003be8c0f2ec8df9fb30b66ac` (v0.1.0) | `46d65c9c2bdcd065e8c0123391b5748133ccfe40245de02b24ad8187027007e7` (v0.1.1) |
| `manifests/licenses.yaml` | `92a8a63a31dc18b91b113a5ee5b44e7da106a229858bdba3aae79355afd157c4` (3 entries) | `ad6736459ff574c12f2efca9fb053cad1a92754354dcc8ccc89be218bf5a4cf3` (8 entries) |
| `requirements-lock.txt` | *(did not exist)* | `4be89c45f8c26963decc4833951dbba44c2220b4be26ee654cfd0e7f6199d341` |

`config/architecture.yaml` is **unchanged** — its hash
`ec763ed8e135f3697b2e4a1fec79df11694c5e2245f9c209160a40d12bc4f55b` is
identical to the review candidate's. No architecture value was altered during
remediation, and the parameter arithmetic was re-verified to close at exactly
5,004,032.

**Historical reports** — `SELF_REVIEW.md`, `PHASE0_REPORT.md`, and
`RECOVERY_TEST_REPORT.md` were annotated with correction notices and
"claims corrected" tables. Their original text is preserved verbatim. The
repository deliberately retains the record that the first candidate failed.

---

## Second full audit of all 22 evaluation answers

Because one "hand-verified" answer proved wrong, every case was re-audited
independently rather than spot-checked. All 18 deterministic cases were
recomputed with exact rational arithmetic and cross-checked against the values
asserted in `tests/test_verification.py`; the 4 semantic cases were checked for
classification coherence.

| Case | Check | Result |
|---|---|---|
| arith-001 | `12 + 7` | 19 ✓ |
| arith-002 | `3 + 4*2` (precedence) | 11 ✓ |
| neg-001 | `-8 + 15` | 7 ✓ |
| dec-001 | `2.5 + 3.75` | exactly 25/4 ✓ |
| frac-001 | `1/2 + 1/3` | exactly 5/6 ✓ |
| pct-001 | `15% of 240` | 36 ✓ |
| ratio-001 | `3:2`, 9 flour | 6 ✓ |
| prop-001 | inverse proportion `4*12/6` | 8 ✓ |
| alg-001 | `2x + 5 = 17` | 6 ✓ |
| units-001 | `3.5 km → m` | 3500 ✓ |
| cur-001 | `3 × 4.25` | exactly 51/4 ✓ |
| sci-001 | `3.2e4 × 2e3` | 64,000,000 ✓ |
| word-001 | `24 − 24/3 + 5` | 21 ✓ |
| est-001 | `998 × 4` = 3992 vs recorded 4000, tol 200 | within band ✓ |
| **tool-001** | `84317 × 9926` | **836,930,542 — CORRECTED** |
| direct-001 | `9 + 10` | 19 ✓ |
| wrong-001 | `7 × 8 = 54?` | False ✓ |
| err-001 | `10% of 50` | 5 ✓ |
| amb-001 | no operands → `request_clarification` | semantic ✓ |
| miss-001 | speed absent → `flag_missing_information` | semantic ✓ |
| undef-001 | `5 / 0` → `flag_undefined` | semantic, **reclassified** ✓ |
| unsup-001 | plotting → `refuse_unsupported` | semantic ✓ |

No further ground-truth defect was found.

---

## Security regression

The new verifier is the main new attack surface and was designed to have none:
it never calls `eval`, `exec`, `compile`, or `pickle`, never executes prompt
text, and dispatches only on a closed 8-operation allowlist. Unknown
operations, extra keys, malformed nodes, division by zero, and fractional
exponents all raise rather than degrade. Tests assert that
`{"op": "__import__"}` and `{"op": "system"}` are rejected.

`yaml.safe_load` remains the only YAML entry point. `scripts/bootstrap.sh`
gained diagnostics but no privileged behavior — tests assert it never invokes
`sudo` or a package manager in command position. No secrets were introduced;
the security scan was repeated over the tree and full history.

---

## Verdict

All 11 findings **RESOLVED**. Of the 5 notes: 3 resolved, 1 accepted with
documentation, 1 deferred to Phase 1 as genuinely later-phase work.

Final verification results are recorded in
`reports/PHASE0_FINAL_APPROVAL.md`.
