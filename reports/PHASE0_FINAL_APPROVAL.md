# Juniper Math 1 — Phase 0 Final Approval

## Identification

| Field | Value |
|---|---|
| **Project** | Juniper Math 1 |
| **Phase** | 0 — Foundation and Recovery |
| **Final reviewer / remediator** | Claude Opus 5 |
| **Original candidate SHA** | `b39e60031e85822a293ca60b5676ce5b7286a66f` — **failed** independent review |
| **Original candidate tag** | `phase-0-review-candidate` (unchanged, still points at the failed candidate) |
| **Remediation SHA** | `39132dedfb54e6a7ea99097188ea8743a3bc5156` |
| **Final approved commit** | the commit tagged `phase-0-foundation` (this report is part of it) |
| **Final Phase 0 tag** | `phase-0-foundation` |
| **Repository** | `https://github.com/Cinqic/juniper-math-1` |
| **Date** | 2026-08-22 |
| **Verdict** | **APPROVED** |

Cinqic completed the final human Phase 0 review and superseded the
engineer/reviewer separation for this remediation pass, authorizing Opus 5 to
remediate its own findings directly rather than returning them to Sonnet 5.

---

## Original audit result

```
CHANGES REQUIRED
BLOCKER: 0
HIGH:    1
MEDIUM:  4
LOW:     6
NOTE:    5
```

The blocking defect was a wrong answer in the frozen, hashed evaluation
suite: case `tool-001` recorded `84317 * 9926` as `837042742` when the correct
product is `836930542`. Nothing in the repository could detect it, because
evaluation validation checked schema only while the documentation claimed
answers had been "hand-verified".

Full detail: `reports/OPUS5_PHASE0_REVIEW.md`.

---

## Remediation summary

All 11 findings **RESOLVED**. Of 5 notes: 3 resolved, 1 accepted with
documentation, 1 deferred to Phase 1 as genuinely later-phase work.
Per-finding fixes, tests, and verification: `reports/PHASE0_REMEDIATION.md`.

| ID | Sev | Resolution |
|---|---|---|
| F-01 | HIGH | `tool-001` corrected to `836930542`; suite `0.1.0` → `0.1.1`; rehashed; manifest updated |
| F-02 | MED | `juniper_math.verification` added; all 22 cases carry structured verification metadata; `evals validate` now checks ground truth; `evals verify` added |
| F-03 | MED | `requirements-lock.txt` — 42 exact pins; install verified to reproduce the environment byte-for-byte |
| F-04 | MED | `python3-venv`/`python3-pip` documented; bootstrap `ensurepip` preflight with actionable message |
| F-05 | MED | licenses 3 → 8 entries from upstream metadata; `deps-check` cross-validation enforced |
| F-06 | LOW | recovery re-run against the approved commit |
| F-07 | LOW | git state now `clean`/`dirty`/`unknown`; failure never reports clean |
| F-08 | LOW | `flag_undefined` added for `undef-001`; unused `refuse_ambiguous` removed |
| F-09 | LOW | tolerance documented and implemented as absolute |
| F-10 | LOW | `data/*/.gitkeep` tracked; skeleton survives a clone |
| F-11 | LOW | handoff pins the exact candidate SHA |
| NOTE-A | — | Python support narrowed to the tested 3.12 |
| NOTE-B | — | mypy scope accepted and documented |
| NOTE-C | — | GPU-marked test deferred to Phase 1 (no GPU code exists in Phase 0) |
| NOTE-D | — | editable-install requirement documented |
| NOTE-E | — | lock added as a tracked hashed artifact; remainder documented |

---

## Architecture

Re-verified independently during final regression:

```
Token embedding (tied)   4096 × 256        =  1,048,576
Attention per block      4 × 256²          =    262,144
SwiGLU per block         3 × 256 × 688     =    528,384
RMSNorm per block        2 × 256           =        512
per block                                  =    791,040
× 5 layers                                 =  3,955,200
Final RMSNorm                              =        256
TOTAL                                      =  5,004,032   ✓ exact
```

`config/architecture.yaml` is **byte-identical** to the review candidate
(`ec763ed8e135f3697b2e4a1fec79df11694c5e2245f9c209160a40d12bc4f55b`). No
frozen architecture value — vocabulary, `d_model`, layers, head counts,
`d_ff`, tying, context, RoPE theta, dropout — was altered during remediation.
Values remain consistent across `config/architecture.yaml`,
`docs/ARCHITECTURE.md`, `README.md`, `config/project.yaml`,
`src/juniper_math/architecture.py`, and `tests/test_architecture.py`.

---

## Evaluation suite

| Field | Value |
|---|---|
| Suite ID | `phase0_baseline` (unchanged) |
| Version | `0.1.1` (was `0.1.0`) |
| Cases | 22 — 18 deterministic, 4 semantic |
| Categories | 22, one case each |
| `tool-001` | **`836930542`** (was `837042742`) |
| SHA-256 | `46d65c9c2bdcd065e8c0123391b5748133ccfe40245de02b24ad8187027007e7` |

All 22 answers were re-audited independently — not spot-checked — using exact
rational arithmetic. One defect was found (`tool-001`) and corrected; no
further ground-truth error exists. Full per-case table in
`reports/PHASE0_REMEDIATION.md`.

**Deterministic validation: PASS.** `evals validate` and `evals verify` both
recompute every deterministic answer. `tests/test_verification.py` re-injects
the original wrong value `837042742` and asserts rejection, so the exact
defect class cannot silently recur.

The verifier uses a closed 8-operation allowlist over `fractions.Fraction`. It
never calls `eval`, `exec`, `compile`, or `pickle`, and never executes prompt
text.

---

## Dependency reproducibility

**Mechanism:** `requirements-lock.txt` — 42 exact `==` pins covering all direct
and transitive dependencies, tracked and hashed as a frozen artifact
(`4be89c45f8c26963decc4833951dbba44c2220b4be26ee654cfd0e7f6199d341`).

**Fresh-install verification:** in a brand-new virtual environment in a fresh
clone, `pip install -r requirements-lock.txt` resolved to an environment
**identical** to the lock — `diff` of `pip freeze` against the lock produced
zero differences across all 42 packages.

Exact validated environment:

```
Python 3.12.3   torch 2.13.0+cu130   numpy 2.5.2       PyYAML 6.0.3
pytest 8.4.2    ruff 0.16.4          mypy 1.20.2       types-PyYAML 6.0.12.20260815
```

**Scope honesty:** the lock reproduces the **Python layer only**. It does not
pin — and is not described as pinning — the Linux kernel, the NVIDIA driver,
the CUDA driver stack, or firmware. Those are system prerequisites documented
separately in `docs/ENVIRONMENT.md` and `docs/RECOVERY.md`.

---

## Licensing

`manifests/licenses.yaml`: **8 entries**, scope stated explicitly (project code
plus every direct runtime and development dependency). Every SPDX expression
was read from installed upstream package metadata, not guessed.

Two errors beyond the original F-05 finding were found and corrected during
remediation:

- **PyTorch** was recorded as plain `BSD-3-Clause`; the distributed wheel
  declares an aggregate expression (`Apache-2.0 AND Apache-2.0 WITH
  LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT`)
  because it vendors third-party components.
- **types-PyYAML** would naturally be assumed MIT by analogy with PyYAML; it
  is typeshed and therefore **Apache-2.0**.

`deps-check`: **PASS — 7/7 direct dependencies licensed.** The cross-check is
enforced in `manifests-validate`, in CI, and by 9 tests including one that
removes NumPy from a manifest copy and asserts failure.

---

## Recovery

**Method:** fresh `git clone` of the canonical remote into a new temporary
directory, checked out at the approved commit, new virtual environment,
dependencies installed from `requirements-lock.txt`. No file, cache, or
environment reused from the development tree.

**System-prerequisite layer exercised.** The host genuinely lacks
`python3-venv`, so the F-04 condition was reproduced rather than simulated.
`scripts/bootstrap.sh` detected it before attempting `venv` and printed:

```
FAIL: this Python cannot create virtual environments — the 'ensurepip' module
      is missing, so 'python3 -m venv' will fail.
      ... sudo apt install python3-venv python3-pip ...
      This script will not install system packages for you.
```

exiting **1**. That is the intended behavior: a normal Debian-family packaging
condition now produces an actionable instruction instead of an unexplained
Python traceback, and the script never runs `sudo` or a package manager itself.

**Results in the fresh clone:**

```
validate-env        → 7/7 PASS (Overall: PASS), real RTX 2060 detected
validate-config     → PASS, parameter_target=5,004,032 estimated=5,004,032
pytest -v           → 130 passed, 0 failed, 0 skipped
hash verify         → 6/6 PASS
evals validate      → PASS schema + PASS ground truth (18 deterministic / 4 semantic)
evals verify        → PASS
manifests-validate  → PASS (sources 0, licenses 8) + cross-check PASS
deps-check          → PASS, 7/7
status              → Phase 0, COMPLETE, clean tree, commit matches
ruff check .        → All checks passed!
ruff format --check → 64 files already formatted
mypy                → Success, no issues in 14 source files
```

**Limitation, stated precisely.** This test ran on the **same physical host**
as development. It validates the complete documented procedure — prerequisite
preflight, fresh clone from the canonical remote, fresh virtual environment,
lock-based install, full gate — against the approved commit. It does **not**
prove survival of an OS wipe or hardware loss, which would require a separate
machine or isolated VM. **No clean-machine recovery claim is made.**

---

## Tests

```
pytest -v              → 130 passed, 0 failed, 0 skipped, 0 xfail
ruff check .           → All checks passed!
ruff format --check .  → 64 files already formatted
mypy                   → Success: no issues found in 14 source files
validate-config        → PASS
manifests-validate     → PASS
deps-check             → PASS
hash verify            → 6/6 PASS
evals validate/verify  → PASS
```

63 → 130 tests. The additions are regression tests for specific defects, not
count inflation: 25 for deterministic verification, 11 for the lock and
bootstrap preflight, 9 for dependency licensing, 8 for git-state honesty, plus
8 added to existing suites.

---

## CI

GitHub Actions run `32582367118` on the remediation commit: **success**
(1m45s, `foundation-checks`). CI now installs from `requirements-lock.txt`
rather than resolving ranges, pins Python to 3.12, and additionally runs
`deps-check` and `evals verify`.

---

## Security

Repeated after remediation:

- No secrets, credentials, tokens, or private keys in the working tree or in
  any historical blob.
- `yaml.safe_load` remains the only YAML entry point (3 call sites).
- No `eval`, `exec`, `compile`, `pickle`, `os.system`, or `shell=True`
  anywhere in `src/`. A dynamic `__import__('os')` call in `environment.py`
  was replaced with a normal import while auditing.
- The new verifier — the main new attack surface — executes nothing. Tests
  assert `{"op": "__import__"}` and `{"op": "system"}` are rejected, along
  with malformed nodes, unexpected keys, division by zero, and fractional
  exponents.
- `scripts/bootstrap.sh` gained diagnostics but no privileged behavior; tests
  assert it never invokes `sudo` or a package manager in command position.

**Security regression: PASS.**

---

## Artifact hash verification

All six frozen artifacts verified by the project's own tooling **and**
independently cross-checked with the OS `sha256sum` utility in the fresh
clone. All six match.

| Artifact | SHA-256 |
|---|---|
| `config/architecture.yaml` | `ec763ed8e135f3697b2e4a1fec79df11694c5e2245f9c209160a40d12bc4f55b` |
| `config/project.yaml` | `f257621c371946635d272d17b47b38643252fdbbcdb7ac0608c002cf3e0a0ebf` |
| `evals/phase0_v1.json` | `46d65c9c2bdcd065e8c0123391b5748133ccfe40245de02b24ad8187027007e7` |
| `manifests/sources.yaml` | `3f2cbc883776f347a29a652cf52d1874ab3de0aa9a6d6f89feec99565d12d29d` |
| `requirements-lock.txt` | `4be89c45f8c26963decc4833951dbba44c2220b4be26ee654cfd0e7f6199d341` |
| `manifests/licenses.yaml` | `ad6736459ff574c12f2efca9fb053cad1a92754354dcc8ccc89be218bf5a4cf3` |

---

## GitHub recovery

**No critical Phase 0 state depends on the local installation.** The working
tree has no untracked non-ignored files. A fresh clone of the canonical remote
contains every config, frozen artifact, manifest, hash, lock file, test, ADR
(8/8), report, CI workflow, bootstrap script, and recovery document required
to rebuild and validate Phase 0 — confirmed by running the entire gate in that
clone with nothing copied from the development machine.

System-layer packages (kernel, driver, `python3-venv`) correctly do not live in
Git, but the **instructions to restore them do**.

---

## Provenance chain

```
phase-0-review-candidate  →  b39e600  (independent review: CHANGES REQUIRED)
                                 ↓
                          39132de   (remediation: all 11 findings resolved)
                                 ↓
phase-0-foundation        →  final approved commit
```

`b39e600` was never amended, reset, or force-pushed. The record that the first
candidate failed review is deliberately preserved — in the tag, in the commit
history, in `reports/OPUS5_PHASE0_REVIEW.md`, and in correction notices
appended to (not substituted for) the original Sonnet 5 reports.

---

## Final verdict

**APPROVED**

## Phase status

**PHASE 0 COMPLETE**

## Next phase

**PHASE 1 AUTHORIZED** — Architecture. Not started. No model, attention, RoPE,
RMSNorm, SwiGLU, Transformer block, tokenizer, dataset, checkpoint, or
training code exists in this repository. Phase 1 begins as a separate, clean
development stage.
