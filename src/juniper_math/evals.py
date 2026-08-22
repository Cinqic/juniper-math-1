"""Fixed evaluation suite loading and integrity validation.

Phase 0 freezes a small baseline suite (evals/phase0_v1.json) BEFORE any
model exists. Its purpose at this stage is integrity — not scoring a model.
Scoring infrastructure that actually runs a model against these cases is
later-phase work.

Two distinct kinds of validation are provided, and the distinction matters:

* **Schema validation** (:func:`load_eval_suite`) — required fields, unique
  IDs, known categories/difficulties/behaviors, well-formed types.
* **Deterministic ground-truth validation**
  (:mod:`juniper_math.verification`) — recomputes each deterministic case's
  answer from structured ``verification`` metadata and compares it against
  the recorded ``expected_answer``.

Suite version 0.1.0 had schema validation only. That was insufficient: it
declared answers "hand-verified" while case ``tool-001`` carried an
arithmetically wrong product. Version 0.1.1 corrects the answer and adds the
``verification`` block that makes such an error detectable automatically.
See ``reports/OPUS5_PHASE0_REVIEW.md`` (F-01, F-02).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from juniper_math.errors import JuniperConfigError
from juniper_math.paths import EVALS_DIR
from juniper_math.verification import VALID_VERIFICATION_MODES, VerificationResult, verify_suite

DEFAULT_SUITE_PATH = EVALS_DIR / "phase0_v1.json"

_VALID_CATEGORIES = {
    "arithmetic_interpretation",
    "operator_precedence",
    "negative_values",
    "decimals",
    "fractions",
    "percentages",
    "ratios",
    "proportions",
    "basic_algebra",
    "units",
    "currency",
    "scientific_notation",
    "word_problem",
    "estimation",
    "ambiguity",
    "missing_information",
    "undefined_operation",
    "unsupported_capability",
    "tool_required",
    "direct_answer",
    "incorrect_supplied_answer",
    "error_recognition",
}

# Closed behavior vocabulary. `flag_undefined` was added in suite 0.1.1 to
# distinguish undefined mathematics (5 / 0) from missing information (an
# operand that was simply never supplied) — conflating them would train and
# measure the wrong distinction. The unused `refuse_ambiguous` was removed in
# the same version; `request_clarification` already covers that behavior.
# See reports/OPUS5_PHASE0_REVIEW.md (F-08).
_VALID_EXPECTED_BEHAVIORS = {
    "answer",
    "request_clarification",
    "refuse_unsupported",
    "flag_missing_information",
    "flag_undefined",
    "flag_incorrect_answer",
    "invoke_tool",
}

_REQUIRED_CASE_FIELDS = {
    "id",
    "category",
    "difficulty",
    "prompt",
    "expected_behavior",
    "expected_answer",
    "tolerance",
    "tool_required",
    "verification",
    "provenance",
    "notes",
}

_VALID_DIFFICULTIES = {"trivial", "easy", "medium", "hard"}


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: str
    difficulty: str
    prompt: str
    expected_behavior: str
    expected_answer: Any
    tolerance: float | None
    tool_required: bool
    verification: dict[str, Any]
    provenance: str
    notes: str


@dataclass(frozen=True)
class EvalSuite:
    suite_version: str
    suite_id: str
    cases: list[EvalCase]

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self.cases:
            counts[case.category] = counts.get(case.category, 0) + 1
        return counts


def _validate_case(raw: dict, index: int, source: Path) -> None:
    missing = _REQUIRED_CASE_FIELDS - raw.keys()
    if missing:
        raise JuniperConfigError(f"{source}: case[{index}] missing field(s): {sorted(missing)}")
    if not isinstance(raw["id"], str) or not raw["id"]:
        raise JuniperConfigError(f"{source}: case[{index}] 'id' must be a non-empty string")
    if raw["category"] not in _VALID_CATEGORIES:
        raise JuniperConfigError(
            f"{source}: case[{index}] id={raw['id']!r} has unknown category {raw['category']!r}"
        )
    if raw["difficulty"] not in _VALID_DIFFICULTIES:
        raise JuniperConfigError(
            f"{source}: case[{index}] id={raw['id']!r} has unknown difficulty {raw['difficulty']!r}"
        )
    if raw["expected_behavior"] not in _VALID_EXPECTED_BEHAVIORS:
        raise JuniperConfigError(
            f"{source}: case[{index}] id={raw['id']!r} has unknown expected_behavior "
            f"{raw['expected_behavior']!r}"
        )
    if not isinstance(raw["tool_required"], bool):
        raise JuniperConfigError(f"{source}: case[{index}] id={raw['id']!r} 'tool_required' must be bool")
    if raw["tolerance"] is not None and not isinstance(raw["tolerance"], (int, float)):
        raise JuniperConfigError(
            f"{source}: case[{index}] id={raw['id']!r} 'tolerance' must be null or numeric"
        )
    verification = raw["verification"]
    if not isinstance(verification, dict):
        raise JuniperConfigError(f"{source}: case[{index}] id={raw['id']!r} 'verification' must be a mapping")
    if verification.get("mode") not in VALID_VERIFICATION_MODES:
        raise JuniperConfigError(
            f"{source}: case[{index}] id={raw['id']!r} has unknown verification mode "
            f"{verification.get('mode')!r}; expected one of {sorted(VALID_VERIFICATION_MODES)}"
        )
    if "expression" not in verification:
        raise JuniperConfigError(
            f"{source}: case[{index}] id={raw['id']!r} 'verification' is missing 'expression'"
        )


def load_eval_suite(path: Path | None = None) -> EvalSuite:
    source = path or DEFAULT_SUITE_PATH
    if not source.is_file():
        raise JuniperConfigError(f"Evaluation suite not found at {source}.")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise JuniperConfigError(f"{source}: invalid JSON ({exc}).") from exc

    for key in ("suite_version", "suite_id", "cases"):
        if key not in raw:
            raise JuniperConfigError(f"{source}: missing top-level field '{key}'.")
    if not isinstance(raw["cases"], list) or not raw["cases"]:
        raise JuniperConfigError(f"{source}: 'cases' must be a non-empty list.")

    seen_ids: set[str] = set()
    cases: list[EvalCase] = []
    for index, raw_case in enumerate(raw["cases"]):
        _validate_case(raw_case, index, source)
        if raw_case["id"] in seen_ids:
            raise JuniperConfigError(f"{source}: duplicate case id {raw_case['id']!r}")
        seen_ids.add(raw_case["id"])
        cases.append(
            EvalCase(
                id=raw_case["id"],
                category=raw_case["category"],
                difficulty=raw_case["difficulty"],
                prompt=raw_case["prompt"],
                expected_behavior=raw_case["expected_behavior"],
                expected_answer=raw_case["expected_answer"],
                tolerance=raw_case["tolerance"],
                tool_required=raw_case["tool_required"],
                verification=raw_case["verification"],
                provenance=raw_case["provenance"],
                notes=raw_case["notes"],
            )
        )

    return EvalSuite(suite_version=raw["suite_version"], suite_id=raw["suite_id"], cases=cases)


def verify_suite_ground_truth(suite: EvalSuite) -> list[VerificationResult]:
    """Recompute every deterministic case's answer and report the results.

    Schema validity says nothing about arithmetic correctness; this is the
    check that does. Returns one result per case (semantic cases report as
    verified with an explanatory detail).
    """
    return verify_suite(suite)
