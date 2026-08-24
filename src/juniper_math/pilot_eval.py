"""Phase 6 model-scoring evaluation for the three frozen v2 suites Phase 5 left unscored.

Phase 5's only model-facing scorer (`juniper_math.tool_format_eval`) checks
generation against `evals/phase4_tool_use_v2.json` for well-formed
`<tool_call>{...}` syntax — a pipeline-mechanics check, not a capability
measurement, and it says nothing about `phase4_math_v2.json`,
`phase4_calibration_v2.json`, or `phase4_adversarial_v2.json` (Sec. 18-19 of
the Phase 6 instructions require category-broken-out numbers on all four).
This module scores those three (and can score tool_use's answer-correctness
alongside `tool_format_eval`'s syntax-only check).

Ground truth
------------
Every case uses the same `Example` schema the training corpus uses. The
"correct" terminal output for a case is defined by
`juniper_math.dataset.shard.expected_completion` — the exact same
tag/value rule `render_training_text` uses to render training targets, not
a second hand-invented notion of correctness. A generation is scored
correct only if:
  - the expected tag is `final` (covers `answer`, `flag_incorrect_answer`,
    and tool-required cases whose ground truth resolves to a value): the
    generated text contains a `<final>` tag, and the value after it is
    numerically equal (via `juniper_math.verification`'s exact
    `fractions.Fraction` comparison, honoring the case's recorded
    tolerance — never float `==`) to `expected_answer`.
  - the expected tag is `unsupported` or `error` (refusal/clarification
    behaviors): that literal tag appears in the generated text.
A case with no recognizable tag, or a `<final>` value that fails to parse
as a number, is scored incorrect and still counted in the denominator —
0% on an early checkpoint is data (Sec. 19, Sec. 30's "no false PASS on
silently skipped input" rule applies here too).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

from juniper_math.dataset.io import example_from_dict
from juniper_math.dataset.schema import Example
from juniper_math.dataset.shard import expected_completion
from juniper_math.errors import JuniperConfigError
from juniper_math.generation import generate
from juniper_math.model import JuniperMathModel
from juniper_math.tokenizer import JuniperTokenizer
from juniper_math.verification import to_exact

_TAG_PATTERN = re.compile(r"<(tool_call|tool_result|final|unsupported|error)>")


def extract_tagged_value(generated_text: str, tag: str) -> tuple[bool, str | None]:
    """Returns (tag_present, text_between_the_first_matching_tag_and_the_next_tag_or_end)."""
    match = _TAG_PATTERN.search(generated_text)
    while match is not None:
        if match.group(1) == tag:
            start = match.end()
            rest = generated_text[start:]
            next_tag = _TAG_PATTERN.search(rest)
            end = next_tag.start() if next_tag is not None else len(rest)
            return True, rest[:end].strip()
        match = _TAG_PATTERN.search(generated_text, match.end())
    return False, None


def numeric_matches(example: Example, generated_value_text: str) -> tuple[bool, str]:
    try:
        expected = to_exact(example.expected_answer, "expected_answer")
    except JuniperConfigError as exc:
        return False, f"expected_answer not numeric: {exc}"
    try:
        got = to_exact(generated_value_text, "generated <final> value")
    except JuniperConfigError:
        return False, f"generated value {generated_value_text!r} did not parse as a number"
    tolerance = Fraction(0) if example.tolerance is None else to_exact(example.tolerance, "tolerance")
    verified = abs(got - expected) <= tolerance
    detail = f"got {got} vs expected {expected} (tolerance {tolerance})"
    return verified, detail


@dataclass
class PilotEvalCaseResult:
    example_id: str
    category: str
    expected_tag: str
    expected_value: str | None
    generated_text: str
    emitted_expected_tag: bool
    correct: bool
    detail: str


@dataclass
class PilotEvalSuiteReport:
    suite_id: str
    n_cases: int
    n_correct: int
    category_counts: dict[str, int] = field(default_factory=dict)
    category_correct: dict[str, int] = field(default_factory=dict)
    results: list[PilotEvalCaseResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "n_cases": self.n_cases,
            "n_correct": self.n_correct,
            "accuracy": self.n_correct / self.n_cases if self.n_cases else 0.0,
            "category_counts": self.category_counts,
            "category_accuracy": {
                cat: (self.category_correct.get(cat, 0) / count if count else 0.0)
                for cat, count in sorted(self.category_counts.items())
            },
        }


def run_capability_evaluation(
    model: JuniperMathModel,
    tokenizer: JuniperTokenizer,
    suite_path: Path,
    device: torch.device,
    max_new_tokens: int,
    sample_size: int | None = None,
) -> PilotEvalSuiteReport:
    raw = json.loads(suite_path.read_text(encoding="utf-8"))
    cases_raw = raw["cases"]
    if not isinstance(cases_raw, list) or not cases_raw:
        raise JuniperConfigError(f"{suite_path}: 'cases' must be a non-empty list.")
    if sample_size is not None:
        cases_raw = cases_raw[:sample_size]

    results: list[PilotEvalCaseResult] = []
    category_counts: dict[str, int] = {}
    category_correct: dict[str, int] = {}
    n_correct = 0

    for raw_case in cases_raw:
        example = example_from_dict(raw_case)
        category_counts[example.category] = category_counts.get(example.category, 0) + 1
        try:
            tag, value = expected_completion(example)
        except ValueError as exc:
            results.append(
                PilotEvalCaseResult(
                    example.example_id, example.category, "?", None, "", False, False, str(exc)
                )
            )
            continue

        gen = generate(model, tokenizer, example.prompt, max_new_tokens, device, temperature=0.0)
        emitted, gen_value = extract_tagged_value(gen.text, tag)

        if tag == "final":
            if emitted and gen_value:
                correct, detail = numeric_matches(example, gen_value)
            else:
                correct, detail = False, "no <final> tag emitted"
        else:
            correct, detail = emitted, ("tag present" if emitted else "tag absent")

        if correct:
            n_correct += 1
            category_correct[example.category] = category_correct.get(example.category, 0) + 1
        results.append(
            PilotEvalCaseResult(
                example.example_id, example.category, tag, value, gen.text, emitted, correct, detail
            )
        )

    return PilotEvalSuiteReport(
        suite_id=raw.get("suite_id", suite_path.stem),
        n_cases=len(cases_raw),
        n_correct=n_correct,
        category_counts=category_counts,
        category_correct=category_correct,
        results=results,
    )


__all__ = [
    "PilotEvalCaseResult",
    "PilotEvalSuiteReport",
    "extract_tagged_value",
    "numeric_matches",
    "run_capability_evaluation",
]
