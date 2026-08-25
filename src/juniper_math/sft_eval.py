"""Phase 8 Sec. 23 tool-metrics harness: runs the held-out
`evals/phase8_instruction_v1.json` suite through the full end-to-end
`tool_interaction.run_tool_interaction` loop and reports every numerator and
denominator Sec. 23 requires, plus direct-vs-tool routing and final-answer
correctness — never only a single "contains <tool_call>" rate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

from juniper_math.dataset.shard import BEHAVIOR_TAG
from juniper_math.model import JuniperMathModel
from juniper_math.tokenizer import JuniperTokenizer
from juniper_math.tool_interaction import InteractionTrace, run_tool_interaction
from juniper_math.tools.config import load_tools_config
from juniper_math.tools.runtime import ToolRuntime

_LEADING_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def _parse_number(text: str) -> Fraction | None:
    match = _LEADING_NUMBER.match(text.strip())
    if match is None:
        return None
    token = match.group(0)
    try:
        if "/" in token:
            num, den = token.split("/")
            return Fraction(int(num), int(den))
        if "." in token:
            return Fraction(token)
        return Fraction(int(token))
    except (ValueError, ZeroDivisionError):
        return None


def _final_answer_correct(case: dict[str, Any], trace: InteractionTrace) -> bool | None:
    """None means "not applicable" (no concrete expected answer, e.g. a
    clarification/unsupported case scored on tag match instead). Numeric
    comparison uses exact `Fraction` arithmetic within `tolerance` (matching
    `dataset.verify.verify_deterministic`'s comparison rule); falls back to
    exact stripped-string equality when either side doesn't parse as a
    number (e.g. an expected string not written to appear numeric)."""
    expected_answer = case.get("expected_answer")
    if expected_answer is None:
        return None
    if trace.final_text is None:
        return False
    expected_num = _parse_number(str(expected_answer))
    actual_num = _parse_number(trace.final_text)
    if expected_num is not None and actual_num is not None:
        tolerance = Fraction(str(case.get("tolerance") or 0))
        return abs(actual_num - expected_num) <= tolerance
    return trace.final_text.strip() == str(expected_answer).strip()


@dataclass
class CaseMetrics:
    case_id: str
    category: str
    tool_required: bool
    expected_tool: str | None
    tool_invocation_required: bool
    model_invoked_tool: bool
    correct_routing: bool
    emitted_tool_call: bool
    call_parsed: bool
    tool_name_correct: bool | None
    execution_successful: bool | None
    final_answer_consistent_with_result: bool | None
    final_answer_correct: bool | None
    unnecessary_tool_call: bool
    missing_required_call: bool
    fabricated_result_attempted: bool
    terminal_tag: str | None
    terminal_tag_correct: bool | None
    stopped_reason: str


@dataclass
class ToolInteractionReport:
    suite_id: str
    n_cases: int
    cases: list[CaseMetrics] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        n = self.n_cases

        def rate(pred) -> dict[str, Any]:
            num = sum(1 for c in self.cases if pred(c))
            return {"numerator": num, "denominator": n, "rate": num / n if n else 0.0}

        tool_cases = [c for c in self.cases if c.tool_invocation_required]
        direct_cases = [c for c in self.cases if not c.tool_invocation_required]

        def rate_over(cases: list[CaseMetrics], pred) -> dict[str, Any]:
            num = sum(1 for c in cases if pred(c))
            d = len(cases)
            return {"numerator": num, "denominator": d, "rate": num / d if d else None}

        return {
            "suite_id": self.suite_id,
            "n_cases": n,
            "n_tool_required_cases": len(tool_cases),
            "n_direct_cases": len(direct_cases),
            "correct_routing": rate(lambda c: c.correct_routing),
            "emitted_tool_call": rate(lambda c: c.emitted_tool_call),
            "call_parsed_valid": rate(lambda c: c.call_parsed),
            "tool_name_correct": rate_over(
                [c for c in self.cases if c.tool_name_correct is not None],
                lambda c: bool(c.tool_name_correct),
            ),
            "argument_execution_successful": rate_over(
                [c for c in self.cases if c.call_parsed], lambda c: bool(c.execution_successful)
            ),
            "end_to_end_success_on_tool_required": rate_over(
                tool_cases, lambda c: bool(c.execution_successful) and bool(c.final_answer_correct)
            ),
            "final_answer_correct_overall": rate_over(
                [c for c in self.cases if c.final_answer_correct is not None],
                lambda c: bool(c.final_answer_correct),
            ),
            "direct_answer_correct": rate_over(
                [c for c in direct_cases if c.final_answer_correct is not None],
                lambda c: bool(c.final_answer_correct),
            ),
            "unnecessary_tool_call": rate_over(direct_cases, lambda c: c.unnecessary_tool_call),
            "missing_required_call": rate_over(tool_cases, lambda c: c.missing_required_call),
            "fabricated_result_attempted": rate(lambda c: c.fabricated_result_attempted),
            "terminal_tag_correct": rate_over(
                [c for c in self.cases if c.terminal_tag_correct is not None],
                lambda c: bool(c.terminal_tag_correct),
            ),
            "warning": (
                "Sec. 23 numerator/denominator tool metrics — every rate is null-safe "
                "(0-case denominators report rate=None, never a fabricated 0.0 or 1.0)."
            ),
        }


def _load_suite_cases(path: Path, sample_size: int | None) -> tuple[str, list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = raw["cases"]
    if sample_size is not None:
        cases = cases[:sample_size]
    return raw.get("suite_id", path.stem), cases


def run_phase8_eval_suite(
    model: JuniperMathModel,
    tokenizer: JuniperTokenizer,
    suite_path: Path,
    device: torch.device,
    max_new_tokens_per_turn: int,
    sample_size: int | None = None,
) -> ToolInteractionReport:
    suite_id, cases = _load_suite_cases(suite_path, sample_size)
    runtime = ToolRuntime(load_tools_config())

    results: list[CaseMetrics] = []
    for case in cases:
        trace = run_tool_interaction(
            model, tokenizer, case["prompt"], runtime, device, max_new_tokens_per_turn
        )
        tool_required = bool(case.get("tool_required"))
        expected_tool = case.get("tool_name")
        model_invoked = trace.emitted_tool_call
        first_attempt = trace.tool_calls[0] if trace.tool_calls else None
        # Only meaningful when a tool was actually expected AND the model
        # actually named one: `None == None` (no tool expected, parse
        # failed so tool_name is None) must never count as a "correct"
        # match — that would silently reward failure on non-tool cases.
        tool_name_correct = (
            (first_attempt.tool_name == expected_tool)
            if (
                first_attempt is not None
                and expected_tool is not None
                and first_attempt.tool_name is not None
            )
            else None
        )
        execution_successful = (
            (first_attempt.result is not None and first_attempt.result.get("status") == "success")
            if first_attempt is not None
            else None
        )
        final_correct = _final_answer_correct(case, trace)

        # Mirrors `dataset.shard.expected_completion`'s tag rule, operating on
        # the raw case dict directly (no Example round-trip needed here):
        # a concrete expected_answer -> "final"; one of the four answerless
        # behaviors -> its mapped tag; "invoke_tool" with no expected_answer
        # -> no terminal tag is expected at all (None, degrades gracefully).
        if case.get("expected_answer") is not None:
            expected_tag = "final"
        elif case.get("expected_behavior") in BEHAVIOR_TAG:
            expected_tag = BEHAVIOR_TAG[case["expected_behavior"]]
        else:
            expected_tag = None
        terminal_tag_correct = (trace.terminal_tag == expected_tag) if expected_tag is not None else None

        results.append(
            CaseMetrics(
                case_id=case.get("example_id", "?"),
                category=case.get("category", "?"),
                tool_required=tool_required,
                expected_tool=expected_tool,
                tool_invocation_required=tool_required,
                model_invoked_tool=model_invoked,
                correct_routing=(model_invoked == tool_required),
                emitted_tool_call=model_invoked,
                call_parsed=bool(first_attempt is not None and first_attempt.parsed),
                tool_name_correct=tool_name_correct,
                execution_successful=execution_successful,
                final_answer_consistent_with_result=(
                    execution_successful and final_correct if execution_successful is not None else None
                ),
                final_answer_correct=final_correct,
                unnecessary_tool_call=(model_invoked and not tool_required),
                missing_required_call=(tool_required and not model_invoked),
                fabricated_result_attempted=trace.fabricated_result_discarded,
                terminal_tag=trace.terminal_tag,
                terminal_tag_correct=terminal_tag_correct,
                stopped_reason=trace.stopped_reason,
            )
        )

    return ToolInteractionReport(suite_id=suite_id, n_cases=len(cases), cases=results)


__all__ = ["CaseMetrics", "ToolInteractionReport", "run_phase8_eval_suite"]
