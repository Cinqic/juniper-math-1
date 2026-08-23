"""Phase 5 tool-format evaluation: does the evaluation *infrastructure* work end to end.

Runs a smoke checkpoint's generations against the frozen v2 tool-use
evaluation suite (`evals/phase4_tool_use_v2.json`) and checks whether
generated text contains a well-formed `<tool_call>{...}` block per the
approved protocol (`juniper_math.tools.protocol.parse_tool_call`).

This is explicitly SMOKE PIPELINE VALIDATION ONLY (Sec. 19, 26). A smoke
model is expected to perform badly or randomly; what must not happen is the
evaluator crashing, mis-parsing, or silently skipping malformed output. The
protocol itself is never loosened to accommodate bad generations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from juniper_math.generation import generate
from juniper_math.model import JuniperMathModel
from juniper_math.tokenizer import JuniperTokenizer
from juniper_math.tools.config import load_tools_config
from juniper_math.tools.errors import ToolProtocolError
from juniper_math.tools.protocol import parse_tool_call
from juniper_math.tools.runtime import ToolRuntime

_TAG_PATTERN = re.compile(r"<(tool_call|tool_result|final|unsupported|error)>")


def load_v2_suite_cases(path: Path) -> list[dict[str, Any]]:
    """Loads a frozen Phase 4 v2 suite's raw cases (Example-shaped, not the phase0 EvalCase schema)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path}: 'cases' must be a non-empty list.")
    return cases


def extract_tool_call_text(generated_text: str) -> str | None:
    """Returns the JSON substring after the first <tool_call> tag, or None if absent."""
    match = _TAG_PATTERN.search(generated_text)
    while match is not None:
        if match.group(1) == "tool_call":
            start = match.end()
            rest = generated_text[start:]
            next_tag = _TAG_PATTERN.search(rest)
            end = next_tag.start() if next_tag is not None else len(rest)
            return rest[:end].strip()
        match = _TAG_PATTERN.search(generated_text, match.end())
    return None


@dataclass
class ToolFormatCaseResult:
    case_id: str
    expected_tool: str | None
    generated_text: str
    extracted_call_text: str | None
    parsed: bool
    parse_error: str | None
    tool_name_match: bool


@dataclass
class ToolFormatReport:
    suite_id: str
    n_cases: int
    n_emitted_tool_call: int
    n_parsed_valid: int
    n_tool_name_match: int
    results: list[ToolFormatCaseResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "n_cases": self.n_cases,
            "n_emitted_tool_call": self.n_emitted_tool_call,
            "n_parsed_valid": self.n_parsed_valid,
            "n_tool_name_match": self.n_tool_name_match,
            "emitted_rate": self.n_emitted_tool_call / self.n_cases if self.n_cases else 0.0,
            "valid_rate": self.n_parsed_valid / self.n_cases if self.n_cases else 0.0,
            "tool_name_match_rate": self.n_tool_name_match / self.n_cases if self.n_cases else 0.0,
            "warning": "SMOKE PIPELINE VALIDATION ONLY — not a capability measurement.",
        }


def run_tool_format_evaluation(
    model: JuniperMathModel,
    tokenizer: JuniperTokenizer,
    suite_path: Path,
    device: torch.device,
    max_new_tokens: int,
    sample_size: int | None = None,
) -> ToolFormatReport:
    raw = json.loads(suite_path.read_text(encoding="utf-8"))
    cases = raw["cases"]
    if sample_size is not None:
        cases = cases[:sample_size]

    tools_config = load_tools_config()
    limits = ToolRuntime(tools_config).limits

    results: list[ToolFormatCaseResult] = []
    n_emitted = 0
    n_valid = 0
    n_match = 0
    for case in cases:
        result = generate(model, tokenizer, case["prompt"], max_new_tokens, device)
        call_text = extract_tool_call_text(result.text)
        parsed = False
        parse_error: str | None = None
        tool_name_match = False
        if call_text is not None:
            n_emitted += 1
            try:
                call = parse_tool_call(call_text, limits)
                parsed = True
                n_valid += 1
                if call.tool == case.get("tool_name"):
                    tool_name_match = True
                    n_match += 1
            except ToolProtocolError as exc:
                parse_error = f"{exc.code}: {exc.message}"
        results.append(
            ToolFormatCaseResult(
                case_id=case.get("example_id", "?"),
                expected_tool=case.get("tool_name"),
                generated_text=result.text,
                extracted_call_text=call_text,
                parsed=parsed,
                parse_error=parse_error,
                tool_name_match=tool_name_match,
            )
        )

    return ToolFormatReport(
        suite_id=raw.get("suite_id", suite_path.stem),
        n_cases=len(cases),
        n_emitted_tool_call=n_emitted,
        n_parsed_valid=n_valid,
        n_tool_name_match=n_match,
        results=results,
    )


__all__ = [
    "ToolFormatCaseResult",
    "ToolFormatReport",
    "extract_tool_call_text",
    "load_v2_suite_cases",
    "run_tool_format_evaluation",
]
