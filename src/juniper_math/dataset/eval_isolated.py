"""Evaluation-only Phase 4 case constructors.

This module does not depend on the training generator registry or training
generator implementations. It may reuse non-generative shared formatting
helpers. Its identities, natural-language templates, and compositions are
held out from the training corpus rather than being a different random seed.
"""

from __future__ import annotations

import json
import random

from juniper_math.dataset.build import ground_truth_ok
from juniper_math.dataset.clean import normalize_text
from juniper_math.dataset.config import DatasetConfig
from juniper_math.dataset.generators.common import fmt_frac
from juniper_math.dataset.idgen import derive_id, derive_seed
from juniper_math.dataset.schema import Example, ToolTrace, validate_example
from juniper_math.dataset.shard import render_training_text
from juniper_math.dataset.verify import evaluate_expression
from juniper_math.errors import JuniperConfigError
from juniper_math.hashing import sha256_bytes
from juniper_math.tokenizer import JuniperTokenizer
from juniper_math.tools.runtime import ToolRuntime

EVAL_GENERATOR_ID = "phase4_evaluation_only"
EVAL_GENERATOR_VERSION = "2.0.0"

SUITE_DEFINITIONS: dict[str, dict] = {
    "phase4_math_v2": {
        "suite_id": "phase4-math-v2",
        "suite_version": "2.0.0",
        "description": "Evaluation-only core mathematics; no training generator or template is used.",
        "categories": {
            "arithmetic": 20,
            "operator_precedence": 15,
            "negative_values": 15,
            "decimals": 15,
            "fractions": 15,
            "percentages": 18,
            "ratios_proportions": 15,
            "scientific_notation": 12,
            "basic_algebra": 18,
            "expression_translation": 12,
            "word_problem": 18,
            "estimation": 12,
            "numerical_comparison": 15,
            "multi_step": 15,
        },
    },
    "phase4_tool_use_v2": {
        "suite_id": "phase4-tool-use-v2",
        "suite_version": "2.0.0",
        "description": (
            "Evaluation-only tool-use suite; every trace is executed through the real Phase 3 runtime."
        ),
        "categories": {
            "unit_conversion": 45,
            "financial_math": 45,
            "tool_use": 40,
            "incorrect_tool_call": 30,
            "tool_error": 25,
        },
    },
    "phase4_calibration_v2": {
        "suite_id": "phase4-calibration-v2",
        "suite_version": "2.0.0",
        "description": (
            "Evaluation-only calibration suite with independently worded premise checks "
            "and direct mathematics."
        ),
        "categories": {
            "incorrect_supplied_answer": 60,
            "arithmetic": 30,
            "percentages": 20,
            "basic_algebra": 20,
        },
    },
    "phase4_adversarial_v2": {
        "suite_id": "phase4-adversarial-v2",
        "suite_version": "2.0.0",
        "description": (
            "Evaluation-only ambiguity, missing-data, undefined-operation, "
            "unsupported-capability, and tool-failure suite."
        ),
        "categories": {
            "ambiguity": 40,
            "missing_information": 40,
            "undefined_operation": 40,
            "unsupported_capability": 40,
            "incorrect_tool_call": 20,
            "tool_error": 15,
        },
    },
}


def _trace(runtime: ToolRuntime, tool: str, arguments: dict) -> ToolTrace:
    call = {"protocol_version": "1.0.0", "tool": tool, "arguments": arguments}
    return ToolTrace(
        call=call,
        result=runtime.execute_text(json.dumps(call, sort_keys=True, separators=(",", ":"))).to_dict(),
    )


def _base(category: str, index: int, seed: int, **overrides: object) -> dict:
    return {
        "example_id": derive_id("phase4-eval-v2", category, index, seed, length=24),
        "generator_id": EVAL_GENERATOR_ID,
        "generator_version": EVAL_GENERATOR_VERSION,
        "family_id": f"held_out_{category}",
        "template_id": f"eval_only_{category}",
        "derivation_id": derive_id("phase4-eval-v2-derivation", category, index, seed),
        "seed": seed,
        "category": category,
        "difficulty": "medium" if index % 3 else "hard",
        "synthetic": True,
        "split": "test",
        "tool_required": False,
        "tool_name": None,
        "tool_traces": (),
        "provenance": "Phase 4 evaluation-only constructor v2.0.0",
        "notes": "held-out evaluation-only template",
        **overrides,
    }


def _direct(category: str, index: int, seed: int) -> Example:
    rng = random.Random(derive_seed("phase4-eval-v2", category, index, seed))
    a, b, c = rng.randint(7, 97), rng.randint(3, 49), rng.randint(2, 19)
    if category == "operator_precedence":
        prompt, tree = (
            f"In a lab notebook, calculate ({a} + {b}) × {c}.",
            {"op": "mul", "args": [{"op": "add", "args": [a, b]}, c]},
        )
    elif category == "negative_values":
        prompt, tree = (
            f"A diver is at −{a} m and descends {b} m more. What is the new signed position?",
            {"op": "sub", "args": [-a, b]},
        )
    elif category == "decimals":
        prompt, tree = (
            f"A scale reads {a}.5 g, then {b}.25 g is added. Report the total.",
            {"op": "add", "args": [f"{a}.5", f"{b}.25"]},
        )
    elif category == "fractions":
        prompt, tree = (
            f"Combine {a}/{c} and {b}/{c}; give the exact result.",
            {"op": "add", "args": [{"op": "div", "args": [a, c]}, {"op": "div", "args": [b, c]}]},
        )
    elif category == "percentages":
        prompt, tree = (
            f"A survey has {a * 10} responses; {b}% meet a criterion. How many responses is that?",
            {"op": "percent_of", "args": [b, a * 10]},
        )
    elif category == "ratios_proportions":
        prompt, tree = (
            f"A dye recipe uses {a}:{b}. If the first component is {a * c}, "
            "what amount of the second preserves the ratio?",
            {"op": "mul", "args": [b, c]},
        )
    elif category == "scientific_notation":
        prompt, tree = (
            f"A sensor records {a} × 10^{c}. What is that quantity multiplied by 10?",
            {"op": "mul", "args": [a * (10**c), 10]},
        )
    elif category == "basic_algebra":
        prompt, tree = (
            f"Solve the equation {c}x + {b} = {c * a + b} for x.",
            {"op": "div", "args": [c * a, c]},
        )
    elif category == "expression_translation":
        prompt, tree = (
            f"Translate and evaluate: subtract {b} from three times {a}.",
            {"op": "sub", "args": [{"op": "mul", "args": [3, a]}, b]},
        )
    elif category == "word_problem":
        prompt, tree = (
            f"A library shelves {a} books on Monday and {b} on Tuesday, then lends out {c}. How many remain?",
            {"op": "sub", "args": [{"op": "add", "args": [a, b]}, c]},
        )
    elif category == "estimation":
        prompt, tree = (
            f"Estimate {a} × {b} after rounding both values to the nearest ten.",
            {"op": "mul", "args": [round(a, -1), round(b, -1)]},
        )
    elif category == "numerical_comparison":
        prompt, tree = (
            f"Which value is larger: {a} × {c} or {b} × {c}? State the larger value.",
            {"op": "max", "args": [a * c, b * c]},
        )
    elif category == "multi_step":
        prompt, tree = (
            f"A tank has {a} L, receives {b} L, and is split across {c} vessels. How much is in each?",
            {"op": "div", "args": [{"op": "add", "args": [a, b]}, c]},
        )
    else:
        prompt, tree = (
            f"A reviewer needs the total of {a} and {b}. Calculate it directly.",
            {"op": "add", "args": [a, b]},
        )
    answer = evaluate_expression(tree)
    assert not isinstance(answer, bool)
    return Example(
        **_base(
            category,
            index,
            seed,
            prompt=prompt,
            expected_behavior="answer",
            expected_answer=fmt_frac(answer),
            tolerance=0,
            verification={"mode": "deterministic", "expression": tree},
        )
    )


def _make_case(category: str, index: int, seed: int, runtime: ToolRuntime) -> Example:
    if category in {"ambiguity", "missing_information", "undefined_operation", "unsupported_capability"}:
        prompt = {
            "ambiguity": f"Jordan has {index + 2} marbles. What is the answer?",
            "missing_information": f"A rectangle has area {index + 12}. What is its perimeter?",
            "undefined_operation": f"Evaluate {index + 1} divided by zero.",
            "unsupported_capability": "Prove the Riemann hypothesis in a single calculator response.",
        }[category]
        behavior = {
            "ambiguity": "request_clarification",
            "missing_information": "flag_missing_information",
            "undefined_operation": "flag_undefined",
            "unsupported_capability": "refuse_unsupported",
        }[category]
        return Example(
            **_base(
                category,
                index,
                seed,
                prompt=prompt,
                expected_behavior=behavior,
                expected_answer=None,
                tolerance=None,
                verification={"mode": "semantic", "expression": None},
            )
        )
    if category in {"unit_conversion", "financial_math", "tool_use", "incorrect_tool_call", "tool_error"}:
        value = index + 11
        if category == "unit_conversion":
            trace, prompt, behavior = (
                _trace(
                    runtime,
                    "calculator.convert",
                    {"category": "length", "from_unit": "meter", "to_unit": "foot", "value": value},
                ),
                f"For a field report, use the conversion tool to express {value} meters in feet.",
                "invoke_tool",
            )
        elif category == "financial_math":
            trace, prompt, behavior = (
                _trace(
                    runtime,
                    "calculator.finance",
                    {
                        "operation": "simple_interest",
                        "principal": value * 100,
                        "annual_rate_percent": 5,
                        "years": 2,
                    },
                ),
                f"Use the finance tool: find two years of simple interest on ${value * 100} at 5% annually.",
                "invoke_tool",
            )
        elif category == "incorrect_tool_call":
            trace, prompt, behavior = (
                _trace(runtime, "calculator.evaluate", {"expression": f"{value} * 4"}),
                f"A draft tool call added {value} and 4, but the request was to multiply them. "
                "Identify the correct result.",
                "flag_incorrect_answer",
            )
        elif category == "tool_error":
            trace, prompt, behavior = (
                _trace(runtime, "calculator.evaluate", {"expression": f"{value} / 0"}),
                f"Send this expression to the calculator and report its real outcome: {value} / 0.",
                "invoke_tool",
            )
        else:
            trace, prompt, behavior = (
                _trace(runtime, "calculator.evaluate", {"expression": f"({value} * {value}) - {index}"}),
                f"Use the calculator for the audit expression ({value} × {value}) − {index}.",
                "invoke_tool",
            )
        expected = trace.result["result"]["value"] if trace.result["status"] == "success" else None
        return Example(
            **_base(
                category,
                index,
                seed,
                prompt=prompt,
                expected_behavior=behavior,
                expected_answer=expected,
                tolerance=0 if expected is not None else None,
                tool_required=True,
                tool_name=trace.call["tool"],
                tool_traces=(trace,),
                verification={"mode": "tool", "expression": None},
            )
        )
    if category == "incorrect_supplied_answer":
        ex = _direct("arithmetic", index, seed)
        return Example(
            **{
                **ex.__dict__,
                "category": category,
                "family_id": "held_out_incorrect_premise",
                "template_id": "eval_only_incorrect_premise",
                "prompt": f"A worksheet claims the answer is 0; independently check this: {ex.prompt}",
                "expected_behavior": "flag_incorrect_answer",
            }
        )
    return _direct(category, index, seed)


def build_independent_eval_suite(
    name: str, config: DatasetConfig, tokenizer: JuniperTokenizer, runtime: ToolRuntime
):
    from juniper_math.dataset.eval_suites import SuiteBuildResult

    if name not in SUITE_DEFINITIONS:
        raise JuniperConfigError(f"Unknown Phase 4 eval suite {name!r}. Known: {sorted(SUITE_DEFINITIONS)}")
    spec = SUITE_DEFINITIONS[name]
    seed = config.master_seed + config.split.eval_suite_seed_offset + sum(ord(c) for c in name)
    cases: list[dict] = []
    for category, count in spec["categories"].items():
        for index in range(count):
            ex = _make_case(category, index, seed, runtime)
            ex = Example(**{**ex.__dict__, "prompt": normalize_text(ex.prompt, config.normalization)})
            validate_example(ex)
            ok, detail = ground_truth_ok(ex)
            if not ok:
                raise JuniperConfigError(f"evaluation-only {category}/{index}: {detail}")
            ex = ex.with_token_count(len(tokenizer.encode(render_training_text(ex))))
            cases.append(ex.to_dict())
    payload = {
        "suite_id": spec["suite_id"],
        "suite_version": spec["suite_version"],
        "description": spec["description"],
        "record_schema": "juniper_math.dataset.schema.Example",
        "cases": sorted(cases, key=lambda c: c["example_id"]),
    }
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return SuiteBuildResult(
        name=name,
        payload=payload,
        text=text,
        sha256=sha256_bytes(text.encode("utf-8")),
        example_count=len(cases),
        prompts=[c["prompt"] for c in cases],
    )
