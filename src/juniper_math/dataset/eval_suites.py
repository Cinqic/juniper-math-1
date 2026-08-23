"""Phase 4 frozen evaluation suite construction (Sec. 12).

Each suite is a JSON file using the SAME record schema as the training
corpus (juniper_math.dataset.schema.Example) rather than the narrower
juniper_math.evals schema the frozen Phase 0 suite uses — Phase 3 already
established that different evaluation suites may use different, purpose-fit
schemas (evals/phase3_tools_v1.json uses its own call/expected_status
schema, not juniper_math.evals's). The Phase 0 baseline's math-only,
8-operation-allowlist schema was never designed to represent a tool-required
or tool-error case; forcing Phase 4's tool-use and adversarial suites into
that shape would be a worse fit than giving them their own frozen format.

Contamination isolation from train/validation/test (Sec. 13): every suite is
generated using ``config.split.eval_suite_seed_offset`` folded into the
seed, giving eval-suite generation a disjoint seed namespace from the
corpus build — the same generator/family/index inputs a train-split example
might use are never drawn here, and the standalone `dataset
contamination-check` command double-checks this after the fact with actual
prompt-similarity comparison (defense in depth, not just seed-namespace
trust).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from juniper_math.dataset.build import ground_truth_ok
from juniper_math.dataset.clean import normalize_text
from juniper_math.dataset.config import DatasetConfig
from juniper_math.dataset.generators.registry import build_registry
from juniper_math.dataset.schema import Example, validate_example
from juniper_math.dataset.shard import render_training_text
from juniper_math.errors import JuniperConfigError
from juniper_math.hashing import sha256_bytes
from juniper_math.tokenizer import JuniperTokenizer
from juniper_math.tools.runtime import ToolRuntime

SUITE_DEFINITIONS: dict[str, dict] = {
    "phase4_math_v1": {
        "suite_id": "phase4-math-v1",
        "suite_version": "1.0.0",
        "description": "Frozen Phase 4 core-mathematics evaluation suite (no tool involvement).",
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
    "phase4_tool_use_v1": {
        "suite_id": "phase4-tool-use-v1",
        "suite_version": "1.0.0",
        "description": "Frozen Phase 4 tool-use evaluation suite: every case is executed against the "
        "real Phase 3 ToolRuntime.",
        "categories": {
            "unit_conversion": 45,
            "financial_math": 45,
            "tool_use": 40,
            "incorrect_tool_call": 30,
            "tool_error": 25,
        },
    },
    "phase4_calibration_v1": {
        "suite_id": "phase4-calibration-v1",
        "suite_version": "1.0.0",
        "description": "Frozen Phase 4 calibration/truthfulness suite: a mix of correct direct "
        "answers and deliberately incorrect claimed answers, testing whether the model asserts "
        "confidence appropriately rather than agreeing with whatever the prompt states.",
        "categories": {
            "incorrect_supplied_answer": 60,
            "arithmetic": 30,
            "percentages": 20,
            "basic_algebra": 20,
        },
    },
    "phase4_adversarial_v1": {
        "suite_id": "phase4-adversarial-v1",
        "suite_version": "1.0.0",
        "description": "Frozen Phase 4 adversarial/error-handling suite: ambiguous, underspecified, "
        "mathematically undefined, and out-of-scope requests, plus malformed tool-use scenarios.",
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


@dataclass(frozen=True)
class SuiteBuildResult:
    name: str
    payload: dict
    text: str
    sha256: str
    example_count: int
    prompts: list[str]


def build_eval_suite(
    name: str,
    config: DatasetConfig,
    tokenizer: JuniperTokenizer,
    runtime: ToolRuntime,
) -> SuiteBuildResult:
    if name not in SUITE_DEFINITIONS:
        raise JuniperConfigError(f"Unknown Phase 4 eval suite {name!r}. Known: {sorted(SUITE_DEFINITIONS)}")
    spec = SUITE_DEFINITIONS[name]
    registry = build_registry()

    # Disjoint from every train/validation/test seed: config-level
    # eval_suite_seed_offset plus a per-suite-name offset, both fixed
    # integers (never Python's hash() — that is PYTHONHASHSEED-randomized
    # per process, see Sec. 29 — so reproducibility comes only from these
    # deterministic sums).
    suite_seed = config.master_seed + config.split.eval_suite_seed_offset + sum(ord(c) for c in name)

    cases: list[dict] = []
    for category, count in spec["categories"].items():
        families = registry[category]
        produced = 0
        attempts = 0
        idx = 0
        seen_ids: set[str] = set()
        while produced < count and attempts < count * 50 + 100:
            attempts += 1
            generator_id, maker = families[attempts % len(families)]
            try:
                ex = maker(idx, suite_seed, runtime)
            except Exception:  # noqa: BLE001
                idx += 1
                continue
            idx += 1
            if ex.example_id in seen_ids:
                continue
            try:
                normalized = normalize_text(ex.prompt, config.normalization)
                ex = Example(**{**ex.__dict__, "prompt": normalized})
                validate_example(ex)
            except JuniperConfigError:
                continue
            ok, _detail = ground_truth_ok(ex)
            if not ok:
                continue
            rendered = render_training_text(ex)
            token_count = len(tokenizer.encode(rendered))
            if token_count > config.token_budget.max_example_tokens:
                continue
            ex = ex.with_token_count(token_count)
            seen_ids.add(ex.example_id)
            cases.append(ex.to_dict())
            produced += 1

    payload = {
        "suite_id": spec["suite_id"],
        "suite_version": spec["suite_version"],
        "description": spec["description"],
        "record_schema": "juniper_math.dataset.schema.Example",
        "cases": sorted(cases, key=lambda c: c["example_id"]),
    }
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    digest = sha256_bytes(text.encode("utf-8"))
    return SuiteBuildResult(
        name=name,
        payload=payload,
        text=text,
        sha256=digest,
        example_count=len(cases),
        prompts=[c["prompt"] for c in cases],
    )
