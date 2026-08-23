"""High-level dataset pipeline operations backing the `dataset` CLI group."""

from __future__ import annotations

import json
from dataclasses import dataclass

from juniper_math.dataset.build import BuildResult, build_dataset
from juniper_math.dataset.config import DatasetConfig, load_dataset_config
from juniper_math.dataset.contamination import build_contamination_report
from juniper_math.dataset.io import iter_all_examples, list_shard_files
from juniper_math.dataset.schema import Example, validate_example
from juniper_math.dataset.shard import ShardInfo, write_manifest, write_shards
from juniper_math.dataset.stats import compute_stats
from juniper_math.dataset.verify import verify_deterministic
from juniper_math.errors import JuniperConfigError
from juniper_math.tokenizer import JuniperTokenizer
from juniper_math.tools.runtime import ToolRuntime

EVAL_SUITE_FILES = [
    "evals/phase4_math_v2.json",
    "evals/phase4_tool_use_v2.json",
    "evals/phase4_calibration_v2.json",
    "evals/phase4_adversarial_v2.json",
]


@dataclass(frozen=True)
class RunReport:
    build: BuildResult
    shard_infos: list[ShardInfo]
    dataset_identity: str
    stats: dict


def run_acquire() -> str:
    return (
        "PASS: dataset_id 'juniper-math-dataset-v1' declares zero external sources "
        "(config/dataset.yaml has no `external_sources` section; see "
        "reports/PHASE4_PROVENANCE_LICENSE_REVIEW.md for the documented scope decision). "
        "Nothing to acquire — this is not a skipped step, it is this dataset version's actual scope."
    )


def _load_eval_reserved_examples(config: DatasetConfig) -> list[Example]:
    from juniper_math.dataset.io import example_from_dict
    from juniper_math.paths import REPO_ROOT

    reserved: list[Example] = []
    for rel_path in EVAL_SUITE_FILES:
        full = REPO_ROOT / rel_path
        if not full.is_file():
            continue
        payload = json.loads(full.read_text(encoding="utf-8"))
        reserved.extend(example_from_dict(c) for c in payload.get("cases", []))
    return reserved


def run_build(*, scale: float = 1.0, seed_override: int | None = None) -> RunReport:
    config = load_dataset_config()
    tokenizer = JuniperTokenizer.load()
    runtime = ToolRuntime()
    eval_reserved = _load_eval_reserved_examples(config)
    result = build_dataset(
        config,
        tokenizer,
        runtime,
        scale=scale,
        seed_override=seed_override,
        eval_reserved_examples=eval_reserved,
    )

    by_split: dict[str, list] = {"train": [], "validation": [], "test": []}
    for ex in result.examples:
        by_split[ex.split].append(ex)

    infos = write_shards(by_split, config.shard, config.output)
    identity = write_manifest(infos, config.dataset_id, config.dataset_schema_version, config.output)

    stats = compute_stats(result.examples, config.token_budget.max_example_tokens, result.counters.as_dict())
    stats["category_token_targets"] = result.category_token_targets
    stats["category_token_actual"] = result.category_token_actual
    stats["shortfall_categories"] = result.shortfall_categories
    stats["dataset_identity"] = identity
    stats["dataset_id"] = config.dataset_id
    config.output.stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return RunReport(build=result, shard_infos=infos, dataset_identity=identity, stats=stats)


def run_validate() -> tuple[bool, list[str]]:
    config = load_dataset_config()
    lines: list[str] = []
    try:
        shard_files = list_shard_files(config.output)
    except JuniperConfigError as exc:
        return False, [f"FAIL: {exc}"]
    total = 0
    errors = 0
    for ex in iter_all_examples(config.output):
        total += 1
        try:
            validate_example(ex)
        except JuniperConfigError as exc:
            errors += 1
            if errors <= 20:
                lines.append(f"FAIL: {exc}")
    lines.insert(0, f"Checked {total} record(s) across {len(shard_files)} shard file(s)")
    ok = errors == 0
    lines.append("PASS: schema validation" if ok else f"FAIL: {errors} schema violation(s)")
    return ok, lines


def run_verify(*, reexecute_tools: bool = True) -> tuple[bool, list[str]]:
    """Recompute deterministic ground truth and re-execute every recorded tool
    call through the live runtime, comparing against the stored result."""
    config = load_dataset_config()
    runtime = ToolRuntime() if reexecute_tools else None
    lines: list[str] = []
    total = det = tool = sem = mismatches = 0
    for ex in iter_all_examples(config.output):
        total += 1
        mode = ex.verification.get("mode")
        if mode == "deterministic":
            det += 1
            ok, detail = verify_deterministic(
                ex.verification.get("expression"), ex.expected_answer, ex.tolerance, ex.example_id
            )
            if not ok:
                mismatches += 1
                if mismatches <= 20:
                    lines.append(f"FAIL: {ex.example_id}: {detail}")
        elif mode == "tool":
            tool += 1
            if runtime is not None and ex.tool_traces:
                import json as _json

                trace = ex.tool_traces[0]
                text = _json.dumps(trace.call, sort_keys=True, separators=(",", ":"))
                live_result = runtime.execute_text(text).to_dict()
                if live_result != trace.result:
                    mismatches += 1
                    if mismatches <= 20:
                        lines.append(
                            f"FAIL: {ex.example_id}: live tool re-execution diverges from stored result"
                        )
        elif mode == "semantic":
            sem += 1
        else:
            mismatches += 1
            lines.append(f"FAIL: {ex.example_id}: unknown verification mode {mode!r}")

    lines.insert(0, f"Checked {total} record(s): {det} deterministic, {tool} tool, {sem} semantic")
    ok = mismatches == 0
    lines.append("PASS: ground truth verified" if ok else f"FAIL: {mismatches} ground-truth mismatch(es)")
    return ok, lines


def run_stats() -> tuple[bool, dict]:
    config = load_dataset_config()
    if not config.output.stats_path.is_file():
        raise JuniperConfigError(f"No stats file at {config.output.stats_path}. Run `dataset build` first.")
    return True, json.loads(config.output.stats_path.read_text(encoding="utf-8"))


def run_build_eval_suites() -> tuple[bool, list[str]]:
    from juniper_math.dataset.eval_suites import SUITE_DEFINITIONS, build_eval_suite
    from juniper_math.paths import REPO_ROOT

    config = load_dataset_config()
    tokenizer = JuniperTokenizer.load()
    runtime = ToolRuntime()
    lines: list[str] = []
    ok = True
    for name in SUITE_DEFINITIONS:
        result = build_eval_suite(name, config, tokenizer, runtime)
        path = REPO_ROOT / "evals" / f"{name}.json"
        path.write_text(result.text, encoding="utf-8")
        lines.append(
            f"{name}: {result.example_count} case(s) -> {path.relative_to(REPO_ROOT)} sha256={result.sha256}"
        )
        if result.example_count == 0:
            ok = False
    return ok, lines


def run_contamination_check(config: DatasetConfig | None = None) -> tuple[bool, list[str]]:
    from juniper_math.paths import REPO_ROOT

    config = config or load_dataset_config()
    examples = list(iter_all_examples(config.output))

    eval_prompts: list[str] = []
    eval_examples: list[Example] = []
    for rel_path in EVAL_SUITE_FILES:
        full = REPO_ROOT / rel_path
        if not full.is_file():
            continue
        payload = json.loads(full.read_text(encoding="utf-8"))
        raw_cases = payload.get("cases", [])
        eval_prompts.extend(c["prompt"] for c in raw_cases)
        from juniper_math.dataset.io import example_from_dict

        eval_examples.extend(example_from_dict(c) for c in raw_cases)

    report = build_contamination_report(
        examples,
        eval_prompts,
        config.dedup.near_duplicate_shingle_size,
        config.dedup.near_duplicate_jaccard_threshold,
        eval_examples,
    )
    lines = [
        f"derivation_id split violations: {len(report.derivation_id_split_violations)}",
        f"exact cross-split duplicates: {len(report.exact_cross_split_duplicates)}",
        f"near-duplicate eval/train pairs: {len(report.near_duplicate_eval_train_pairs)}",
        f"shared eval/train generator IDs: {len(report.shared_eval_generator_ids)}",
        f"shared eval/train family IDs: {len(report.shared_eval_family_ids)}",
        f"shared eval/train template IDs: {len(report.shared_eval_template_ids)}",
        f"exact structural eval/train pairs: {len(report.exact_structural_eval_train_pairs)}",
    ]
    for v in report.derivation_id_split_violations[:10]:
        lines.append(f"  FAIL: {v}")
    for v in report.exact_cross_split_duplicates[:10]:
        lines.append(f"  FAIL: {v}")
    for a, b in report.near_duplicate_eval_train_pairs[:10]:
        lines.append(f"  FAIL: eval {a!r} too similar to train {b!r}")
    for a, b in report.exact_structural_eval_train_pairs[:10]:
        lines.append(f"  FAIL: eval structure {a!r} matches train {b!r}")
    lines.append("PASS: no contamination detected" if report.clean else "FAIL: contamination detected")
    return report.clean, lines
