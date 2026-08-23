"""Full Phase 4 dataset build pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

from juniper_math.dataset.clean import normalize_text
from juniper_math.dataset.config import DatasetConfig
from juniper_math.dataset.dedup import ExactDeduplicator, NearDeduplicator, exact_key
from juniper_math.dataset.generators.registry import build_registry
from juniper_math.dataset.schema import Example, validate_example
from juniper_math.dataset.split import assign_split
from juniper_math.dataset.verify import verify_deterministic
from juniper_math.errors import JuniperConfigError
from juniper_math.tokenizer import JuniperTokenizer
from juniper_math.tools.runtime import ToolRuntime

# Safety backstop: if a category cannot reach its token target within this
# many generation attempts, stop and report a shortfall rather than loop
# forever (e.g. a bug makes every draw fail validation).
_MAX_ATTEMPTS_PER_TARGET_TOKEN = 160


@dataclass
class BuildCounters:
    generated: int = 0
    rejected_ground_truth_mismatch: int = 0
    rejected_schema_invalid: int = 0
    rejected_exceeds_context: int = 0
    rejected_exact_duplicate: int = 0
    rejected_near_duplicate: int = 0
    rejected_diversity_cap: int = 0
    accepted: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "generated": self.generated,
            "rejected_ground_truth_mismatch": self.rejected_ground_truth_mismatch,
            "rejected_schema_invalid": self.rejected_schema_invalid,
            "rejected_exceeds_context": self.rejected_exceeds_context,
            "rejected_exact_duplicate": self.rejected_exact_duplicate,
            "rejected_near_duplicate": self.rejected_near_duplicate,
            "rejected_diversity_cap": self.rejected_diversity_cap,
            "accepted": self.accepted,
        }


@dataclass
class BuildResult:
    examples: list[Example] = field(default_factory=list)
    counters: BuildCounters = field(default_factory=BuildCounters)
    category_token_targets: dict[str, int] = field(default_factory=dict)
    category_token_actual: dict[str, int] = field(default_factory=dict)
    shortfall_categories: list[str] = field(default_factory=list)
    sample_errors: list[str] = field(default_factory=list)


def ground_truth_ok(ex: Example) -> tuple[bool, str]:
    mode = ex.verification.get("mode")
    if mode == "deterministic":
        return verify_deterministic(
            ex.verification.get("expression"), ex.expected_answer, ex.tolerance, f"example {ex.example_id}"
        )
    if mode == "tool":
        if not ex.tool_traces:
            return False, "mode 'tool' but no tool_traces recorded"
        return True, "ground truth executed via real ToolRuntime"
    if mode == "semantic":
        return True, "semantic case — no deterministic answer to recompute"
    return False, f"unknown verification mode {mode!r}"


def build_dataset(
    config: DatasetConfig,
    tokenizer: JuniperTokenizer,
    runtime: ToolRuntime,
    *,
    scale: float = 1.0,
    seed_override: int | None = None,
    eval_reserved_examples: tuple[Example, ...] | list[Example] = (),
) -> BuildResult:
    """Run the full generation -> verify -> clean -> dedup -> split pipeline.

    ``scale`` multiplies the configured token target (0 < scale <= 1 for a
    smaller test/CI build; 1.0 for the full configured envelope).

    ``eval_reserved_examples`` are pre-seeded into the exact/near
    deduplicators before generation starts, so anything exactly or nearly
    matching a frozen evaluation-suite case is naturally excluded from
    train/validation/test (Sec. 13) by the same dedup machinery that already
    keeps the corpus itself clean — not a second, separately-maintained
    exclusion list that could silently drift out of sync.
    """
    from juniper_math.dataset.shard import render_training_text

    if not 0 < scale <= 1:
        raise JuniperConfigError(f"build_dataset: scale must be in (0, 1], got {scale}")

    master_seed = seed_override if seed_override is not None else config.master_seed
    registry = build_registry()
    total_target = int(config.token_budget.target * scale)

    exact_dedup = ExactDeduplicator()
    near_dedup = NearDeduplicator(
        config.dedup.near_duplicate_shingle_size, config.dedup.near_duplicate_jaccard_threshold
    )
    for reserved in eval_reserved_examples:
        exact_dedup.seed(exact_key(reserved.prompt, reserved.expected_answer))
        near_dedup.seed(f"{reserved.generator_id}/{reserved.family_id}", reserved.prompt)
    counters = BuildCounters()
    result = BuildResult(counters=counters)

    family_indices: dict[tuple[str, str], int] = {}
    template_counts: dict[tuple[str, str, str], int] = {}
    family_totals: dict[tuple[str, str], int] = {}

    for category, proportion in sorted(config.category_mixture.items()):
        target_tokens = int(total_target * proportion)
        result.category_token_targets[category] = target_tokens
        families = registry[category]
        accumulated = 0
        attempts = 0
        max_attempts = max(200, (target_tokens // 8) * _MAX_ATTEMPTS_PER_TARGET_TOKEN // 100 + 200)
        round_robin = 0

        while accumulated < target_tokens and attempts < max_attempts:
            attempts += 1
            generator_id, maker = families[round_robin % len(families)]
            round_robin += 1
            fam_key = (generator_id, category)
            idx = family_indices.get(fam_key, 0)
            family_indices[fam_key] = idx + 1

            try:
                ex = maker(idx, master_seed, runtime)
            except Exception as exc:  # noqa: BLE001 - a broken generator must not crash the whole build
                counters.rejected_schema_invalid += 1
                if len(result.sample_errors) < 20:
                    result.sample_errors.append(f"{generator_id}/{category}: {type(exc).__name__}: {exc}")
                continue
            counters.generated += 1

            try:
                normalized_prompt = normalize_text(ex.prompt, config.normalization)
                ex = Example(**{**ex.__dict__, "prompt": normalized_prompt})
                validate_example(ex)
            except JuniperConfigError:
                counters.rejected_schema_invalid += 1
                continue

            ok, _detail = ground_truth_ok(ex)
            if not ok:
                counters.rejected_ground_truth_mismatch += 1
                continue

            fam_total_key = (ex.generator_id, ex.family_id)
            template_key = (ex.generator_id, ex.family_id, ex.template_id)
            fam_total = family_totals.get(fam_total_key, 0)
            tmpl_total = template_counts.get(template_key, 0)
            if (
                fam_total >= 30
                and (tmpl_total + 1) / (fam_total + 1)
                > config.diversity_caps.max_template_share_within_family
            ):
                counters.rejected_diversity_cap += 1
                continue

            key = exact_key(ex.prompt, ex.expected_answer)
            if exact_dedup.is_duplicate(key):
                counters.rejected_exact_duplicate += 1
                continue
            if near_dedup.is_near_duplicate(f"{ex.generator_id}/{ex.family_id}", ex.prompt):
                counters.rejected_near_duplicate += 1
                continue

            rendered = render_training_text(ex)
            token_count = len(tokenizer.encode(rendered))
            if token_count > config.token_budget.max_example_tokens:
                counters.rejected_exceeds_context += 1
                continue
            ex = ex.with_token_count(token_count)

            split = assign_split(ex.generator_id, ex.family_id, ex.derivation_id, master_seed, config.split)
            ex = ex.with_split(split)

            result.examples.append(ex)
            counters.accepted += 1
            accumulated += token_count
            family_totals[fam_total_key] = fam_total + 1
            template_counts[template_key] = tmpl_total + 1

        result.category_token_actual[category] = accumulated
        if accumulated < target_tokens * 0.9:
            result.shortfall_categories.append(category)

    return result
