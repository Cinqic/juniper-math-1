"""Phase 4 dataset engineering tests."""

from __future__ import annotations

import json

import pytest

from juniper_math.dataset.build import build_dataset, ground_truth_ok
from juniper_math.dataset.clean import normalize_text
from juniper_math.dataset.config import load_dataset_config
from juniper_math.dataset.contamination import (
    build_contamination_report,
    check_derivation_id_isolation,
    check_exact_cross_split_duplicates,
)
from juniper_math.dataset.dedup import ExactDeduplicator, NearDeduplicator, exact_key, jaccard, shingles
from juniper_math.dataset.generators.registry import build_registry
from juniper_math.dataset.idgen import derive_id, derive_seed
from juniper_math.dataset.schema import VALID_CATEGORIES, Example, ToolTrace, validate_example
from juniper_math.dataset.shard import render_training_text
from juniper_math.dataset.split import assign_split
from juniper_math.dataset.verify import evaluate_expression, verify_deterministic
from juniper_math.errors import JuniperConfigError
from juniper_math.tokenizer import JuniperTokenizer
from juniper_math.tools.runtime import ToolRuntime


@pytest.fixture(scope="module")
def dataset_config():
    return load_dataset_config()


@pytest.fixture(scope="module")
def tokenizer():
    return JuniperTokenizer.load()


@pytest.fixture(scope="module")
def runtime():
    return ToolRuntime()


# --------------------------------------------------------------------------
# config parsing
# --------------------------------------------------------------------------


def test_dataset_config_loads_and_mixture_sums_to_one(dataset_config):
    assert abs(sum(dataset_config.category_mixture.values()) - 1.0) < 1e-9
    assert set(dataset_config.category_mixture) == VALID_CATEGORIES


def test_dataset_config_rejects_bad_mixture(tmp_path):
    bad = tmp_path / "dataset.yaml"
    bad.write_text(
        "dataset_id: x\ndataset_schema_version: '1.0.0'\nmaster_seed: 1\n"
        "tokenizer_identity: t\ntool_protocol_identity: p\ntool_protocol_version: '1.0.0'\n"
        "token_budget: {envelope_min: 1, envelope_max: 2, target: 1, max_example_tokens: 10}\n"
        "category_mixture: {arithmetic: 0.5}\n"
        "diversity_caps: {max_template_share_within_family: 0.5, max_family_share_of_corpus: 0.5}\n"
        "split: {train: 0.9, validation: 0.05, test: 0.05, eval_suite_seed_offset: 1}\n"
        "dedup: {exact_key: x, near_duplicate_method: y, near_duplicate_shingle_size: 5, "
        "near_duplicate_jaccard_threshold: 0.9}\n"
        "normalization: {unicode_form: NFC, collapse_repeated_whitespace: true, "
        "strip_surrounding_whitespace: true, preserve_math_unicode: [], reject_control_characters: true}\n"
        "shard: {format: jsonl, records_per_shard: 10, "
        "filename_pattern: 'x.{split}.{shard_index:05d}.jsonl'}\n"
        "output: {processed_dir: x, manifest_file: x, stats_file: x, dataset_identity_file: x}\n",
        encoding="utf-8",
    )
    with pytest.raises(JuniperConfigError, match="sum to 1.0"):
        load_dataset_config(bad)


# --------------------------------------------------------------------------
# deterministic seed/id derivation
# --------------------------------------------------------------------------


def test_derive_seed_is_deterministic():
    assert derive_seed("a", "b", 1) == derive_seed("a", "b", 1)
    assert derive_seed("a", "b", 1) != derive_seed("a", "b", 2)


def test_derive_id_is_deterministic_and_distinct():
    assert derive_id("a", "b") == derive_id("a", "b")
    assert derive_id("a", "b") != derive_id("a", "c")


# --------------------------------------------------------------------------
# closed-allowlist verification
# --------------------------------------------------------------------------


def test_evaluate_expression_basic_ops():
    assert evaluate_expression({"op": "add", "args": [2, 3]}) == 5
    assert evaluate_expression({"op": "mul", "args": [4, 5]}) == 20
    assert evaluate_expression({"op": "percent_of", "args": [50, 200]}) == 100


def test_evaluate_expression_rejects_unknown_op():
    with pytest.raises(JuniperConfigError, match="unknown operation"):
        evaluate_expression({"op": "eval", "args": [1]})


def test_evaluate_expression_rejects_division_by_zero():
    with pytest.raises(JuniperConfigError, match="division by zero"):
        evaluate_expression({"op": "div", "args": [1, 0]})


def test_verify_deterministic_catches_wrong_recorded_answer():
    """Regression guard for exactly the Phase 0 tool-001 defect class
    (Opus 5 F-01): a generator that hardcodes a wrong answer must be
    caught by re-verification, not merely trusted."""
    ok, detail = verify_deterministic({"op": "mul", "args": [84317, 9926]}, "837042742", 0, "ctx")
    assert not ok
    assert "836930542" in detail

    ok, _ = verify_deterministic({"op": "mul", "args": [84317, 9926]}, "836930542", 0, "ctx")
    assert ok


# --------------------------------------------------------------------------
# schema validation
# --------------------------------------------------------------------------


def _minimal_example(**overrides) -> Example:
    base = dict(
        example_id="abc123",
        generator_id="g",
        generator_version="1.0.0",
        family_id="f",
        template_id="t0",
        derivation_id="d0",
        seed=1,
        category="arithmetic",
        difficulty="easy",
        synthetic=True,
        split="train",
        prompt="What is 2 + 2?",
        expected_behavior="answer",
        expected_answer="4",
        tolerance=0,
        tool_required=False,
        tool_name=None,
        tool_traces=(),
        verification={"mode": "deterministic", "expression": {"op": "add", "args": [2, 2]}},
        provenance="test",
        notes="",
    )
    base.update(overrides)
    return Example(**base)


def test_validate_example_accepts_well_formed_case():
    validate_example(_minimal_example())


def test_validate_example_rejects_unknown_category():
    with pytest.raises(JuniperConfigError, match="unknown category"):
        validate_example(_minimal_example(category="not_a_category"))


def test_validate_example_rejects_tool_required_without_trace():
    with pytest.raises(JuniperConfigError, match="no tool_traces recorded"):
        validate_example(_minimal_example(tool_required=True, tool_name="calculator.evaluate"))


def test_validate_example_rejects_semantic_with_nonnull_answer():
    with pytest.raises(JuniperConfigError, match="semantic verification requires"):
        validate_example(
            _minimal_example(verification={"mode": "semantic", "expression": None}, expected_answer="4")
        )


def test_validate_example_accepts_tool_trace():
    trace = ToolTrace(
        call={"protocol_version": "1.0.0", "tool": "calculator.evaluate", "arguments": {"expression": "2+2"}},
        result={
            "protocol_version": "1.0.0",
            "tool": "calculator.evaluate",
            "status": "success",
            "result": {"value": "4", "exact": True},
            "error": None,
        },
    )
    validate_example(
        _minimal_example(
            tool_required=True,
            tool_name="calculator.evaluate",
            tool_traces=(trace,),
            verification={"mode": "tool", "expression": None},
        )
    )


# --------------------------------------------------------------------------
# cleaning / normalization
# --------------------------------------------------------------------------


def test_normalize_text_preserves_math_unicode(dataset_config):
    text = normalize_text("What is 5 × 3 − 1 ≤ 20?", dataset_config.normalization)
    assert "×" in text and "−" in text and "≤" in text


def test_normalize_text_collapses_whitespace(dataset_config):
    text = normalize_text("What   is\t\t2 + 2?  ", dataset_config.normalization)
    assert text == "What is 2 + 2?"


def test_normalize_text_rejects_control_characters(dataset_config):
    with pytest.raises(JuniperConfigError, match="control character"):
        normalize_text("What is 2\x01 + 2?", dataset_config.normalization)


def test_normalize_text_rejects_empty_result(dataset_config):
    with pytest.raises(JuniperConfigError, match="empty"):
        normalize_text("   \t  ", dataset_config.normalization)


# --------------------------------------------------------------------------
# dedup
# --------------------------------------------------------------------------


def test_exact_deduplicator_catches_repeats():
    dedup = ExactDeduplicator()
    key = exact_key("same prompt", "4")
    assert not dedup.is_duplicate(key)
    assert dedup.is_duplicate(key)
    assert dedup.removed == 1


def test_near_deduplicator_catches_template_variants():
    dedup = NearDeduplicator(shingle_size=3, threshold=0.5)
    assert not dedup.is_near_duplicate("fam", "What is the sum of twelve and five today")
    assert dedup.is_near_duplicate("fam", "What is the sum of twelve and five today")


def test_jaccard_of_disjoint_sets_is_zero():
    assert jaccard({"a", "b"}, {"c", "d"}) == 0.0


def test_shingles_short_text_falls_back_to_whole_text():
    assert shingles("a b", 5) == {"a b"}


# --------------------------------------------------------------------------
# split determinism and family isolation
# --------------------------------------------------------------------------


def test_split_assignment_is_deterministic(dataset_config):
    a = assign_split("gen", "fam", "deriv1", 42, dataset_config.split)
    b = assign_split("gen", "fam", "deriv1", 42, dataset_config.split)
    assert a == b
    assert a in {"train", "validation", "test"}


def test_split_assignment_groups_by_derivation_id(dataset_config):
    """The same derivation_id must always land in the same split, regardless
    of any other example-level field."""
    splits = {assign_split("gen", "fam", "shared-deriv", 42, dataset_config.split) for _ in range(20)}
    assert len(splits) == 1


def test_check_derivation_id_isolation_flags_violation():
    ex1 = _minimal_example(example_id="e1", derivation_id="shared", split="train")
    ex2 = _minimal_example(example_id="e2", derivation_id="shared", split="test")
    violations = check_derivation_id_isolation([ex1, ex2])
    assert len(violations) == 1


def test_check_exact_cross_split_duplicates_flags_violation():
    ex1 = _minimal_example(example_id="e1", split="train")
    ex2 = _minimal_example(example_id="e2", split="test")
    violations = check_exact_cross_split_duplicates([ex1, ex2])
    assert len(violations) == 1


def test_build_contamination_report_clean_when_no_overlap():
    ex1 = _minimal_example(example_id="e1", prompt="What is 9 + 9?", split="train")
    report = build_contamination_report(
        [ex1], eval_prompts=["completely unrelated text here"], shingle_size=3, threshold=0.9
    )
    assert report.clean


def test_build_contamination_report_flags_near_duplicate_eval_leak():
    ex1 = _minimal_example(example_id="e1", prompt="What is the sum of twelve and five today", split="train")
    report = build_contamination_report(
        [ex1], eval_prompts=["What is the sum of twelve and five today"], shingle_size=3, threshold=0.5
    )
    assert not report.clean
    assert report.near_duplicate_eval_train_pairs


# --------------------------------------------------------------------------
# generator registry / real generation
# --------------------------------------------------------------------------


def test_registry_covers_every_category():
    registry = build_registry()
    assert set(registry) == VALID_CATEGORIES


def test_generators_are_deterministic_given_same_seed(runtime):
    registry = build_registry()
    for category, families in registry.items():
        generator_id, maker = families[0]
        ex_a = maker(0, 12345, runtime)
        ex_b = maker(0, 12345, runtime)
        assert ex_a.prompt == ex_b.prompt, f"{category} generator is non-deterministic"
        assert ex_a.example_id == ex_b.example_id


def test_generators_produce_valid_ground_truth(runtime):
    """Every registered family's ground truth must be self-consistent."""
    registry = build_registry()
    for category, families in registry.items():
        for generator_id, maker in families:
            for i in range(5):
                ex = maker(i, 999, runtime)
                ok, detail = ground_truth_ok(ex)
                assert ok, f"{category}/{generator_id}: {detail}"
                validate_example(ex)


# --------------------------------------------------------------------------
# tokenizer / rendering
# --------------------------------------------------------------------------


def test_render_training_text_is_deterministic():
    ex = _minimal_example()
    assert render_training_text(ex) == render_training_text(ex)


def test_render_training_text_includes_tool_trace():
    trace = ToolTrace(
        call={"protocol_version": "1.0.0", "tool": "calculator.evaluate", "arguments": {"expression": "2+2"}},
        result={
            "protocol_version": "1.0.0",
            "tool": "calculator.evaluate",
            "status": "success",
            "result": {"value": "4", "exact": True},
            "error": None,
        },
    )
    ex = _minimal_example(
        tool_required=True,
        tool_name="calculator.evaluate",
        tool_traces=(trace,),
        verification={"mode": "tool", "expression": None},
    )
    text = render_training_text(ex)
    assert "<tool_call>" in text and "<tool_result>" in text


def test_tokenizer_encode_is_deterministic(tokenizer):
    text = "What is 2 + 2? <final>4"
    assert tokenizer.encode(text) == tokenizer.encode(text)


# --------------------------------------------------------------------------
# end-to-end tiny build (the "full tiny end-to-end dataset build" gate)
# --------------------------------------------------------------------------


def test_tiny_end_to_end_build_is_reproducible(dataset_config, tokenizer, runtime):
    result_a = build_dataset(dataset_config, tokenizer, runtime, scale=0.00005, seed_override=777)
    result_b = build_dataset(dataset_config, tokenizer, runtime, scale=0.00005, seed_override=777)

    ids_a = sorted(e.example_id for e in result_a.examples)
    ids_b = sorted(e.example_id for e in result_b.examples)
    assert ids_a == ids_b
    assert result_a.counters.as_dict() == result_b.counters.as_dict()


def test_tiny_end_to_end_build_examples_all_validate(dataset_config, tokenizer, runtime):
    result = build_dataset(dataset_config, tokenizer, runtime, scale=0.00005, seed_override=555)
    assert result.examples
    for ex in result.examples:
        validate_example(ex)
        ok, detail = ground_truth_ok(ex)
        assert ok, detail
        assert ex.token_count is not None
        assert ex.token_count <= dataset_config.token_budget.max_example_tokens


def test_tiny_end_to_end_build_respects_family_split_isolation(dataset_config, tokenizer, runtime):
    result = build_dataset(dataset_config, tokenizer, runtime, scale=0.0002, seed_override=333)
    violations = check_derivation_id_isolation(result.examples)
    assert not violations


def test_eval_reserved_examples_are_excluded_from_build(dataset_config, tokenizer, runtime):
    result = build_dataset(dataset_config, tokenizer, runtime, scale=0.0002, seed_override=333)
    reserved = result.examples[:5]
    result2 = build_dataset(
        dataset_config, tokenizer, runtime, scale=0.0002, seed_override=333, eval_reserved_examples=reserved
    )
    produced_prompts = {e.prompt for e in result2.examples}
    for r in reserved:
        assert r.prompt not in produced_prompts


# --------------------------------------------------------------------------
# CLI exit codes / failure honesty
# --------------------------------------------------------------------------


def test_list_shard_files_fails_honestly_without_a_build(dataset_config):
    from juniper_math.dataset.config import OutputConfig
    from juniper_math.dataset.io import list_shard_files

    missing_output = OutputConfig(
        processed_dir="data/processed/__no_such_dataset__",
        manifest_file="data/processed/__no_such_dataset__/shard_manifest.json",
        stats_file="data/processed/__no_such_dataset__/stats.json",
        dataset_identity_file="data/processed/__no_such_dataset__/DATASET_IDENTITY.sha256",
    )
    with pytest.raises(JuniperConfigError, match="Run `dataset build` first"):
        list_shard_files(missing_output)


# --------------------------------------------------------------------------
# frozen Phase 4 eval suites — ongoing re-verification (Sec. 24: "Every
# deterministic evaluation answer must be reverified automatically"). This
# is the automated check that would catch a hand-edited or stale suite file
# on every regular `pytest` run, not just at suite-generation time.
# --------------------------------------------------------------------------

_EVAL_SUITE_NAMES = [
    "phase4_math_v1",
    "phase4_tool_use_v1",
    "phase4_calibration_v1",
    "phase4_adversarial_v1",
]


@pytest.mark.parametrize("suite_name", _EVAL_SUITE_NAMES)
def test_frozen_phase4_eval_suite_reverifies(suite_name, runtime):
    from juniper_math.paths import REPO_ROOT

    path = REPO_ROOT / "evals" / f"{suite_name}.json"
    if not path.is_file():
        pytest.skip(f"{path} not present (run `dataset eval-suites-build` first)")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["cases"], f"{suite_name} has no cases"

    for raw_case in payload["cases"]:
        from juniper_math.dataset.io import example_from_dict

        ex = example_from_dict(raw_case)
        validate_example(ex)
        mode = ex.verification.get("mode")
        if mode == "tool":
            # Re-execute live, not just trust the recorded trace.
            trace = ex.tool_traces[0]
            call_text = json.dumps(trace.call, sort_keys=True, separators=(",", ":"))
            live_result = runtime.execute_text(call_text).to_dict()
            assert live_result == trace.result, f"{ex.example_id}: live tool result diverges"
        else:
            ok, detail = ground_truth_ok(ex)
            assert ok, f"{ex.example_id}: {detail}"
