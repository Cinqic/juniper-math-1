from __future__ import annotations

import json

import pytest

from juniper_math.errors import JuniperConfigError
from juniper_math.evals import DEFAULT_SUITE_PATH, load_eval_suite


def test_loads_frozen_suite():
    suite = load_eval_suite()
    assert suite.suite_id == "phase0_baseline"
    assert suite.suite_version == "0.1.0"
    assert len(suite.cases) == 22


def test_all_case_ids_unique():
    suite = load_eval_suite()
    ids = [case.id for case in suite.cases]
    assert len(ids) == len(set(ids))


def test_every_category_represented_at_least_once():
    suite = load_eval_suite()
    counts = suite.category_counts()
    assert len(counts) >= 20  # 22 categories in the fixed set, one case each


def test_duplicate_id_rejected(tmp_path):
    raw = json.loads(DEFAULT_SUITE_PATH.read_text(encoding="utf-8"))
    raw["cases"].append(dict(raw["cases"][0]))  # exact duplicate id
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(JuniperConfigError, match="duplicate case id"):
        load_eval_suite(bad)


def test_unknown_category_rejected(tmp_path):
    raw = json.loads(DEFAULT_SUITE_PATH.read_text(encoding="utf-8"))
    raw["cases"][0]["category"] = "not_a_real_category"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(JuniperConfigError, match="unknown category"):
        load_eval_suite(bad)


def test_missing_field_rejected(tmp_path):
    raw = json.loads(DEFAULT_SUITE_PATH.read_text(encoding="utf-8"))
    del raw["cases"][0]["expected_behavior"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(JuniperConfigError, match="missing field"):
        load_eval_suite(bad)


def test_empty_cases_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"suite_version": "0.0.1", "suite_id": "x", "cases": []}), encoding="utf-8")
    with pytest.raises(JuniperConfigError, match="non-empty"):
        load_eval_suite(bad)


def test_missing_file_raises(tmp_path):
    with pytest.raises(JuniperConfigError, match="not found"):
        load_eval_suite(tmp_path / "nope.json")
