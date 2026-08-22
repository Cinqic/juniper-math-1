from __future__ import annotations

from juniper_math.tokenizer_corpus import (
    CATEGORY_PROPORTIONS,
    category_counts,
    generate_corpus,
    write_corpus,
)


def test_category_proportions_sum_to_one():
    assert abs(sum(CATEGORY_PROPORTIONS.values()) - 1.0) < 1e-9


def test_category_counts_sum_close_to_total():
    counts = category_counts(10_000)
    # Rounding per-category can drift the total by at most one per category.
    assert abs(sum(counts.values()) - 10_000) <= len(CATEGORY_PROPORTIONS)


def test_generation_is_deterministic():
    a = list(generate_corpus(seed=1, total_lines=2_000))
    b = list(generate_corpus(seed=1, total_lines=2_000))
    assert [line.text for line in a] == [line.text for line in b]


def test_different_seeds_diverge():
    a = [line.text for line in generate_corpus(seed=1, total_lines=2_000)]
    b = [line.text for line in generate_corpus(seed=2, total_lines=2_000)]
    assert a != b


def test_all_categories_present():
    lines = list(generate_corpus(seed=1, total_lines=2_000))
    categories = {line.category for line in lines}
    assert categories == set(CATEGORY_PROPORTIONS)


def test_write_corpus_is_deterministic(tmp_path):
    p1 = tmp_path / "a.txt"
    p2 = tmp_path / "b.txt"
    n1 = write_corpus(p1, seed=5, total_lines=1_000)
    n2 = write_corpus(p2, seed=5, total_lines=1_000)
    assert n1 == n2
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")


def test_write_corpus_no_blank_lines(tmp_path):
    p = tmp_path / "c.txt"
    write_corpus(p, seed=5, total_lines=1_000)
    lines = p.read_text(encoding="utf-8").splitlines()
    assert all(line.strip() for line in lines)


def test_tool_syntax_category_present():
    lines = list(generate_corpus(seed=1, total_lines=10_000))
    tool_lines = [line.text for line in lines if line.category == "tool_syntax"]
    joined = " ".join(tool_lines)
    assert "<tool_call>" in joined or "<tool_result>" in joined
