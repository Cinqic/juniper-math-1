"""Phase 2 token-efficiency benchmark: per-category, plus an informational
general-purpose baseline comparison (Sec. 54-57).

The evaluation set is frozen and deterministically generated with a seed
distinct from both the tokenizer-training corpus and the validation
property-test set, so efficiency numbers are not measured against text the
tokenizer was trained on.
"""

from __future__ import annotations

from typing import Any

from juniper_math.tokenizer import JuniperTokenizer
from juniper_math.tokenizer_corpus import _GENERATORS, _rng_for

_EVAL_SEED = 771177
_EVAL_LINES_PER_CATEGORY = 150

_BASELINE_IDENTITY = "tiktoken gpt2 (OpenAI, 50257 tokens) — informational only, not vocab-size-matched"


def _frozen_eval_set() -> dict[str, list[str]]:
    """A held-out evaluation set: same generators, a seed the training corpus never uses."""
    out: dict[str, list[str]] = {}
    for category in sorted(_GENERATORS):
        rng = _rng_for(category, _EVAL_SEED)
        generator = _GENERATORS[category]
        out[category] = [line for line in generator(rng, _EVAL_LINES_PER_CATEGORY) if line.strip()]
    return out


def _category_stats(lines: list[str], encode_fn) -> dict[str, float]:
    total_tokens = 0
    total_chars = 0
    for line in lines:
        total_tokens += len(encode_fn(line))
        total_chars += len(line)
    return {
        "tokens_per_char": total_tokens / total_chars if total_chars else 0.0,
        "tokens_per_sample": total_tokens / len(lines) if lines else 0.0,
        "samples": len(lines),
    }


def run_benchmark(tok: JuniperTokenizer) -> dict[str, Any]:
    eval_set = _frozen_eval_set()
    categories = {cat: _category_stats(lines, tok.encode) for cat, lines in eval_set.items()}

    result: dict[str, Any] = {
        "evaluation_set": {
            "seed": _EVAL_SEED,
            "lines_per_category": _EVAL_LINES_PER_CATEGORY,
            "total_lines": sum(len(v) for v in eval_set.values()),
        },
        "categories": categories,
        "baseline": None,
        "baseline_error": None,
    }

    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        baseline_categories = {cat: _category_stats(lines, enc.encode) for cat, lines in eval_set.items()}
        result["baseline"] = {"identity": _BASELINE_IDENTITY, "categories": baseline_categories}
    except Exception as exc:  # noqa: BLE001 - baseline is informational; never block the run
        result["baseline_error"] = f"{type(exc).__name__}: {exc}"

    return result
