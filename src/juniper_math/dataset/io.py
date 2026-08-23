"""Streaming shard I/O: read JSONL shards back into Example objects without
holding the whole corpus in memory at once (Sec. 19).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from juniper_math.dataset.config import OutputConfig
from juniper_math.dataset.schema import Example, ToolTrace
from juniper_math.errors import JuniperConfigError


def example_from_dict(d: dict) -> Example:
    traces = tuple(ToolTrace(call=t["call"], result=t["result"]) for t in d.get("tool_traces", []))
    return Example(
        example_id=d["example_id"],
        generator_id=d["generator_id"],
        generator_version=d["generator_version"],
        family_id=d["family_id"],
        template_id=d["template_id"],
        derivation_id=d["derivation_id"],
        seed=d["seed"],
        category=d["category"],
        difficulty=d["difficulty"],
        synthetic=d["synthetic"],
        split=d["split"],
        prompt=d["prompt"],
        expected_behavior=d["expected_behavior"],
        expected_answer=d["expected_answer"],
        tolerance=d["tolerance"],
        tool_required=d["tool_required"],
        tool_name=d["tool_name"],
        tool_traces=traces,
        verification=d["verification"],
        provenance=d["provenance"],
        notes=d["notes"],
        token_count=d.get("token_count"),
    )


def list_shard_files(output: OutputConfig) -> list[Path]:
    if not output.processed_path.is_dir():
        raise JuniperConfigError(
            f"No dataset build found at {output.processed_path}. Run `dataset build` first."
        )
    return sorted(output.processed_path.glob("*.jsonl"))


def iter_shard_examples(path: Path) -> Iterator[Example]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JuniperConfigError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
            yield example_from_dict(d)


def iter_all_examples(output: OutputConfig) -> Iterator[Example]:
    for path in list_shard_files(output):
        yield from iter_shard_examples(path)
