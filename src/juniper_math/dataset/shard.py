"""Deterministic sharding, serialization, and shard-manifest construction (Sec. 21)."""

from __future__ import annotations

import json
from dataclasses import dataclass

from juniper_math.dataset.config import OutputConfig, ShardConfig
from juniper_math.dataset.schema import Example
from juniper_math.hashing import sha256_file
from juniper_math.tools.protocol import CANONICAL_SEPARATORS

BEHAVIOR_TAG = {
    "refuse_unsupported": "unsupported",
    "flag_undefined": "error",
    "flag_missing_information": "unsupported",
    "request_clarification": "unsupported",
}

# Back-compat alias for the original module-private name.
_BEHAVIOR_TAG = BEHAVIOR_TAG


def expected_completion(ex: Example) -> tuple[str, str | None]:
    """The canonical terminal control tag and value `render_training_text` appends.

    Returns ``("final", str(expected_answer))`` when the example carries a
    concrete expected answer (covers ``answer``, ``flag_incorrect_answer``,
    and any tool-required example whose ground truth resolves to a final
    value), or ``(BEHAVIOR_TAG[behavior], None)`` for the four
    answer-less refusal/clarification behaviors. Exists so evaluation code
    (e.g. a model-scoring pass) can check generated output against exactly
    the same ground-truth rule ``render_training_text`` uses, rather than a
    second, potentially-drifting reimplementation of this mapping.
    """
    if ex.expected_answer is not None:
        return "final", str(ex.expected_answer)
    if ex.expected_behavior in BEHAVIOR_TAG:
        return BEHAVIOR_TAG[ex.expected_behavior], None
    raise ValueError(
        f"example {ex.example_id!r}: no expected_answer and expected_behavior "
        f"{ex.expected_behavior!r} has no known terminal tag."
    )


def render_training_text(ex: Example) -> str:
    """Render one example into the exact text a token-counting pass sees.

    Deterministic and stable: same example, same string, always. Tool traces
    are rendered using the canonical wire format already frozen by Phase 3
    (``juniper_math.tools.protocol.wire_tool_call``/``wire_tool_result``) so
    a token-count pass over this text measures exactly what the model would
    see at inference time, not an ad hoc approximation.
    """
    parts = [ex.prompt]
    for trace in ex.tool_traces:
        call_json = json.dumps(
            trace.call, sort_keys=True, separators=CANONICAL_SEPARATORS, ensure_ascii=False
        )
        result_json = json.dumps(
            trace.result, sort_keys=True, separators=CANONICAL_SEPARATORS, ensure_ascii=False
        )
        parts.append(f"<tool_call>{call_json}")
        parts.append(f"<tool_result>{result_json}")
    if ex.expected_answer is not None:
        parts.append(f"<final>{ex.expected_answer}")
    elif ex.expected_behavior in {
        "refuse_unsupported",
        "flag_undefined",
        "flag_missing_information",
        "request_clarification",
    }:
        parts.append(f"<{BEHAVIOR_TAG[ex.expected_behavior]}>")
    return "\n".join(parts)


def record_dict(ex: Example) -> dict:
    d = ex.to_dict()
    return d


@dataclass(frozen=True)
class ShardInfo:
    split: str
    shard_index: int
    filename: str
    record_count: int
    token_count: int
    byte_size: int
    sha256: str


def write_shards(
    examples_by_split: dict[str, list[Example]],
    shard_config: ShardConfig,
    output: OutputConfig,
) -> list[ShardInfo]:
    output.processed_path.mkdir(parents=True, exist_ok=True)
    infos: list[ShardInfo] = []
    for split, examples in examples_by_split.items():
        # Deterministic ordering: sort by example_id (a stable hash-derived
        # id), never by generation/iteration order, which can vary with
        # concurrency or dict ordering across Python versions.
        ordered = sorted(examples, key=lambda e: e.example_id)
        for shard_index, start in enumerate(range(0, len(ordered), shard_config.records_per_shard)):
            chunk = ordered[start : start + shard_config.records_per_shard]
            filename = shard_config.filename_pattern.format(split=split, shard_index=shard_index)
            path = output.processed_path / filename
            token_count = 0
            with path.open("w", encoding="utf-8") as handle:
                for ex in chunk:
                    token_count += ex.token_count or 0
                    handle.write(json.dumps(record_dict(ex), sort_keys=True, ensure_ascii=False))
                    handle.write("\n")
            infos.append(
                ShardInfo(
                    split=split,
                    shard_index=shard_index,
                    filename=filename,
                    record_count=len(chunk),
                    token_count=token_count,
                    byte_size=path.stat().st_size,
                    sha256=sha256_file(path),
                )
            )
    return infos


def write_manifest(infos: list[ShardInfo], dataset_id: str, schema_version: str, output: OutputConfig) -> str:
    import hashlib

    payload = {
        "dataset_id": dataset_id,
        "dataset_schema_version": schema_version,
        "shards": [
            {
                "split": i.split,
                "shard_index": i.shard_index,
                "filename": i.filename,
                "record_count": i.record_count,
                "token_count": i.token_count,
                "byte_size": i.byte_size,
                "sha256": i.sha256,
            }
            for i in sorted(infos, key=lambda i: (i.split, i.shard_index))
        ],
    }
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    output.manifest_path.write_text(text, encoding="utf-8")

    # Whole-dataset identity: the ordered concatenation of every shard's own
    # sha256 (never a hash of a directory listing, which is filesystem-order
    # dependent) — see Sec. 21 "compute a deterministic whole-dataset
    # identity derived from the ordered shard identities."
    ordered_hashes = "\n".join(i.sha256 for i in sorted(infos, key=lambda i: (i.split, i.shard_index)))
    identity = hashlib.sha256(ordered_hashes.encode("utf-8")).hexdigest()
    output.dataset_identity_path.write_text(f"{identity}  {dataset_id}\n", encoding="utf-8")
    return identity
