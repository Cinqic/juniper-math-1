"""Phase 8 assistant-focused loss rendering: turns one frozen `Example` into
token ids plus a parallel label sequence where only assistant-authored
content is loss-bearing (Sec. 11).

Reuses `dataset.shard.render_training_text`'s exact tag conventions
(`<tool_call>`, `<tool_result>`, `<final>`, `<unsupported>`, `<error>`) —
this module does not invent a second textual format, it only adds a
per-segment supervision label to the same text Phase 5-7 already tokenize
as one undifferentiated blob (appropriate for base pretraining; wrong for
SFT, where the user's prompt and the runtime's tool result must never be
treated as something the model should learn to *produce*).

Segment-wise tokenization (never re-tokenizing the joined string) is the
load-bearing design choice here: it makes segment boundaries exact at the
token level by construction. `tests/test_sft_rendering.py` verifies this
introduces no drift relative to `pilot_data.tokenize_examples`' joint-string
tokenization on a representative sample.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from juniper_math.dataset.schema import Example
from juniper_math.dataset.shard import BEHAVIOR_TAG
from juniper_math.tokenizer import JuniperTokenizer
from juniper_math.tools.protocol import CANONICAL_SEPARATORS

Role = Literal["context", "supervised"]

_ANSWERLESS_BEHAVIORS = frozenset(
    {"refuse_unsupported", "flag_undefined", "flag_missing_information", "request_clarification"}
)


class SftRenderingError(ValueError):
    """Raised when an example cannot be rendered into a valid masked sequence."""


@dataclass(frozen=True)
class Segment:
    text: str
    role: Role


def render_segments(ex: Example) -> list[Segment]:
    """The same content and tag ordering as `dataset.shard.render_training_text`,
    split into (text, role) segments instead of one joined string.

    Context-only segments: the user prompt, every `<tool_result>{...}` block
    (the runtime's own output — never something the assistant is trained to
    *generate*, per ADR 0004 and the fabricated-result-resistance
    requirement, Sec. 8/26).
    Supervised segments: every `<tool_call>{...}` block (the assistant's
    decision to invoke a tool) and the terminal `<final>`/`<unsupported>`/
    `<error>` tag (the assistant's concluding action).
    """
    segments = [Segment(ex.prompt, "context")]
    for trace in ex.tool_traces:
        call_json = json.dumps(
            trace.call, sort_keys=True, separators=CANONICAL_SEPARATORS, ensure_ascii=False
        )
        result_json = json.dumps(
            trace.result, sort_keys=True, separators=CANONICAL_SEPARATORS, ensure_ascii=False
        )
        segments.append(Segment(f"\n<tool_call>{call_json}", "supervised"))
        segments.append(Segment(f"\n<tool_result>{result_json}", "context"))
    if ex.expected_answer is not None:
        segments.append(Segment(f"\n<final>{ex.expected_answer}", "supervised"))
    elif ex.expected_behavior in _ANSWERLESS_BEHAVIORS:
        segments.append(Segment(f"\n<{BEHAVIOR_TAG[ex.expected_behavior]}>", "supervised"))
    elif ex.expected_behavior == "invoke_tool":
        # Matches dataset.shard.render_training_text exactly: a tool-required
        # example with no separate expected_answer appends no terminal tag —
        # the sequence ends after the last <tool_result> block. Not an error.
        pass
    else:
        raise SftRenderingError(
            f"example {ex.example_id!r}: no expected_answer and expected_behavior "
            f"{ex.expected_behavior!r} has no known terminal tag."
        )
    return segments


def rendered_text(ex: Example) -> str:
    """Concatenation of `render_segments`' text, for parity checks against
    `dataset.shard.render_training_text` (which joins with '\\n'; segments
    here already carry their own leading '\\n' except the first)."""
    return "".join(seg.text for seg in render_segments(ex))


@dataclass(frozen=True)
class MaskedTokenization:
    ids: list[int]  # bos + body + eos
    labels: list[int]  # same length; -100 where context-only


def tokenize_and_mask(
    ex: Example, tokenizer: JuniperTokenizer, max_sequence_length: int
) -> MaskedTokenization:
    """Segment-wise tokenize + mask. Raises SftRenderingError (never
    truncates) if the full BOS+body+EOS sequence exceeds max_sequence_length
    — Sec. 12 forbids silently truncating a tool call, a tool result, or a
    final answer in a way that would corrupt supervision meaning; rejecting
    an oversized example is the safe alternative to clipping mid-structure.
    """
    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")

    ids: list[int] = [bos_id]
    labels: list[int] = [-100]  # BOS is never a supervised target
    for seg in render_segments(ex):
        seg_ids = tokenizer.encode(seg.text)
        ids.extend(seg_ids)
        if seg.role == "supervised":
            labels.extend(seg_ids)
        else:
            labels.extend([-100] * len(seg_ids))
    ids.append(eos_id)
    # EOS is supervised: it is the correct "stop" signal immediately
    # following a supervised segment (every example ends on a supervised
    # segment by construction — see render_segments).
    labels.append(eos_id)

    if len(ids) > max_sequence_length:
        raise SftRenderingError(
            f"example {ex.example_id!r}: rendered length {len(ids)} exceeds "
            f"max_sequence_length={max_sequence_length}; rejected rather than truncated."
        )
    return MaskedTokenization(ids=ids, labels=labels)


__all__ = [
    "MaskedTokenization",
    "Role",
    "Segment",
    "SftRenderingError",
    "render_segments",
    "rendered_text",
    "tokenize_and_mask",
]
