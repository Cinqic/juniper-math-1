"""Phase 8 end-to-end tool-interaction harness (Sec. 10).

Unlike `tool_format_eval.py` (Phase 5: does the model emit a well-formed
`<tool_call>` at all, single shot, never executed), this module drives the
*complete* loop: generate -> detect `<tool_call>` -> parse via the frozen
protocol -> execute through the real `ToolRuntime` -> append the runtime's
own `<tool_result>` as context -> resume generation -> extract the final
answer.

Trust boundary (Sec. 10/26, restated from `tools/runtime.py`'s own
docstring): the model may request a tool; the host decides whether a
syntactically valid request executes; the model never constructs a trusted
tool result. Concretely: whatever the model generates immediately after its
own `<tool_call>{...}` block (including any `<tool_result>...` text it may
hallucinate) is **discarded**, never inspected as if it were an execution
outcome. Generation always resumes from `prompt + tool_call_text +
<REAL wire_tool_result>`, never from the model's own continuation past the
call. `tests/test_tool_interaction.py` asserts this directly: a model that
free-generates a fabricated `<tool_result>` claiming a different value than
the real runtime produces still gets the *real* result appended, and the
fabricated text never appears in the trace fed back into the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch

from juniper_math.generation import generate
from juniper_math.model import JuniperMathModel
from juniper_math.tokenizer import JuniperTokenizer
from juniper_math.tools.errors import ToolProtocolError
from juniper_math.tools.protocol import ToolResult, parse_tool_call, wire_tool_result
from juniper_math.tools.runtime import ToolRuntime

_TAG_PATTERN = re.compile(r"<(tool_call|tool_result|final|unsupported|error)>")

DEFAULT_MAX_TOOL_CALLS = 2  # loop safeguard (Sec. 17/23 "repeated/looping calls")


def _first_tag(text: str) -> tuple[str, int, int] | None:
    """Returns (tag_name, tag_start, content_start) for the first control tag
    in `text`, or None if there is none."""
    match = _TAG_PATTERN.search(text)
    if match is None:
        return None
    return match.group(1), match.start(), match.end()


def _extract_first_call_block(text: str) -> tuple[str, int] | None:
    """If `text` starts (after any leading whitespace) with a `<tool_call>`
    block, returns (call_json_text, end_index_of_that_block). Only ever
    looks at the *first* tag — everything the model generated after that,
    including a self-authored `<tool_result>`, is the model overrunning its
    turn and is never treated as trusted execution evidence."""
    first = _first_tag(text)
    if first is None or first[0] != "tool_call":
        return None
    _, _, content_start = first
    rest = text[content_start:]
    next_tag = _first_tag(rest)
    end = next_tag[1] if next_tag is not None else len(rest)
    return rest[:end].strip(), content_start + end


@dataclass
class ToolCallAttempt:
    raw_text: str
    parsed: bool
    parse_error: str | None
    tool_name: str | None
    result: dict[str, Any] | None  # the REAL runtime result, never model text


@dataclass
class InteractionTrace:
    prompt: str
    turns_text: list[str] = field(default_factory=list)  # model-generated text per turn (pre-truncation)
    tool_calls: list[ToolCallAttempt] = field(default_factory=list)
    final_text: str | None = None  # text after a <final>/<unsupported>/<error> tag, or None
    terminal_tag: str | None = None
    stopped_reason: str = "unknown"  # "terminal_tag" | "no_tool_call" | "max_tool_calls" | "max_new_tokens"
    fabricated_result_discarded: bool = False

    @property
    def emitted_tool_call(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def any_valid_call(self) -> bool:
        return any(c.parsed for c in self.tool_calls)

    @property
    def any_successful_execution(self) -> bool:
        return any(c.result is not None and c.result.get("status") == "success" for c in self.tool_calls)


def run_tool_interaction(
    model: JuniperMathModel,
    tokenizer: JuniperTokenizer,
    prompt: str,
    runtime: ToolRuntime,
    device: torch.device,
    max_new_tokens_per_turn: int,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> InteractionTrace:
    trace = InteractionTrace(prompt=prompt)
    context = prompt

    for _ in range(max_tool_calls + 1):
        gen = generate(model, tokenizer, context, max_new_tokens_per_turn, device, temperature=0.0)
        # `generate()` decodes the FULL sequence (prompt tokens + newly
        # generated tokens), not just the continuation — see its docstring
        # and `GenerationResult.text`. Slicing at the token level (rather
        # than re-encoding/string-slicing `context`, which is not guaranteed
        # to round-trip byte-for-byte through BPE) isolates exactly the new
        # continuation, so repeated turns never re-append the prior context.
        prefix_len = len(tokenizer.encode(context))
        new_text = tokenizer.decode(gen.token_ids[prefix_len:])
        trace.turns_text.append(new_text)

        terminal = _first_tag(new_text)
        if terminal is not None and terminal[0] in {"final", "unsupported", "error"}:
            tag_name, _, content_start = terminal
            trace.terminal_tag = tag_name
            trace.final_text = new_text[content_start:].strip()
            # Anything after this point (including a further <tool_call>) is
            # post-terminal overrun and is deliberately not inspected.
            trace.stopped_reason = "terminal_tag"
            return trace

        call_block = _extract_first_call_block(new_text)
        if call_block is None:
            trace.stopped_reason = "no_tool_call"
            return trace

        call_text, block_end = call_block
        if len(trace.tool_calls) >= max_tool_calls:
            trace.stopped_reason = "max_tool_calls"
            return trace

        # Detect (for reporting only, never for trust decisions) whether the
        # model also hallucinated its own <tool_result> in this same turn —
        # that text is discarded below regardless of what it claims.
        if "<tool_result>" in new_text[block_end:]:
            trace.fabricated_result_discarded = True

        try:
            call = parse_tool_call(call_text, runtime.limits)
            result: ToolResult = runtime.execute_call(call)
            attempt = ToolCallAttempt(
                raw_text=call_text,
                parsed=True,
                parse_error=None,
                tool_name=call.tool,
                result=result.to_dict(),
            )
            wire_result = wire_tool_result(result)
        except ToolProtocolError as exc:
            attempt = ToolCallAttempt(
                raw_text=call_text,
                parsed=False,
                parse_error=f"{exc.code}: {exc.message}",
                tool_name=None,
                result=None,
            )
            # A malformed call still gets a real (error) runtime response,
            # never silence and never a fabricated success — reuse the same
            # trusted error-construction path `ToolRuntime.execute_text` uses.
            wire_result = wire_tool_result(runtime.execute_text(call_text))

        trace.tool_calls.append(attempt)
        # Resume strictly from context + the model's own <tool_call> block
        # (up through `block_end` of the newly generated continuation) + the
        # REAL wire result — never from the continuation past that point,
        # which may contain a hallucinated <tool_result>.
        context = context + new_text[:block_end] + wire_result

    trace.stopped_reason = "max_tool_calls"
    return trace


__all__ = [
    "DEFAULT_MAX_TOOL_CALLS",
    "InteractionTrace",
    "ToolCallAttempt",
    "run_tool_interaction",
]
