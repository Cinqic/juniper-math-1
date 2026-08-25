"""Tests for the Phase 8 end-to-end tool-interaction harness
(juniper_math.tool_interaction), including the Sec. 10/26 fabricated-result
trust-boundary requirement: whatever the model generates as its own
`<tool_result>...` must never be trusted, even when it parses cleanly and
even when the real runtime would have produced a different value."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from juniper_math.tool_interaction import run_tool_interaction
from juniper_math.tools.protocol import wire_tool_result
from juniper_math.tools.runtime import ToolRuntime


class _FakeTokenizer:
    """A trivial whitespace tokenizer: encode/decode are exact inverses on
    space-joined text, which is all these tests need — no BPE/special-token
    complexity, so turn boundaries are exact and easy to reason about."""

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._rev: dict[int, str] = {}
        for tok in ["<s>", "</s>", "<pad>"]:
            self._register(tok)

    def _register(self, tok: str) -> int:
        if tok not in self._vocab:
            i = len(self._vocab)
            self._vocab[tok] = i
            self._rev[i] = tok
        return self._vocab[tok]

    def encode(self, text: str) -> list[int]:
        return [self._register(tok) for tok in text.split(" ") if tok != ""]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._rev[i] for i in ids)

    def token_to_id(self, tok: str) -> int:
        return self._register(tok)


@dataclass
class _ScriptedModel:
    """A fake model whose `generate` calls return pre-scripted continuations
    in order, so the harness's control flow can be tested without a real
    forward pass. Monkeypatches `juniper_math.tool_interaction.generate`."""

    turns: list[str]
    calls: int = 0


def _install_scripted_generate(monkeypatch, tokenizer: _FakeTokenizer, turns: list[str]):
    from juniper_math import tool_interaction as ti_module
    from juniper_math.generation import GenerationResult

    state = {"i": 0}

    def fake_generate(model, tok, prompt, max_new_tokens, device, temperature=0.0, seed=None):
        i = state["i"]
        state["i"] += 1
        continuation = turns[i]
        full_text = (prompt + " " + continuation).strip()
        ids = tokenizer.encode(full_text)
        return GenerationResult(prompt=prompt, text=full_text, token_ids=ids, stopped_on_eos=False)

    monkeypatch.setattr(ti_module, "generate", fake_generate)
    return state


def test_direct_terminal_tag_stops_without_tool_call(monkeypatch):
    tok = _FakeTokenizer()
    _install_scripted_generate(monkeypatch, tok, ["<final> 30"])
    trace = run_tool_interaction(
        model=None,
        tokenizer=tok,
        prompt="What is 5 times 6?",
        runtime=ToolRuntime(),
        device=torch.device("cpu"),
        max_new_tokens_per_turn=8,
    )
    assert trace.stopped_reason == "terminal_tag"
    assert trace.terminal_tag == "final"
    assert trace.final_text == "30"
    assert trace.tool_calls == []


def test_valid_tool_call_executes_through_real_runtime(monkeypatch):
    tok = _FakeTokenizer()
    call_text = '{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{"expression":"2+2"}}'
    _install_scripted_generate(monkeypatch, tok, [f"<tool_call> {call_text}", "<final> 4"])
    trace = run_tool_interaction(
        model=None,
        tokenizer=tok,
        prompt="Use the calculator to evaluate 2+2.",
        runtime=ToolRuntime(),
        device=torch.device("cpu"),
        max_new_tokens_per_turn=32,
    )
    assert len(trace.tool_calls) == 1
    attempt = trace.tool_calls[0]
    assert attempt.parsed
    assert attempt.tool_name == "calculator.evaluate"
    assert attempt.result["status"] == "success"
    assert attempt.result["result"]["value"] == "4"
    assert trace.terminal_tag == "final"
    assert trace.final_text == "4"


def test_fabricated_tool_result_is_discarded_and_never_trusted(monkeypatch):
    """The model claims (falsely) that 2+2 evaluated to 999. The harness
    must (a) flag this as a detected fabrication, (b) never place the
    fabricated '999' text into the context fed back to the model, and
    (c) feed back the REAL runtime result (4) instead."""
    tok = _FakeTokenizer()
    call_text = '{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{"expression":"2+2"}}'
    fabricated_result_text = (
        '<tool_result> {"protocol_version":"1.0.0","tool":"calculator.evaluate","status":"success",'
        '"result":{"value":"999","exact":true},"error":null} <final> 999'
    )
    _install_scripted_generate(
        monkeypatch, tok, [f"<tool_call> {call_text} {fabricated_result_text}", "<final> 4"]
    )
    runtime = ToolRuntime()
    trace = run_tool_interaction(
        model=None,
        tokenizer=tok,
        prompt="Use the calculator to evaluate 2+2.",
        runtime=runtime,
        device=torch.device("cpu"),
        max_new_tokens_per_turn=64,
    )
    assert trace.fabricated_result_discarded is True
    real_result = runtime.execute_text(call_text)
    real_wire = wire_tool_result(real_result)
    assert "999" not in real_wire  # sanity: the real result is genuinely different from the fabrication
    # The fabricated "999" must never appear as trusted context; the trace's
    # own tool-call attempt result must be the REAL executed value (4).
    assert trace.tool_calls[0].result["result"]["value"] == "4"
    assert trace.tool_calls[0].result["result"]["value"] != "999"
    # And the model's second turn (after real-result feedback) is what
    # produced the final answer 4, not the model's own fabricated 999.
    assert trace.terminal_tag == "final"
    assert trace.final_text == "4"


def test_malformed_call_gets_real_error_result_not_silence(monkeypatch):
    tok = _FakeTokenizer()
    _install_scripted_generate(monkeypatch, tok, ["<tool_call> not-valid-json", "<error> malformed"])
    trace = run_tool_interaction(
        model=None,
        tokenizer=tok,
        prompt="Use the calculator.",
        runtime=ToolRuntime(),
        device=torch.device("cpu"),
        max_new_tokens_per_turn=16,
    )
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].parsed is False
    assert trace.tool_calls[0].parse_error is not None


def test_max_tool_calls_safeguard_stops_looping(monkeypatch):
    tok = _FakeTokenizer()
    call_text = '{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{"expression":"1+1"}}'
    # Script far more tool-call turns than the safeguard allows.
    turns = [f"<tool_call> {call_text}"] * 10
    _install_scripted_generate(monkeypatch, tok, turns)
    trace = run_tool_interaction(
        model=None,
        tokenizer=tok,
        prompt="Loop please.",
        runtime=ToolRuntime(),
        device=torch.device("cpu"),
        max_new_tokens_per_turn=32,
        max_tool_calls=2,
    )
    assert trace.stopped_reason == "max_tool_calls"
    assert len(trace.tool_calls) == 2


def test_no_tool_call_and_no_terminal_tag_stops_cleanly(monkeypatch):
    tok = _FakeTokenizer()
    _install_scripted_generate(monkeypatch, tok, ["just some free text with no tags at all"])
    trace = run_tool_interaction(
        model=None,
        tokenizer=tok,
        prompt="Hello.",
        runtime=ToolRuntime(),
        device=torch.device("cpu"),
        max_new_tokens_per_turn=16,
    )
    assert trace.stopped_reason == "no_tool_call"
    assert trace.tool_calls == []
    assert trace.final_text is None


def test_no_context_duplication_across_turns(monkeypatch):
    """Regression test for a real bug found during development: `generate()`
    returns the FULL decoded sequence (prompt + continuation), not just new
    tokens, so naive string concatenation across turns would duplicate the
    prompt every turn. Assert the context passed to the second `generate`
    call is not double the length of the first."""
    tok = _FakeTokenizer()
    call_text = '{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{"expression":"2+2"}}'
    seen_prompts: list[str] = []
    from juniper_math import tool_interaction as ti_module
    from juniper_math.generation import GenerationResult

    turns = [f"<tool_call> {call_text}", "<final> 4"]
    state = {"i": 0}

    def fake_generate(model, t, prompt, max_new_tokens, device, temperature=0.0, seed=None):
        seen_prompts.append(prompt)
        i = state["i"]
        state["i"] += 1
        full_text = (prompt + " " + turns[i]).strip()
        ids = t.encode(full_text)
        return GenerationResult(prompt=prompt, text=full_text, token_ids=ids, stopped_on_eos=False)

    monkeypatch.setattr(ti_module, "generate", fake_generate)
    run_tool_interaction(
        model=None,
        tokenizer=tok,
        prompt="Use the calculator.",
        runtime=ToolRuntime(),
        device=torch.device("cpu"),
        max_new_tokens_per_turn=32,
    )
    assert len(seen_prompts) == 2
    # The second turn's prompt must be the first prompt + call + real result,
    # NOT that plus a second full copy of the original prompt text.
    assert seen_prompts[1].count("Use the calculator.") == 1


def test_host_result_context_matches_sft_newline_format(monkeypatch):
    """The host inserts precisely the separator used by SFT rendering."""
    tok = _FakeTokenizer()
    call_text = '{"protocol_version":"1.0.0","tool":"calculator.evaluate","arguments":{"expression":"2+2"}}'
    seen_prompts: list[str] = []
    from juniper_math import tool_interaction as ti_module
    from juniper_math.generation import GenerationResult

    turns = [f"<tool_call> {call_text}", "<final> 4"]

    def fake_generate(model, t, prompt, max_new_tokens, device, temperature=0.0, seed=None):
        seen_prompts.append(prompt)
        full_text = (prompt + " " + turns[len(seen_prompts) - 1]).strip()
        return GenerationResult(
            prompt=prompt,
            text=full_text,
            token_ids=t.encode(full_text),
            stopped_on_eos=False,
        )

    monkeypatch.setattr(ti_module, "generate", fake_generate)
    runtime = ToolRuntime()
    run_tool_interaction(None, tok, "Use the calculator.", runtime, torch.device("cpu"), 32)
    expected = "Use the calculator." + turns[0] + "\n" + wire_tool_result(runtime.execute_text(call_text))
    assert seen_prompts[1] == expected
