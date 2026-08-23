"""Minimal autoregressive generation for Phase 5 smoke validation.

No KV cache: recomputes the full forward pass each step. That is fine at
smoke scale (short prompts, `max_new_tokens` on the order of tens of
tokens, a 5M-parameter model) and keeps this module simple and easy to
verify. A cached, batched generation path is later-phase optimization
work, not a Phase 5 requirement.

This module exists to (a) demonstrate that training changed model
behavior relative to initialization (Sec. 18) and (b) drive the
tool-format evaluation (Sec. 19) — it makes no claim about generation
*quality*.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from juniper_math.model import JuniperMathModel
from juniper_math.tokenizer import JuniperTokenizer


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    text: str
    token_ids: list[int]
    stopped_on_eos: bool


@torch.no_grad()
def generate(
    model: JuniperMathModel,
    tokenizer: JuniperTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 0.0,
    seed: int | None = None,
) -> GenerationResult:
    """Greedy (temperature=0) or temperature-sampled generation from a text prompt.

    `seed` seeds a local generator for sampling only — it never touches
    global RNG state, so callers doing fixed-seed generation comparisons
    are not perturbed by an unrelated call elsewhere.
    """
    was_training = model.training
    model.eval()
    try:
        bos_id = tokenizer.token_to_id("<s>")
        eos_id = tokenizer.token_to_id("</s>")
        ids = [bos_id, *tokenizer.encode(prompt)]
        local_generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None

        stopped_on_eos = False
        for _ in range(max_new_tokens):
            if len(ids) >= model.max_context_length:
                break
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(input_ids).logits[0, -1, :]
            if temperature <= 0.0:
                next_id = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits / temperature, dim=-1).to("cpu")
                next_id = int(torch.multinomial(probs, num_samples=1, generator=local_generator).item())
            ids.append(next_id)
            if next_id == eos_id:
                stopped_on_eos = True
                break

        generated_ids = ids[1:]  # drop bos from the reported token list
        text = tokenizer.decode(generated_ids)
        return GenerationResult(
            prompt=prompt, text=text, token_ids=generated_ids, stopped_on_eos=stopped_on_eos
        )
    finally:
        model.train(was_training)


__all__ = ["GenerationResult", "generate"]
