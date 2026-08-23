"""Phase 5 generation module tests."""

from __future__ import annotations

import torch

from juniper_math.architecture import load_architecture_config
from juniper_math.generation import generate
from juniper_math.model import build_model
from juniper_math.seed import set_global_seed
from juniper_math.tokenizer import JuniperTokenizer


def test_generate_greedy_is_deterministic_and_in_vocab():
    set_global_seed(0, deterministic_algorithms=False)
    config = load_architecture_config()
    model = build_model(config)
    tokenizer = JuniperTokenizer.load()
    device = torch.device("cpu")

    first = generate(model, tokenizer, "2 + 2 =", max_new_tokens=8, device=device, temperature=0.0)
    second = generate(model, tokenizer, "2 + 2 =", max_new_tokens=8, device=device, temperature=0.0)

    assert first.token_ids == second.token_ids
    assert all(0 <= tid < config.vocab_size for tid in first.token_ids)
    assert isinstance(first.text, str)


def test_generate_respects_max_new_tokens_budget():
    config = load_architecture_config()
    model = build_model(config)
    tokenizer = JuniperTokenizer.load()
    prompt = "1 + 1 ="
    prompt_len = len(tokenizer.encode(prompt))

    result = generate(model, tokenizer, prompt, max_new_tokens=5, device=torch.device("cpu"), temperature=0.0)
    assert len(result.token_ids) <= prompt_len + 5


def test_generate_restores_model_training_mode():
    config = load_architecture_config()
    model = build_model(config)
    model.train()
    tokenizer = JuniperTokenizer.load()
    generate(model, tokenizer, "3 + 3 =", max_new_tokens=2, device=torch.device("cpu"))
    assert model.training is True
