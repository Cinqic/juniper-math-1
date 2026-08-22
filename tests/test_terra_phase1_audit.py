"""Independent adversarial regression tests added during the Phase 1 Terra audit.

These references deliberately do not call the production RoPE or masking
helpers, so a shared helper cannot make both sides agree while being wrong.
"""

from __future__ import annotations

import math

import torch

from juniper_math.architecture import load_architecture_config
from juniper_math.model import CausalSelfAttention, RotaryEmbedding

CONFIG = load_architecture_config()


def _rope_reference(x: torch.Tensor, theta: float) -> torch.Tensor:
    """Independent half-split RoPE reference for [B, H, T, D] tensors."""
    _, _, seq_len, head_dim = x.shape
    positions = torch.arange(seq_len, dtype=torch.float64)
    frequencies = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    angles = positions[:, None] * frequencies[None, :]
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(dtype=x.dtype, device=x.device)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(dtype=x.dtype, device=x.device)
    first, second = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    rotated = torch.cat((-second, first), dim=-1)
    return x * cos[None, None] + rotated * sin[None, None]


def test_terra_rope_reference_position_zero_and_nonzero():
    torch.manual_seed(41)
    rotary = RotaryEmbedding(head_dim=64, theta=10_000, max_positions=1024)
    x = torch.randn(2, 4, 7, 64)
    cos, sin = rotary(7, torch.device("cpu"), torch.float32)
    from juniper_math.model import apply_rotary_pos_emb

    got, _ = apply_rotary_pos_emb(x, x, cos, sin)
    expected = _rope_reference(x, 10_000)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)
    assert torch.equal(got[:, :, 0], x[:, :, 0])


def test_terra_attention_matches_manual_causal_padding_reference():
    torch.manual_seed(42)
    rotary = RotaryEmbedding(CONFIG.head_dim, CONFIG.rope_theta, CONFIG.max_context_length)
    attention = CausalSelfAttention(CONFIG, rotary)
    with torch.no_grad():
        for projection in (
            attention.q_proj,
            attention.k_proj,
            attention.v_proj,
            attention.o_proj,
        ):
            projection.weight.copy_(torch.eye(CONFIG.d_model))

    x = torch.randn(1, 4, CONFIG.d_model)
    mask = torch.tensor([[1, 1, 0, 1]], dtype=torch.int64)
    got = attention(x, mask)

    q = x.reshape(1, 4, 4, 64).transpose(1, 2)
    k = q.clone()
    v = q.clone()
    q = _rope_reference(q, 10_000)
    k = _rope_reference(k, 10_000)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(64)
    causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    key_valid = mask.bool().reshape(1, 1, 1, 4)
    scores = scores.masked_fill(~(causal.reshape(1, 1, 4, 4) & key_valid), float("-inf"))
    expected = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(1, 4, CONFIG.d_model)

    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)
