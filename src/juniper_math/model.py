"""Juniper Math 1 — Phase 1 frozen architecture implementation.

Implements the decoder-only causal Transformer declared in
config/architecture.yaml as real `nn.Module` code: token embedding, RMSNorm,
RoPE, causal self-attention, SwiGLU, and weight-tied LM output. This module
is the authoritative parameter-count source — `count_trainable_parameters`
counts actual instantiated `nn.Parameter` objects, not a configured target.

Attention mask convention: `attention_mask` (when provided) has shape
[B, T] with True/1 meaning "this position is a real token, attend to it"
and False/0 meaning "this position is padding, do not attend to it" — the
same convention used by Hugging Face `attention_mask`. Padding never affects
causality: position t can never attend to t' > t regardless of the mask.

Tokenizer/dataset independence: this module has no dependency on any
tokenizer or dataset artifact. Tests use synthetic integer token IDs in
[0, vocab_size). Binding real text to the vocabulary is Phase 2 work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from juniper_math.architecture import ArchitectureConfig, load_architecture_config

RMS_NORM_EPS = 1e-6


class JuniperModelError(ValueError):
    """Raised for invalid inputs to the model (shape, range, or sequence-length errors)."""


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization. No mean subtraction, no bias."""

    def __init__(self, dim: int, eps: float = RMS_NORM_EPS) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x.to(dtype)) * self.weight


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin tables for head_dim, theta, max positions."""

    def __init__(self, head_dim: int, theta: float, max_positions: int) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise JuniperModelError(f"RoPE requires an even head_dim, got {head_dim}.")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_positions, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # [max_positions, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_positions, head_dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_cached = cast(torch.Tensor, self.cos_cached)
        sin_cached = cast(torch.Tensor, self.sin_cached)
        cos = cos_cached[:seq_len].to(device=device, dtype=dtype)
        sin = sin_cached[:seq_len].to(device=device, dtype=dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k: [B, H, T, head_dim]; cos, sin: [T, head_dim]."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rotated = (q * cos) + (_rotate_half(q) * sin)
    k_rotated = (k * cos) + (_rotate_half(k) * sin)
    return q_rotated, k_rotated


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention. n_query_heads == n_kv_heads (plain MHA)."""

    def __init__(self, config: ArchitectureConfig, rotary: RotaryEmbedding) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_query_heads
        self.head_dim = config.head_dim
        self.rotary = rotary

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(t, x.device, q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if attention_mask is None:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # attention_mask: [B, T] True=keep. Combine with causal mask, pass additive bias.
            causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))
            key_valid = attention_mask.bool().view(b, 1, 1, t)
            combined = causal.view(1, 1, t, t) & key_valid
            bias = torch.zeros(b, 1, t, t, dtype=q.dtype, device=x.device)
            bias = bias.masked_fill(~combined, float("-inf"))
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=False)

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.o_proj(attn_out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ArchitectureConfig, rotary: RotaryEmbedding) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config, rotary)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class JuniperMathModel(nn.Module):
    """Decoder-only causal Transformer, frozen per config/architecture.yaml."""

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        if config.n_query_heads != config.n_kv_heads:
            raise JuniperModelError(
                "This implementation is plain MHA and requires n_query_heads == n_kv_heads "
                f"(got {config.n_query_heads} vs {config.n_kv_heads}); GQA/MQA are out of scope."
            )
        self.config = config
        self.vocab_size = config.vocab_size
        self.max_context_length = config.max_context_length

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        rotary = RotaryEmbedding(config.head_dim, config.rope_theta, config.max_context_length)
        self.blocks = nn.ModuleList([TransformerBlock(config, rotary) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)

        self.apply(self._init_weights)

    @property
    def lm_head_weight(self) -> torch.Tensor:
        """Weight tying: real storage-level sharing (same Parameter object), not value copying.

        Defined as a class-level property rather than an instance attribute assignment —
        assigning an nn.Parameter directly to an attribute would make nn.Module register it
        as a second entry in _parameters, which would duplicate the tensor in state_dict()
        even though count_trainable_parameters() dedupes by storage identity. A property
        has no such registration, so exactly one parameter (embed_tokens.weight) exists.
        """
        return self.embed_tokens.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _validate_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> None:
        if input_ids.dim() != 2:
            raise JuniperModelError(f"input_ids must be rank 2 [B, T], got rank {input_ids.dim()}.")
        if not torch.is_floating_point(input_ids) and input_ids.dtype not in (
            torch.int32,
            torch.int64,
            torch.int16,
        ):
            raise JuniperModelError(f"input_ids must be an integer dtype, got {input_ids.dtype}.")
        if torch.is_floating_point(input_ids):
            raise JuniperModelError(f"input_ids must be an integer dtype, got {input_ids.dtype}.")

        t = input_ids.shape[1]
        if t == 0:
            raise JuniperModelError("input_ids sequence length T must be >= 1 (got T=0).")
        if t > self.max_context_length:
            raise JuniperModelError(
                f"sequence length {t} exceeds max_context_length={self.max_context_length}."
            )

        if input_ids.numel() > 0:
            min_id = int(input_ids.min().item())
            max_id = int(input_ids.max().item())
            if min_id < 0 or max_id >= self.vocab_size:
                raise JuniperModelError(
                    f"token ids must be in [0, {self.vocab_size}), got range [{min_id}, {max_id}]."
                )

        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise JuniperModelError(
                    f"attention_mask shape {tuple(attention_mask.shape)} must match "
                    f"input_ids shape {tuple(input_ids.shape)}."
                )
            if attention_mask.device != input_ids.device:
                raise JuniperModelError("attention_mask must be on the same device as input_ids.")
            if attention_mask.dtype is not torch.bool and not torch.is_floating_point(attention_mask):
                allowed = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
                if attention_mask.dtype not in allowed:
                    raise JuniperModelError(
                        f"attention_mask must have boolean or integer dtype, got {attention_mask.dtype}."
                    )
            if torch.is_floating_point(attention_mask):
                raise JuniperModelError(
                    f"attention_mask must have boolean or integer dtype, got {attention_mask.dtype}."
                )
            if attention_mask.numel() and not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
                raise JuniperModelError("attention_mask values must be 0/1 (False/True).")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> ModelOutput:
        self._validate_inputs(input_ids, attention_mask)

        x = self.embed_tokens(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.final_norm(x)
        logits = F.linear(x, self.lm_head_weight)

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise JuniperModelError(
                    f"labels shape {tuple(labels.shape)} must match input_ids shape {tuple(input_ids.shape)}."
                )
            if labels.device != input_ids.device:
                raise JuniperModelError("labels must be on the same device as input_ids.")
            if torch.is_floating_point(labels) or labels.dtype not in (torch.int16, torch.int32, torch.int64):
                raise JuniperModelError(f"labels must be an integer dtype, got {labels.dtype}.")
            valid_labels = (labels == -100) | ((labels >= 0) & (labels < self.vocab_size))
            if not bool(valid_labels.all()):
                raise JuniperModelError(f"labels must be -100 or token ids in [0, {self.vocab_size}).")
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            flat_labels = shift_labels.view(-1)
            if bool((flat_labels != -100).any()):
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.vocab_size).float(),
                    flat_labels,
                    ignore_index=-100,
                )
            else:
                # F.cross_entropy(reduction="mean") over an all-ignored batch divides 0/0 and
                # returns NaN, which would silently poison training. Defined behavior instead:
                # an all-ignored batch contributes exactly zero loss (and zero gradient, since
                # the multiply-by-zero keeps the graph connected to shift_logits).
                loss = shift_logits.sum() * 0.0

        return ModelOutput(logits=logits, loss=loss)


def build_model(config: ArchitectureConfig | None = None) -> JuniperMathModel:
    return JuniperMathModel(config or load_architecture_config())


def count_trainable_parameters(model: nn.Module) -> int:
    """Counts actual instantiated nn.Parameter objects, deduplicated by storage identity.

    Tied parameters (e.g. embed_tokens.weight == lm_head_weight) must be counted once,
    since named_parameters() would otherwise list the same underlying tensor under two names.
    """
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        key = id(p)
        if key in seen:
            continue
        seen.add(key)
        total += p.numel()
    return total


def verify_parameter_count(model: nn.Module, expected: int) -> None:
    actual = count_trainable_parameters(model)
    if actual != expected:
        raise JuniperModelError(f"trainable parameter count mismatch: expected {expected:,}, got {actual:,}.")


__all__ = [
    "RMS_NORM_EPS",
    "CausalSelfAttention",
    "JuniperMathModel",
    "JuniperModelError",
    "ModelOutput",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "TransformerBlock",
    "apply_rotary_pos_emb",
    "build_model",
    "count_trainable_parameters",
    "verify_parameter_count",
]
