"""Phase 1 architecture tests: construction, parameter count, shapes, causality,
weight tying, loss semantics, forward/backward, gradients, determinism.
"""

from __future__ import annotations

import pytest
import torch

from juniper_math.architecture import load_architecture_config
from juniper_math.model import (
    JuniperModelError,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    TransformerBlock,
    apply_rotary_pos_emb,
    build_model,
    count_trainable_parameters,
)
from juniper_math.seed import set_global_seed

CONFIG = load_architecture_config()


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------


def test_exact_parameter_count():
    model = build_model(CONFIG)
    assert count_trainable_parameters(model) == 5_004_032


def test_parameter_count_detects_mismatch(tmp_path):
    """Deliberately alter a dimension and prove the counter catches the mismatch —
    the configured target must not be allowed to validate itself."""
    import dataclasses

    altered = dataclasses.replace(CONFIG, d_ff=700)  # not the frozen 688
    model = build_model(altered)
    assert count_trainable_parameters(model) != CONFIG.parameter_target


def test_no_duplicate_lm_head_parameter():
    model = build_model(CONFIG)
    names = [n for n, _ in model.named_parameters()]
    assert "embed_tokens.weight" in names
    assert "lm_head_weight" not in names  # property alias, not a separately registered nn.Parameter


# ---------------------------------------------------------------------------
# RMSNorm — independent reference
# ---------------------------------------------------------------------------


def _reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def test_rmsnorm_matches_independent_reference():
    torch.manual_seed(0)
    dim = 256
    norm = RMSNorm(dim)
    nn_weight = torch.nn.Parameter(torch.randn(dim))
    norm.weight = nn_weight
    x = torch.randn(4, 8, dim)
    got = norm(x)
    expected = _reference_rmsnorm(x, nn_weight, norm.eps)
    assert torch.allclose(got, expected, atol=1e-5)


def test_rmsnorm_is_not_layernorm_no_mean_subtraction():
    """A constant offset should not be removed by RMSNorm (unlike LayerNorm)."""
    dim = 16
    norm = RMSNorm(dim)
    with torch.no_grad():
        norm.weight.fill_(1.0)
    x = torch.ones(1, 1, dim) * 5.0  # constant vector, mean=5, var=0
    out = norm(x)
    # RMSNorm scales by rsqrt(mean(x^2)); output should not be all-zero (mean subtraction
    # would drive a constant vector to exactly zero).
    assert not torch.allclose(out, torch.zeros_like(out))


@pytest.mark.parametrize("shape", [(1, 1, 256), (2, 8, 256), (4, 32, 256)])
def test_rmsnorm_shape_preservation(shape):
    norm = RMSNorm(shape[-1])
    x = torch.randn(*shape)
    assert norm(x).shape == x.shape


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def test_rope_position_dependence():
    """Same content at different positions must produce different rotated Q/K."""
    rotary = RotaryEmbedding(head_dim=64, theta=10000, max_positions=1024)
    cos, sin = rotary(seq_len=4, device=torch.device("cpu"), dtype=torch.float32)
    q = torch.randn(1, 1, 4, 64)
    q_repeated = q[:, :, 0:1, :].expand(-1, -1, 4, -1).clone()  # identical content at every position
    k = torch.zeros_like(q_repeated)
    rotated_q, _ = apply_rotary_pos_emb(q_repeated, k, cos, sin)
    # position 0 vs position 1 must differ despite identical input content
    assert not torch.allclose(rotated_q[:, :, 0, :], rotated_q[:, :, 1, :])


def test_rope_no_trainable_parameters():
    rotary = RotaryEmbedding(head_dim=64, theta=10000, max_positions=1024)
    assert list(rotary.parameters()) == []


def test_rope_rejects_odd_head_dim():
    with pytest.raises(JuniperModelError, match="even head_dim"):
        RotaryEmbedding(head_dim=63, theta=10000, max_positions=128)


def test_rope_covers_full_context():
    rotary = RotaryEmbedding(head_dim=64, theta=10000, max_positions=1024)
    cos, sin = rotary(seq_len=1024, device=torch.device("cpu"), dtype=torch.float32)
    assert cos.shape == (1024, 64)
    assert torch.isfinite(cos).all() and torch.isfinite(sin).all()


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------


def test_swiglu_shape_and_param_count():
    ffn = SwiGLU(d_model=256, d_ff=688)
    x = torch.randn(2, 8, 256)
    out = ffn(x)
    assert out.shape == x.shape
    n_params = sum(p.numel() for p in ffn.parameters())
    assert n_params == 3 * 256 * 688  # gate + up + down, no bias


def test_swiglu_no_bias_parameters():
    ffn = SwiGLU(d_model=256, d_ff=688)
    for module in ffn.modules():
        if isinstance(module, torch.nn.Linear):
            assert module.bias is None


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


def test_block_has_exactly_two_norms():
    rotary = RotaryEmbedding(CONFIG.head_dim, CONFIG.rope_theta, CONFIG.max_context_length)
    block = TransformerBlock(CONFIG, rotary)
    norms = [m for m in block.modules() if isinstance(m, RMSNorm)]
    assert len(norms) == 2


def test_block_residual_shape_preserved():
    rotary = RotaryEmbedding(CONFIG.head_dim, CONFIG.rope_theta, CONFIG.max_context_length)
    block = TransformerBlock(CONFIG, rotary)
    x = torch.randn(2, 8, CONFIG.d_model)
    out = block(x, attention_mask=None)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Full model shapes / sequence boundaries / token range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("b,t", [(1, 1), (1, 16), (2, 32), (4, 128)])
def test_forward_shapes(b, t):
    model = build_model(CONFIG)
    x = torch.randint(0, CONFIG.vocab_size, (b, t))
    out = model(x)
    assert out.logits.shape == (b, t, CONFIG.vocab_size)
    assert out.loss is None


def test_forward_full_context_length():
    model = build_model(CONFIG)
    x = torch.randint(0, CONFIG.vocab_size, (1, 1024))
    out = model(x)
    assert out.logits.shape == (1, 1024, CONFIG.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_context_length_1025_rejected():
    model = build_model(CONFIG)
    x = torch.randint(0, CONFIG.vocab_size, (1, 1025))
    with pytest.raises(JuniperModelError, match="exceeds max_context_length"):
        model(x)


def test_zero_length_sequence_rejected():
    model = build_model(CONFIG)
    x = torch.randint(0, CONFIG.vocab_size, (1, 0))
    with pytest.raises(JuniperModelError, match="T=0"):
        model(x)


def test_wrong_rank_rejected():
    model = build_model(CONFIG)
    x = torch.randint(0, CONFIG.vocab_size, (1, 4, 4))  # rank 3, should be rank 2
    with pytest.raises(JuniperModelError, match="rank 2"):
        model(x)


def test_float_token_ids_rejected():
    model = build_model(CONFIG)
    x = torch.randint(0, CONFIG.vocab_size, (1, 4)).float()
    with pytest.raises(JuniperModelError, match="integer dtype"):
        model(x)


def test_negative_token_id_rejected():
    model = build_model(CONFIG)
    x = torch.zeros(1, 4, dtype=torch.long)
    x[0, 0] = -1
    with pytest.raises(JuniperModelError, match=r"\[0, 4096\)"):
        model(x)


def test_token_id_at_vocab_size_rejected():
    model = build_model(CONFIG)
    x = torch.zeros(1, 4, dtype=torch.long)
    x[0, 0] = CONFIG.vocab_size  # exactly out of range (valid range is [0, vocab_size))
    with pytest.raises(JuniperModelError, match=r"\[0, 4096\)"):
        model(x)


# ---------------------------------------------------------------------------
# Causal masking — behavioral test, not just triangular-mask inspection
# ---------------------------------------------------------------------------


def test_future_tokens_cannot_affect_past_logits():
    torch.manual_seed(0)
    model = build_model(CONFIG)
    model.eval()
    t = 12
    x = torch.randint(0, CONFIG.vocab_size, (1, t))
    with torch.no_grad():
        out1 = model(x).logits

    x_mutated = x.clone()
    cutoff = 5
    x_mutated[0, cutoff + 1 :] = (x_mutated[0, cutoff + 1 :] + 1) % CONFIG.vocab_size
    with torch.no_grad():
        out2 = model(x_mutated).logits

    # logits at positions <= cutoff must be unchanged; positions after may differ.
    assert torch.allclose(out1[0, : cutoff + 1], out2[0, : cutoff + 1], atol=1e-5)
    assert not torch.allclose(out1[0, cutoff + 1 :], out2[0, cutoff + 1 :], atol=1e-5)


# ---------------------------------------------------------------------------
# Padding mask
# ---------------------------------------------------------------------------


def test_padding_mask_shape_validation():
    model = build_model(CONFIG)
    x = torch.randint(0, CONFIG.vocab_size, (2, 8))
    bad_mask = torch.ones(2, 4, dtype=torch.bool)  # wrong T
    with pytest.raises(JuniperModelError, match="attention_mask shape"):
        model(x, attention_mask=bad_mask)


def test_padding_mask_dtype_and_values_validation():
    model = build_model(CONFIG)
    x = torch.randint(0, CONFIG.vocab_size, (1, 4))
    with pytest.raises(JuniperModelError, match="boolean or integer dtype"):
        model(x, attention_mask=torch.ones(1, 4, dtype=torch.float32))
    with pytest.raises(JuniperModelError, match="values must be 0/1"):
        model(x, attention_mask=torch.tensor([[1, 1, 2, 0]]))


def test_padding_positions_not_attended():
    """Changing a padded (masked-out) token's id must not change any valid position's logits."""
    torch.manual_seed(1)
    model = build_model(CONFIG)
    model.eval()
    t = 10
    x = torch.randint(0, CONFIG.vocab_size, (1, t))
    mask = torch.ones(1, t, dtype=torch.bool)
    mask[0, -3:] = False  # last 3 positions are padding

    with torch.no_grad():
        out1 = model(x, attention_mask=mask).logits

    x_mutated = x.clone()
    x_mutated[0, -3:] = (x_mutated[0, -3:] + 1) % CONFIG.vocab_size
    with torch.no_grad():
        out2 = model(x_mutated, attention_mask=mask).logits

    # Valid (non-padded) positions must be unaffected by changes to padded content.
    assert torch.allclose(out1[0, :-3], out2[0, :-3], atol=1e-5)


def test_all_padding_row_produces_finite_output():
    model = build_model(CONFIG)
    model.eval()
    t = 4
    x = torch.randint(0, CONFIG.vocab_size, (1, t))
    mask = torch.zeros(1, t, dtype=torch.bool)  # everything padded
    with torch.no_grad():
        out = model(x, attention_mask=mask)
    # Position 0 always attends to itself under causal masking even if flagged padding,
    # since causal & key_valid both gate on position 0 seeing only position 0's key.
    assert torch.isfinite(out.logits).all()


# ---------------------------------------------------------------------------
# Loss: shift semantics, masking, all-ignored edge case
# ---------------------------------------------------------------------------


def test_loss_shift_semantics():
    torch.manual_seed(0)
    model = build_model(CONFIG)
    x = torch.randint(0, CONFIG.vocab_size, (2, 8))
    out = model(x, labels=x)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    assert out.loss.item() > 0


def test_loss_none_when_no_labels():
    model = build_model(CONFIG)
    x = torch.randint(0, CONFIG.vocab_size, (1, 4))
    out = model(x)
    assert out.loss is None


def test_ignored_labels_excluded():
    """Masking out all-but-one label position should change the loss (proves masking is live)."""
    torch.manual_seed(0)
    model = build_model(CONFIG)
    model.eval()
    x = torch.randint(0, CONFIG.vocab_size, (1, 8))
    labels_full = x.clone()
    labels_masked = x.clone()
    labels_masked[0, 2:] = -100  # ignore all but position 0,1 (which shift to targets 1)

    with torch.no_grad():
        loss_full = model(x, labels=labels_full).loss
        loss_masked = model(x, labels=labels_masked).loss

    assert torch.isfinite(loss_full)
    assert torch.isfinite(loss_masked)
    assert loss_full.item() != pytest.approx(loss_masked.item())


def test_all_ignored_labels_documented_behavior():
    """All labels == -100: F.cross_entropy(reduction="mean") divides 0/0 and returns NaN
    on an all-ignored batch. The model explicitly special-cases this to a defined 0.0 loss
    (see model.py) rather than letting an unexplained NaN pass silently into training.
    """
    model = build_model(CONFIG)
    x = torch.randint(0, CONFIG.vocab_size, (1, 4))
    labels = torch.full_like(x, -100)
    out = model(x, labels=labels)
    assert torch.isfinite(out.loss)
    assert out.loss.item() == pytest.approx(0.0)


def test_labels_shape_mismatch_rejected():
    model = build_model(CONFIG)
    x = torch.randint(0, CONFIG.vocab_size, (1, 8))
    bad_labels = torch.randint(0, CONFIG.vocab_size, (1, 4))
    with pytest.raises(JuniperModelError, match="labels shape"):
        model(x, labels=bad_labels)


def test_labels_dtype_and_range_rejected():
    model = build_model(CONFIG)
    x = torch.randint(0, CONFIG.vocab_size, (1, 8))
    with pytest.raises(JuniperModelError, match="integer dtype"):
        model(x, labels=x.float())
    labels = x.clone()
    labels[0, 0] = -1
    with pytest.raises(JuniperModelError, match="-100 or token ids"):
        model(x, labels=labels)


# ---------------------------------------------------------------------------
# Weight tying
# ---------------------------------------------------------------------------


def test_weight_tying_identity():
    model = build_model(CONFIG)
    assert model.embed_tokens.weight is model.lm_head_weight


def test_weight_tying_survives_state_dict_roundtrip():
    model = build_model(CONFIG)
    state = model.state_dict()
    model2 = build_model(CONFIG)
    model2.load_state_dict(state)
    assert model2.embed_tokens.weight is model2.lm_head_weight


def test_updating_embedding_changes_lm_projection():
    model = build_model(CONFIG)
    with torch.no_grad():
        model.embed_tokens.weight[0, 0] += 123.0
    assert model.lm_head_weight[0, 0].item() == pytest.approx(model.embed_tokens.weight[0, 0].item())


def test_state_dict_has_no_duplicate_output_weight():
    model = build_model(CONFIG)
    keys = list(model.state_dict().keys())
    assert "lm_head_weight" not in keys  # tied buffer/param is not separately serialized
    assert "embed_tokens.weight" in keys


# ---------------------------------------------------------------------------
# Forward / backward / gradients / parameter update
# ---------------------------------------------------------------------------


def test_backward_produces_finite_gradients_across_all_layers():
    torch.manual_seed(0)
    model = build_model(CONFIG)
    x = torch.randint(0, CONFIG.vocab_size, (2, 16))
    out = model(x, labels=x)
    out.loss.backward()

    checked = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"missing gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"
            checked += 1
    assert checked > 0

    # spot-check representative locations explicitly
    assert model.embed_tokens.weight.grad is not None
    assert model.blocks[0].attn.q_proj.weight.grad is not None
    assert model.blocks[len(model.blocks) // 2].ffn.gate_proj.weight.grad is not None
    assert model.blocks[-1].ffn.down_proj.weight.grad is not None
    assert model.final_norm.weight.grad is not None


def test_no_parameter_is_accidentally_frozen():
    model = build_model(CONFIG)
    for name, p in model.named_parameters():
        assert p.requires_grad, f"{name} unexpectedly has requires_grad=False"


def test_optimizer_step_updates_parameters():
    torch.manual_seed(0)
    model = build_model(CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    x = torch.randint(0, CONFIG.vocab_size, (2, 16))
    out = model(x, labels=x)
    out.loss.backward()
    optimizer.step()

    changed = 0
    for n, p in model.named_parameters():
        if not torch.equal(before[n], p.detach()):
            changed += 1
        assert torch.isfinite(p).all(), f"{n} became non-finite after update"
    assert changed > 0
    assert model.embed_tokens.weight is model.lm_head_weight  # tying survives update


# ---------------------------------------------------------------------------
# Deterministic initialization
# ---------------------------------------------------------------------------


def test_deterministic_initialization_same_seed():
    set_global_seed(42, deterministic_algorithms=False)
    m1 = build_model(CONFIG)
    set_global_seed(42, deterministic_algorithms=False)
    m2 = build_model(CONFIG)
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters(), strict=True):
        assert n1 == n2
        assert torch.equal(p1, p2), f"{n1} differs across identically-seeded construction"


def test_different_seed_yields_different_weights():
    set_global_seed(42, deterministic_algorithms=False)
    m1 = build_model(CONFIG)
    set_global_seed(43, deterministic_algorithms=False)
    m2 = build_model(CONFIG)
    any_diff = any(
        not torch.equal(p1, p2)
        for (_, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters(), strict=True)
    )
    assert any_diff


# ---------------------------------------------------------------------------
# Train/eval mode
# ---------------------------------------------------------------------------


def test_train_eval_mode_toggle():
    model = build_model(CONFIG)
    model.train()
    assert model.training
    model.eval()
    assert not model.training


# ---------------------------------------------------------------------------
# Model save/load — identical logits in eval mode
# ---------------------------------------------------------------------------


def test_model_save_load_identical_logits(tmp_path):
    torch.manual_seed(0)
    model = build_model(CONFIG)
    model.eval()
    x = torch.randint(0, CONFIG.vocab_size, (1, 16))
    with torch.no_grad():
        out1 = model(x).logits

    save_path = tmp_path / "model_state.pt"
    torch.save(model.state_dict(), save_path)

    model2 = build_model(CONFIG)
    model2.load_state_dict(torch.load(save_path, weights_only=True))
    model2.eval()
    with torch.no_grad():
        out2 = model2(x).logits

    assert torch.equal(out1, out2)


def test_state_dict_reload_no_missing_or_unexpected_keys():
    model = build_model(CONFIG)
    state = model.state_dict()
    model2 = build_model(CONFIG)
    result = model2.load_state_dict(state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


# ---------------------------------------------------------------------------
# Device transfer (CPU only here; CUDA-specific tests live in test_model_cuda.py)
# ---------------------------------------------------------------------------


def test_cpu_to_cpu_transfer_noop():
    model = build_model(CONFIG)
    model.to("cpu")
    x = torch.randint(0, CONFIG.vocab_size, (1, 4))
    out = model(x)
    assert out.logits.device.type == "cpu"


# ---------------------------------------------------------------------------
# Numerical sanity
# ---------------------------------------------------------------------------


def test_initial_logits_are_not_pathological():
    torch.manual_seed(0)
    model = build_model(CONFIG)
    model.eval()
    x = torch.randint(0, CONFIG.vocab_size, (2, 32))
    with torch.no_grad():
        logits = model(x).logits
    assert torch.isfinite(logits).all()
    # Initial logits at std=0.02 init should be modest, not order-of-magnitude huge.
    assert logits.abs().max().item() < 50.0
