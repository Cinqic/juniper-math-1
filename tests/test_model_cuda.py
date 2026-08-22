"""GPU-only Phase 1 tests: CUDA forward/backward, device transfer, mixed precision,
CUDA checkpoint/resume smoke test. Auto-skipped on CPU-only machines (see conftest.py).
Run locally on the RTX 2060 target hardware — GitHub Actions CI has no GPU.
"""

from __future__ import annotations

import pytest
import torch

from juniper_math.architecture import load_architecture_config
from juniper_math.checkpoint import build_checkpoint, load_checkpoint, save_checkpoint_atomic
from juniper_math.model import build_model
from juniper_math.seed import set_global_seed

CONFIG = load_architecture_config()
pytestmark = pytest.mark.gpu


def test_cuda_forward_backward_fp32():
    model = build_model(CONFIG).cuda()
    x = torch.randint(0, CONFIG.vocab_size, (2, 64), device="cuda")
    out = model(x, labels=x)
    assert out.logits.device.type == "cuda"
    assert out.logits.dtype == torch.float32
    assert torch.isfinite(out.logits).all()
    out.loss.backward()
    assert model.embed_tokens.weight.grad is not None
    assert torch.isfinite(model.embed_tokens.weight.grad).all()


def test_cuda_full_context_length():
    model = build_model(CONFIG).cuda()
    x = torch.randint(0, CONFIG.vocab_size, (1, 1024), device="cuda")
    out = model(x, labels=x)
    assert out.logits.shape == (1, 1024, CONFIG.vocab_size)
    assert torch.isfinite(out.logits).all()


def test_device_transfer_cpu_to_cuda_and_back():
    model = build_model(CONFIG)
    model = model.to("cuda")
    x = torch.randint(0, CONFIG.vocab_size, (1, 8), device="cuda")
    out = model(x)
    assert out.logits.device.type == "cuda"

    model = model.to("cpu")
    x_cpu = torch.randint(0, CONFIG.vocab_size, (1, 8))
    out_cpu = model(x_cpu)
    assert out_cpu.logits.device.type == "cpu"


def test_rope_cache_follows_device_transfer():
    """RoPE cos/sin buffers must move with .to(device) — a common bug source for cached state."""
    model = build_model(CONFIG).cuda()
    rotary = model.blocks[0].attn.rotary
    assert rotary.cos_cached.device.type == "cuda"
    x = torch.randint(0, CONFIG.vocab_size, (1, 16), device="cuda")
    out = model(x)  # would crash with a device-mismatch error if the cache were stale
    assert out.logits.device.type == "cuda"


def test_cuda_mixed_precision_fp16_finite_with_grad_scaler():
    model = build_model(CONFIG).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randint(0, CONFIG.vocab_size, (2, 32), device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(x, labels=x)
    assert torch.isfinite(out.loss)

    scaler.scale(out.loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_cuda_peak_vram_comfortably_within_6gb_budget():
    torch.cuda.reset_peak_memory_stats()
    model = build_model(CONFIG).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, CONFIG.vocab_size, (4, 1024), device="cuda")
    out = model(x, labels=x)
    out.loss.backward()
    optimizer.step()
    peak_bytes = torch.cuda.max_memory_allocated()
    peak_mb = peak_bytes / (1024 * 1024)
    # RTX 2060 has 6144MB; this configuration must leave large headroom, not barely avoid OOM.
    assert peak_mb < 2048, f"peak VRAM {peak_mb:.1f} MB is not comfortably within the 6GB budget"


def test_cuda_checkpoint_save_restore_continue_smoke(tmp_path):
    """Operational smoke test, not bitwise equality — CUDA kernels are not guaranteed
    bit-deterministic across saves. Verifies save/restore/continue produces finite,
    sensible state rather than proving exact numerical reproduction (see test_checkpoint.py
    for the strict CPU equivalence test).
    """
    set_global_seed(7, deterministic_algorithms=False)
    model = build_model(CONFIG).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(2):
        x = torch.randint(0, CONFIG.vocab_size, (2, 32), device="cuda")
        out = model(x, labels=x)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    ckpt = build_checkpoint(
        architecture=CONFIG,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        step=2,
        tokens_seen=2 * 2 * 32,
        training_config={"lr": 1e-3},
        data_stream_position={},
        seed=7,
        git_commit="deadbeef",
    )
    path = tmp_path / "cuda_ckpt.pt"
    save_checkpoint_atomic(ckpt, path)

    fresh_model = build_model(CONFIG).cuda()
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    loaded = load_checkpoint(path, CONFIG, model=fresh_model, optimizer=fresh_optimizer)
    assert loaded.step == 2

    x = torch.randint(0, CONFIG.vocab_size, (2, 32), device="cuda")
    out = fresh_model(x, labels=x)
    out.loss.backward()
    optimizer_before = [p.detach().clone() for p in fresh_model.parameters()]
    fresh_optimizer.step()

    assert torch.isfinite(out.loss)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(optimizer_before, fresh_model.parameters(), strict=True)
    )
    for p in fresh_model.parameters():
        assert torch.isfinite(p).all()
