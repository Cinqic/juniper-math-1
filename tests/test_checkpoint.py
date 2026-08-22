"""Checkpoint save/load, compatibility, atomicity, and resume-equivalence tests."""

from __future__ import annotations

import dataclasses
import random

import pytest
import torch

from juniper_math.architecture import load_architecture_config
from juniper_math.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    build_checkpoint,
    inspect_checkpoint_metadata,
    load_checkpoint,
    load_checkpoint_raw,
    save_checkpoint_atomic,
)
from juniper_math.model import build_model
from juniper_math.seed import set_global_seed

CONFIG = load_architecture_config()


def _make_checkpoint(model, optimizer, step=10, tokens_seen=1000, seed=42):
    return build_checkpoint(
        architecture=CONFIG,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        step=step,
        tokens_seen=tokens_seen,
        training_config={"lr": 1e-3, "purpose": "architecture_validation_only"},
        data_stream_position={"synthetic_offset": 7},
        seed=seed,
        git_commit="deadbeef",
    )


def test_save_load_roundtrip_all_state(tmp_path):
    set_global_seed(1, deterministic_algorithms=False)
    model = build_model(CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(0, CONFIG.vocab_size, (2, 8))
    out = model(x, labels=x)
    out.loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    ckpt = _make_checkpoint(model, optimizer)
    path = tmp_path / "ckpt.pt"
    save_checkpoint_atomic(ckpt, path)
    assert path.is_file()

    fresh_model = build_model(CONFIG)
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    loaded = load_checkpoint(path, CONFIG, model=fresh_model, optimizer=fresh_optimizer)

    assert loaded.step == 10
    assert loaded.tokens_seen == 1000
    assert loaded.seed == 42
    assert loaded.data_stream_position == {"synthetic_offset": 7}

    for (n1, p1), (n2, p2) in zip(model.named_parameters(), fresh_model.named_parameters(), strict=True):
        assert n1 == n2
        assert torch.equal(p1, p2)

    assert optimizer.state_dict()["param_groups"] == fresh_optimizer.state_dict()["param_groups"]


def test_checkpoint_incompatible_architecture_rejected(tmp_path):
    model = build_model(CONFIG)
    ckpt = _make_checkpoint(model, None)
    path = tmp_path / "ckpt.pt"
    save_checkpoint_atomic(ckpt, path)

    bad_config = dataclasses.replace(CONFIG, d_model=128)  # incompatible
    bad_model = build_model(bad_config)
    with pytest.raises(CheckpointError, match="does not match"):
        load_checkpoint(path, bad_config, model=bad_model)


def test_checkpoint_missing_schema_version_rejected(tmp_path):
    path = tmp_path / "not_a_checkpoint.pt"
    torch.save({"some_other_key": 1}, path)
    with pytest.raises(CheckpointError, match="does not look like"):
        load_checkpoint_raw(path)


def test_checkpoint_wrong_schema_version_rejected(tmp_path):
    model = build_model(CONFIG)
    ckpt = _make_checkpoint(model, None)
    raw = ckpt.to_dict()
    raw["schema_version"] = "999.0.0"
    path = tmp_path / "ckpt.pt"
    torch.save(raw, path)
    from juniper_math.checkpoint import verify_checkpoint_compatibility

    with pytest.raises(CheckpointError, match="incompatible"):
        verify_checkpoint_compatibility(load_checkpoint_raw(path), CONFIG)


def test_checkpoint_nonexistent_file_rejected(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint_raw(tmp_path / "does_not_exist.pt")


def test_checkpoint_corrupted_file_rejected(tmp_path):
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"not a valid torch checkpoint at all")
    with pytest.raises(CheckpointError, match="Failed to deserialize"):
        load_checkpoint_raw(path)


def test_checkpoint_restore_failure_rolls_back_model_state(tmp_path):
    source = build_model(CONFIG)
    checkpoint = _make_checkpoint(source, None)
    raw = checkpoint.to_dict()
    raw["model_state_dict"] = {"embed_tokens.weight": source.embed_tokens.weight.detach().clone()}
    path = tmp_path / "incomplete_model_state.pt"
    torch.save(raw, path)

    target = build_model(CONFIG)
    before = {name: parameter.detach().clone() for name, parameter in target.named_parameters()}
    with pytest.raises(CheckpointError, match="prior state was preserved"):
        load_checkpoint(path, CONFIG, model=target)
    for name, parameter in target.named_parameters():
        assert torch.equal(parameter, before[name]), f"{name} changed after failed restore"


def test_checkpoint_missing_required_payload_rejected_before_restore(tmp_path):
    model = build_model(CONFIG)
    raw = _make_checkpoint(model, None).to_dict()
    del raw["rng_state"]
    path = tmp_path / "missing_rng.pt"
    torch.save(raw, path)
    with pytest.raises(CheckpointError, match="rng_state"):
        load_checkpoint(path, CONFIG, model=build_model(CONFIG))


def test_atomic_save_does_not_clobber_good_checkpoint_on_failure(tmp_path, monkeypatch):
    model = build_model(CONFIG)
    ckpt = _make_checkpoint(model, None)
    path = tmp_path / "ckpt.pt"
    save_checkpoint_atomic(ckpt, path)
    good_bytes = path.read_bytes()

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated interruption during save")

    monkeypatch.setattr(torch, "save", _boom)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        save_checkpoint_atomic(ckpt, path)

    # target file must be untouched — still the last good checkpoint
    assert path.read_bytes() == good_bytes
    # no leftover temp files
    leftovers = list(tmp_path.glob(".ckpt.pt.*.tmp"))
    assert leftovers == []


def test_checkpoint_metadata_inspection_no_state_mutation(tmp_path):
    model = build_model(CONFIG)
    ckpt = _make_checkpoint(model, None, step=5, tokens_seen=500)
    path = tmp_path / "ckpt.pt"
    save_checkpoint_atomic(ckpt, path)

    meta = inspect_checkpoint_metadata(path)
    assert meta["step"] == 5
    assert meta["tokens_seen"] == 500
    assert meta["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert meta["architecture_identity"]["parameter_target"] == 5_004_032
    assert meta["has_optimizer_state"] is False


def test_checkpoint_size_smaller_than_full_training_checkpoint(tmp_path):
    """Model-only state should serialize smaller than a full training checkpoint with optimizer state."""
    model = build_model(CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, CONFIG.vocab_size, (2, 8))
    out = model(x, labels=x)
    out.loss.backward()
    optimizer.step()  # populate AdamW's per-parameter state (exp_avg, exp_avg_sq)

    model_only_path = tmp_path / "model_only.pt"
    torch.save(model.state_dict(), model_only_path)

    full_ckpt = _make_checkpoint(model, optimizer)
    full_path = tmp_path / "full.pt"
    save_checkpoint_atomic(full_ckpt, full_path)

    assert full_path.stat().st_size > model_only_path.stat().st_size


# ---------------------------------------------------------------------------
# Exact resume equivalence (deterministic CPU control experiment)
# ---------------------------------------------------------------------------


def _train_steps(model, optimizer, steps, batches):
    losses = []
    for i in range(steps):
        x = batches[i]
        out = model(x, labels=x)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(out.loss.item())
    return losses


def test_interrupted_resume_matches_uninterrupted_control(tmp_path):
    """Train N steps uninterrupted (control) vs K steps -> checkpoint -> reload -> N-K more
    steps (resume), using identical synthetic data ordering. Final parameters must match
    exactly on CPU with deterministic algorithms enabled.
    """
    seed = 1234
    n_steps = 6
    k_steps = 3
    batch_size, seq_len = 2, 16

    set_global_seed(seed, deterministic_algorithms=True)
    rng = random.Random(seed)
    batches = [
        torch.randint(
            0,
            CONFIG.vocab_size,
            (batch_size, seq_len),
            generator=torch.Generator().manual_seed(rng.randint(0, 2**31)),
        )
        for _ in range(n_steps)
    ]

    # Control: uninterrupted
    set_global_seed(seed, deterministic_algorithms=True)
    control_model = build_model(CONFIG)
    control_optimizer = torch.optim.AdamW(control_model.parameters(), lr=1e-3)
    control_losses = _train_steps(control_model, control_optimizer, n_steps, batches)

    # Resume: K steps, checkpoint, destroy, reload, remaining N-K steps
    set_global_seed(seed, deterministic_algorithms=True)
    resume_model = build_model(CONFIG)
    resume_optimizer = torch.optim.AdamW(resume_model.parameters(), lr=1e-3)
    _train_steps(resume_model, resume_optimizer, k_steps, batches[:k_steps])

    ckpt = build_checkpoint(
        architecture=CONFIG,
        model=resume_model,
        optimizer=resume_optimizer,
        scheduler=None,
        scaler=None,
        step=k_steps,
        tokens_seen=k_steps * batch_size * seq_len,
        training_config={"lr": 1e-3},
        data_stream_position={"batch_index": k_steps},
        seed=seed,
        git_commit="deadbeef",
    )
    path = tmp_path / "resume.pt"
    save_checkpoint_atomic(ckpt, path)

    # destroy and reconstruct fresh objects
    del resume_model, resume_optimizer
    reloaded_model = build_model(CONFIG)
    reloaded_optimizer = torch.optim.AdamW(reloaded_model.parameters(), lr=1e-3)
    loaded = load_checkpoint(path, CONFIG, model=reloaded_model, optimizer=reloaded_optimizer)
    assert loaded.step == k_steps

    _train_steps(reloaded_model, reloaded_optimizer, n_steps - k_steps, batches[k_steps:])

    for (n1, p1), (n2, p2) in zip(
        control_model.named_parameters(), reloaded_model.named_parameters(), strict=True
    ):
        assert n1 == n2
        assert torch.equal(p1, p2), f"parameter {n1} diverged between control and resumed run"

    assert len(control_losses) == n_steps
