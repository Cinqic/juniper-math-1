"""Phase 1 training-state checkpointing.

Trust model: checkpoints saved by this module are project-generated training
artifacts, not arbitrary third-party data. Loading uses `torch.load` with
`weights_only=False` because a full training checkpoint carries optimizer
state, RNG state, and Python-native metadata (dicts, ints) that
`weights_only=True` cannot deserialize — that mode only supports plain
tensors/state_dicts. `weights_only=False` uses `pickle`, which can execute
arbitrary code for a maliciously crafted file. Do not load a checkpoint from
an untrusted or unverified source; only load checkpoints this project
produced (or that were sha256-verified against a manifest entry produced by
this project). See docs/CHECKPOINT_POLICY.md.

Atomicity: saves write to a temporary file in the same directory as the
target, flush+close it, then `os.replace` it into place. `os.replace` is
atomic on POSIX and Windows for same-filesystem renames, so a process killed
mid-save can never leave a half-written file at the target path — the target
either has the old checkpoint or the new one, never a corrupt partial one.
"""

from __future__ import annotations

import copy
import os
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from juniper_math.architecture import ArchitectureConfig

CHECKPOINT_SCHEMA_VERSION = "1.0.0"


class CheckpointError(ValueError):
    """Raised for malformed, incompatible, or unsafe checkpoint operations."""


@dataclass
class ArchitectureIdentity:
    """The subset of architecture identity a checkpoint must match to be safely restorable."""

    architecture_version: str
    vocab_size: int
    d_model: int
    n_layers: int
    parameter_target: int

    @classmethod
    def from_config(cls, config: ArchitectureConfig) -> ArchitectureIdentity:
        return cls(
            architecture_version=config.architecture_version,
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            parameter_target=config.parameter_target,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "architecture_version": self.architecture_version,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "parameter_target": self.parameter_target,
        }

    def matches(self, other: dict[str, Any]) -> list[str]:
        """Returns a list of human-readable mismatch descriptions (empty = compatible)."""
        mismatches = []
        for key, expected in self.as_dict().items():
            actual = other.get(key)
            if actual != expected:
                mismatches.append(f"{key}: checkpoint has {actual!r}, current architecture has {expected!r}")
        return mismatches


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda_all"] = None
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python_random"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("numpy") is not None:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:
            pass
    if state.get("torch_cuda_all") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


@dataclass
class TrainingCheckpoint:
    schema_version: str
    architecture_identity: dict[str, Any]
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any] | None
    scheduler_state_dict: dict[str, Any] | None
    scaler_state_dict: dict[str, Any] | None
    rng_state: dict[str, Any]
    step: int
    tokens_seen: int
    training_config: dict[str, Any]
    data_stream_position: dict[str, Any]
    seed: int
    git_commit: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "architecture_identity": self.architecture_identity,
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "scheduler_state_dict": self.scheduler_state_dict,
            "scaler_state_dict": self.scaler_state_dict,
            "rng_state": self.rng_state,
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "training_config": self.training_config,
            "data_stream_position": self.data_stream_position,
            "seed": self.seed,
            "git_commit": self.git_commit,
            "extra": self.extra,
        }


def build_checkpoint(
    *,
    architecture: ArchitectureConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: torch.amp.GradScaler | None,
    step: int,
    tokens_seen: int,
    training_config: dict[str, Any],
    data_stream_position: dict[str, Any],
    seed: int,
    git_commit: str,
    extra: dict[str, Any] | None = None,
) -> TrainingCheckpoint:
    return TrainingCheckpoint(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        architecture_identity=ArchitectureIdentity.from_config(architecture).as_dict(),
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict() if optimizer is not None else None,
        scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
        scaler_state_dict=scaler.state_dict() if scaler is not None else None,
        rng_state=capture_rng_state(),
        step=step,
        tokens_seen=tokens_seen,
        training_config=training_config,
        data_stream_position=data_stream_position,
        seed=seed,
        git_commit=git_commit,
        extra=extra or {},
    )


def save_checkpoint_atomic(checkpoint: TrainingCheckpoint, path: Path) -> None:
    """Write via temp-file-then-rename so an interrupted save never corrupts `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(checkpoint.to_dict(), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def load_checkpoint_raw(path: Path) -> dict[str, Any]:
    """Load raw checkpoint dict. Trusted-source only — see module docstring."""
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found at {path}.")
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - surface any deserialization failure clearly
        raise CheckpointError(f"Failed to deserialize checkpoint at {path}: {exc}") from exc
    if not isinstance(raw, dict) or "schema_version" not in raw:
        raise CheckpointError(
            f"{path} does not look like a Juniper Math 1 checkpoint (missing schema_version)."
        )
    return raw


def verify_checkpoint_compatibility(raw: dict[str, Any], architecture: ArchitectureConfig) -> None:
    schema_version = raw.get("schema_version")
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            f"Checkpoint schema_version {schema_version!r} is incompatible with the "
            f"currently supported schema {CHECKPOINT_SCHEMA_VERSION!r}."
        )
    identity = raw.get("architecture_identity")
    if not isinstance(identity, dict):
        raise CheckpointError("Checkpoint is missing architecture_identity metadata.")
    mismatches = ArchitectureIdentity.from_config(architecture).matches(identity)
    if mismatches:
        raise CheckpointError(
            "Checkpoint architecture identity does not match the current architecture config: "
            + "; ".join(mismatches)
        )


def _validate_checkpoint_payload(raw: dict[str, Any]) -> None:
    """Reject incomplete checkpoint payloads before mutating caller-owned state."""
    required = {
        "model_state_dict": dict,
        "rng_state": dict,
        "step": int,
        "tokens_seen": int,
        "training_config": dict,
        "data_stream_position": dict,
        "seed": int,
    }
    for key, expected_type in required.items():
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, expected_type):
            raise CheckpointError(
                f"Checkpoint has invalid or missing {key!r}; expected {expected_type.__name__}."
            )
    for key in ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict"):
        value = raw.get(key)
        if value is not None and not isinstance(value, dict):
            raise CheckpointError(f"Checkpoint field {key!r} must be a dict or null.")


def load_checkpoint(
    path: Path,
    architecture: ArchitectureConfig,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
    restore_rng: bool = True,
) -> TrainingCheckpoint:
    """Load a compatible checkpoint transactionally into the given objects.

    State is restored only after payload validation. If any individual restore
    operation fails, this function restores the caller-owned objects and RNG
    state to their pre-load snapshots before raising ``CheckpointError``.
    """
    raw = load_checkpoint_raw(path)
    verify_checkpoint_compatibility(raw, architecture)
    _validate_checkpoint_payload(raw)

    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    scheduler_before = copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None
    scaler_before = copy.deepcopy(scaler.state_dict()) if scaler is not None else None
    rng_before = capture_rng_state() if restore_rng else None

    try:
        model.load_state_dict(raw["model_state_dict"], strict=True)
        if optimizer is not None and raw.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(raw["optimizer_state_dict"])
        if scheduler is not None and raw.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(raw["scheduler_state_dict"])
        if scaler is not None and raw.get("scaler_state_dict") is not None:
            scaler.load_state_dict(raw["scaler_state_dict"])
        if restore_rng:
            restore_rng_state(raw["rng_state"])
    except Exception as exc:  # noqa: BLE001 - normalize framework-specific restore errors
        model.load_state_dict(model_before, strict=True)
        if optimizer is not None and optimizer_before is not None:
            optimizer.load_state_dict(optimizer_before)
        if scheduler is not None and scheduler_before is not None:
            scheduler.load_state_dict(scheduler_before)
        if scaler is not None and scaler_before is not None:
            scaler.load_state_dict(scaler_before)
        if rng_before is not None:
            restore_rng_state(rng_before)
        raise CheckpointError(f"Checkpoint restoration failed; prior state was preserved: {exc}") from exc

    return TrainingCheckpoint(
        schema_version=raw["schema_version"],
        architecture_identity=raw["architecture_identity"],
        model_state_dict=raw["model_state_dict"],
        optimizer_state_dict=raw.get("optimizer_state_dict"),
        scheduler_state_dict=raw.get("scheduler_state_dict"),
        scaler_state_dict=raw.get("scaler_state_dict"),
        rng_state=raw.get("rng_state", {}),
        step=raw["step"],
        tokens_seen=raw["tokens_seen"],
        training_config=raw.get("training_config", {}),
        data_stream_position=raw.get("data_stream_position", {}),
        seed=raw["seed"],
        git_commit=raw.get("git_commit", "unknown"),
        extra=raw.get("extra", {}),
    )


def inspect_checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Report checkpoint metadata without touching model/optimizer/RNG state (safe inspection)."""
    raw = load_checkpoint_raw(path)
    size_bytes = Path(path).stat().st_size
    return {
        "path": str(path),
        "size_bytes": size_bytes,
        "schema_version": raw.get("schema_version"),
        "architecture_identity": raw.get("architecture_identity"),
        "step": raw.get("step"),
        "tokens_seen": raw.get("tokens_seen"),
        "seed": raw.get("seed"),
        "git_commit": raw.get("git_commit"),
        "training_config": raw.get("training_config"),
        "has_optimizer_state": raw.get("optimizer_state_dict") is not None,
        "has_scheduler_state": raw.get("scheduler_state_dict") is not None,
        "has_scaler_state": raw.get("scaler_state_dict") is not None,
    }


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "ArchitectureIdentity",
    "CheckpointError",
    "TrainingCheckpoint",
    "build_checkpoint",
    "capture_rng_state",
    "inspect_checkpoint_metadata",
    "load_checkpoint",
    "load_checkpoint_raw",
    "restore_rng_state",
    "save_checkpoint_atomic",
    "verify_checkpoint_compatibility",
]
