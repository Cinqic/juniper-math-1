"""Phase 5 smoke-training loop: the actual pipeline mechanics being validated.

Deliberately simple and synchronous — no mixed precision, no multi-GPU, no
gradient checkpointing, no data-loader worker processes. Phase 5 exists to
prove the pipeline works, not to be fast. See docs/TRAINING.md.

Token accounting: `tokens_seen` counts exactly the loss-bearing (shifted,
non-padding) token positions the model's own loss computation uses
(`JuniperMathModel.forward` shifts labels by one and ignores -100), not raw
input length — see Sec. 23.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from juniper_math.architecture import ArchitectureConfig
from juniper_math.checkpoint import (
    CheckpointError,
    build_checkpoint,
    load_checkpoint,
    save_checkpoint_atomic,
)
from juniper_math.model import (
    JuniperMathModel,
    build_model,
    verify_parameter_count,
)
from juniper_math.smoke_data import TokenizedSmokeDataset, collate_smoke_batch, epoch_order
from juniper_math.training_config import TrainingConfig


class TrainingNumericalError(RuntimeError):
    """Raised when loss, gradients, or parameters become non-finite."""


@dataclass
class TrainState:
    model: JuniperMathModel
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    device: torch.device
    step: int = 0
    tokens_seen: int = 0
    epoch: int = 0
    position_in_epoch: int = 0
    loss_history: list[dict[str, float]] = field(default_factory=list)


def build_lr_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float):
    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return _lr_lambda


def init_state(
    architecture: ArchitectureConfig, training_config: TrainingConfig, device: torch.device
) -> TrainState:
    """Construct a fresh model/optimizer/scheduler. Caller must have seeded RNG beforehand."""
    model = build_model(architecture).to(device)
    verify_parameter_count(model, architecture.parameter_target)
    opt_cfg = training_config.optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.learning_rate,
        betas=(opt_cfg.beta1, opt_cfg.beta2),
        eps=opt_cfg.eps,
        weight_decay=opt_cfg.weight_decay,
    )
    lr_lambda = build_lr_lambda(
        training_config.scheduler.warmup_steps,
        training_config.schedule.total_steps,
        training_config.scheduler.min_lr_ratio,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return TrainState(model=model, optimizer=optimizer, scheduler=scheduler, device=device)


def _assert_finite(tensor: torch.Tensor, label: str) -> None:
    if not torch.isfinite(tensor).all():
        raise TrainingNumericalError(f"{label} contains non-finite values.")


def assert_model_finite(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            raise TrainingNumericalError(f"Parameter {name!r} contains non-finite values.")


def loss_bearing_tokens(labels: torch.Tensor) -> int:
    """Matches JuniperMathModel.forward's shift-by-one + ignore_index=-100 semantics exactly."""
    return int((labels[:, 1:] != -100).sum().item())


def make_loader(dataset: TokenizedSmokeDataset, indices: list[int], batch_size: int) -> DataLoader:
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_smoke_batch)


def _next_micro_batches(
    state: TrainState, dataset: TokenizedSmokeDataset, training_config: TrainingConfig, n: int
) -> list[dict[str, torch.Tensor]]:
    """Advance the deterministic epoch cursor by n micro-batches, wrapping epochs as needed."""
    batches: list[dict[str, torch.Tensor]] = []
    for _ in range(n):
        order = epoch_order(len(dataset), training_config.seed, state.epoch, training_config.data.shuffle)
        remaining = len(order) - state.position_in_epoch
        take = min(training_config.data.micro_batch_size, remaining)
        indices = order[state.position_in_epoch : state.position_in_epoch + take]
        state.position_in_epoch += take
        if state.position_in_epoch >= len(order):
            state.epoch += 1
            state.position_in_epoch = 0
        items = [dataset[i] for i in indices]
        batches.append(collate_smoke_batch(items))
    return batches


def train_one_optimizer_step(
    state: TrainState, dataset: TokenizedSmokeDataset, training_config: TrainingConfig
) -> dict[str, float]:
    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)

    micro_batches = _next_micro_batches(
        state, dataset, training_config, training_config.data.gradient_accumulation_steps
    )
    total_loss = 0.0
    total_tokens = 0
    for batch in micro_batches:
        input_ids = batch["input_ids"].to(state.device)
        labels = batch["labels"].to(state.device)
        attention_mask = batch["attention_mask"].to(state.device)
        out = state.model(input_ids, attention_mask=attention_mask, labels=labels)
        _assert_finite(out.loss.detach(), "training loss")
        n_tokens = loss_bearing_tokens(labels)
        # Scale so accumulated gradients equal the mean loss over ALL
        # loss-bearing tokens in this optimizer step, not per-microbatch mean.
        (out.loss * max(1, n_tokens)).backward()
        total_loss += out.loss.item() * max(1, n_tokens)
        total_tokens += n_tokens

    for name, param in state.model.named_parameters():
        if param.grad is not None:
            _assert_finite(param.grad.detach(), f"gradient for {name!r}")

    if total_tokens > 0:
        for param in state.model.parameters():
            if param.grad is not None:
                param.grad.div_(total_tokens)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        state.model.parameters(), training_config.optimizer.grad_clip_norm
    )
    state.optimizer.step()
    state.scheduler.step()
    assert_model_finite(state.model)

    state.step += 1
    state.tokens_seen += total_tokens
    mean_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    metrics: dict[str, float] = {
        "step": float(state.step),
        "loss": mean_loss,
        "grad_norm": grad_norm.item(),
        "lr": float(state.scheduler.get_last_lr()[0]),
        "tokens_seen": float(state.tokens_seen),
    }
    state.loss_history.append(metrics)
    return metrics


@torch.no_grad()
def validate(state: TrainState, dataset: TokenizedSmokeDataset, batch_size: int) -> dict[str, float]:
    state.model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_smoke_batch)
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(state.device)
        labels = batch["labels"].to(state.device)
        attention_mask = batch["attention_mask"].to(state.device)
        out = state.model(input_ids, attention_mask=attention_mask, labels=labels)
        n_tokens = loss_bearing_tokens(labels)
        total_loss += out.loss.item() * max(1, n_tokens)
        total_tokens += n_tokens
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("nan")
    return {"validation_loss": avg_loss, "validation_tokens": total_tokens}


def data_stream_position(state: TrainState) -> dict[str, Any]:
    return {"epoch": state.epoch, "position_in_epoch": state.position_in_epoch}


def save_state(
    state: TrainState,
    architecture: ArchitectureConfig,
    training_config: TrainingConfig,
    path: Path,
    git_commit: str,
    extra: dict[str, Any] | None = None,
) -> None:
    checkpoint = build_checkpoint(
        architecture=architecture,
        model=state.model,
        optimizer=state.optimizer,
        scheduler=state.scheduler,
        scaler=None,
        step=state.step,
        tokens_seen=state.tokens_seen,
        training_config=training_config.raw,
        data_stream_position=data_stream_position(state),
        seed=training_config.seed,
        git_commit=git_commit,
        extra=extra or {},
    )
    save_checkpoint_atomic(checkpoint, path)


def load_state(state: TrainState, architecture: ArchitectureConfig, path: Path) -> None:
    """Restore step/tokens_seen/epoch/position/model/optimizer/scheduler/RNG in place."""
    try:
        loaded = load_checkpoint(
            path,
            architecture,
            model=state.model,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            scaler=None,
            restore_rng=True,
        )
    except CheckpointError:
        raise
    state.step = loaded.step
    state.tokens_seen = loaded.tokens_seen
    state.epoch = loaded.data_stream_position.get("epoch", 0)
    state.position_in_epoch = loaded.data_stream_position.get("position_in_epoch", 0)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def run_training(
    state: TrainState,
    train_dataset: TokenizedSmokeDataset,
    val_dataset: TokenizedSmokeDataset,
    architecture: ArchitectureConfig,
    training_config: TrainingConfig,
    end_step: int,
    checkpoint_dir: Path,
    log_path: Path,
    git_commit: str,
    on_metrics=None,
) -> dict[str, Any]:
    """Train `state` from its current step up to (and including) `end_step`.

    Checkpoints/validation/generation cadences use the *global* step number
    against training_config.schedule intervals, so this works whether the
    caller runs 0->total_steps in one call or resumes partway through.
    """
    sched = training_config.schedule
    while state.step < end_step:
        start = time.perf_counter()
        metrics = train_one_optimizer_step(state, train_dataset, training_config)
        metrics["step_seconds"] = time.perf_counter() - start

        if state.step % sched.logging_interval == 0 or state.step == end_step:
            append_jsonl(log_path, {"event": "train_step", **metrics})
            if on_metrics is not None:
                on_metrics(metrics)

        if sched.validation_interval and state.step % sched.validation_interval == 0:
            val_metrics = validate(state, val_dataset, training_config.data.micro_batch_size)
            append_jsonl(log_path, {"event": "validation", "step": state.step, **val_metrics})

        if sched.checkpoint_interval and state.step % sched.checkpoint_interval == 0:
            path = checkpoint_dir / f"step_{state.step:06d}.pt"
            save_state(state, architecture, training_config, path, git_commit)
            append_jsonl(log_path, {"event": "checkpoint", "step": state.step, "path": str(path)})

    return {"final_step": state.step, "tokens_seen": state.tokens_seen, "loss_history": state.loss_history}


__all__ = [
    "TrainState",
    "TrainingNumericalError",
    "append_jsonl",
    "assert_model_finite",
    "build_lr_lambda",
    "data_stream_position",
    "init_state",
    "load_state",
    "loss_bearing_tokens",
    "run_training",
    "save_state",
    "train_one_optimizer_step",
    "validate",
]
