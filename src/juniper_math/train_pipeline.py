"""Phase 5 smoke-pretraining orchestration: the functions the CLI's `train`,
`evaluate`, and `infer` commands call into. Mirrors the existing
`juniper_math.dataset.pipeline` house style (thin CLI handlers, real logic
here, dataclass reports).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from juniper_math.architecture import ArchitectureConfig, load_architecture_config
from juniper_math.dataset.config import load_dataset_config
from juniper_math.errors import JuniperConfigError
from juniper_math.generation import GenerationResult, generate
from juniper_math.model import JuniperMathModel, count_trainable_parameters
from juniper_math.paths import REPO_ROOT
from juniper_math.seed import set_global_seed
from juniper_math.smoke_data import TokenizedSmokeDataset, select_and_record_smoke_subset
from juniper_math.tokenizer import JuniperTokenizer
from juniper_math.tool_format_eval import ToolFormatReport, run_tool_format_evaluation
from juniper_math.trainer import (
    TrainState,
    append_jsonl,
    assert_model_finite,
    init_state,
    load_state,
    run_training,
    save_state,
    train_one_optimizer_step,
    validate,
)
from juniper_math.training_config import TrainingConfig, load_training_config


def _git_commit() -> str:
    from juniper_math.cli import describe_git_state  # lazy: avoids import-order cycles

    commit, _tree_state = describe_git_state()
    return commit


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print(
            "WARNING: config/training.yaml requests device=cuda but CUDA is not available; "
            "falling back to cpu. This invalidates the intended GPU hardware validation."
        )
        return torch.device("cpu")
    return torch.device(requested)


def _load_common(config_path: Path | None) -> tuple[TrainingConfig, ArchitectureConfig, JuniperTokenizer]:
    training_config = load_training_config(config_path)
    architecture = load_architecture_config()
    if architecture.architecture_version != training_config.architecture_identity:
        raise JuniperConfigError(
            f"training config architecture_identity {training_config.architecture_identity!r} does not "
            f"match config/architecture.yaml architecture_version {architecture.architecture_version!r}."
        )
    tokenizer = JuniperTokenizer.load()
    return training_config, architecture, tokenizer


def _build_datasets(
    training_config: TrainingConfig, tokenizer: JuniperTokenizer
) -> tuple[TokenizedSmokeDataset, TokenizedSmokeDataset, Any]:
    dataset_config = load_dataset_config()
    if dataset_config.dataset_id != training_config.dataset_identity:
        raise JuniperConfigError(
            f"training config dataset_identity {training_config.dataset_identity!r} does not match "
            f"config/dataset.yaml dataset_id {dataset_config.dataset_id!r}."
        )
    selections, manifest = select_and_record_smoke_subset(training_config, dataset_config)
    train_ds = TokenizedSmokeDataset(
        selections["train"], tokenizer, training_config.smoke_subset.max_sequence_length
    )
    val_ds = TokenizedSmokeDataset(
        selections["validation"], tokenizer, training_config.smoke_subset.max_sequence_length
    )
    return train_ds, val_ds, manifest


def _generate_fixed_prompts(
    state: TrainState, tokenizer: JuniperTokenizer, training_config: TrainingConfig
) -> list[GenerationResult]:
    return [
        generate(
            state.model,
            tokenizer,
            prompt,
            training_config.generation_max_new_tokens,
            state.device,
            temperature=0.0,
        )
        for prompt in training_config.fixed_generation_prompts
    ]


@dataclass
class SmokeRunReport:
    training_config: TrainingConfig
    architecture: ArchitectureConfig
    smoke_manifest: Any
    device: str
    parameter_count: int
    initial_generations: list[GenerationResult]
    final_generations: list[GenerationResult]
    initial_validation: dict[str, float]
    final_validation: dict[str, float]
    initial_train_loss: float
    train_result: dict[str, Any]
    final_checkpoint_path: str
    log_path: str
    elapsed_seconds: float
    peak_cuda_memory_bytes: int | None


def run_smoke_train(
    config_path: Path | None = None,
    max_steps: int | None = None,
    run_evaluate: bool = False,
) -> SmokeRunReport:
    training_config, architecture, tokenizer = _load_common(config_path)
    device = _resolve_device(training_config.device)
    set_global_seed(training_config.seed)

    train_ds, val_ds, manifest = _build_datasets(training_config, tokenizer)

    experiment_dir = training_config.output.experiment_path
    checkpoint_dir = training_config.output.checkpoint_path
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = experiment_dir / "train_log.jsonl"

    state = init_state(architecture, training_config, device)
    parameter_count = count_trainable_parameters(state.model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    append_jsonl(
        log_path,
        {
            "event": "run_start",
            "run_id": training_config.run_id,
            "git_commit": _git_commit(),
            "parameter_count": parameter_count,
            "device": str(device),
            "smoke_manifest": manifest.as_dict(),
        },
    )

    initial_generations = _generate_fixed_prompts(state, tokenizer, training_config)
    initial_validation = validate(state, val_ds, training_config.data.micro_batch_size)
    append_jsonl(
        log_path,
        {
            "event": "initial_validation",
            "step": 0,
            **initial_validation,
            "generations": [{"prompt": g.prompt, "text": g.text} for g in initial_generations],
        },
    )
    initial_step_metrics = train_one_optimizer_step(state, train_ds, training_config)
    initial_train_loss = initial_step_metrics["loss"]
    # That first optimizer step already advanced state by one step; the loop below continues from there.
    end_step = max_steps if max_steps is not None else training_config.schedule.total_steps

    start = time.perf_counter()
    train_result = run_training(
        state,
        train_ds,
        val_ds,
        architecture,
        training_config,
        end_step,
        checkpoint_dir,
        log_path,
        _git_commit(),
    )
    elapsed = time.perf_counter() - start

    final_checkpoint_path = checkpoint_dir / f"step_{state.step:06d}_final.pt"
    save_state(
        state, architecture, training_config, final_checkpoint_path, _git_commit(), extra={"final": True}
    )

    final_validation = validate(state, val_ds, training_config.data.micro_batch_size)
    final_generations = _generate_fixed_prompts(state, tokenizer, training_config)
    assert_model_finite(state.model)

    peak_mem = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None

    append_jsonl(
        log_path,
        {
            "event": "run_end",
            "final_step": state.step,
            "tokens_seen": state.tokens_seen,
            "final_validation": final_validation,
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_bytes": peak_mem,
            "generations": [{"prompt": g.prompt, "text": g.text} for g in final_generations],
        },
    )

    if run_evaluate:
        tool_report = evaluate_tool_use_suite(final_checkpoint_path, config_path=config_path, sample_size=20)
        append_jsonl(log_path, {"event": "tool_format_evaluation", **tool_report.as_dict()})

    return SmokeRunReport(
        training_config=training_config,
        architecture=architecture,
        smoke_manifest=manifest,
        device=str(device),
        parameter_count=parameter_count,
        initial_generations=initial_generations,
        final_generations=final_generations,
        initial_validation=initial_validation,
        final_validation=final_validation,
        initial_train_loss=initial_train_loss,
        train_result=train_result,
        final_checkpoint_path=str(final_checkpoint_path),
        log_path=str(log_path),
        elapsed_seconds=elapsed,
        peak_cuda_memory_bytes=peak_mem,
    )


@dataclass
class ResumeComparisonReport:
    run_a_final_step: int
    run_b_final_step: int
    run_a_tokens_seen: int
    run_b_tokens_seen: int
    run_a_final_loss: float
    run_b_final_loss: float
    loss_history_max_abs_diff: float
    max_param_abs_diff: float
    generations_match: bool
    equivalent: bool
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_a_final_step": self.run_a_final_step,
            "run_b_final_step": self.run_b_final_step,
            "run_a_tokens_seen": self.run_a_tokens_seen,
            "run_b_tokens_seen": self.run_b_tokens_seen,
            "run_a_final_loss": self.run_a_final_loss,
            "run_b_final_loss": self.run_b_final_loss,
            "loss_history_max_abs_diff": self.loss_history_max_abs_diff,
            "max_param_abs_diff": self.max_param_abs_diff,
            "generations_match": self.generations_match,
            "equivalent": self.equivalent,
            "notes": self.notes,
        }


def _param_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu() for p in model.parameters()])


def run_resume_test(config_path: Path | None = None) -> ResumeComparisonReport:
    """Sec. 22: uninterrupted Run A vs. interrupted-then-resumed Run B, same init."""
    training_config, architecture, tokenizer = _load_common(config_path)
    device = _resolve_device(training_config.device)
    total_steps = training_config.schedule.total_steps
    interrupt_step = training_config.resume_test.interrupt_step
    if not (0 < interrupt_step < total_steps):
        raise JuniperConfigError("resume_test.interrupt_step must be strictly between 0 and total_steps.")

    set_global_seed(training_config.seed)
    train_ds, val_ds, _manifest = _build_datasets(training_config, tokenizer)

    scratch_dir = training_config.output.checkpoint_path / "resume_test"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    init_ckpt = scratch_dir / "init.pt"
    interrupt_ckpt = scratch_dir / "interrupt.pt"
    log_path = training_config.output.experiment_path / "resume_test_log.jsonl"

    # Build the shared initial state once and freeze it to disk.
    state_a = init_state(architecture, training_config, device)
    save_state(state_a, architecture, training_config, init_ckpt, _git_commit())

    # Run A: uninterrupted, straight through.
    run_training(
        state_a,
        train_ds,
        val_ds,
        architecture,
        training_config,
        total_steps,
        scratch_dir,
        log_path,
        _git_commit(),
    )
    gen_a = _generate_fixed_prompts(state_a, tokenizer, training_config)

    # Run B stage 1: fresh objects, restore init, train to interrupt_step, checkpoint.
    state_b1 = init_state(architecture, training_config, device)
    load_state(state_b1, architecture, init_ckpt)
    run_training(
        state_b1,
        train_ds,
        val_ds,
        architecture,
        training_config,
        interrupt_step,
        scratch_dir,
        log_path,
        _git_commit(),
    )
    save_state(state_b1, architecture, training_config, interrupt_ckpt, _git_commit())
    del state_b1

    # Run B stage 2: brand-new objects simulating a new process, resume, finish.
    state_b2 = init_state(architecture, training_config, device)
    load_state(state_b2, architecture, interrupt_ckpt)
    run_training(
        state_b2,
        train_ds,
        val_ds,
        architecture,
        training_config,
        total_steps,
        scratch_dir,
        log_path,
        _git_commit(),
    )
    gen_b = _generate_fixed_prompts(state_b2, tokenizer, training_config)

    loss_a = {m["step"]: m["loss"] for m in state_a.loss_history}
    loss_b1_and_b2 = {m["step"]: m["loss"] for m in state_b2.loss_history}
    common_steps = sorted(set(loss_a) & set(loss_b1_and_b2))
    max_loss_diff = max((abs(loss_a[s] - loss_b1_and_b2[s]) for s in common_steps), default=float("nan"))

    param_diff = float((_param_vector(state_a.model) - _param_vector(state_b2.model)).abs().max().item())
    generations_match = [g.text for g in gen_a] == [g.text for g in gen_b]

    notes = []
    if device.type == "cuda":
        notes.append(
            "Comparison run on CUDA: torch.use_deterministic_algorithms(warn_only=True) is a "
            "best-effort determinism request, not a bitwise guarantee; small floating-point "
            "differences from kernel nondeterminism are possible and are tolerance-checked, not "
            "required to be exactly zero."
        )
    equivalent = (
        state_a.step == state_b2.step
        and state_a.tokens_seen == state_b2.tokens_seen
        and max_loss_diff < 1e-2
        and param_diff < 1e-2
    )

    report = ResumeComparisonReport(
        run_a_final_step=state_a.step,
        run_b_final_step=state_b2.step,
        run_a_tokens_seen=state_a.tokens_seen,
        run_b_tokens_seen=state_b2.tokens_seen,
        run_a_final_loss=state_a.loss_history[-1]["loss"] if state_a.loss_history else float("nan"),
        run_b_final_loss=state_b2.loss_history[-1]["loss"] if state_b2.loss_history else float("nan"),
        loss_history_max_abs_diff=max_loss_diff,
        max_param_abs_diff=param_diff,
        generations_match=generations_match,
        equivalent=equivalent,
        notes=notes,
    )
    append_jsonl(log_path, {"event": "resume_comparison_result", **report.as_dict()})
    return report


def _load_checkpoint_for_inference(
    checkpoint_path: Path, architecture: ArchitectureConfig, device: torch.device
) -> JuniperMathModel:
    from juniper_math.model import build_model

    model = build_model(architecture)
    model.to(device)
    from juniper_math.checkpoint import load_checkpoint

    load_checkpoint(checkpoint_path, architecture, model=model, restore_rng=False)
    model.eval()
    return model


def run_infer(
    checkpoint_path: Path, prompt: str, max_new_tokens: int, config_path: Path | None = None
) -> GenerationResult:
    training_config, architecture, tokenizer = _load_common(config_path)
    device = _resolve_device(training_config.device)
    model = _load_checkpoint_for_inference(checkpoint_path, architecture, device)
    return generate(model, tokenizer, prompt, max_new_tokens, device, temperature=0.0)


def evaluate_tool_use_suite(
    checkpoint_path: Path, config_path: Path | None = None, sample_size: int | None = 20
) -> ToolFormatReport:
    training_config, architecture, tokenizer = _load_common(config_path)
    device = _resolve_device(training_config.device)
    model = _load_checkpoint_for_inference(checkpoint_path, architecture, device)
    suite_path = REPO_ROOT / "evals" / "phase4_tool_use_v2.json"
    return run_tool_format_evaluation(
        model, tokenizer, suite_path, device, training_config.generation_max_new_tokens, sample_size
    )


__all__ = [
    "ResumeComparisonReport",
    "SmokeRunReport",
    "evaluate_tool_use_suite",
    "run_infer",
    "run_resume_test",
    "run_smoke_train",
]
