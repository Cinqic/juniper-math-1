"""Phase 7 full base-pretraining orchestration: the functions the CLI's
`train full-run`, `train full-resume-test`, `full-evaluate`, and `full-infer`
commands call into. Mirrors `juniper_math.pilot_pipeline`'s house style and
reuses the same underlying primitives (`juniper_math.trainer`,
`juniper_math.pilot_data.PackedPilotDataset`, `juniper_math.smoke_data.
TokenizedSmokeDataset`) unchanged — Phase 7's only real difference from
Phase 6 is *what* dataset it trains/validates on (the entire frozen train/
validation splits via `juniper_math.full_data`, not a pilot subsample) and
that its run length is expressed in epochs over that full data via
`schedule.total_steps`, not a fixed pilot-scale step count.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from juniper_math.architecture import ArchitectureConfig, load_architecture_config
from juniper_math.dataset.config import load_dataset_config
from juniper_math.dataset.schema import Example
from juniper_math.errors import JuniperConfigError
from juniper_math.full_data import FullManifest, select_and_record_full_dataset
from juniper_math.full_training_config import FullTrainingConfig, load_full_training_config
from juniper_math.generation import GenerationResult, generate
from juniper_math.model import JuniperMathModel, count_trainable_parameters
from juniper_math.paths import REPO_ROOT
from juniper_math.pilot_data import PackedPilotDataset
from juniper_math.pilot_eval import run_capability_evaluation
from juniper_math.seed import set_global_seed
from juniper_math.smoke_data import TokenizedSmokeDataset
from juniper_math.tokenizer import JuniperTokenizer
from juniper_math.tool_format_eval import run_tool_format_evaluation
from juniper_math.trainer import (
    TrainState,
    append_jsonl,
    assert_model_finite,
    init_state,
    load_state,
    run_training,
    save_state,
    validate,
)

CAPABILITY_SUITE_FILES = {
    "math": "phase4_math_v2.json",
    "calibration": "phase4_calibration_v2.json",
    "adversarial": "phase4_adversarial_v2.json",
}
TOOL_USE_SUITE_FILE = "phase4_tool_use_v2.json"

# Same rationale as pilot_pipeline.VALIDATION_MAX_SEQUENCE_LENGTH: frozen
# corpus examples are short (median 27, p99 194 tokens); 256 covers the p99
# with margin without reintroducing training-scale padding waste for
# individually-padded validation examples.
VALIDATION_MAX_SEQUENCE_LENGTH = 256


def _git_commit() -> str:
    from juniper_math.cli import describe_git_state  # lazy: avoids import-order cycles

    commit, _tree_state = describe_git_state()
    return commit


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print(
            "WARNING: config/training_phase7_full.yaml requests device=cuda but CUDA is not "
            "available; falling back to cpu. This invalidates the intended GPU hardware validation."
        )
        return torch.device("cpu")
    return torch.device(requested)


def _load_common(
    config_path: Path | None,
) -> tuple[FullTrainingConfig, ArchitectureConfig, JuniperTokenizer]:
    training_config = load_full_training_config(config_path)
    architecture = load_architecture_config()
    if architecture.architecture_version != training_config.architecture_identity:
        raise JuniperConfigError(
            f"full training config architecture_identity {training_config.architecture_identity!r} "
            f"does not match config/architecture.yaml architecture_version "
            f"{architecture.architecture_version!r}."
        )
    if training_config.full_subset.max_sequence_length > architecture.max_context_length:
        raise JuniperConfigError(
            "full_subset.max_sequence_length exceeds frozen architecture context length."
        )
    tokenizer = JuniperTokenizer.load()
    return training_config, architecture, tokenizer


@dataclass
class FullDatasets:
    train: PackedPilotDataset
    validation: TokenizedSmokeDataset
    validation_examples: list[Example]
    manifest: FullManifest


def _build_datasets(training_config: FullTrainingConfig, tokenizer: JuniperTokenizer) -> FullDatasets:
    dataset_config = load_dataset_config()
    if dataset_config.dataset_id != training_config.dataset_identity:
        raise JuniperConfigError(
            f"full training config dataset_identity {training_config.dataset_identity!r} does not "
            f"match config/dataset.yaml dataset_id {dataset_config.dataset_id!r}."
        )
    fs = training_config.full_subset
    selections, manifest = select_and_record_full_dataset(
        dataset_id=training_config.dataset_identity,
        tokenizer_identity=training_config.tokenizer_identity,
        seed=training_config.seed,
        max_sequence_length=fs.max_sequence_length,
        pack_sequences_flag=fs.pack_sequences,
        output_dir=training_config.output.full_dataset_path,
        dataset_config=dataset_config,
    )
    train_ds = PackedPilotDataset(
        selections["train"], tokenizer, fs.max_sequence_length, pack_sequences_flag=fs.pack_sequences
    )
    val_ds = TokenizedSmokeDataset(selections["validation"], tokenizer, VALIDATION_MAX_SEQUENCE_LENGTH)
    return FullDatasets(
        train=train_ds, validation=val_ds, validation_examples=selections["validation"], manifest=manifest
    )


def compute_validation_metrics(
    state: TrainState, datasets: FullDatasets, batch_size: int
) -> tuple[dict[str, float], dict[str, float]]:
    """Overall validation loss (entire frozen validation split) plus a per-category breakdown."""
    overall = validate(state, datasets.validation, batch_size)
    by_category: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(datasets.validation_examples):
        by_category[ex.category].append(i)
    category_losses: dict[str, float] = {}
    for category, indices in sorted(by_category.items()):
        subset = torch.utils.data.Subset(datasets.validation, indices)
        metrics = validate(state, subset, batch_size)
        category_losses[category] = metrics["validation_loss"]
    return overall, category_losses


def _generate_fixed_prompts(
    state: TrainState, tokenizer: JuniperTokenizer, training_config: FullTrainingConfig
) -> list[dict[str, str]]:
    out = []
    for entry in training_config.fixed_generation_prompts:
        result = generate(
            state.model,
            tokenizer,
            entry["prompt"],
            training_config.generation_max_new_tokens,
            state.device,
            temperature=0.0,
        )
        out.append({"category": entry["category"], "prompt": result.prompt, "text": result.text})
    return out


def run_capability_suites(
    model: JuniperMathModel,
    tokenizer: JuniperTokenizer,
    device: torch.device,
    max_new_tokens: int,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Same four frozen v2 suites Phase 6 evaluates; identical settings across milestones."""
    reports: dict[str, Any] = {}
    for name, filename in CAPABILITY_SUITE_FILES.items():
        suite_path = REPO_ROOT / "evals" / filename
        report = run_capability_evaluation(model, tokenizer, suite_path, device, max_new_tokens, sample_size)
        reports[name] = report.as_dict()
    tool_report = run_tool_format_evaluation(
        model, tokenizer, REPO_ROOT / "evals" / TOOL_USE_SUITE_FILE, device, max_new_tokens, sample_size
    )
    reports["tool_use_format"] = tool_report.as_dict()
    return reports


@dataclass
class MilestoneReport:
    step: int
    fraction: float
    validation_loss: float
    category_validation_loss: dict[str, float]
    capability: dict[str, Any]
    generations: list[dict[str, str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "fraction": self.fraction,
            "validation_loss": self.validation_loss,
            "category_validation_loss": self.category_validation_loss,
            "capability": self.capability,
            "generations": self.generations,
        }


def run_milestone(
    state: TrainState,
    datasets: FullDatasets,
    training_config: FullTrainingConfig,
    tokenizer: JuniperTokenizer,
    fraction: float,
    eval_sample_size: int | None,
) -> MilestoneReport:
    overall, category_losses = compute_validation_metrics(
        state, datasets, training_config.data.micro_batch_size
    )
    capability = run_capability_suites(
        state.model, tokenizer, state.device, training_config.generation_max_new_tokens, eval_sample_size
    )
    generations = _generate_fixed_prompts(state, tokenizer, training_config)
    return MilestoneReport(
        step=state.step,
        fraction=fraction,
        validation_loss=overall["validation_loss"],
        category_validation_loss=category_losses,
        capability=capability,
        generations=generations,
    )


@dataclass
class FullRunReport:
    training_config: FullTrainingConfig
    architecture: ArchitectureConfig
    full_manifest: FullManifest
    device: str
    parameter_count: int
    milestones: list[MilestoneReport]
    final_checkpoint_path: str
    log_path: str
    elapsed_seconds: float
    peak_cuda_memory_bytes: int | None
    train_padding_fraction: float
    train_packed_sequences: int
    train_total_loss_tokens: int


def run_full_train(
    config_path: Path | None = None,
    max_steps: int | None = None,
    eval_sample_size: int | None = None,
    milestone_eval: bool = True,
    resume_from: Path | None = None,
) -> FullRunReport:
    """Runs Phase 7 full base pretraining from fresh random initialization, per
    reports/PHASE6_FINAL_APPROVAL.md: "must use fresh random initialization, never the pilot
    weights". Pass `resume_from` (a checkpoint saved by an earlier call to this same function, same
    config) to continue an interrupted canonical run: `juniper_math.trainer.load_state` restores
    model/optimizer/scheduler/RNG/step/tokens_seen/data-stream-position, and training resumes from
    that exact point — already-completed milestones are not rerun, and the JSONL log is appended
    to rather than truncated.
    """
    training_config, architecture, tokenizer = _load_common(config_path)
    source_commit, source_tree_state = __import__(
        "juniper_math.cli", fromlist=["describe_git_state"]
    ).describe_git_state()
    device = _resolve_device(training_config.device)
    set_global_seed(training_config.seed)

    datasets = _build_datasets(training_config, tokenizer)

    experiment_dir = training_config.output.experiment_path
    checkpoint_dir = training_config.output.checkpoint_path
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = experiment_dir / "train_log.jsonl"

    state = init_state(architecture, training_config, device)
    if resume_from is not None:
        load_state(state, architecture, resume_from)
    else:
        log_path.unlink(missing_ok=True)
    parameter_count = count_trainable_parameters(state.model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    append_jsonl(
        log_path,
        {
            "event": "run_start" if resume_from is None else "run_resume",
            "run_id": training_config.run_id,
            "git_commit": source_commit,
            "source_tree_state": source_tree_state,
            "parameter_count": parameter_count,
            "device": str(device),
            "full_manifest": datasets.manifest.as_dict(),
            "train_packed_sequences": len(datasets.train),
            "train_padding_fraction": datasets.train.padding_fraction,
            "train_total_loss_tokens": datasets.train.total_loss_tokens,
            "resume_from_step": state.step if resume_from is not None else None,
        },
    )

    end_step = max_steps if max_steps is not None else training_config.schedule.total_steps
    milestones: list[MilestoneReport] = []
    milestone_steps = sorted({round(f * end_step) for f in training_config.milestone_fractions})

    start = time.perf_counter()

    def _maybe_milestone(current_fraction_step: int) -> None:
        if not milestone_eval:
            return
        for f in training_config.milestone_fractions:
            target_step = round(f * end_step)
            if target_step == current_fraction_step:
                report = run_milestone(state, datasets, training_config, tokenizer, f, eval_sample_size)
                milestones.append(report)
                append_jsonl(log_path, {"event": "milestone", **report.as_dict()})

    if state.step == 0:
        _maybe_milestone(0)  # step 0 / initialization milestone, before any training

    for target_step in [s for s in milestone_steps if s > state.step]:
        run_training(
            state,
            datasets.train,
            datasets.validation,
            architecture,
            training_config,
            target_step,
            checkpoint_dir,
            log_path,
            source_commit,
        )
        _maybe_milestone(target_step)

    elapsed = time.perf_counter() - start

    final_checkpoint_path = checkpoint_dir / f"step_{state.step:06d}_final.pt"
    save_state(
        state, architecture, training_config, final_checkpoint_path, source_commit, extra={"final": True}
    )
    assert_model_finite(state.model)
    peak_mem = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None

    append_jsonl(
        log_path,
        {
            "event": "run_end",
            "final_step": state.step,
            "tokens_seen": state.tokens_seen,
            "elapsed_seconds": elapsed,
            "peak_cuda_memory_bytes": peak_mem,
            "final_checkpoint": str(final_checkpoint_path),
        },
    )

    return FullRunReport(
        training_config=training_config,
        architecture=architecture,
        full_manifest=datasets.manifest,
        device=str(device),
        parameter_count=parameter_count,
        milestones=milestones,
        final_checkpoint_path=str(final_checkpoint_path),
        log_path=str(log_path),
        elapsed_seconds=elapsed,
        peak_cuda_memory_bytes=peak_mem,
        train_padding_fraction=datasets.train.padding_fraction,
        train_packed_sequences=len(datasets.train),
        train_total_loss_tokens=datasets.train.total_loss_tokens,
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


def _generate_fixed_prompts_state(
    state: TrainState, tokenizer: JuniperTokenizer, training_config: FullTrainingConfig
) -> list[GenerationResult]:
    return [
        generate(
            state.model,
            tokenizer,
            entry["prompt"],
            training_config.generation_max_new_tokens,
            state.device,
            temperature=0.0,
        )
        for entry in training_config.fixed_generation_prompts
    ]


def run_full_resume_test(config_path: Path | None = None) -> ResumeComparisonReport:
    """Same Sec. 24-style gate as `pilot_pipeline.run_pilot_resume_test`, run at Phase 7 scale
    against the full-dataset config, over a short window (resume_test.interrupt_step /
    schedule.total_steps as configured) so this evidence-producing check stays bounded."""
    training_config, architecture, tokenizer = _load_common(config_path)
    device = _resolve_device(training_config.device)
    total_steps = training_config.schedule.total_steps
    interrupt_step = training_config.resume_test.interrupt_step
    if not (0 < interrupt_step < total_steps):
        raise JuniperConfigError("resume_test.interrupt_step must be strictly between 0 and total_steps.")

    set_global_seed(training_config.seed)
    datasets = _build_datasets(training_config, tokenizer)

    scratch_dir = training_config.output.checkpoint_path / "resume_test"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    init_ckpt = scratch_dir / "init.pt"
    interrupt_ckpt = scratch_dir / "interrupt.pt"
    log_path = training_config.output.experiment_path / "resume_test_log.jsonl"
    log_path.unlink(missing_ok=True)

    state_a = init_state(architecture, training_config, device)
    save_state(state_a, architecture, training_config, init_ckpt, _git_commit())

    run_training(
        state_a,
        datasets.train,
        datasets.validation,
        architecture,
        training_config,
        total_steps,
        scratch_dir,
        log_path,
        _git_commit(),
    )
    gen_a = _generate_fixed_prompts_state(state_a, tokenizer, training_config)

    state_b1 = init_state(architecture, training_config, device)
    load_state(state_b1, architecture, init_ckpt)
    run_training(
        state_b1,
        datasets.train,
        datasets.validation,
        architecture,
        training_config,
        interrupt_step,
        scratch_dir,
        log_path,
        _git_commit(),
    )
    save_state(state_b1, architecture, training_config, interrupt_ckpt, _git_commit())
    b1_loss_history = list(state_b1.loss_history)
    del state_b1

    state_b2 = init_state(architecture, training_config, device)
    load_state(state_b2, architecture, interrupt_ckpt)
    run_training(
        state_b2,
        datasets.train,
        datasets.validation,
        architecture,
        training_config,
        total_steps,
        scratch_dir,
        log_path,
        _git_commit(),
    )
    gen_b = _generate_fixed_prompts_state(state_b2, tokenizer, training_config)

    loss_a = {m["step"]: m["loss"] for m in state_a.loss_history}
    loss_b1_and_b2 = {m["step"]: m["loss"] for m in b1_loss_history + state_b2.loss_history}
    common_steps = sorted(set(loss_a) & set(loss_b1_and_b2))
    max_loss_diff = max((abs(loss_a[s] - loss_b1_and_b2[s]) for s in common_steps), default=float("nan"))

    param_diff = float((_param_vector(state_a.model) - _param_vector(state_b2.model)).abs().max().item())
    generations_match = [g.text for g in gen_a] == [g.text for g in gen_b]

    notes = []
    if device.type == "cuda":
        notes.append(
            "Comparison run on CUDA: torch.use_deterministic_algorithms(warn_only=True) is a "
            "best-effort determinism request, not a bitwise guarantee — see "
            "reports/PHASE6_RESULTS.md 'Resume result'. Small floating-point differences from "
            "kernel nondeterminism are possible and are tolerance-checked (<1e-2), not required "
            "to be exactly zero; generation divergence on CUDA is expected and not itself a "
            "resume failure."
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
    from juniper_math.checkpoint import load_checkpoint
    from juniper_math.model import build_model

    model = build_model(architecture)
    model.to(device)
    load_checkpoint(checkpoint_path, architecture, model=model, restore_rng=False)
    model.eval()
    return model


def run_full_infer(
    checkpoint_path: Path, prompt: str, max_new_tokens: int, config_path: Path | None = None
) -> GenerationResult:
    training_config, architecture, tokenizer = _load_common(config_path)
    device = _resolve_device(training_config.device)
    model = _load_checkpoint_for_inference(checkpoint_path, architecture, device)
    return generate(model, tokenizer, prompt, max_new_tokens, device, temperature=0.0)


def evaluate_full_checkpoint(
    checkpoint_path: Path, config_path: Path | None = None, sample_size: int | None = None
) -> dict[str, Any]:
    """Runs all four frozen v2 suites against an arbitrary saved Phase 7 checkpoint."""
    training_config, architecture, tokenizer = _load_common(config_path)
    device = _resolve_device(training_config.device)
    model = _load_checkpoint_for_inference(checkpoint_path, architecture, device)
    return run_capability_suites(
        model, tokenizer, device, training_config.generation_max_new_tokens, sample_size
    )


__all__ = [
    "CAPABILITY_SUITE_FILES",
    "FullDatasets",
    "FullRunReport",
    "MilestoneReport",
    "ResumeComparisonReport",
    "TOOL_USE_SUITE_FILE",
    "compute_validation_metrics",
    "evaluate_full_checkpoint",
    "run_capability_suites",
    "run_full_infer",
    "run_full_resume_test",
    "run_full_train",
    "run_milestone",
]
