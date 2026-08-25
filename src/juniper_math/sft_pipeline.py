"""Phase 8 SFT orchestration: the functions the CLI's `train sft-run`,
`train sft-resume-test`, `sft-evaluate`, and `sft-infer` commands call into.

Mirrors `full_pipeline.py`'s house style and reuses
`juniper_math.trainer`/`juniper_math.checkpoint` completely unchanged (see
reports/PHASE8_PLAN.md Sec. 6). The only genuinely new mechanics here are:
(1) initializing from the verified Phase 7 Base's *weights only*, with a
freshly constructed optimizer/scheduler (`init_sft_state` — Sec. 4, never
resumes Phase 7's optimizer trajectory), and (2) the Phase 8-specific
milestone evaluation, which adds the end-to-end tool-interaction harness
(`juniper_math.tool_interaction`) and Sec. 23's tool metrics
(`juniper_math.sft_eval`) on top of the frozen-suite regression check Phase
7 already performs.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from juniper_math.architecture import ArchitectureConfig, load_architecture_config
from juniper_math.checkpoint import load_checkpoint
from juniper_math.dataset.schema import Example
from juniper_math.errors import JuniperConfigError
from juniper_math.full_pipeline import run_capability_suites
from juniper_math.generation import GenerationResult, generate
from juniper_math.model import build_model, count_trainable_parameters
from juniper_math.paths import REPO_ROOT
from juniper_math.seed import set_global_seed
from juniper_math.sft_data import SFT_DATASET_ID, MaskedSftDataset, SftManifest, select_and_record_sft_subset
from juniper_math.sft_eval import run_phase8_eval_suite
from juniper_math.sft_training_config import (
    SftTrainingConfig,
    load_sft_training_config,
    verify_parent_checkpoint,
)
from juniper_math.smoke_data import TokenizedSmokeDataset, select_smoke_examples
from juniper_math.tokenizer import JuniperTokenizer
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

PHASE8_EVAL_SUITE_FILE = "phase8_instruction_v1.json"


def _git_commit() -> str:
    from juniper_math.cli import describe_git_state  # lazy: avoids import-order cycles

    commit, _tree_state = describe_git_state()
    return commit


def require_clean_source_tree(source_commit: str, source_tree_state: str) -> None:
    """Reject approval-candidate SFT runs without immutable source provenance."""
    unavailable = "unavailable (git not found or not a repository)"
    if source_tree_state != "clean" or source_commit in {"unknown", unavailable}:
        raise JuniperConfigError(
            "Phase 8 approval-candidate training requires a clean committed source tree. "
            "Use a separately labeled development workflow for exploratory dirty-tree work."
        )


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print(
            "WARNING: config/training_phase8_sft.yaml requests device=cuda but CUDA is not "
            "available; falling back to cpu. This invalidates the intended GPU hardware validation."
        )
        return torch.device("cpu")
    return torch.device(requested)


def _load_common(config_path: Path | None) -> tuple[SftTrainingConfig, ArchitectureConfig, JuniperTokenizer]:
    training_config = load_sft_training_config(config_path)
    architecture = load_architecture_config()
    if architecture.architecture_version != training_config.architecture_identity:
        raise JuniperConfigError(
            f"SFT training config architecture_identity {training_config.architecture_identity!r} "
            f"does not match config/architecture.yaml architecture_version "
            f"{architecture.architecture_version!r}."
        )
    if training_config.sft_subset.max_sequence_length > architecture.max_context_length:
        raise JuniperConfigError("sft_subset.max_sequence_length exceeds frozen architecture context length.")
    from juniper_math.tools.config import load_tools_config

    tools_config = load_tools_config()
    if (
        tools_config.protocol_version != "1.0.0"
        or training_config.tool_protocol_identity != "juniper-tool-protocol-v1"
    ):
        raise JuniperConfigError(
            "SFT training config tool_protocol_identity does not match the frozen tool protocol."
        )
    tokenizer = JuniperTokenizer.load()
    verify_parent_checkpoint(training_config)
    if training_config.dataset_identity != SFT_DATASET_ID:
        raise JuniperConfigError(
            f"SFT training config dataset_identity {training_config.dataset_identity!r} does not match "
            f"the current derived representation {SFT_DATASET_ID!r}."
        )
    return training_config, architecture, tokenizer


@dataclass
class SftDatasets:
    train: MaskedSftDataset
    validation: MaskedSftDataset
    validation_examples: list[Example]
    base_regression_validation: TokenizedSmokeDataset
    manifest: SftManifest


def _build_datasets(
    training_config: SftTrainingConfig, tokenizer: JuniperTokenizer, dataset_config=None
) -> SftDatasets:
    from juniper_math.dataset.config import load_dataset_config

    dataset_config = dataset_config or load_dataset_config()
    if dataset_config.dataset_id != "juniper-math-dataset-v1":
        raise JuniperConfigError(
            f"Phase 8 SFT data must derive from the frozen 'juniper-math-dataset-v1' corpus; "
            f"config/dataset.yaml declares {dataset_config.dataset_id!r}."
        )
    ss = training_config.sft_subset
    selections, manifest = select_and_record_sft_subset(
        tokenizer_identity=training_config.tokenizer_identity,
        seed=training_config.seed,
        train_target_per_category=ss.train_target_per_category,
        validation_target_per_category=ss.validation_target_per_category,
        max_sequence_length=ss.max_sequence_length,
        output_dir=training_config.output.sft_dataset_path,
        tokenizer=tokenizer,
        dataset_config=dataset_config,
        category_weight_overrides=ss.category_weight_overrides or None,
    )
    train_ds = MaskedSftDataset(selections["train"], tokenizer, ss.max_sequence_length)
    val_ds = MaskedSftDataset(selections["validation"], tokenizer, ss.max_sequence_length)
    base_validation_examples = select_smoke_examples(
        dataset_config,
        "validation",
        training_config.base_regression_validation_examples,
        training_config.seed,
    )
    base_regression_validation = TokenizedSmokeDataset(
        base_validation_examples, tokenizer, ss.max_sequence_length
    )
    return SftDatasets(
        train=train_ds,
        validation=val_ds,
        validation_examples=selections["validation"],
        base_regression_validation=base_regression_validation,
        manifest=manifest,
    )


def init_sft_state(
    architecture: ArchitectureConfig, training_config: SftTrainingConfig, device: torch.device
) -> TrainState:
    """Loads the verified Phase 7 Base's *model weights only* (never its
    optimizer/scheduler/RNG trajectory — Sec. 4), then builds a fresh AdamW +
    warmup/cosine schedule exactly like `trainer.init_state` does for a
    from-scratch run. Reuses `trainer.init_state`'s optimizer/scheduler
    construction by delegating to it and then overwriting only the model
    weights, so there is a single source of truth for how an AdamW/scheduler
    pair gets built from an SftTrainingConfig-shaped config."""
    state = init_state(architecture, training_config, device)
    checkpoint_path = REPO_ROOT / training_config.parent_checkpoint_path
    model = build_model(architecture)
    load_checkpoint(
        checkpoint_path, architecture, model=model, optimizer=None, scheduler=None, restore_rng=False
    )
    state.model.load_state_dict(model.state_dict(), strict=True)
    state.model.to(device)
    return state


def compute_validation_metrics(
    state: TrainState, datasets: SftDatasets, batch_size: int
) -> tuple[dict[str, float], dict[str, float]]:
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
    state: TrainState, tokenizer: JuniperTokenizer, training_config: SftTrainingConfig
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


@dataclass
class MilestoneReport:
    step: int
    fraction: float
    validation_loss: float
    base_regression_validation_loss: float
    category_validation_loss: dict[str, float]
    capability: dict[str, Any]
    tool_interaction: dict[str, Any]
    generations: list[dict[str, str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "fraction": self.fraction,
            "validation_loss": self.validation_loss,
            "base_regression_validation_loss": self.base_regression_validation_loss,
            "category_validation_loss": self.category_validation_loss,
            "capability": self.capability,
            "tool_interaction": self.tool_interaction,
            "generations": self.generations,
        }


def run_milestone(
    state: TrainState,
    datasets: SftDatasets,
    training_config: SftTrainingConfig,
    tokenizer: JuniperTokenizer,
    fraction: float,
    eval_sample_size: int | None,
) -> MilestoneReport:
    overall, category_losses = compute_validation_metrics(
        state, datasets, training_config.data.micro_batch_size
    )
    base_regression = validate(
        state, datasets.base_regression_validation, training_config.data.micro_batch_size
    )
    capability = run_capability_suites(
        state.model, tokenizer, state.device, training_config.generation_max_new_tokens, eval_sample_size
    )
    tool_report = run_phase8_eval_suite(
        state.model,
        tokenizer,
        REPO_ROOT / "evals" / PHASE8_EVAL_SUITE_FILE,
        state.device,
        training_config.generation_max_new_tokens,
        eval_sample_size,
    )
    generations = _generate_fixed_prompts(state, tokenizer, training_config)
    return MilestoneReport(
        step=state.step,
        fraction=fraction,
        validation_loss=overall["validation_loss"],
        base_regression_validation_loss=base_regression["validation_loss"],
        category_validation_loss=category_losses,
        capability=capability,
        tool_interaction=tool_report.as_dict(),
        generations=generations,
    )


@dataclass
class SftRunReport:
    training_config: SftTrainingConfig
    architecture: ArchitectureConfig
    sft_manifest: SftManifest
    device: str
    parameter_count: int
    milestones: list[MilestoneReport]
    final_checkpoint_path: str
    log_path: str
    elapsed_seconds: float
    peak_cuda_memory_bytes: int | None
    train_padding_fraction: float
    train_total_loss_tokens: int


def run_sft_train(
    config_path: Path | None = None,
    max_steps: int | None = None,
    eval_sample_size: int | None = None,
    milestone_eval: bool = True,
    resume_from: Path | None = None,
) -> SftRunReport:
    training_config, architecture, tokenizer = _load_common(config_path)
    source_commit, source_tree_state = __import__(
        "juniper_math.cli", fromlist=["describe_git_state"]
    ).describe_git_state()
    require_clean_source_tree(source_commit, source_tree_state)
    device = _resolve_device(training_config.device)
    set_global_seed(training_config.seed)

    datasets = _build_datasets(training_config, tokenizer)
    train_padding_fraction = datasets.train.padding_fraction_for_order(
        training_config.seed,
        epoch=0,
        shuffle=training_config.data.shuffle,
        micro_batch_size=training_config.data.micro_batch_size,
    )

    experiment_dir = training_config.output.experiment_path
    checkpoint_dir = training_config.output.checkpoint_path
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = experiment_dir / "train_log.jsonl"

    if resume_from is not None:
        state = init_state(architecture, training_config, device)
        load_state(state, architecture, resume_from)
    else:
        state = init_sft_state(architecture, training_config, device)
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
            "parent_checkpoint_path": training_config.parent_checkpoint_path,
            "parent_checkpoint_sha256": training_config.parent_checkpoint_sha256,
            "parent_phase7_tag": training_config.parent_phase7_tag,
            "sft_manifest": datasets.manifest.as_dict(),
            "sft_identity": datasets.manifest.sft_identity,
            "train_examples": len(datasets.train),
            "train_padding_fraction": train_padding_fraction,
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
        _maybe_milestone(0)

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
        state,
        architecture,
        training_config,
        final_checkpoint_path,
        source_commit,
        extra={
            "final": True,
            "parent_checkpoint_path": training_config.parent_checkpoint_path,
            "parent_checkpoint_sha256": training_config.parent_checkpoint_sha256,
            "parent_phase7_tag": training_config.parent_phase7_tag,
            "sft_identity": datasets.manifest.sft_identity,
        },
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

    return SftRunReport(
        training_config=training_config,
        architecture=architecture,
        sft_manifest=datasets.manifest,
        device=str(device),
        parameter_count=parameter_count,
        milestones=milestones,
        final_checkpoint_path=str(final_checkpoint_path),
        log_path=str(log_path),
        elapsed_seconds=elapsed,
        peak_cuda_memory_bytes=peak_mem,
        train_padding_fraction=train_padding_fraction,
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
    state: TrainState, tokenizer: JuniperTokenizer, training_config: SftTrainingConfig
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


def run_sft_resume_test(config_path: Path | None = None) -> ResumeComparisonReport:
    """Sec. 18's mandatory Phase-8-specific resume check: Phase 7's own resume
    proof does not establish that the new SFT data stream / masked-label
    Dataset resumes correctly, so this repeats the Sec. 24-style
    interrupted-vs-uninterrupted equivalence gate against the actual Phase 8
    pipeline (masked SFT dataset, Base-initialized weights, fresh optimizer)."""
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

    state_a = init_sft_state(architecture, training_config, device)
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
            "Comparison run on CUDA: best-effort determinism only (see full_pipeline's identical "
            "note); small floating-point differences from kernel nondeterminism are "
            "tolerance-checked (<1e-2), not required to be exactly zero."
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
) -> Any:
    model = build_model(architecture)
    model.to(device)
    load_checkpoint(checkpoint_path, architecture, model=model, restore_rng=False)
    model.eval()
    return model


def run_sft_infer(
    checkpoint_path: Path, prompt: str, max_new_tokens: int, config_path: Path | None = None
) -> GenerationResult:
    training_config, architecture, tokenizer = _load_common(config_path)
    device = _resolve_device(training_config.device)
    model = _load_checkpoint_for_inference(checkpoint_path, architecture, device)
    return generate(model, tokenizer, prompt, max_new_tokens, device, temperature=0.0)


def evaluate_sft_checkpoint(
    checkpoint_path: Path, config_path: Path | None = None, sample_size: int | None = None
) -> dict[str, Any]:
    """Runs the four frozen v2 suites plus the Phase 8 tool-interaction suite
    against an arbitrary saved Phase 8 checkpoint."""
    training_config, architecture, tokenizer = _load_common(config_path)
    device = _resolve_device(training_config.device)
    model = _load_checkpoint_for_inference(checkpoint_path, architecture, device)
    capability = run_capability_suites(
        model, tokenizer, device, training_config.generation_max_new_tokens, sample_size
    )
    tool_report = run_phase8_eval_suite(
        model,
        tokenizer,
        REPO_ROOT / "evals" / PHASE8_EVAL_SUITE_FILE,
        device,
        training_config.generation_max_new_tokens,
        sample_size,
    )
    capability["tool_interaction"] = tool_report.as_dict()
    return capability


__all__ = [
    "PHASE8_EVAL_SUITE_FILE",
    "MilestoneReport",
    "ResumeComparisonReport",
    "SftDatasets",
    "SftRunReport",
    "compute_validation_metrics",
    "evaluate_sft_checkpoint",
    "init_sft_state",
    "run_milestone",
    "run_sft_infer",
    "run_sft_resume_test",
    "run_sft_train",
]
