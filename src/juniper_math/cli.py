"""Juniper Math 1 canonical command-line interface.

Usage: `python -m juniper_math <command> ...` or the installed
`juniper-math <command> ...` console script (same entry point — pick one
style and use it consistently; both resolve here).

Commands are split into two honest categories:
  - Fully functional now (Phase 0/1): status, validate-env, validate-config,
    seed-test, evals validate, evals verify, hash verify,
    manifests-validate, deps-check, model, checkpoint inspect.
  - Not yet implemented (later phases): tokenizer, dataset, train,
    evaluate, infer, tool-test. These print an explicit
    "not implemented until Phase N" message and exit non-zero — they never
    silently pretend to succeed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from juniper_math import __version__
from juniper_math.architecture import load_architecture_config
from juniper_math.environment import CheckStatus, run_environment_validation
from juniper_math.errors import JuniperConfigError, JuniperManifestError
from juniper_math.evals import load_eval_suite, verify_suite_ground_truth
from juniper_math.hashing import sha256_file
from juniper_math.logging_utils import get_logger
from juniper_math.manifests import (
    check_dependency_licenses,
    load_licenses_manifest,
    load_sources_manifest,
    verify_artifacts_manifest,
)
from juniper_math.metadata import load_project_metadata
from juniper_math.seed import DEFAULT_PROJECT_SEED, set_global_seed

_MODEL_IMPORT_ERROR: Exception | None
try:
    import torch

    from juniper_math.checkpoint import CheckpointError, inspect_checkpoint_metadata
    from juniper_math.model import JuniperModelError, build_model, count_trainable_parameters

    _MODEL_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised only without torch installed
    _MODEL_IMPORT_ERROR = exc

logger = get_logger(__name__)

_NOT_IMPLEMENTED = {
    "tokenizer": 2,
    "dataset": 4,
    "train": 1,
    "evaluate": 1,
    "infer": 1,
    "tool-test": 3,
}


GIT_UNKNOWN = "unknown"
GIT_UNAVAILABLE = "unavailable (git not found or not a repository)"


def describe_git_state(cwd: Path | None = None) -> tuple[str, str]:
    """Return ``(commit, tree_state)``, never converting a failure into success.

    An earlier version reported ``clean`` whenever ``git status --porcelain``
    produced no stdout — which is exactly what happens when the command
    *fails* (not a repository, git missing, timeout). A failed interrogation
    was therefore indistinguishable from a genuinely clean tree. Tree state is
    now one of ``clean`` / ``dirty`` / ``unknown``, and ``unknown`` is
    reported whenever git could not actually answer.
    See reports/OPUS5_PHASE0_REVIEW.md (F-07).
    """

    def _run(args: list[str]) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=5, check=False, cwd=cwd)
        except (OSError, subprocess.SubprocessError):
            return None

    head = _run(["git", "rev-parse", "HEAD"])
    if head is None:
        return GIT_UNAVAILABLE, GIT_UNKNOWN
    commit = head.stdout.strip() if head.returncode == 0 and head.stdout.strip() else GIT_UNKNOWN

    status = _run(["git", "status", "--porcelain"])
    if status is None or status.returncode != 0:
        return commit, GIT_UNKNOWN
    return commit, "dirty" if status.stdout.strip() else "clean"


def _cmd_status(_args: argparse.Namespace) -> int:
    try:
        meta = load_project_metadata()
    except JuniperConfigError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"Project:        {meta.project_name}")
    print(f"Phase:          {meta.current_phase} — {meta.phase_name}")
    print(f"Phase status:   {meta.phase_status}")
    print(f"Architecture:   v{meta.architecture_version} (parameter target {meta.parameter_target:,})")

    commit, tree_state = describe_git_state()
    print(f"Git commit:     {commit}")
    print(f"Git tree state: {tree_state}")

    return 0


def _cmd_validate_env(_args: argparse.Namespace) -> int:
    report = run_environment_validation()
    for check in report.checks:
        print(f"[{check.status.value:7}] {check.name}: {check.detail}")
    print(f"\nOverall: {report.overall_status.value}")
    return 1 if report.overall_status == CheckStatus.FAIL else 0


def _cmd_validate_config(_args: argparse.Namespace) -> int:
    try:
        arch = load_architecture_config()
        meta = load_project_metadata()
    except JuniperConfigError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    estimate = arch.estimated_parameter_count()
    print(f"PASS: architecture config valid (v{arch.architecture_version})")
    print(f"PASS: project metadata valid (phase {meta.current_phase}, status {meta.phase_status})")
    print(f"      parameter_target={arch.parameter_target:,} estimated={estimate:,}")
    if estimate != arch.parameter_target:
        print(
            f"WARNING: estimate does not exactly match parameter_target "
            f"(diff={estimate - arch.parameter_target:+,}); this is a rough arithmetic check only, "
            f"authoritative verification is Phase 1 work."
        )
    return 0


def _cmd_seed_test(args: argparse.Namespace) -> int:
    report = set_global_seed(args.seed)
    print(f"seed={report.seed}")
    print(f"python_random_seeded={report.python_random_seeded}")
    print(f"numpy_seeded={report.numpy_seeded}")
    print(f"torch_cpu_seeded={report.torch_cpu_seeded}")
    print(f"torch_cuda_seeded={report.torch_cuda_seeded}")
    print(f"deterministic_algorithms_requested={report.deterministic_algorithms_requested}")
    return 0


def _cmd_evals_validate(_args: argparse.Namespace) -> int:
    try:
        suite = load_eval_suite()
    except JuniperConfigError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: schema valid — suite {suite.suite_id} v{suite.suite_version}, {len(suite.cases)} cases")
    for category, count in sorted(suite.category_counts().items()):
        print(f"  {category}: {count}")

    # Schema validity says nothing about arithmetic correctness. Recompute.
    return _report_ground_truth(suite)


def _report_ground_truth(suite) -> int:  # noqa: ANN001
    try:
        results = verify_suite_ground_truth(suite)
    except JuniperConfigError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    failures = [r for r in results if not r.verified]
    deterministic = [r for r in results if r.mode == "deterministic"]
    semantic = [r for r in results if r.mode == "semantic"]
    for result in failures:
        print(f"[FAIL] {result.case_id}: {result.detail}", file=sys.stderr)
    if failures:
        print(
            f"FAIL: ground-truth verification — {len(failures)} of {len(deterministic)} "
            f"deterministic case(s) do not match their recorded answer.",
            file=sys.stderr,
        )
        return 1
    print(
        f"PASS: ground truth verified — {len(deterministic)} deterministic case(s) recomputed, "
        f"{len(semantic)} semantic case(s) checked for classification consistency"
    )
    return 0


def _cmd_evals_verify(_args: argparse.Namespace) -> int:
    try:
        suite = load_eval_suite()
    except JuniperConfigError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return _report_ground_truth(suite)


def _cmd_hash_file(args: argparse.Namespace) -> int:
    path = Path(args.path)
    try:
        digest = sha256_file(path)
    except FileNotFoundError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(digest)
    return 0


def _cmd_hash_verify(_args: argparse.Namespace) -> int:
    try:
        results = verify_artifacts_manifest()
    except JuniperManifestError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    ok = True
    for artifact_id, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {artifact_id}: {detail}")
        ok = ok and passed
    return 0 if ok else 1


def _cmd_manifests_validate(_args: argparse.Namespace) -> int:
    try:
        sources = load_sources_manifest()
        licenses = load_licenses_manifest()
    except JuniperManifestError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: sources manifest valid — {len(sources)} entries")
    print(f"PASS: licenses manifest valid — {len(licenses)} entries")
    return _cmd_deps_check(_args)


def _cmd_deps_check(_args: argparse.Namespace) -> int:
    """Fail if a declared direct dependency has no license manifest entry."""
    try:
        results = check_dependency_licenses()
    except JuniperManifestError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    failures = [r for r in results if not r[1]]
    for package, ok, detail in results:
        if not ok:
            print(f"[FAIL] {package}: {detail}", file=sys.stderr)
    if failures:
        print(
            f"FAIL: dependency/license cross-check — {len(failures)} of {len(results)} "
            f"declared dependencies lack correct licensing metadata.",
            file=sys.stderr,
        )
        return 1
    print(f"PASS: dependency/license cross-check — {len(results)} direct dependencies all licensed")
    return 0


def _cmd_model(args: argparse.Namespace) -> int:
    if _MODEL_IMPORT_ERROR is not None:
        print(
            f"FAIL: model construction requires PyTorch, which is not importable: {_MODEL_IMPORT_ERROR}",
            file=sys.stderr,
        )
        return 1
    try:
        config = load_architecture_config()
    except JuniperConfigError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    try:
        model = build_model(config)
    except JuniperModelError as exc:
        print(f"FAIL: model construction failed: {exc}", file=sys.stderr)
        return 1

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(device_str)
        model = model.to(device)
    except (RuntimeError, ValueError) as exc:
        print(f"FAIL: could not move model to device {device_str!r}: {exc}", file=sys.stderr)
        return 1

    actual = count_trainable_parameters(model)
    print(f"Architecture:        {config.architecture_class} v{config.architecture_version}")
    print(
        f"d_model={config.d_model} n_layers={config.n_layers} "
        f"n_heads={config.n_query_heads} d_ff={config.d_ff} vocab={config.vocab_size} "
        f"context={config.max_context_length}"
    )
    print(f"Trainable parameters: {actual:,}")
    print(f"Parameter target:     {config.parameter_target:,}")
    print(f"Device:               {device}")

    if actual != config.parameter_target:
        print(
            f"FAIL: parameter count mismatch (expected {config.parameter_target:,}, got {actual:,})",
            file=sys.stderr,
        )
        return 1
    print("PASS: parameter count matches frozen target exactly")

    if args.forward_check:
        try:
            model.eval()
            with torch.no_grad():
                sample = torch.randint(0, config.vocab_size, (1, 8), device=device)
                out = model(sample)
            finite = bool(torch.isfinite(out.logits).all())
            print(
                f"Synthetic forward pass: logits shape={tuple(out.logits.shape)}, "
                f"dtype={out.logits.dtype}, finite={finite}"
            )
            if not finite:
                print("FAIL: forward pass produced non-finite logits", file=sys.stderr)
                return 1
            print("PASS: synthetic forward pass succeeded")
        except JuniperModelError as exc:
            print(f"FAIL: synthetic forward pass raised: {exc}", file=sys.stderr)
            return 1

    return 0


def _cmd_checkpoint_inspect(args: argparse.Namespace) -> int:
    if _MODEL_IMPORT_ERROR is not None:
        print(f"FAIL: checkpoint inspection requires PyTorch: {_MODEL_IMPORT_ERROR}", file=sys.stderr)
        return 1
    try:
        meta = inspect_checkpoint_metadata(Path(args.path))
    except CheckpointError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    for key, value in meta.items():
        print(f"{key}: {value}")
    return 0


def _make_not_implemented(command: str, phase: int):
    def _handler(_args: argparse.Namespace) -> int:
        print(
            f"'{command}' is not implemented until Phase {phase}. "
            f"Phase 0 establishes configuration structure only.",
            file=sys.stderr,
        )
        return 2

    return _handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="juniper-math", description="Juniper Math 1 project CLI")
    parser.add_argument("--version", action="version", version=f"juniper-math-1 {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Report current project phase/status").set_defaults(func=_cmd_status)

    subparsers.add_parser(
        "validate-env", help="Validate the local environment (Python/PyTorch/CUDA/hardware)"
    ).set_defaults(func=_cmd_validate_env)

    subparsers.add_parser(
        "validate-config", help="Validate architecture and project configuration"
    ).set_defaults(func=_cmd_validate_config)

    seed_parser = subparsers.add_parser("seed-test", help="Exercise the deterministic seed helper")
    seed_parser.add_argument("--seed", type=int, default=DEFAULT_PROJECT_SEED)
    seed_parser.set_defaults(func=_cmd_seed_test)

    evals_parser = subparsers.add_parser("evals", help="Evaluation suite operations")
    evals_sub = evals_parser.add_subparsers(dest="evals_command", required=True)
    evals_sub.add_parser(
        "validate", help="Validate the frozen evaluation suite (schema + deterministic ground truth)"
    ).set_defaults(func=_cmd_evals_validate)
    evals_sub.add_parser(
        "verify", help="Recompute deterministic evaluation answers only (ground-truth check)"
    ).set_defaults(func=_cmd_evals_verify)

    hash_parser = subparsers.add_parser("hash", help="Artifact hashing operations")
    hash_sub = hash_parser.add_subparsers(dest="hash_command", required=True)
    hash_file_parser = hash_sub.add_parser("file", help="Print the SHA-256 of a file")
    hash_file_parser.add_argument("path")
    hash_file_parser.set_defaults(func=_cmd_hash_file)
    hash_sub.add_parser("verify", help="Verify all artifacts in manifests/artifacts.yaml").set_defaults(
        func=_cmd_hash_verify
    )

    subparsers.add_parser(
        "manifests-validate",
        help="Validate source and license manifests (includes the dependency/license cross-check)",
    ).set_defaults(func=_cmd_manifests_validate)

    subparsers.add_parser(
        "deps-check",
        help="Cross-check pyproject direct dependencies against manifests/licenses.yaml",
    ).set_defaults(func=_cmd_deps_check)

    model_parser = subparsers.add_parser(
        "model", help="Construct the frozen architecture and verify its parameter count"
    )
    model_parser.add_argument("--device", default=None, help="cpu or cuda (default: auto-detect)")
    model_parser.add_argument(
        "--forward-check",
        action="store_true",
        default=True,
        help="run a harmless synthetic forward pass (default: on)",
    )
    model_parser.add_argument(
        "--no-forward-check", dest="forward_check", action="store_false", help="skip the forward pass check"
    )
    model_parser.set_defaults(func=_cmd_model)

    checkpoint_parser = subparsers.add_parser("checkpoint", help="Checkpoint operations")
    checkpoint_sub = checkpoint_parser.add_subparsers(dest="checkpoint_command", required=True)
    checkpoint_inspect_parser = checkpoint_sub.add_parser(
        "inspect", help="Safely report checkpoint metadata without restoring state"
    )
    checkpoint_inspect_parser.add_argument("path")
    checkpoint_inspect_parser.set_defaults(func=_cmd_checkpoint_inspect)

    for command, phase in _NOT_IMPLEMENTED.items():
        sub = subparsers.add_parser(command, help=f"(Phase {phase}) not yet implemented")
        sub.set_defaults(func=_make_not_implemented(command, phase))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
