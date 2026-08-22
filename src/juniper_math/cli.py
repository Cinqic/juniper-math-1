"""Juniper Math 1 canonical command-line interface.

Usage: `python -m juniper_math <command> ...` or the installed
`juniper-math <command> ...` console script (same entry point — pick one
style and use it consistently; both resolve here).

Commands are split into two honest categories:
  - Fully functional now (Phase 0): status, validate-env, validate-config,
    seed-test, evals validate, hash verify.
  - Not yet implemented (later phases): tokenizer, dataset, model, train,
    evaluate, infer, tool-test, checkpoint. These print an explicit
    "not implemented until Phase N" message and exit non-zero — they never
    silently pretend to succeed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from juniper_math import __version__
from juniper_math.architecture import load_architecture_config
from juniper_math.environment import CheckStatus, run_environment_validation
from juniper_math.errors import JuniperConfigError, JuniperManifestError
from juniper_math.evals import load_eval_suite
from juniper_math.hashing import sha256_file
from juniper_math.logging_utils import get_logger
from juniper_math.manifests import (
    load_licenses_manifest,
    load_sources_manifest,
    verify_artifacts_manifest,
)
from juniper_math.metadata import load_project_metadata
from juniper_math.seed import DEFAULT_PROJECT_SEED, set_global_seed

logger = get_logger(__name__)

_NOT_IMPLEMENTED = {
    "tokenizer": 2,
    "dataset": 4,
    "model": 1,
    "train": 1,
    "evaluate": 1,
    "infer": 1,
    "tool-test": 3,
    "checkpoint": 1,
}


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

    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5, check=False
        ).stdout.strip()
        print(f"Git commit:     {commit or 'unknown'}")
        print(f"Git tree state: {'dirty' if dirty else 'clean'}")
    except (OSError, FileNotFoundError):
        print("Git commit:     unavailable (git not found)")

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
    print(f"PASS: suite {suite.suite_id} v{suite.suite_version} — {len(suite.cases)} cases")
    for category, count in sorted(suite.category_counts().items()):
        print(f"  {category}: {count}")
    return 0


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
    evals_sub.add_parser("validate", help="Validate the frozen evaluation suite").set_defaults(
        func=_cmd_evals_validate
    )

    hash_parser = subparsers.add_parser("hash", help="Artifact hashing operations")
    hash_sub = hash_parser.add_subparsers(dest="hash_command", required=True)
    hash_file_parser = hash_sub.add_parser("file", help="Print the SHA-256 of a file")
    hash_file_parser.add_argument("path")
    hash_file_parser.set_defaults(func=_cmd_hash_file)
    hash_sub.add_parser("verify", help="Verify all artifacts in manifests/artifacts.yaml").set_defaults(
        func=_cmd_hash_verify
    )

    subparsers.add_parser("manifests-validate", help="Validate source and license manifests").set_defaults(
        func=_cmd_manifests_validate
    )

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
