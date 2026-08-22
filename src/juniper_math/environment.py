"""Environment validation.

Collects facts about the current machine and classifies them as
PASS / WARNING / FAIL. Required conditions (supported Python, importable
package) are distinguished from optional accelerator availability (CUDA/GPU)
-- a CPU-only machine should not FAIL environment validation, but it must
be reported honestly as CPU-only, never faked as CUDA-available.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from enum import StrEnum

# Must stay in sync with pyproject.toml's requires-python. Narrowed from 3.10
# during Opus 5 Phase 0 remediation so declared support equals tested support
# (see reports/OPUS5_PHASE0_REVIEW.md NOTE-A).
_MIN_PYTHON = (3, 12)
_MAX_PYTHON_EXCLUSIVE = (3, 13)
_SUPPORTED_RANGE = "3.12"


class CheckStatus(StrEnum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    detail: str


@dataclass
class EnvironmentReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def overall_status(self) -> CheckStatus:
        statuses = {c.status for c in self.checks}
        if CheckStatus.FAIL in statuses:
            return CheckStatus.FAIL
        if CheckStatus.WARNING in statuses:
            return CheckStatus.WARNING
        return CheckStatus.PASS

    def add(self, name: str, status: CheckStatus, detail: str) -> None:
        self.checks.append(CheckResult(name, status, detail))


def _check_python(report: EnvironmentReport) -> None:
    version = sys.version_info[:3]
    version_str = ".".join(str(v) for v in version)
    if _MIN_PYTHON <= version[:2] < _MAX_PYTHON_EXCLUSIVE:
        report.add(
            "python_version",
            CheckStatus.PASS,
            f"Python {version_str} (supported: {_SUPPORTED_RANGE})",
        )
    else:
        report.add(
            "python_version",
            CheckStatus.FAIL,
            f"Python {version_str} is outside the supported range ({_SUPPORTED_RANGE})",
        )


def _check_torch(report: EnvironmentReport) -> object | None:
    try:
        import torch
    except ImportError:
        report.add("torch_import", CheckStatus.FAIL, "PyTorch is not installed")
        return None
    report.add("torch_import", CheckStatus.PASS, f"torch {torch.__version__}")
    return torch


def _check_cuda(report: EnvironmentReport, torch_module) -> None:
    if torch_module is None:
        report.add("cuda_availability", CheckStatus.WARNING, "Skipped — PyTorch not importable")
        return

    if torch_module.cuda.is_available():
        gpu_name = torch_module.cuda.get_device_name(0)
        cuda_version = torch_module.version.cuda
        gpu_count = torch_module.cuda.device_count()
        report.add(
            "cuda_availability",
            CheckStatus.PASS,
            f"CUDA available — {gpu_count} device(s), GPU 0: {gpu_name}, torch CUDA build: {cuda_version}",
        )
    else:
        report.add(
            "cuda_availability",
            CheckStatus.WARNING,
            "CUDA not available — running CPU-only. This is expected on non-GPU machines and "
            "is not a failure, but training-related work requires the RTX 2060 target hardware "
            "(or another CUDA device) to be practical.",
        )


def _check_yaml(report: EnvironmentReport) -> None:
    try:
        import yaml  # noqa: F401
    except ImportError:
        report.add("pyyaml_import", CheckStatus.FAIL, "PyYAML is not installed")
        return
    report.add("pyyaml_import", CheckStatus.PASS, "PyYAML importable")


def _check_git(report: EnvironmentReport) -> None:
    try:
        result = subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=5, check=False)
        if result.returncode == 0:
            report.add("git_available", CheckStatus.PASS, result.stdout.strip())
        else:
            report.add("git_available", CheckStatus.WARNING, "git command failed")
    except (OSError, subprocess.TimeoutExpired):
        report.add("git_available", CheckStatus.WARNING, "git executable not found on PATH")


def _check_hardware(report: EnvironmentReport) -> None:
    report.add(
        "cpu_info",
        CheckStatus.PASS,
        f"{platform.processor() or platform.machine()} ({os.cpu_count()} logical cores)",
    )
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    report.add("system_memory", CheckStatus.PASS, f"{kb / (1024 * 1024):.1f} GiB total RAM")
                    break
    except OSError:
        report.add("system_memory", CheckStatus.WARNING, "Could not read /proc/meminfo (non-Linux host?)")


def run_environment_validation() -> EnvironmentReport:
    report = EnvironmentReport()
    _check_python(report)
    torch_module = _check_torch(report)
    _check_cuda(report, torch_module)
    _check_yaml(report)
    _check_git(report)
    _check_hardware(report)
    return report
