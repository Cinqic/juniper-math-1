from __future__ import annotations

from juniper_math.environment import CheckStatus, run_environment_validation


def test_report_has_python_check():
    report = run_environment_validation()
    names = {check.name for check in report.checks}
    assert "python_version" in names
    assert "cuda_availability" in names


def test_python_check_passes_on_supported_interpreter():
    report = run_environment_validation()
    python_check = next(c for c in report.checks if c.name == "python_version")
    assert python_check.status == CheckStatus.PASS


def test_cuda_check_never_fails_only_warns_or_passes():
    # CUDA absence must be a WARNING, never a FAIL, and must never be
    # silently reported as available when it is not.
    report = run_environment_validation()
    cuda_check = next(c for c in report.checks if c.name == "cuda_availability")
    assert cuda_check.status in (CheckStatus.PASS, CheckStatus.WARNING)


def test_overall_status_is_worst_of_checks():
    report = run_environment_validation()
    statuses = {c.status for c in report.checks}
    if CheckStatus.FAIL in statuses:
        assert report.overall_status == CheckStatus.FAIL
    elif CheckStatus.WARNING in statuses:
        assert report.overall_status == CheckStatus.WARNING
    else:
        assert report.overall_status == CheckStatus.PASS
