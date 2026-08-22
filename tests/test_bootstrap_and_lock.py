"""Tests for the environment lock and bootstrap prerequisite diagnostics.

Covers Opus 5 Phase 0 review findings F-03 (no exact dependency lock) and
F-04 (documented recovery fails on Debian/Ubuntu/Linux Mint because
`python3 -m venv` needs the separately-packaged ensurepip support).
"""

from __future__ import annotations

import re
import subprocess

import pytest

from juniper_math.manifests import declared_dependencies, normalize_package_name
from juniper_math.paths import REPO_ROOT

LOCK_PATH = REPO_ROOT / "requirements-lock.txt"
BOOTSTRAP_PATH = REPO_ROOT / "scripts" / "bootstrap.sh"


def _locked_packages() -> dict[str, str]:
    pins = {}
    for line in LOCK_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, version = line.partition("==")
        pins[normalize_package_name(name)] = version
    return pins


# --- F-03: exact dependency lock -------------------------------------------


def test_lock_file_exists():
    assert LOCK_PATH.is_file(), "requirements-lock.txt must exist for reproducible recovery"


def test_lock_pins_every_entry_exactly():
    """No ranges, no unpinned entries — every line is an exact '==' pin."""
    for line in LOCK_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert "==" in line, f"lock entry is not exactly pinned: {line!r}"
        assert not re.search(r"[<>~]|!=", line), f"lock entry contains a range operator: {line!r}"


def test_lock_covers_every_declared_direct_dependency():
    pins = _locked_packages()
    declared = declared_dependencies()
    missing = (declared["runtime"] | declared["dev"]) - set(pins)
    assert missing == set(), f"declared dependencies absent from the lock: {sorted(missing)}"


def test_lock_pins_the_quality_gate_tools():
    """ruff/mypy drift can break `ruff format --check` and `mypy` on unchanged code."""
    pins = _locked_packages()
    for tool in ("ruff", "mypy", "pytest", "torch", "numpy", "pyyaml", "types-pyyaml"):
        assert tool in pins, f"{tool} must be pinned in the lock"
        assert re.match(r"^\d+\.\d+", pins[tool]), f"{tool} pin looks malformed: {pins[tool]!r}"


def test_lock_documents_that_it_does_not_pin_system_layer():
    """The lock must not overclaim: it reproduces Python packages, not drivers."""
    header = LOCK_PATH.read_text(encoding="utf-8")
    lowered = header.lower()
    assert "driver" in lowered
    assert "does not pin" in lowered or "not pin" in lowered


# --- F-04: bootstrap prerequisite diagnostics ------------------------------


def test_bootstrap_checks_for_ensurepip_before_creating_venv():
    script = BOOTSTRAP_PATH.read_text(encoding="utf-8")
    assert "import ensurepip" in script
    ensurepip_at = script.index("import ensurepip")
    venv_at = script.index("-m venv .venv")
    assert ensurepip_at < venv_at, "the ensurepip preflight must run before `python -m venv`"


def test_bootstrap_names_the_debian_family_package():
    script = BOOTSTRAP_PATH.read_text(encoding="utf-8")
    assert "python3-venv" in script
    assert "Linux Mint" in script


def _executable_lines(text: str) -> list[tuple[int, str]]:
    """Script lines that bash actually executes, excluding heredoc payloads."""
    lines: list[tuple[int, str]] = []
    inside_heredoc = False
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if inside_heredoc:
            if stripped == "EOF":
                inside_heredoc = False
            continue
        if re.search(r"<<-?'?EOF'?", stripped):
            inside_heredoc = True
            continue
        if not stripped or stripped.startswith("#"):
            continue
        lines.append((number, line))
    return lines


def test_bootstrap_never_invokes_sudo_itself():
    """It may print sudo instructions; it must never execute them."""
    invocation = re.compile(r"(^|[;&|]\s*|\$\(|`)\s*sudo\b")
    for number, line in _executable_lines(BOOTSTRAP_PATH.read_text(encoding="utf-8")):
        if "sudo" not in line:
            continue
        assert not invocation.search(line.strip()), (
            f"bootstrap must not execute sudo (line {number}): {line!r}"
        )


def test_bootstrap_does_not_install_system_packages():
    """The script must not run a package manager on the user's behalf."""
    managers = re.compile(r"(^|[;&|]\s*|\$\(|`)\s*(apt|apt-get|dnf|yum|pacman)\b")
    for number, line in _executable_lines(BOOTSTRAP_PATH.read_text(encoding="utf-8")):
        assert not managers.search(line.strip()), (
            f"bootstrap must not invoke a package manager (line {number}): {line!r}"
        )


def test_bootstrap_is_syntactically_valid():
    result = subprocess.run(["bash", "-n", str(BOOTSTRAP_PATH)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("doc", ["RECOVERY.md", "ENVIRONMENT.md"])
def test_recovery_docs_document_the_venv_prerequisite(doc):
    text = (REPO_ROOT / "docs" / doc).read_text(encoding="utf-8")
    assert "python3-venv" in text, f"docs/{doc} must document the python3-venv prerequisite"


def test_recovery_doc_documents_the_lock_install():
    text = (REPO_ROOT / "docs" / "RECOVERY.md").read_text(encoding="utf-8")
    assert "requirements-lock.txt" in text
