"""Tests that a failed git interrogation is never reported as a clean tree.

The Phase 0 review candidate printed "Git tree state: clean" whenever
`git status --porcelain` produced no stdout — which is also what happens when
the command *fails*. Running `status` outside a repository therefore claimed a
clean tree that had never been examined.
See reports/OPUS5_PHASE0_REVIEW.md (F-07).
"""

from __future__ import annotations

import subprocess

import pytest

from juniper_math.cli import GIT_UNKNOWN, describe_git_state, main


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def clean_repo(tmp_path):
    repo = tmp_path / "clean"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    (repo / "file.txt").write_text("content\n", encoding="utf-8")
    _git("add", "file.txt", cwd=repo)
    _git("commit", "-q", "-m", "initial", cwd=repo)
    return repo


def test_clean_repository_reports_clean(clean_repo):
    commit, tree_state = describe_git_state(cwd=clean_repo)
    assert tree_state == "clean"
    assert len(commit) == 40


def test_dirty_repository_reports_dirty(clean_repo):
    (clean_repo / "file.txt").write_text("modified\n", encoding="utf-8")
    _, tree_state = describe_git_state(cwd=clean_repo)
    assert tree_state == "dirty"


def test_untracked_file_reports_dirty(clean_repo):
    (clean_repo / "new.txt").write_text("new\n", encoding="utf-8")
    _, tree_state = describe_git_state(cwd=clean_repo)
    assert tree_state == "dirty"


def test_non_git_directory_never_reports_clean(tmp_path):
    """The original defect: a non-repository must not look clean."""
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    commit, tree_state = describe_git_state(cwd=plain)
    assert tree_state == GIT_UNKNOWN
    assert tree_state != "clean"
    assert commit != ""


def test_missing_git_executable_never_reports_clean(monkeypatch, tmp_path):
    def _boom(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", _boom)
    commit, tree_state = describe_git_state(cwd=tmp_path)
    assert tree_state == GIT_UNKNOWN
    assert "unavailable" in commit


def test_git_timeout_never_reports_clean(monkeypatch, tmp_path):
    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=5)

    monkeypatch.setattr(subprocess, "run", _timeout)
    _, tree_state = describe_git_state(cwd=tmp_path)
    assert tree_state == GIT_UNKNOWN


def test_status_command_prints_unknown_outside_repository(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["status"]) == 0
    output = capsys.readouterr().out
    assert "Git tree state:" in output
    assert "clean" not in output
    assert GIT_UNKNOWN in output
