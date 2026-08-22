from __future__ import annotations

import pytest

from juniper_math.paths import REPO_ROOT, RepositoryRootNotFoundError, find_repo_root


def test_repo_root_contains_pyproject():
    assert (REPO_ROOT / "pyproject.toml").is_file()


def test_repo_root_is_not_hardcoded_to_a_personal_path():
    # The root must be derived by walking up from this file, not hardcoded
    # to any particular developer's home directory.
    assert (
        REPO_ROOT.name in {"juniper-math-1", "Juniper Math 1"}
        or (REPO_ROOT / "src" / "juniper_math").is_dir()
    )


def test_missing_marker_raises(tmp_path):
    with pytest.raises(RepositoryRootNotFoundError):
        find_repo_root(tmp_path / "some_file.py")
