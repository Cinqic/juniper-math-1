"""Repository path resolution.

Every path used by this project is resolved relative to the repository root,
never to a hidden local absolute path. The root is located by walking up
from this file until a marker (pyproject.toml) is found, so the project
works identically regardless of where it is cloned.
"""

from __future__ import annotations

from pathlib import Path


class RepositoryRootNotFoundError(RuntimeError):
    pass


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RepositoryRootNotFoundError(
        "Could not locate repository root (no pyproject.toml found in any parent directory)."
    )


REPO_ROOT = find_repo_root()
CONFIG_DIR = REPO_ROOT / "config"
EVALS_DIR = REPO_ROOT / "evals"
MANIFESTS_DIR = REPO_ROOT / "manifests"
DOCS_DIR = REPO_ROOT / "docs"
