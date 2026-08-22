"""Reusable SHA-256 artifact hashing utility.

Used to make frozen Phase 0 artifacts (architecture config, evaluation
suite, manifests) verifiable byte-for-byte after a fresh clone.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024  # 1 MiB streaming reads — safe for large files


class ArtifactNotFoundError(FileNotFoundError):
    pass


def sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of a file's bytes.

    Streams the file in chunks so this works for arbitrarily large files
    without loading them fully into memory. Never modifies the file.
    """
    if not path.is_file():
        raise ArtifactNotFoundError(f"Cannot hash missing file: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_file(path: Path, expected_sha256: str) -> bool:
    return sha256_file(path) == expected_sha256.lower()
