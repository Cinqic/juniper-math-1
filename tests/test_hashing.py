from __future__ import annotations

import hashlib

import pytest

from juniper_math.hashing import ArtifactNotFoundError, sha256_bytes, sha256_file, verify_file


def test_sha256_file_matches_hashlib(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_bytes(b"juniper math 1 phase 0")
    expected = hashlib.sha256(b"juniper math 1 phase 0").hexdigest()
    assert sha256_file(target) == expected


def test_sha256_file_streams_large_content(tmp_path):
    target = tmp_path / "large.bin"
    data = b"x" * (5 * 1024 * 1024 + 37)  # not an exact multiple of the chunk size
    target.write_bytes(data)
    assert sha256_file(target) == hashlib.sha256(data).hexdigest()


def test_missing_file_raises(tmp_path):
    with pytest.raises(ArtifactNotFoundError):
        sha256_file(tmp_path / "does_not_exist.bin")


def test_sha256_bytes_matches_hashlib():
    assert sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_verify_file_true_and_false(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_bytes(b"data")
    correct = hashlib.sha256(b"data").hexdigest()
    assert verify_file(target, correct) is True
    assert verify_file(target, "0" * 64) is False


def test_hashing_does_not_modify_file(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_bytes(b"unchanged")
    before = target.read_bytes()
    sha256_file(target)
    after = target.read_bytes()
    assert before == after
