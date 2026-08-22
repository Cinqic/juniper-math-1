from __future__ import annotations

import pytest

from juniper_math.cli import main


def test_status_exits_zero(capsys):
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "Juniper Math 1" in out
    assert "AWAITING_OPUS_5_REVIEW" in out


def test_validate_config_exits_zero(capsys):
    assert main(["validate-config"]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out


def test_seed_test_exits_zero():
    assert main(["seed-test", "--seed", "1"]) == 0


def test_evals_validate_exits_zero(capsys):
    assert main(["evals", "validate"]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out


def test_manifests_validate_exits_zero():
    assert main(["manifests-validate"]) == 0


def test_hash_verify_exits_zero():
    assert main(["hash", "verify"]) == 0


def test_hash_file_prints_digest(capsys, tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("abc", encoding="utf-8")
    assert main(["hash", "file", str(target)]) == 0
    digest = capsys.readouterr().out.strip()
    assert len(digest) == 64

    import hashlib

    assert digest == hashlib.sha256(b"abc").hexdigest()


def test_hash_file_missing_returns_nonzero(capsys, tmp_path):
    assert main(["hash", "file", str(tmp_path / "missing.bin")]) == 1


def test_unimplemented_commands_are_honest(capsys):
    for command in ["model", "train", "evaluate", "infer", "tokenizer", "tool-test", "dataset", "checkpoint"]:
        exit_code = main([command])
        assert exit_code == 2, f"{command} should exit 2"
        err = capsys.readouterr().err
        assert "not implemented until Phase" in err


def test_unknown_command_rejected():
    with pytest.raises(SystemExit):
        main(["definitely-not-a-real-command"])


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
