from __future__ import annotations

import pytest

from juniper_math.cli import main


def test_status_exits_zero(capsys):
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "Juniper Math 1" in out
    assert "COMPLETE" in out


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
    # `dataset` moved out of this list in Phase 4 (config/project.yaml
    # current_phase 4) — it is a real command group now, not a placeholder;
    # see test_dataset_cli.py.
    for command in ["train", "evaluate", "infer"]:
        exit_code = main([command])
        assert exit_code == 2, f"{command} should exit 2"
        err = capsys.readouterr().err
        assert "not implemented until Phase" in err


def test_model_command_exits_zero_and_verifies_param_count(capsys):
    assert main(["model", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "5,004,032" in out
    assert "PASS: parameter count matches frozen target exactly" in out
    assert "PASS: synthetic forward pass succeeded" in out


def test_model_command_no_forward_check(capsys):
    assert main(["model", "--device", "cpu", "--no-forward-check"]) == 0
    out = capsys.readouterr().out
    assert "synthetic forward pass" not in out


def test_checkpoint_inspect_missing_file_returns_nonzero(capsys, tmp_path):
    exit_code = main(["checkpoint", "inspect", str(tmp_path / "missing.pt")])
    assert exit_code == 1
    assert "FAIL" in capsys.readouterr().err


def test_checkpoint_inspect_reports_metadata(capsys, tmp_path):

    from juniper_math.architecture import load_architecture_config
    from juniper_math.checkpoint import build_checkpoint, save_checkpoint_atomic
    from juniper_math.model import build_model

    config = load_architecture_config()
    model = build_model(config)
    ckpt = build_checkpoint(
        architecture=config,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        step=3,
        tokens_seen=99,
        training_config={},
        data_stream_position={},
        seed=1,
        git_commit="abc123",
    )
    path = tmp_path / "ckpt.pt"
    save_checkpoint_atomic(ckpt, path)

    assert main(["checkpoint", "inspect", str(path)]) == 0
    out = capsys.readouterr().out
    assert "step: 3" in out
    assert "tokens_seen: 99" in out


def test_unknown_command_rejected():
    with pytest.raises(SystemExit):
        main(["definitely-not-a-real-command"])


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
