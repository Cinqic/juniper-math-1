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


def test_no_placeholder_commands_remain():
    # `dataset` (Phase 4) and `train`/`evaluate`/`infer` (Phase 5) have all
    # moved out of the "not yet implemented" placeholder set — every command
    # in build_parser() is now a real implementation. See test_train_cli.py.
    from juniper_math.cli import _NOT_IMPLEMENTED

    assert _NOT_IMPLEMENTED == {}


def test_train_evaluate_infer_are_real_subcommands():
    # These now require real arguments (a config subcommand or a
    # --checkpoint path) rather than accepting no arguments and printing a
    # placeholder message — argparse itself rejects the bare invocation.
    for argv in (["train"], ["evaluate"], ["infer"]):
        with pytest.raises(SystemExit) as exc_info:
            main(argv)
        assert exc_info.value.code == 2


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
