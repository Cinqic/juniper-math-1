"""Phase 6 pilot CLI end-to-end smoke test: train pilot-run -> checkpoint
inspect -> pilot-infer -> pilot-evaluate.

Requires a real local `dataset build` (the 1.6M-example corpus is
disposable and not committed) and the frozen tokenizer artifacts. Skips
honestly rather than faking success when those aren't present, matching
tests/test_train_cli.py's convention.
"""

from __future__ import annotations

import pytest
import yaml

from juniper_math.cli import main
from juniper_math.dataset.config import load_dataset_config
from juniper_math.pilot_training_config import PILOT_TRAINING_CONFIG_PATH


def _dataset_available() -> bool:
    try:
        cfg = load_dataset_config()
        return (
            cfg.output.manifest_path.is_file()
            and cfg.output.processed_path.is_dir()
            and any(cfg.output.processed_path.glob("*.jsonl"))
        )
    except Exception:  # noqa: BLE001
        return False


requires_dataset = pytest.mark.skipif(not _dataset_available(), reason="local dataset build not present")


@pytest.fixture
def tmp_pilot_config(tmp_path):
    raw = yaml.safe_load(PILOT_TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["schedule"]["total_steps"] = 2
    raw["schedule"]["checkpoint_interval"] = 2
    raw["schedule"]["validation_interval"] = 2
    raw["schedule"]["logging_interval"] = 1
    raw["scheduler"]["warmup_steps"] = 0
    raw["resume_test"]["interrupt_step"] = 1
    raw["device"] = "cpu"
    raw["pilot_subset"]["target_train_tokens"] = 3_000_000  # minimum of the enforced envelope
    raw["pilot_subset"]["validation_examples"] = 40
    raw["pilot_subset"]["min_category_examples"] = 1
    raw["pilot_subset"]["min_category_examples_validation"] = 1
    raw["pilot_subset"]["max_sequence_length"] = 64
    raw["milestone_fractions"] = [0.0, 1.0]
    raw["output"] = {
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "experiment_dir": str(tmp_path / "experiments"),
        "pilot_dataset_dir": str(tmp_path / "data"),
    }
    path = tmp_path / "training_phase6_pilot.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@requires_dataset
def test_train_pilot_run_end_to_end(tmp_pilot_config, capsys, tmp_path):
    exit_code = main(
        [
            "train",
            "pilot-run",
            "--config",
            str(tmp_pilot_config),
            "--eval-sample-size",
            "2",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "PASS: pilot training run" in out
    assert "milestone(s) evaluated" in out

    checkpoints = list((tmp_path / "checkpoints").glob("*.pt"))
    assert checkpoints, "expected at least one checkpoint to be written"


@requires_dataset
def test_pilot_infer_and_checkpoint_inspect(tmp_pilot_config, capsys, tmp_path):
    assert main(["train", "pilot-run", "--config", str(tmp_pilot_config), "--no-milestone-eval"]) == 0
    capsys.readouterr()

    final_ckpt = tmp_path / "checkpoints" / "step_000002_final.pt"
    assert final_ckpt.is_file()

    assert main(["checkpoint", "inspect", str(final_ckpt)]) == 0
    inspect_out = capsys.readouterr().out
    assert "step: 2" in inspect_out

    assert (
        main(
            [
                "pilot-infer",
                "--checkpoint",
                str(final_ckpt),
                "--prompt",
                "1 + 1 =",
                "--max-new-tokens",
                "4",
                "--config",
                str(tmp_pilot_config),
            ]
        )
        == 0
    )
    infer_out = capsys.readouterr().out
    assert infer_out.strip() != ""


@requires_dataset
def test_pilot_evaluate_command_runs_without_crashing(tmp_pilot_config, capsys, tmp_path):
    assert main(["train", "pilot-run", "--config", str(tmp_pilot_config), "--no-milestone-eval"]) == 0
    capsys.readouterr()
    final_ckpt = tmp_path / "checkpoints" / "step_000002_final.pt"

    exit_code = main(
        [
            "pilot-evaluate",
            "--checkpoint",
            str(final_ckpt),
            "--config",
            str(tmp_pilot_config),
            "--sample-size",
            "2",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[math]" in out
    assert "[tool_use_format]" in out
    assert "[calibration]" in out
    assert "[adversarial]" in out


@requires_dataset
def test_train_pilot_resume_test_reports_equivalence(tmp_pilot_config, capsys):
    exit_code = main(["train", "pilot-resume-test", "--config", str(tmp_pilot_config)])
    out = capsys.readouterr().out
    assert "equivalent: " in out
    assert exit_code == 0
