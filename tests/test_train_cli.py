"""Phase 5 CLI end-to-end smoke test: train run -> checkpoint inspect -> infer.

Requires a real local `dataset build` (the 1.6M-example corpus is
disposable and not committed — see docs/RECOVERY.md) and the frozen
tokenizer artifacts. Skips honestly rather than faking success when those
aren't present, matching the existing convention in test_dataset.py.
"""

from __future__ import annotations

import pytest
import yaml

from juniper_math.cli import main
from juniper_math.dataset.config import load_dataset_config
from juniper_math.training_config import TRAINING_CONFIG_PATH


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
def tmp_training_config(tmp_path):
    raw = yaml.safe_load(TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["schedule"]["total_steps"] = 2
    raw["schedule"]["checkpoint_interval"] = 2
    raw["schedule"]["validation_interval"] = 2
    raw["schedule"]["logging_interval"] = 1
    raw["device"] = "cpu"
    raw["smoke_subset"]["train_examples"] = 16
    raw["smoke_subset"]["validation_examples"] = 8
    raw["output"] = {
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "experiment_dir": str(tmp_path / "experiments"),
        "smoke_dataset_dir": str(tmp_path / "data"),
    }
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@requires_dataset
def test_train_run_end_to_end(tmp_training_config, capsys, tmp_path):
    exit_code = main(["train", "run", "--config", str(tmp_training_config)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "PASS: smoke training run" in out
    assert "final step:           2" in out

    checkpoints = list((tmp_path / "checkpoints").glob("*.pt"))
    assert checkpoints, "expected at least one checkpoint to be written"


@requires_dataset
def test_train_then_infer_and_checkpoint_inspect(tmp_training_config, capsys, tmp_path):
    assert main(["train", "run", "--config", str(tmp_training_config)]) == 0
    capsys.readouterr()

    final_ckpt = tmp_path / "checkpoints" / "step_000002_final.pt"
    assert final_ckpt.is_file()

    assert main(["checkpoint", "inspect", str(final_ckpt)]) == 0
    inspect_out = capsys.readouterr().out
    assert "step: 2" in inspect_out

    assert (
        main(
            [
                "infer",
                "--checkpoint",
                str(final_ckpt),
                "--prompt",
                "1 + 1 =",
                "--max-new-tokens",
                "4",
                "--config",
                str(tmp_training_config),
            ]
        )
        == 0
    )
    infer_out = capsys.readouterr().out
    assert infer_out.strip() != ""


@requires_dataset
def test_evaluate_command_runs_without_crashing(tmp_training_config, capsys, tmp_path):
    assert main(["train", "run", "--config", str(tmp_training_config)]) == 0
    capsys.readouterr()
    final_ckpt = tmp_path / "checkpoints" / "step_000002_final.pt"

    exit_code = main(
        [
            "evaluate",
            "--checkpoint",
            str(final_ckpt),
            "--config",
            str(tmp_training_config),
            "--sample-size",
            "3",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "SMOKE PIPELINE VALIDATION ONLY" in out
