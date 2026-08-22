"""Automated coverage for the tiny controlled overfit experiment (scripts/tiny_overfit.py).

Kept cheap deliberately — few steps, tiny synthetic batch. See scripts/tiny_overfit.py
for the full reproducible experiment with the project's real step count.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tiny_overfit import run  # noqa: E402


def test_tiny_overfit_gate_passes_on_cpu():
    result = run(steps=150, device_str="cpu")
    assert result["gate_passed"] is True
    assert result["final_loss"] < result["initial_loss"]
    assert result["final_next_token_accuracy"] >= result["accuracy_gate"]
