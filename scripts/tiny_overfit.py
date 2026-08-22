#!/usr/bin/env python3
"""Reproducible tiny controlled overfit experiment (Phase 1 architecture validation).

Purpose: prove the model + optimizer can deliberately memorize a small,
deterministic, synthetic batch of integer token sequences. This is NOT a
claim of language capability — it is a mechanics check that gradients flow
correctly end to end and that training actually reduces loss on a target
the architecture should trivially be able to fit.

Uses only synthetic token IDs in [0, vocab_size) — no tokenizer or dataset
dependency (out of scope until Phase 2+).

Usage:
    python scripts/tiny_overfit.py [--device cuda|cpu] [--steps 300]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from juniper_math.architecture import load_architecture_config  # noqa: E402
from juniper_math.model import build_model  # noqa: E402
from juniper_math.seed import set_global_seed  # noqa: E402

SEED = 5004032  # project default seed
N_SEQUENCES = 4
SEQ_LEN = 32
LEARNING_RATE = 3e-3
DEFAULT_STEPS = 300
LOSS_RATIO_GATE = 0.05  # final loss must be <= 5% of initial loss
ACCURACY_GATE = 0.99  # next-token accuracy on the memorized batch


def make_synthetic_batch(vocab_size: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    batch = torch.randint(0, vocab_size, (N_SEQUENCES, SEQ_LEN), generator=generator)
    return batch.to(device)


def next_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits[:, :-1, :].argmax(dim=-1)
    targets = labels[:, 1:]
    correct = (preds == targets).float().mean().item()
    return correct


def run(steps: int, device_str: str) -> dict:
    device = torch.device(device_str)
    set_global_seed(SEED, deterministic_algorithms=False)
    config = load_architecture_config()
    model = build_model(config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    batch = make_synthetic_batch(config.vocab_size, device)

    out = model(batch, labels=batch)
    initial_loss = out.loss.item()

    start = time.perf_counter()
    for _ in range(steps):
        out = model(batch, labels=batch)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    elapsed = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        final_out = model(batch, labels=batch)
    final_loss = final_out.loss.item()
    accuracy = next_token_accuracy(final_out.logits, batch)

    loss_ratio = final_loss / initial_loss if initial_loss > 0 else float("nan")
    passed = loss_ratio <= LOSS_RATIO_GATE and accuracy >= ACCURACY_GATE

    result = {
        "seed": SEED,
        "device": device_str,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "steps": steps,
        "n_sequences": N_SEQUENCES,
        "seq_len": SEQ_LEN,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": loss_ratio,
        "loss_ratio_gate": LOSS_RATIO_GATE,
        "final_next_token_accuracy": accuracy,
        "accuracy_gate": ACCURACY_GATE,
        "elapsed_seconds": elapsed,
        "gate_passed": passed,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON only")
    args = parser.parse_args()

    result = run(args.steps, args.device)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
        print()
        print("GATE PASSED" if result["gate_passed"] else "GATE FAILED")

    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
