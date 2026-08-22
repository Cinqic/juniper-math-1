#!/usr/bin/env python3
"""Reproducible Phase 1 benchmark script.

Measures parameter memory, checkpoint size, peak CUDA VRAM, host RAM,
training/inference throughput at representative sequence lengths, on the
actual target hardware (Ryzen 7 5700G / RTX 2060 6GB / 16GB RAM).

Methodology:
  - CUDA timing uses torch.cuda.synchronize() around timed sections.
  - Each configuration runs WARMUP_ITERS untimed iterations before
    MEASURED_ITERS timed iterations (avoids counting CUDA context/kernel
    JIT warmup as steady-state throughput).
  - VRAM is read from torch.cuda.max_memory_allocated()/reserved() after
    resetting peak stats at the start of each configuration — not read from
    nvidia-smi, which reports whole-process (not per-workload) usage.
  - RAM is process RSS via /proc/self/status (Linux), documented as such —
    not total system memory.

Usage:
    python scripts/benchmark_phase1.py [--device cuda|cpu] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from juniper_math import __version__  # noqa: E402
from juniper_math.architecture import load_architecture_config  # noqa: E402
from juniper_math.checkpoint import build_checkpoint, save_checkpoint_atomic  # noqa: E402
from juniper_math.model import build_model, count_trainable_parameters  # noqa: E402

WARMUP_ITERS = 5
MEASURED_ITERS = 20
SEQ_LENS = [128, 512, 1024]
TRAIN_BATCH_SIZE = 4
INFER_BATCH_SIZE = 1


def _process_rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024
    except OSError:
        return None
    return None


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_training(model, config, device, seq_len, batch_size) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    for _ in range(WARMUP_ITERS):
        out = model(x, labels=x)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(MEASURED_ITERS):
        out = model(x, labels=x)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    _sync(device)
    elapsed = time.perf_counter() - start

    tokens_per_iter = batch_size * seq_len
    total_tokens = tokens_per_iter * MEASURED_ITERS
    result = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "iters": MEASURED_ITERS,
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
    }
    if device.type == "cuda":
        result["peak_vram_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        result["peak_vram_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 * 1024)
    return result


def benchmark_inference(model, config, device, seq_len, batch_size) -> dict:
    model.eval()
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            model(x)
        _sync(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        for _ in range(MEASURED_ITERS):
            model(x)
        _sync(device)
        elapsed = time.perf_counter() - start

    model.train()
    tokens_per_iter = batch_size * seq_len
    total_tokens = tokens_per_iter * MEASURED_ITERS
    result = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "iters": MEASURED_ITERS,
        "elapsed_seconds": elapsed,
        "forward_tokens_per_second": total_tokens / elapsed,
    }
    if device.type == "cuda":
        result["peak_vram_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        result["peak_vram_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 * 1024)
    return result


def run(device_str: str, tmp_dir: Path) -> dict:
    device = torch.device(device_str)
    config = load_architecture_config()

    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() and device.type == "cuda" else None
        ),
        "git_commit": _git_commit(),
        "juniper_math_version": __version__,
        "device_under_test": device_str,
    }

    model = build_model(config).to(device)
    param_count = count_trainable_parameters(model)
    param_memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    training_results = [
        benchmark_training(model, config, device, seq_len, TRAIN_BATCH_SIZE) for seq_len in SEQ_LENS
    ]
    inference_results = [
        benchmark_inference(model, config, device, seq_len, INFER_BATCH_SIZE) for seq_len in SEQ_LENS
    ]

    model_only_path = tmp_dir / "bench_model_only.pt"
    torch.save(model.state_dict(), model_only_path)
    model_only_size_mb = model_only_path.stat().st_size / (1024 * 1024)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, config.vocab_size, (2, 32), device=device)
    out = model(x, labels=x)
    out.loss.backward()
    optimizer.step()
    ckpt = build_checkpoint(
        architecture=config,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        step=1,
        tokens_seen=64,
        training_config={"purpose": "benchmark_full_checkpoint"},
        data_stream_position={},
        seed=0,
        git_commit=env["git_commit"],
    )
    full_ckpt_path = tmp_dir / "bench_full.pt"
    save_checkpoint_atomic(ckpt, full_ckpt_path)
    full_checkpoint_size_mb = full_ckpt_path.stat().st_size / (1024 * 1024)

    return {
        "environment": env,
        "methodology": {
            "warmup_iters": WARMUP_ITERS,
            "measured_iters": MEASURED_ITERS,
            "cuda_sync_policy": "torch.cuda.synchronize() before timing start and after timing end",
            "vram_source": "torch.cuda.max_memory_allocated/reserved (peak reset per configuration)",
            "ram_source": "/proc/self/status VmRSS (process RSS, not total system RAM)",
        },
        "model": {
            "trainable_parameters": param_count,
            "parameter_memory_mb_fp32": param_memory_mb,
        },
        "checkpoint_sizes": {
            "model_state_only_mb": model_only_size_mb,
            "full_training_checkpoint_mb": full_checkpoint_size_mb,
            "note": "full checkpoint is larger because it also serializes AdamW optimizer "
            "state (exp_avg, exp_avg_sq per parameter — roughly 2x parameter memory) plus "
            "RNG state and metadata, not just model weights.",
        },
        "host_ram_process_rss_mb": _process_rss_mb(),
        "training": training_results,
        "inference": inference_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json", type=str, default=None, help="write results to this JSON file")
    parser.add_argument("--tmp-dir", type=str, default=None)
    args = parser.parse_args()

    import tempfile

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix="juniper_bench_"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results = run(args.device, tmp_dir)
    text = json.dumps(results, indent=2)
    print(text)
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
