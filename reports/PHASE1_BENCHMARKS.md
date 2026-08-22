# Phase 1 Benchmarks

**Project:** Juniper Math 1
**Phase:** 1 — Architecture
**Engineer:** Claude Sonnet 5
**Date:** 2026-08-22
**Script:** [`scripts/benchmark_phase1.py`](../scripts/benchmark_phase1.py)
**Raw output:** [`reports/artifacts/phase1_benchmark_cuda.json`](artifacts/phase1_benchmark_cuda.json),
[`reports/artifacts/phase1_benchmark_cpu.json`](artifacts/phase1_benchmark_cpu.json)

## Hardware / environment

| | |
|---|---|
| CPU | AMD Ryzen 7 5700G |
| GPU | NVIDIA GeForce RTX 2060 (6GB VRAM) |
| System RAM | 16 GB (15.0 GiB reported by `/proc/meminfo`) |
| OS | Linux 7.0.0-30-generic, x86_64 |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| CUDA (torch build) | 13.0 |
| Driver | NVIDIA driver present, `nvidia-smi` confirms RTX 2060 |
| Commit under test | `f9e3659efe37183bae60ffafc2762b3d342cd047` (Phase 0 foundation; Phase 1 candidate commit recorded in `PHASE1_REPORT.md`) |

## Methodology

- **Warm-up:** 5 untimed iterations per configuration before timing starts, so CUDA context
  initialization and kernel JIT compilation are not counted as steady-state throughput.
- **Measured iterations:** 20 timed iterations per configuration.
- **CUDA synchronization:** `torch.cuda.synchronize()` is implicit in `.item()`/host reads during
  warm-up; explicit `torch.cuda.synchronize()` calls bound each timed section so asynchronous
  kernel launch latency is never mistaken for completed work.
- **VRAM:** `torch.cuda.max_memory_allocated()` / `max_memory_reserved()`, with peak stats reset
  at the start of each configuration. This is **not** `nvidia-smi` output — `nvidia-smi` reports
  whole-process VRAM (including the CUDA context itself, ~300-400MB baseline on this driver/CUDA
  combination), not the workload's own allocation.
- **RAM:** process RSS from `/proc/self/status` (`VmRSS`) — the Python process's own resident
  memory, not total system memory and not a measure of dataset/OS cache.
- **Batch/sequence configurations:** representative lengths T ∈ {128, 512, 1024}, training batch
  size 4, inference (forward-only) batch size 1. Not an exhaustive sweep to the OOM boundary —
  a conservative, comfortable-headroom configuration, per the Phase 1 hardware acceptance gate.
- No dropout, no gradient accumulation, no mixed precision in the reported numbers below (FP32
  reference path); FP16 mixed-precision correctness is verified separately in
  `tests/test_model_cuda.py::test_cuda_mixed_precision_fp16_finite_with_grad_scaler` but not
  benchmarked for throughput here — that is future work, not a Phase 1 gate.

## Model

| | |
|---|---|
| Trainable parameters | 5,004,032 (exact, programmatically verified) |
| Parameter memory (FP32) | 19.09 MB |

## Checkpoint sizes

| Checkpoint | Size |
|---|---|
| Model state only (`model.state_dict()`) | 19.1 MB |
| Full training checkpoint (model + AdamW optimizer state + RNG + metadata) | 57.3 MB |

The full checkpoint is roughly 3x the model-only size because AdamW stores two extra
per-parameter tensors (`exp_avg`, `exp_avg_sq`), each the same size as the parameter itself,
plus RNG state and training metadata — not because of any inefficiency in the checkpoint format.

## CUDA (RTX 2060) — training (forward + backward + AdamW step, batch=4)

| Seq len | Tokens/sec | Peak VRAM allocated | Peak VRAM reserved | % of 6GB budget (reserved) |
|---|---|---|---|---|
| 128 | 43,599 | 159.3 MB | 188 MB | 3.1% |
| 512 | 68,285 | 417.7 MB | 476 MB | 7.7% |
| 1024 (full context) | 64,054 | 757.6 MB | 840 MB | 13.7% |

Even at the full 1,024-token context length with batch size 4, peak VRAM usage is under 14% of
the RTX 2060's 6GB budget — comfortable headroom, not a barely-avoided OOM. This configuration
was chosen deliberately conservative per the Phase 1 hardware acceptance gate (item 50); the
crash boundary was not explored, since a valuable-headroom operating point matters more than
finding the exact ceiling.

## CUDA (RTX 2060) — inference (forward only, batch=1)

| Seq len | Forward tokens/sec | Peak VRAM allocated |
|---|---|---|
| 128 | 46,661 | 38.0 MB |
| 512 | 166,607 | 44.3 MB |
| 1024 | 199,381 | 52.8 MB |

No KV cache exists in this Phase 1 implementation (see `ARCHITECTURE.md` / item 48 —
out of scope for Phase 1). These numbers are full-forward reference throughput, not
incremental-decode throughput; the low, roughly-constant VRAM reflects that every measurement
runs one full forward pass over the entire sequence, not autoregressive step-by-step decoding.

## CPU (Ryzen 7 5700G) — training (forward + backward + AdamW step, batch=4)

| Seq len | Tokens/sec |
|---|---|
| 128 | 7,346 |
| 512 | 7,721 |
| 1024 | 6,906 |

CPU training is roughly 9x slower than CUDA at T=1024 — expected and unremarkable; CPU
execution exists to support debugging and the strict deterministic resume-equivalence test
(`tests/test_checkpoint.py::test_interrupted_resume_matches_uninterrupted_control`), not as a
practical training path.

## Host RAM (process RSS)

Recorded via `/proc/self/status` during the benchmark run: on the order of ~1.2 GB process RSS
for the benchmark process itself (dominated by the PyTorch/CUDA runtime, not the 19MB model).
See the raw JSON artifacts for the exact per-run figure — this is not total system memory usage
and should not be read as "the model needs 1.2GB of RAM."

## Limitations

- These are microbenchmarks of forward/backward/optimizer-step mechanics on synthetic random
  token IDs, not a projection of real pretraining throughput (which depends on data loading,
  gradient accumulation strategy, learning-rate schedule overhead, logging, and eventual
  tokenizer/dataset specifics that don't exist yet).
- No FlashAttention-2-specific kernel is guaranteed on Turing-generation hardware; SDPA's
  fallback kernel path is what's actually exercised here (see ADR 0009).
- Inference throughput has no KV cache; it is full-forward reference throughput only.
