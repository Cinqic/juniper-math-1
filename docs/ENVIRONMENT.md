# Environment Setup

## Target hardware

Juniper Math 1's primary development/training hardware is deliberately
modest — infrastructure decisions must remain practical on it, not assume
datacenter resources:

- CPU: AMD Ryzen 7 5700G
- GPU: NVIDIA GeForce RTX 2060 (6 GB VRAM)
- System RAM: 16 GB
- Storage: 256 GB NVMe

## Supported Python

Python `>=3.10,<3.13`, as declared in [`pyproject.toml`](../pyproject.toml).

## System-level assumptions (not covered by Python packaging)

Python package pinning does **not** reproduce NVIDIA drivers or the CUDA
toolkit. These must be installed at the OS level and are **not** guaranteed
reproducible purely from this repository:

- An NVIDIA driver compatible with the PyTorch CUDA build in use (PyTorch
  wheels bundle their own CUDA runtime; a recent driver from the 535+ series
  or later is expected to work with current PyTorch CUDA 12.x wheels).
- `nvidia-smi` should report the GPU when the driver is correctly installed.

## Creating the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

## Validating the environment

```bash
python -m juniper_math validate-env
```

This reports PASS/WARNING/FAIL per check. CUDA unavailability is a
**WARNING**, not a FAIL — CPU-only machines can still run configuration
validation and unit tests. A FAIL indicates an unsupported Python version or
a missing required package (PyTorch, PyYAML).

The command never fabricates GPU availability: if CUDA is not detected, it
is reported as unavailable, plainly.
