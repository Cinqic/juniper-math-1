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

Python `>=3.12,<3.13`, as declared in [`pyproject.toml`](../pyproject.toml).
Narrowed from `>=3.10` during Opus 5 Phase 0 remediation so that declared
support equals *tested* support: the validated dependency set requires 3.12
(`numpy` 2.5.2 needs >= 3.12), so installing on 3.10/3.11 would silently
resolve a different environment.

On Debian, Ubuntu, and Linux Mint, venv and pip support ship separately from
the interpreter and must be installed explicitly:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
```

Without `python3-venv`, `python3 -m venv` fails with
`ensurepip is not available`. `scripts/bootstrap.sh` detects this up front and
tells you what to install; it never runs `sudo` itself.

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

# Exact validated environment (preferred — see requirements-lock.txt):
pip install -r requirements-lock.txt
pip install -e . --no-deps

# Or resolve from pyproject.toml ranges (only when testing upgrades):
# pip install -e ".[dev]"
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
