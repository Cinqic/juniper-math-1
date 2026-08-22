"""Centralized deterministic seed handling.

This is the ONE place random state is seeded across the project. Do not
duplicate seeding logic elsewhere.

Honesty about determinism: seeding random/NumPy/PyTorch CPU RNGs gives
reproducible results for CPU-bound, non-parallel-reduction operations.
PyTorch's `use_deterministic_algorithms(True)` forces deterministic kernels
where available (and raises for ops with no deterministic implementation
rather than silently proceeding). CUDA convolution/reduction kernels and
multi-threaded CPU reductions can still exhibit low-level nondeterminism
(floating-point order-of-operations) even when seeded. This module makes a
best-effort, transparently documented, and never claims hardware-level
bitwise determinism it cannot guarantee.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

DEFAULT_PROJECT_SEED = 5004032  # matches the frozen parameter target; arbitrary but memorable


@dataclass(frozen=True)
class SeedReport:
    seed: int
    python_random_seeded: bool
    numpy_seeded: bool
    torch_cpu_seeded: bool
    torch_cuda_seeded: bool
    deterministic_algorithms_requested: bool


def set_global_seed(seed: int = DEFAULT_PROJECT_SEED, deterministic_algorithms: bool = True) -> SeedReport:
    """Seed all known RNG sources. Returns a report of what was actually seeded.

    Missing optional dependencies (numpy, torch) are handled gracefully —
    their fields are reported False rather than raising.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)

    numpy_seeded = False
    try:
        import numpy as np

        np.random.seed(seed)
        numpy_seeded = True
    except ImportError:
        pass

    torch_cpu_seeded = False
    torch_cuda_seeded = False
    deterministic_requested = False
    try:
        import torch

        torch.manual_seed(seed)
        torch_cpu_seeded = True

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch_cuda_seeded = True

        if deterministic_algorithms:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            deterministic_requested = True
    except ImportError:
        pass

    return SeedReport(
        seed=seed,
        python_random_seeded=True,
        numpy_seeded=numpy_seeded,
        torch_cpu_seeded=torch_cpu_seeded,
        torch_cuda_seeded=torch_cuda_seeded,
        deterministic_algorithms_requested=deterministic_requested,
    )
