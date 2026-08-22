from __future__ import annotations

import random

import pytest

from juniper_math.seed import DEFAULT_PROJECT_SEED, set_global_seed


def test_default_seed_is_deterministic():
    set_global_seed(DEFAULT_PROJECT_SEED)
    first = [random.random() for _ in range(5)]
    set_global_seed(DEFAULT_PROJECT_SEED)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_different_seeds_diverge():
    set_global_seed(1)
    a = [random.random() for _ in range(5)]
    set_global_seed(2)
    b = [random.random() for _ in range(5)]
    assert a != b


def test_negative_seed_rejected():
    with pytest.raises(ValueError):
        set_global_seed(-1)


def test_report_reflects_available_backends():
    report = set_global_seed(42)
    assert report.seed == 42
    assert report.python_random_seeded is True
    # numpy/torch may or may not be installed in a minimal environment;
    # the report must not lie about it either way.
    try:
        import numpy  # noqa: F401

        assert report.numpy_seeded is True
    except ImportError:
        assert report.numpy_seeded is False

    try:
        import torch  # noqa: F401

        assert report.torch_cpu_seeded is True
        assert report.torch_cuda_seeded == torch.cuda.is_available()
    except ImportError:
        assert report.torch_cpu_seeded is False
