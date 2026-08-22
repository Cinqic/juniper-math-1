from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if cuda_available:
        return

    skip_gpu = pytest.mark.skip(reason="CUDA not available on this machine")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
