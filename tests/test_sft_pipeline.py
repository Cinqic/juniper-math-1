"""Phase 8 approval-candidate provenance guards."""

from __future__ import annotations

import pytest

from juniper_math.errors import JuniperConfigError
from juniper_math.sft_pipeline import require_clean_source_tree


def test_clean_source_tree_is_accepted():
    require_clean_source_tree("a" * 40, "clean")


@pytest.mark.parametrize("commit,state", [("a" * 40, "dirty"), ("unknown", "clean"), ("a" * 40, "unknown")])
def test_non_reproducible_source_tree_is_rejected(commit, state):
    with pytest.raises(JuniperConfigError, match="clean committed source tree"):
        require_clean_source_tree(commit, state)
