from __future__ import annotations

import logging

from juniper_math.logging_utils import configure_logging, get_logger


def test_get_logger_returns_namespaced_logger():
    logger = get_logger("juniper_math.test_module")
    assert logger.name == "juniper_math.test_module"


def test_configure_logging_is_idempotent():
    configure_logging()
    configure_logging()
    root = logging.getLogger("juniper_math")
    # No duplicated handlers even after multiple configure calls.
    assert len(root.handlers) <= 2  # console (+ optional file)


def test_logger_does_not_propagate_to_root():
    configure_logging()
    root = logging.getLogger("juniper_math")
    assert root.propagate is False
