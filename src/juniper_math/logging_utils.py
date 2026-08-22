"""Canonical logging setup. The one place logging is configured project-wide.

Uses the Python standard `logging` module — no external dependency is
justified for a 5M-parameter research project's console/file logging.
"""

from __future__ import annotations

import logging
from pathlib import Path

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


def configure_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Idempotent: calling this more than once will not duplicate handlers."""
    global _CONFIGURED
    root = logging.getLogger("juniper_math")
    root.setLevel(level)

    if not _CONFIGURED:
        formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

        root.propagate = False
        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
