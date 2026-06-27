"""Configuracion centralizada de logging."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configura el root logger con formato estructurado simple."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    return logging.getLogger("jarvis_bot")
