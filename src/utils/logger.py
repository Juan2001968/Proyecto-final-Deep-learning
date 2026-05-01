"""Logging estructurado y consistente para todo el proyecto."""

from __future__ import annotations

import logging
import sys
from logging import Logger

_FORMAT = "[%(asctime)s] %(levelname)-7s %(name)s :: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: str | int = "INFO") -> Logger:
    """Devuelve un logger con handler único a stdout y formato consistente."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    logger.addHandler(handler)
    logger.setLevel(level if isinstance(level, int) else level.upper())
    logger.propagate = False
    return logger
