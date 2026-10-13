"""Logging helper (no print statements in library code)."""
from __future__ import annotations

import logging
import os


def get_logger(name: str) -> logging.Logger:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=os.environ.get("FAIRTRAFFIC_LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    return logging.getLogger(name)
