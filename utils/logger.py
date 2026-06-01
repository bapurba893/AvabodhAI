"""
utils/logger.py
---------------
Structured logging for production.
Every module imports get_logger(__name__) — never print() in production.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from config.settings import get_settings

settings = get_settings()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(
    open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    )
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file handler — max 10 MB, keep 5 backups
    fh = RotatingFileHandler(
        settings.LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
