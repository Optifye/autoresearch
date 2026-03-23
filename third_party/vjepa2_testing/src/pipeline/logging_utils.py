"""Logging helpers for the classifier pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src import constants


RESET = "\033[0m"
COLOR_BY_LEVEL = {
    logging.DEBUG: "\033[36m",  # Cyan
    logging.INFO: "\033[32m",  # Green
    logging.WARNING: "\033[33m",  # Yellow
    logging.ERROR: "\033[31m",  # Red
    logging.CRITICAL: "\033[35m",  # Magenta
}


class _ColorFormatter(logging.Formatter):
    def __init__(self, fmt: str, datefmt: str, use_color: bool):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting
        msg = super().format(record)
        if not self.use_color:
            return msg
        color = COLOR_BY_LEVEL.get(record.levelno)
        return f"{color}{msg}{RESET}" if color else msg


def setup_logger() -> logging.Logger:
    """Configure and return the pipeline logger with colored output."""

    log_dir = Path(constants.LOG_FILENAME).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("pipeline")
    logger.setLevel(getattr(logging, constants.LOG_LEVEL.upper(), logging.INFO))

    if logger.handlers:
        return logger

    fmt = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        _ColorFormatter(fmt=fmt, datefmt=datefmt, use_color=sys.stdout.isatty())
    )

    file_handler = logging.FileHandler(constants.LOG_FILENAME, mode="a")
    file_handler.setFormatter(logging.Formatter(fmt=f"{fmt} | %(filename)s:%(lineno)d", datefmt=datefmt))

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    logger.info("Logger initialized. Logs -> %s", constants.LOG_FILENAME)
    return logger
