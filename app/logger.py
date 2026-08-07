from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from app.config import Settings

_LOG = logging.getLogger("app")


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except (ValueError, KeyError):
            level = record.levelno
        frame, depth = logging.currentframe(), 0
        while frame is not None and depth < 10:
            if frame.f_code.co_filename != __file__:
                break
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(settings: Settings, log_file: Optional[Path] = None) -> None:
    level = getattr(logging, settings.logging.level.upper(), logging.INFO)
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    if settings.logging.file:
        file_path = log_file or (settings.dirs.logs / "av1transcode.log")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(str(file_path), level=level, rotation="100 MB", retention="30 days",
                   enqueue=True, serialize=False)
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.WARNING)
    logger.opt(colors=True)


def get_logger(name: str = "app"):
    return logger.bind(name=name)