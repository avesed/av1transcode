from __future__ import annotations

import logging
import sys
import time
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


def cleanup_old_job_logs(settings: Settings, retention_days: Optional[int] = None) -> int:
    """Delete job_*.log files older than retention_days (0/None->config).

    The job logs are written unbounded (one per job, can be tens of MB for a
    long optimizer run), so prune them on startup and periodically. Returns
    the number of files removed.
    """
    days = settings.logging.retention_days if retention_days is None else retention_days
    if not days or days <= 0:
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    logs_dir = settings.dirs.logs
    if not logs_dir.is_dir():
        return 0
    for p in logs_dir.glob("job_*.log"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def prune_job_logs(settings: Settings, keep_ids: Optional[set] = None) -> int:
    """Delete job_*.log files whose job id is not in keep_ids (or all of them)."""
    removed = 0
    logs_dir = settings.dirs.logs
    if not logs_dir.is_dir():
        return 0
    for p in logs_dir.glob("job_*.log"):
        jid = p.stem[len("job_"):]
        if keep_ids is not None and jid in keep_ids:
            continue
        try:
            p.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def get_logger(name: str = "app"):
    return logger.bind(name=name)