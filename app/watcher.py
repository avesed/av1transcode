from __future__ import annotations

import queue as queue_mod
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from app import analyzer
from app.config import Settings


class FileWatcher:
    """Watch a directory for new/staged video files and feed them to the queue.

    Uses a simple polling approach (robust across network mounts) with a
    pending set, plus optional inotify via 'watchdog' when available.
    """

    def __init__(
        self,
        settings: Settings,
        submit: Callable[[str], None],
    ) -> None:
        self.settings = settings
        self.submit = submit
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            self._observer_cls = Observer
            self._handler_cls = FileSystemEventHandler
            self._observer = None
        except ImportError:
            self._observer = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="file-watcher", daemon=True)
        self._thread.start()
        logger.info("File watcher started on {}", self.settings.dirs.input)

    def stop(self) -> None:
        self._stop.set()
        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=3)
            except Exception:  # noqa: BLE001
                pass
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self) -> None:
        inp = self.settings.dirs.input
        inp.mkdir(parents=True, exist_ok=True)
        while not self._stop.is_set():
            try:
                found = self._scan()
                for f in found:
                    self._maybe_submit(f)
            except Exception as e:  # noqa: BLE001
                logger.warning("watcher scan error: {}", e)
            self._stop.wait(5.0)

    def _scan(self) -> list[Path]:
        inp = self.settings.dirs.input
        results: list[Path] = []
        pattern = self.settings.watcher.extensions
        # Directories we must never feed back into the queue (our own output
        # dirs when output is source-relative: `<input>/av1/`, temp work, rpu)
        exclude = {"av1", "work", "rpu"}
        for p in inp.rglob("*") if self.settings.watcher.recursive else sorted(inp.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower().lstrip(".") not in pattern:
                continue
            if any(part in exclude for part in p.relative_to(inp).parts[:-1]):
                continue
            results.append(p)
        return results

    def _maybe_submit(self, p: Path) -> None:
        try:
            if p.stat().st_size < self.settings.watcher.min_size_mb * 1024 * 1024:
                return
        except OSError:
            return
        key = str(p)
        with self._lock:
            if key in self._pending:
                return
            self._pending.add(key)
        try:
            if analyzer.is_stable(key, self.settings.watcher.stable_seconds):
                logger.info("New stable media file detected: {}", p)
                self.submit(key)
            else:
                with self._lock:
                    self._pending.discard(key)
        except Exception as e:  # noqa: BLE001
            logger.debug("watcher submit error: {}", e)
            with self._lock:
                self._pending.discard(key)