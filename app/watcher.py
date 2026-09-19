from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from app import analyzer
from app.config import Settings



# Directories never fed back into the queue: our own output when it is
# source-relative (<input>/av1/, and its rpu/), and temp work.
_OWN_DIRS = {"av1", "work", "rpu"}


def input_videos(settings: Settings, recursive: bool = True) -> list[Path]:
    """Video files under dirs.input that could be sources: the watcher's scan
    and `cli scan` both list them here, so neither walks into av1/."""
    inp = settings.dirs.input
    results: list[Path] = []
    for p in sorted(inp.rglob("*")) if recursive else sorted(inp.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") not in settings.watcher.extensions:
            continue
        if any(part in _OWN_DIRS for part in p.relative_to(inp).parts[:-1]):
            continue
        results.append(p)
    return results

class FileWatcher:
    """Watch a directory for new/staged video files and feed them to the queue.

    Uses a simple polling approach (robust across network mounts) with a
    pending set, plus optional inotify via 'watchdog' when available.
    """

    def __init__(
        self,
        settings: Settings,
        # Returns the new job id, or None when the file was not enqueued -
        # which TranscodeManager.enqueue_new_file does for anything this system
        # has handled before. The watcher does not care either way, but the
        # annotation said None and both callers have always returned an id.
        submit: Callable[[str], Optional[str]],
    ) -> None:
        self.settings = settings
        self.submit = submit
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Paths this process has already handed to submit(). In-process only;
        # the durable answer to "have we seen this file" is the job table, via
        # TranscodeManager.enqueue_new_file.
        self._submitted: set[str] = set()
        # Last (size, mtime) each candidate was seen with, so the next scan can
        # be the second observation instead of a sleep. See _maybe_submit.
        self._last_seen: dict[str, tuple[int, float]] = {}
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
        return input_videos(self.settings, self.settings.watcher.recursive)

    def _maybe_submit(self, p: Path) -> None:
        """Submit `p` once it has stopped changing.

        Stability is two observations that agree, and the second one is the
        NEXT scan rather than a sleep inside this call. The old version slept
        two seconds between its two stats, with the scan loop blocked behind
        it - one file at a time, so a batch of fifty cost a hundred seconds of
        a thread doing nothing. Comparing across scan cycles is both free and a
        longer settling window than the two seconds it replaces.
        """
        key = str(p)
        with self._lock:
            if key in self._submitted:
                return
        seen = analyzer.fingerprint(key)
        if seen is None:
            return
        size, mtime = seen
        if size < self.settings.watcher.min_size_mb * 1024 * 1024:
            return
        if not analyzer.settled_for(mtime, self.settings.watcher.stable_seconds):
            with self._lock:
                self._last_seen[key] = seen
            return
        with self._lock:
            previous = self._last_seen.get(key)
            if previous != seen:
                # First sighting, or it changed since the last scan. Either way
                # this scan is observation one; the next is observation two.
                self._last_seen[key] = seen
                return
            self._submitted.add(key)
            self._last_seen.pop(key, None)
        try:
            logger.info("New stable media file detected: {}", p)
            self.submit(key)
        except Exception as e:  # noqa: BLE001
            logger.debug("watcher submit error: {}", e)
            with self._lock:
                self._submitted.discard(key)