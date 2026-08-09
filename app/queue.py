from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from app import analyzer as analyzer_mod
from app import db, decisions
from app.config import Settings
from app.transcoder import run_full_transcode

ACTIVE = db.ACTIVE


class TranscodeManager:
    """Owns the job queue, spawns worker threads, tracks status in DB."""

    def __init__(self, settings: Settings, store: "db.JobStore") -> None:
        self.settings = settings
        self.store = store
        self._condition = threading.Condition()
        self._running = False
        self._threads: list[threading.Thread] = []
        self._next_idx = 0
        self._cancel: set[str] = set()
        self._shrink_to: Optional[int] = None
        self.on_event: Optional[Callable[[str, str], None]] = None

    # ---------------- control ----------------
    def start(self) -> None:
        with self._condition:
            if self._running:
                return
            self._running = True
        n = max(1, self.settings.workers.concurrency)
        recovered = self.store.reset_interrupted()
        if recovered:
            logger.info("Recovered {} job(s) stuck in running/analyzing -> pending", recovered)
        self._spawn_workers(n)
        logger.info("Started {} transcode worker(s)", n)

    def _spawn_workers(self, count: int) -> None:
        with self._condition:
            for _ in range(count):
                idx = self._next_idx
                self._next_idx += 1
                t = threading.Thread(target=self._worker_loop, args=(idx,),
                                     name=f"worker-{idx}", daemon=True)
                self._threads.append(t)
                t.start()

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        for t in self._threads:
            t.join(timeout=5)
        logger.info("Transcode workers stopped")

    @property
    def running(self) -> bool:
        return self._running

    def worker_count(self) -> int:
        return len(self._threads)

    # ---------------- submission ----------------
    def enqueue_file(self, source: str, preset: str = "",
                     overrides: Optional[dict] = None) -> Optional[str]:
        """Enqueue a file (or a whole directory). Returns first job id created."""
        src = Path(source)
        if not src.exists():
            logger.error("Cannot enqueue missing file: {}", source)
            return None
        if src.is_dir():
            jids = []
            for child in sorted(src.iterdir()):
                if self._is_video(child):
                    jid = self._enqueue_single(child, preset, overrides)
                    if jid:
                        jids.append(jid)
            if jids:
                logger.info("Enqueued {} file(s) from directory {}", len(jids), src)
            return jids[0] if jids else None
        return self._enqueue_single(src, preset, overrides)

    def _is_video(self, p: Path) -> bool:
        return p.is_file() and p.suffix.lower().lstrip(".") in self.settings.watcher.extensions

    def _enqueue_single(self, src: Path, preset: str, overrides: Optional[dict] = None) -> Optional[str]:
        # de-duplicate: only one active job per source
        for existing in self.store.list(status=None):
            if existing["source"] == str(src) and existing["status"] in ACTIVE:
                logger.info("File already queued/active: {}", src)
                return existing["id"]
        jid = self.store.create(
            source=str(src), preset=preset or self.settings.transcode.default_preset,
            overrides=overrides,
        )
        logger.info("Enqueued '{}' -> job {}", src.name, jid)
        with self._condition:
            self._condition.notify_all()
        return jid

    def cancel(self, jid: Optional[str] = None) -> int:
        if jid:
            job = self.store.get(jid)
            if not job:
                return 0
            status = job.get("status")
            # pending/analyzing: mark cancelled immediately so a worker never picks it up
            if status not in (db.RUNNING,):
                self.store.update(jid, status=db.CANCELLED, stage="cancelled",
                                  finished_at=time.time())
                self._emit(jid, "cancelled")
                logger.info("Cancelled job {} (status={})", jid, status)
                return 1
            # running: request async abort at the next check
            self._cancel.add(jid)
            self.store.update(jid, stage="cancelling")
            logger.info("Cancel requested for running job {}", jid)
            return 1
        n = self.store.cancel_pending()
        logger.info("Cancelled {} pending job(s)", n)
        return n

    # ---------------- worker loop ----------------
    def _worker_loop(self, idx: int) -> None:
        while True:
            with self._condition:
                if not self._running:
                    return
                if self._retire_self(idx):
                    return
                jid = self.store.next_pending()
                if not jid:
                    self._condition.wait(timeout=2)
                    continue
            logger.debug("worker-{} picked job {}", idx, jid)
            self._process(jid)
            if self._retire_self(idx):
                return

    def _retire_self(self, idx: int) -> bool:
        """Exit this worker when the pool is shrinking past its target.
        Returns True when this thread should stop."""
        with self._condition:
            if self._shrink_to is not None and idx >= self._shrink_to:
                try:
                    self._threads.remove(threading.current_thread())
                except ValueError:
                    pass
                return True
            return False

    def queued_count(self) -> int:
        return self.store.count_by_status().get("pending", 0)

    def set_concurrency(self, n: int) -> int:
        """Dynamically grow/shrink the running worker pool. Returns new count."""
        n = max(1, int(n))
        with self._condition:
            current = len(self._threads)
            if n > current:
                self._spawn_workers(n - current)
                logger.info("Grew worker pool to {}", n)
            elif n < current:
                self._shrink_to = n
                self._condition.notify_all()
                logger.info("Shrinking worker pool to {} (in-flight jobs finish)", n)
            else:
                self._shrink_to = None
            self._condition.notify_all()
        return len(self._threads)

    # ---------------- processing ----------------
    def _process(self, jid: str) -> None:
        job = self.store.get(jid)
        if not job:
            return
        source = Path(job["source"])
        preset = job["preset"]
        overrides = None
        raw_params = job.get("params")
        if raw_params:
            overrides = (db.loads(raw_params) or {}).get("overrides")
        self.store.update(jid, status=db.ANALYZING, stage="analyzing")
        self._emit(jid, "analyzing")

        try:
            logger.debug("analyzing {} (job {})", source, jid)
            info = analyzer_mod.analyze(self.settings, str(source))
            if jid in self._cancel:
                self._cancel.discard(jid)
                self.store.update(jid, status=db.CANCELLED, stage="cancelled",
                                  finished_at=time.time())
                self._emit(jid, "cancelled")
                logger.info("Job {} cancelled during analysis", jid)
                return
            if info is None:
                logger.warning("analysis returned None for {} (job {})", source, jid)
                self._fail(jid, "analysis failed", job)
                return
            plan = decisions.decide_action(self.settings, info, preset, overrides)
            if plan.skip:
                self.store.update(
                    jid,
                    status=db.SKIPPED,
                    meta=self._meta(info),
                    stage="skipped",
                    error=plan.skip_reason,
                    finished_at=time.time(),
                )
                self._emit(jid, "skipped")
                logger.info("Skip {}: {}", source.name, plan.skip_reason)
                return

            self.store.update(
                jid, status=db.RUNNING, stage="encoding",
                meta=self._meta(info), started_at=time.time(),
                size_before=source.stat().st_size,
                rpu_path=str(plan.rpu_path) if plan.rpu_path else "",
            )
            self._emit(jid, "running")

            out = plan.output_path or (self.settings.dirs.output or source.parent / "av1") / f"{info.path.stem}.av1.mkv"
            out.parent.mkdir(parents=True, exist_ok=True)
            if plan.rpu_path:
                plan.rpu_path.parent.mkdir(parents=True, exist_ok=True)
            log_path = self.settings.dirs.logs / f"job_{jid}.log"

            def progress_cb(pct: float, stats: Optional[dict] = None) -> None:
                # stage is owned by stage_cb: progress updates must not
                # overwrite "scenedetect" back to "encoding"
                fields = {"progress": round(pct, 1)}
                if stats:
                    fields["progress_fps"] = stats.get("fps") or 0
                    fields["progress_done"] = stats.get("done") or 0
                    fields["progress_total"] = stats.get("total") or 0
                self.store.update(jid, **fields)

            def stage_cb(stage: str) -> None:
                self.store.update(jid, stage=stage)

            def cancel_flag() -> bool:
                return jid in self._cancel

            run_full_transcode(
                self.settings, info, plan, source, out, log_path,
                progress_cb=progress_cb, cancel_flag=cancel_flag, stage_cb=stage_cb,
            )

            if jid in self._cancel:
                self._cancel.remove(jid)
                self.store.update(jid, status=db.CANCELLED, finished_at=time.time())
                if out.exists():
                    out.unlink()
                self._emit(jid, "cancelled")
                logger.info("Job {} cancelled", jid)
                return

            self.store.update(
                jid, status=db.DONE, stage="done",
                output_path=str(out), size_after=out.stat().st_size,
                progress=100.0, finished_at=time.time(),
            )
            self._emit(jid, "done")
            logger.info("Job {} done: {}", jid, out)

            self._postprocess(jid, source)

        except Exception as e:  # noqa: BLE001
            if jid in self._cancel:
                # cancelled mid-encode: mark cancelled, not failed
                self._cancel.discard(jid)
                self.store.update(jid, status=db.CANCELLED, error=str(e),
                                 stage="cancelled", finished_at=time.time())
                self._emit(jid, "cancelled")
                logger.info("Job {} cancelled during encoding", jid)
            else:
                self._fail(jid, str(e), job)

    def _fail(self, jid: str, error: str, job: dict) -> None:
        if jid in self._cancel:
            self._cancel.discard(jid)
            self.store.update(jid, status=db.CANCELLED, error=error,
                              stage="cancelled", finished_at=time.time())
            self._emit(jid, "cancelled")
            logger.info("Job {} cancelled (was failing: {})", jid, error)
            return
        retries = int(job.get("retries") or 0)
        if retries < self.settings.workers.max_retries:
            self.store.update(jid, retries=retries + 1, error=error,
                             status=db.PENDING, stage="retry",
                             started_at=None)
            logger.warning("Job {} failed ({}) - will retry ({}/{})",
                          jid, error, retries + 1, self.settings.workers.max_retries)
        else:
            self.store.update(jid, status=db.FAILED, error=error,
                             stage="failed", finished_at=time.time())
            logger.error("Job {} failed after retries: {}", jid, error)
        self._emit(jid, "failed")

    def _postprocess(self, jid: str, source: Path) -> None:
        """Optional: delete the source after a successful encode."""
        if self.settings.transcode.delete_source:
            try:
                source.unlink()
                logger.info("Deleted source after success: {}", source)
            except OSError as e:
                logger.warning("Could not delete source {}: {}", source, e)
        # archive functionality removed - source files are kept in place
        # output already written to av1/ subdirectory

    def _emit(self, jid: str, event: str) -> None:
        if self.on_event:
            try:
                self.on_event(jid, event)
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _meta(info) -> dict:
        return {
            "display": info.display,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "codec": info.video_codec,
            "duration": info.duration,
            "size": info.size,
            "audio_count": info.audio_count,
            "is_hdr": info.is_hdr,
            "is_hlg": info.is_hlg,
            "dovi": {"present": info.dovi.present, "profile": info.dovi.profile},
        }