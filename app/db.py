from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from loguru import logger

from app.config import Settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    title       TEXT,
    preset      TEXT,
    status      TEXT NOT NULL,           -- pending|analyzing|running|done|failed|skipped|cancelled
    progress    REAL DEFAULT 0,
    stage       TEXT DEFAULT '',
    progress_fps    REAL DEFAULT 0,      -- av1an live stats (for the UI)
    progress_done   INTEGER DEFAULT 0,
    progress_total  INTEGER DEFAULT 0,
    error       TEXT,
    meta        TEXT,                    -- JSON probe info
    params      TEXT,                    -- JSON transcode params snapshot
    rpu_path    TEXT,
    output_path TEXT,
    size_before INTEGER DEFAULT 0,
    size_after  INTEGER DEFAULT 0,
    retries     INTEGER DEFAULT 0,
    created_at  REAL,
    started_at  REAL,
    finished_at REAL,
    finalized_at REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source);
"""

# statuses
PENDING = "pending"
ANALYZING = "analyzing"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"
CANCELLED = "cancelled"
ACTIVE = (PENDING, ANALYZING, RUNNING)


def loads(raw: Any) -> Any:
    """Parse a stored JSON column back to a dict; None passthrough."""
    if raw is None:
        return raw
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if raw else raw
    except (TypeError, json.JSONDecodeError):
        return raw


class JobStore:
    """JSON-fallback + option
    Backed by SQLite (default) or single JSON document when sqlite unavailable."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._open_sqlite()

    def _open_sqlite(self) -> None:
        db_path: Path = self.settings.dirs.db
        # Resolve db dir; if the configured path is unwritable (e.g. container
        # path on a dev machine), fall back to in-memory sqlite.
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(SCHEMA)
            logger.warning("DB path {} not writable; using in-memory sqlite", db_path)
            return
        try:
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(SCHEMA)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._migrate_schema()
        except Exception as e:  # pragma: no cover
            logger.warning("SQLite unavailable ({}); falling back to in-memory store", e)
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(SCHEMA)

    def _migrate_schema(self) -> None:
        """Add columns introduced after the first release (CREATE TABLE IF NOT
        EXISTS does not alter existing tables)."""
        if self._conn is None:
            return
        try:
            cur = self._conn.execute("PRAGMA table_info(jobs)")
            have = {r[1] for r in cur.fetchall()}
            if "progress_fps" not in have:
                self._conn.execute("ALTER TABLE jobs ADD COLUMN progress_fps REAL DEFAULT 0")
            if "progress_done" not in have:
                self._conn.execute("ALTER TABLE jobs ADD COLUMN progress_done INTEGER DEFAULT 0")
            if "progress_total" not in have:
                self._conn.execute("ALTER TABLE jobs ADD COLUMN progress_total INTEGER DEFAULT 0")
            self._conn.commit()
        except sqlite3.Error as e:  # pragma: no cover
            logger.warning("Schema migration skipped: {}", e)

    @contextmanager
    def _cursor(self) -> Generator[Any, None, None]:
        if self._conn is None:
            raise RuntimeError("no backend")
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            finally:
                cur.close()

    def create(self, *, source: str, preset: str, meta: Optional[Dict] = None,
               overrides: Optional[Dict] = None) -> str:
        jid = str(uuid.uuid4())
        params = json.dumps({"source": source, "preset": preset,
                             "overrides": overrides or {}}, default=str)
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO jobs (id, source, preset, status, meta, params, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (jid, source, preset, PENDING, json.dumps(meta or {}, default=str), params, time.time()),
            )
            self._conn.commit()
        return jid

    def update(self, jid: str, *, only_from: Optional[tuple] = None, **fields: Any) -> bool:
        """Write columns of one job. With only_from, the write happens only
        while the job's status is still one of those - a worker moving its
        job on must not write over a cancel that landed in the meantime.
        Returns whether a row was written."""
        if not fields:
            return False
        # NB: these must be real column names - anything else is dropped without
        # a word. "size_bytes" used to sit here in place of "size_before", so
        # every job stored 0 for its source size and the API reported 0 too.
        allowed = {"status", "progress", "stage", "error", "meta", "params", "rpu_path",
                   "output_path", "size_before", "size_after", "retries", "started_at",
                   "finished_at", "finalized_at",
                   "progress_fps", "progress_done", "progress_total"}
        f = {k: v for k, v in fields.items() if k in allowed}
        if not f:
            return False
        cols = ", ".join(f"{k}=?" for k in f)
        if "meta" in f:
            f["meta"] = json.dumps(f["meta"], default=str)
        if "params" in f:
            f["params"] = json.dumps(f["params"], default=str)
        vals = list(f.values())
        where, args = "id=?", [jid]
        if only_from:
            where += f" AND status IN ({','.join('?' * len(only_from))})"
            args += list(only_from)
        with self._cursor() as cur:
            cur.execute(f"UPDATE jobs SET {cols} WHERE {where}", [*vals, *args])
            n = cur.rowcount
            self._conn.commit()
        return n > 0

    def get(self, jid: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM jobs WHERE id=?", (jid,))
            row = cur.fetchone()
            return dict(row) if row else None

    def list(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        q = "SELECT * FROM jobs"
        args: list = []
        if status:
            q += " WHERE status=?"
            args.append(status)
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._cursor() as cur:
            cur.execute(q, args)
            return [dict(r) for r in cur.fetchall()]

    def find_active(self, source: str) -> Optional[str]:
        """Id of the active (pending/analyzing/running) job for `source`, if any.

        A targeted query rather than a scan of whatever list() returns. The
        caller is de-duplication, and list() defaults to the 100 NEWEST jobs -
        so once the queue was longer than that, the oldest pending jobs were
        invisible to it and the same file could be enqueued twice.

        That is not a corner case: a restart resets every running/analyzing job
        to pending (see reset_interrupted) and the watcher's in-memory seen-set
        starts empty, so the next scan re-submits the whole input directory.
        Any library with more than 100 files waiting therefore duplicated its
        tail on every restart - two jobs encoding the same source to the same
        output path, and with delete_source the second one failing its analysis
        because the first had already removed the file.
        """
        ph = ",".join("?" * len(ACTIVE))
        with self._cursor() as cur:
            cur.execute(
                f"SELECT id FROM jobs WHERE source=? AND status IN ({ph}) "
                "ORDER BY created_at ASC LIMIT 1",
                (source, *ACTIVE),
            )
            row = cur.fetchone()
            return row["id"] if row else None

    def has_job(self, source: str) -> bool:
        """Whether ANY job exists for `source`, in any state.

        find_active answers "is it in flight", which is the right question for
        a manual re-submit. This answers "has this system ever handled it",
        which is the right question for the watcher: a file it has already
        transcoded must not be picked up again just because the process
        restarted. See TranscodeManager.enqueue_new_file.
        """
        with self._cursor() as cur:
            cur.execute("SELECT 1 FROM jobs WHERE source=? LIMIT 1", (source,))
            return cur.fetchone() is not None

    def next_pending(self) -> Optional[str]:
        """Atomically claim the oldest pending job (sets it analyzing) so
        concurrent workers never pick the same job."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT id FROM jobs WHERE status=? ORDER BY created_at ASC LIMIT 1",
                (PENDING,),
            )
            row = cur.fetchone()
            if not row:
                return None
            jid = row["id"]
            cur.execute(
                "UPDATE jobs SET status=?, stage='' WHERE id=? AND status=?",
                (ANALYZING, jid, PENDING),
            )
            self._conn.commit()
            return jid if cur.rowcount == 1 else None

    def count_by_status(self) -> Dict[str, int]:
        out = {s: 0 for s in ("pending", "analyzing", "running", "done", "failed", "skipped", "cancelled")}
        with self._cursor() as cur:
            cur.execute("SELECT status, COUNT(*) c FROM jobs GROUP BY status")
            for r in cur.fetchall():
                out[r["status"]] = r["c"]
        return out

    def cancel_pending(self) -> int:
        with self._cursor() as cur:
            cur.execute("UPDATE jobs SET status=?, stage='cancelled', finished_at=? "
                        "WHERE status IN (?,?)",
                        (CANCELLED, time.time(), PENDING, ANALYZING))
            n = cur.rowcount
            self._conn.commit()
            return n

    def reset_interrupted(self) -> int:
        """On startup: jobs stuck in running/analyzing belonged to a previous
        process (crash/restart). Reset them to pending so they get picked up."""
        with self._cursor() as cur:
            cur.execute("UPDATE jobs SET status=?, stage='', started_at=NULL "
                        "WHERE status IN (?,?)",
                        (PENDING, ANALYZING, RUNNING))
            n = cur.rowcount
            self._conn.commit()
            return n

    def prune(self, statuses: Optional[List[str]] = None) -> int:
        """Hard-delete finished jobs (done/failed/skipped/cancelled)."""
        allowed = {DONE, FAILED, SKIPPED, CANCELLED}
        statuses = [s for s in (statuses or list(allowed)) if s in allowed]
        if not statuses:
            return 0
        ph = ",".join("?" * len(statuses))
        with self._cursor() as cur:
            cur.execute(f"DELETE FROM jobs WHERE status IN ({ph})", statuses)
            n = cur.rowcount
            self._conn.commit()
            return n

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass