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
        except Exception as e:  # pragma: no cover
            logger.warning("SQLite unavailable ({}); falling back to in-memory store", e)
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(SCHEMA)

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

    def update(self, jid: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {"status", "progress", "stage", "error", "meta", "params", "rpu_path",
                   "output_path", "size_bytes", "size_after", "retries", "started_at",
                   "finished_at", "finalized_at"}
        f = {k: v for k, v in fields.items() if k in allowed}
        if not f:
            return
        cols = ", ".join(f"{k}=?" for k in f)
        if "meta" in f:
            f["meta"] = json.dumps(f["meta"], default=str)
        if "params" in f:
            f["params"] = json.dumps(f["params"], default=str)
        vals = list(f.values())
        with self._cursor() as cur:
            cur.execute(f"UPDATE jobs SET {cols} WHERE id=?", [*vals, jid])
            self._conn.commit()

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

    def next_pending(self) -> Optional[str]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT id FROM jobs WHERE status=? ORDER BY created_at ASC LIMIT 1",
                (PENDING,),
            )
            row = cur.fetchone()
            return row["id"] if row else None

    def count_by_status(self) -> Dict[str, int]:
        out = {s: 0 for s in ("pending", "analyzing", "running", "done", "failed", "skipped", "cancelled")}
        with self._cursor() as cur:
            cur.execute("SELECT status, COUNT(*) c FROM jobs GROUP BY status")
            for r in cur.fetchall():
                out[r["status"]] = r["c"]
        return out

    def cancel_pending(self) -> int:
        with self._cursor() as cur:
            cur.execute("UPDATE jobs SET status=? WHERE status IN (?,?)",
                        (CANCELLED, PENDING, ANALYZING))
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