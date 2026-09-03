"""Queue, work-directory lifecycle, settings persistence and API auth.

These areas had no coverage at all, which is where every defect exercised
below was living.
"""

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("AV1TC_DIRS_INPUT", tempfile.mkdtemp(prefix="av1_in"))
os.environ.setdefault("AV1TC_DIRS_OUTPUT", tempfile.mkdtemp(prefix="av1_out"))
os.environ.setdefault("AV1TC_DIRS_RPU", tempfile.mkdtemp(prefix="av1_rpu"))
os.environ.setdefault("AV1TC_DIRS_WORK", tempfile.mkdtemp(prefix="av1_work"))
os.environ.setdefault("AV1TC_DIRS_DB", tempfile.mkdtemp(prefix="av1_db") + "/test.db")
os.environ.setdefault("AV1TC_DIRS_LOGS", tempfile.mkdtemp(prefix="av1_logs"))
os.environ.setdefault("AV1TC_DIRS_PRESETS_FILE", tempfile.mkdtemp(prefix="av1_presets") + "/presets.json")
os.environ.setdefault("AV1TC_DIRS_SETTINGS_FILE", tempfile.mkdtemp(prefix="av1_settings") + "/settings.json")

from app import db  # noqa: E402
from app.config import load_settings  # noqa: E402


@pytest.fixture()
def settings(tmp_path):
    """Settings pointed at a per-test scratch DB and work dir.

    Deliberately not the module-level one: these tests create hundreds of jobs
    and delete directories, and sharing either with the rest of the suite makes
    the failures order-dependent.
    """
    s = load_settings()
    s.dirs.db = tmp_path / "jobs.db"
    s.dirs.work = tmp_path / "work"
    s.dirs.work.mkdir()
    return s


@pytest.fixture()
def store(settings):
    s = db.JobStore(settings)
    yield s
    s.close()


# ---------------------------------------------------------------- queue ----

def test_find_active_sees_jobs_beyond_the_first_page(store):
    """De-duplication must not be limited to the newest page of jobs.

    It used to scan store.list(), whose default limit is 100, so once the queue
    was longer than that the OLDEST pending jobs were invisible and the same
    source could be enqueued twice. A restart is exactly the case that builds a
    queue that long: reset_interrupted() puts every running job back to pending
    and the watcher's in-memory seen-set starts empty, so the whole input
    directory is re-submitted.
    """
    for i in range(150):
        store.create(source=f"/media/in/file{i:03d}.mkv", preset="balanced")

    oldest = "/media/in/file000.mkv"
    assert oldest not in [j["source"] for j in store.list(status=None)], \
        "the paged read is expected to miss it - that is the bug being fixed"
    assert store.find_active(oldest) is not None


def test_find_active_ignores_finished_jobs(store):
    """A file that finished once must be re-encodable."""
    jid = store.create(source="/media/in/done.mkv", preset="balanced")
    assert store.find_active("/media/in/done.mkv") == jid
    store.update(jid, status=db.DONE)
    assert store.find_active("/media/in/done.mkv") is None


def test_find_active_matches_the_exact_source(store):
    """Path matching must be exact - a prefix is a different file."""
    store.create(source="/media/in/a.mkv", preset="balanced")
    assert store.find_active("/media/in/a.mkv.part") is None
    assert store.find_active("/media/in/a.mk") is None


def test_enqueue_does_not_duplicate_a_long_queued_source(settings, store, tmp_path):
    """The whole point, through the manager rather than the store."""
    from app.queue import TranscodeManager

    src = tmp_path / "movie.mkv"
    src.write_bytes(b"x")
    manager = TranscodeManager(settings, store)

    first = manager.enqueue_file(str(src))
    # bury it under more than one page of later jobs
    for i in range(150):
        store.create(source=f"/media/in/other{i:03d}.mkv", preset="balanced")

    assert manager.enqueue_file(str(src)) == first
    matching = [j for j in store.list(status=None, limit=1000)
                if j["source"] == str(src)]
    assert len(matching) == 1


def test_enqueue_allows_a_requeue_once_the_job_is_done(settings, store, tmp_path):
    from app.queue import TranscodeManager

    src = tmp_path / "movie.mkv"
    src.write_bytes(b"x")
    manager = TranscodeManager(settings, store)
    first = manager.enqueue_file(str(src))
    store.update(first, status=db.DONE)

    second = manager.enqueue_file(str(src))
    assert second is not None and second != first
