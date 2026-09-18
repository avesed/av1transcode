"""Queue, work-directory lifecycle, settings persistence and API auth.

These areas had no coverage at all, which is where every defect exercised
below was living.
"""

import json
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
    s.dirs.presets_file = tmp_path / "presets.json"
    s.dirs.settings_file = tmp_path / "settings.json"
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


# -------------------------------------------------------------- watcher ----

def _watched(settings, store, tmp_path):
    """A watcher over a scratch input dir, wired the way cli.run wires it."""
    from app.queue import TranscodeManager
    from app.watcher import FileWatcher

    settings.dirs.input = tmp_path / "in"
    settings.dirs.input.mkdir(exist_ok=True)
    settings.watcher.min_size_mb = 0
    settings.watcher.stable_seconds = 0
    manager = TranscodeManager(settings, store)
    return manager, FileWatcher(settings, manager.enqueue_new_file)


def _scan_once(watcher):
    """Two scan cycles, because stability now needs two agreeing observations
    and the second one is the next scan rather than a sleep."""
    for _ in range(2):
        for p in watcher._scan():
            watcher._maybe_submit(p)


def test_the_watcher_does_not_resubmit_after_a_restart(settings, store, tmp_path):
    """The in-memory seen-set is empty at startup, so it cannot be what stops
    a second submission - and it was the only thing that did.

    A restart therefore re-enqueued every file still in the input directory,
    including the ones already transcoded: the source stays in place by
    default and the output goes to an excluded av1/ subdirectory, so nothing
    downstream noticed the whole library being encoded again.
    """
    manager, watcher = _watched(settings, store, tmp_path)
    (settings.dirs.input / "movie.mkv").write_bytes(b"x" * 1024)

    _scan_once(watcher)
    jobs = store.list(status=None, limit=100)
    assert len(jobs) == 1
    store.update(jobs[0]["id"], status=db.DONE)

    for _ in range(3):                      # three restarts
        _, restarted = _watched(settings, store, tmp_path)
        _scan_once(restarted)
    assert len(store.list(status=None, limit=100)) == 1


@pytest.mark.parametrize("status", [db.DONE, db.FAILED, db.SKIPPED, db.CANCELLED])
def test_the_watcher_leaves_a_finished_file_alone_whatever_the_outcome(
        settings, store, tmp_path, status):
    """Any job at all counts. A file that failed past max_retries would
    otherwise be retried on every restart forever, and skipped/cancelled were
    decisions rather than accidents."""
    _, watcher = _watched(settings, store, tmp_path)
    (settings.dirs.input / "movie.mkv").write_bytes(b"x" * 1024)
    _scan_once(watcher)
    store.update(store.list(status=None, limit=1)[0]["id"], status=status)

    _, restarted = _watched(settings, store, tmp_path)
    _scan_once(restarted)
    assert len(store.list(status=None, limit=100)) == 1


def test_the_watcher_still_picks_up_a_genuinely_new_file(settings, store, tmp_path):
    _, watcher = _watched(settings, store, tmp_path)
    (settings.dirs.input / "movie.mkv").write_bytes(b"x" * 1024)
    _scan_once(watcher)
    store.update(store.list(status=None, limit=1)[0]["id"], status=db.DONE)

    (settings.dirs.input / "another.mkv").write_bytes(b"y" * 1024)
    _, restarted = _watched(settings, store, tmp_path)
    _scan_once(restarted)

    names = sorted(Path(j["source"]).name for j in store.list(status=None, limit=100))
    assert names == ["another.mkv", "movie.mkv"]


def test_a_person_can_still_resubmit_what_the_watcher_will_not(settings, store, tmp_path):
    """enqueue_new_file is the watcher's rule, not everyone's - the UI and CLI
    still go through enqueue_file."""
    manager, watcher = _watched(settings, store, tmp_path)
    src = settings.dirs.input / "movie.mkv"
    src.write_bytes(b"x" * 1024)
    _scan_once(watcher)
    store.update(store.list(status=None, limit=1)[0]["id"], status=db.DONE)

    assert manager.enqueue_new_file(str(src)) is None
    assert manager.enqueue_file(str(src)) is not None


def test_a_file_still_being_written_is_not_submitted(settings, store, tmp_path):
    """Two observations that agree, or it waits. The second one is the next
    scan, so a file growing between cycles never qualifies."""
    _, watcher = _watched(settings, store, tmp_path)
    src = settings.dirs.input / "copying.mkv"

    for chunk in range(4):                  # four scans, four sizes
        src.write_bytes(b"x" * (1024 * (chunk + 1)))
        for p in watcher._scan():
            watcher._maybe_submit(p)
    assert store.list(status=None, limit=10) == []

    # it stops growing: the next two scans agree and it goes in
    _scan_once(watcher)
    assert len(store.list(status=None, limit=10)) == 1


def test_a_freshly_touched_file_waits_for_stable_seconds(settings, store, tmp_path):
    _, watcher = _watched(settings, store, tmp_path)
    settings.watcher.stable_seconds = 3600          # nothing is old enough
    (settings.dirs.input / "movie.mkv").write_bytes(b"x" * 1024)

    _scan_once(watcher)
    assert store.list(status=None, limit=10) == []

    settings.watcher.stable_seconds = 0
    _scan_once(watcher)
    assert len(store.list(status=None, limit=10)) == 1


def test_a_file_under_min_size_is_ignored(settings, store, tmp_path):
    _, watcher = _watched(settings, store, tmp_path)
    settings.watcher.min_size_mb = 1
    (settings.dirs.input / "sidecar.mkv").write_bytes(b"x" * 1024)

    _scan_once(watcher)
    assert store.list(status=None, limit=10) == []


def test_a_batch_of_files_does_not_serialise_the_scan(settings, store, tmp_path):
    """The stability check used to sleep two seconds per file with the scan
    loop blocked behind it, so this batch would have taken about 80 seconds."""
    import time

    _, watcher = _watched(settings, store, tmp_path)
    for i in range(20):
        (settings.dirs.input / f"m{i:02d}.mkv").write_bytes(b"x" * 1024)

    t0 = time.monotonic()
    _scan_once(watcher)
    elapsed = time.monotonic() - t0

    assert len(store.list(status=None, limit=100)) == 20
    assert elapsed < 2.0, f"20 files took {elapsed:.1f}s; the sleep is back"


def test_has_job_is_broader_than_find_active(store):
    jid = store.create(source="/media/in/x.mkv", preset="balanced")
    store.update(jid, status=db.DONE)
    assert store.find_active("/media/in/x.mkv") is None
    assert store.has_job("/media/in/x.mkv") is True
    assert store.has_job("/media/in/never-seen.mkv") is False


# ------------------------------------------------------- work-dir sweep ----

def test_sweep_removes_what_an_interrupted_job_left(settings):
    """_cleanup_temp runs in a finally, which a SIGKILL never reaches."""
    from app.transcoder import sweep_stale_work

    work = settings.dirs.work
    stale_dir = work / "av1an_1234567890"
    (stale_dir / "probes").mkdir(parents=True)
    (stale_dir / "probes" / "enc_00000.ivf").write_bytes(b"x" * 32)
    (work / "Some Movie.dv_p5.mkv").write_bytes(b"x" * 32)
    (work / "Some Movie.dv_bl.mkv").write_bytes(b"x" * 32)

    assert sweep_stale_work(settings) == 3
    assert not stale_dir.exists()
    assert not (work / "Some Movie.dv_p5.mkv").exists()
    assert not (work / "Some Movie.dv_bl.mkv").exists()


def test_sweep_leaves_everything_else_alone(settings):
    """dirs.work is operator-configured and may hold things we did not put
    there, so the sweep matches only the names this program creates."""
    from app.transcoder import sweep_stale_work

    work = settings.dirs.work
    (work / "notes.txt").write_text("keep me")
    (work / "av1an").mkdir()               # no trailing id: not one of ours
    (work / "holiday.mkv").write_bytes(b"x")

    assert sweep_stale_work(settings) == 0
    assert (work / "notes.txt").exists()
    assert (work / "av1an").is_dir()
    assert (work / "holiday.mkv").exists()


def test_sweep_honours_keep_temp(settings):
    from app.transcoder import sweep_stale_work

    (settings.dirs.work / "av1an_42").mkdir()
    settings.transcode.keep_temp = True
    assert sweep_stale_work(settings) == 0
    assert (settings.dirs.work / "av1an_42").is_dir()


def test_sweep_tolerates_a_missing_work_dir(settings):
    from app.transcoder import sweep_stale_work

    settings.dirs.work = settings.dirs.work / "does" / "not" / "exist"
    assert sweep_stale_work(settings) == 0


# --------------------------------------------- settings persistence ----

def _client(settings, store, api_key=""):
    from fastapi.testclient import TestClient

    from app.api import create_app
    from app.queue import TranscodeManager

    settings.web.api_key = api_key
    manager = TranscodeManager(settings, store)
    return TestClient(create_app(settings, store, manager))


def test_optimizer_settings_save_round_trips(settings, store, tmp_path):
    """Saving from the settings page must actually persist.

    OptimizerSettings carries two Path fields (vszip_plugin, bestsource_plugin)
    and a bare model_dump() yields PosixPath, which json cannot encode - so
    every save from the UI raised inside save_user_settings and came back a
    500, having written nothing. Only ever exercised through the merge helper
    before, never through the route.
    """
    from app.config import load_user_settings

    settings.dirs.settings_file = tmp_path / "settings.json"
    client = _client(settings, store)

    r = client.put("/api/settings/optimizer", json={"probe_preset": 7})
    assert r.status_code == 200, r.text
    assert r.json()["optimizer"]["probe_preset"] == 7

    # persisted, and in a shape load_settings can read back - and ONLY the
    # key that moved, so config.yaml keeps owning everything else
    saved = load_user_settings(settings)
    assert saved["optimizer"]["probe_preset"] == 7
    assert "probe_bracket_width" not in saved["optimizer"]     # never touched: not written
    assert settings.transcode.optimizer.probe_preset == 7
    # a Path field still encodes when it is the one overridden
    r = client.put("/api/settings/optimizer", json={"vszip_plugin": "/opt/vszip.so"})
    assert r.status_code == 200, r.text
    assert isinstance(load_user_settings(settings)["optimizer"]["vszip_plugin"], str)


def test_optimizer_settings_save_keeps_unposted_fields(settings, store, tmp_path):
    """The form posts a subset; the rest must survive the save."""
    settings.dirs.settings_file = tmp_path / "settings.json"
    settings.transcode.optimizer.verify_shots = 3
    client = _client(settings, store)

    r = client.put("/api/settings/optimizer", json={"probe_preset": 7})
    assert r.status_code == 200, r.text
    assert r.json()["optimizer"]["verify_shots"] == 3


# ------------------------------------------------ derived config values ----

def _settings_with_user_preset(monkeypatch, tmp_path, name, **fields):
    """load_settings() with a user preset already on disk, i.e. what a restart
    sees after someone edited a preset in the web UI."""
    import json

    from app.config import VideoParams

    pf = tmp_path / "presets.json"
    pf.write_text(json.dumps({name: VideoParams(**fields).model_dump(mode="json")}))
    monkeypatch.setenv("AV1TC_DIRS_PRESETS_FILE", str(pf))
    monkeypatch.setenv("AV1TC_DIRS_SETTINGS_FILE", str(tmp_path / "settings.json"))
    return load_settings()


def test_a_user_edited_default_preset_reaches_transcode_video(monkeypatch, tmp_path):
    """transcode.video is what preset="custom" starts from, and it is derived
    from the default preset. It used to be derived BEFORE user presets were
    applied, so editing the default preset left it on config.yaml's values and
    a restart did not help."""
    s = _settings_with_user_preset(monkeypatch, tmp_path, "balanced",
                                   crf=19, preset=2)
    assert s.transcode.default_preset == "balanced"
    assert s.transcode.presets["balanced"].crf == 19
    assert s.transcode.video.crf == 19
    assert s.transcode.video.preset == 2


def test_the_builtin_snapshot_survives_a_user_override(monkeypatch, tmp_path):
    """builtin_presets is what `_builtin` reports and what deleting a user
    preset restores, so it must stay config.yaml's version."""
    s = _settings_with_user_preset(monkeypatch, tmp_path, "balanced", crf=19)
    assert s.transcode.builtin_presets["balanced"].crf != 19


def test_editing_the_default_preset_is_live_without_a_restart(settings, store):
    client = _client(settings, store)
    assert settings.transcode.default_preset == "balanced"
    assert settings.transcode.video.crf != 19
    assert client.put("/api/presets/balanced",
                      json={"crf": 19, "preset": 2}).status_code == 200
    assert settings.transcode.video.crf == 19


def test_deleting_a_user_preset_re_derives_from_the_builtin(settings, store):
    client = _client(settings, store)
    builtin_crf = settings.transcode.builtin_presets["balanced"].crf
    client.put("/api/presets/balanced", json={"crf": 19})
    client.delete("/api/presets/balanced")
    assert settings.transcode.video.crf == builtin_crf


def test_av1an_is_given_the_presets_own_extra_split_sec(settings, monkeypatch):
    """Every other flag build_av1an_cmd emits comes from the preset it was
    handed; extra_split_sec read the global transcode.video instead, so a
    preset could not set it at all - every preset got the default preset's."""
    from app import transcoder

    monkeypatch.setattr(transcoder.Settings, "tool_path",
                        lambda self, n: f"/usr/bin/{n}")
    settings.transcode.video.extra_split_sec = 60
    video = settings.transcode.presets["quality"].model_copy(
        update={"extra_split_sec": 77})

    cmd = transcoder.build_av1an_cmd(settings, video, Path("/in.mkv"),
                                     Path("/out.mkv"), Path("/tmp"))
    assert cmd[cmd.index("--extra-split-sec") + 1] == "77"

    # and a preset that turns it off must actually turn it off
    off = video.model_copy(update={"extra_split_sec": 0})
    cmd = transcoder.build_av1an_cmd(settings, off, Path("/in.mkv"),
                                     Path("/out.mkv"), Path("/tmp"))
    assert "--extra-split-sec" not in cmd


# ------------------------------------------------------------- API auth ----

# Every mutating route, plus the one read that walks the host filesystem
# rather than this app's own state. cancel/cancel_all/browse were the three
# that never called _auth.
GATED = [
    ("post", "/api/jobs", {"json": {"path": "/nope.mkv"}}),
    ("post", "/api/jobs/deadbeef/cancel", {}),
    ("post", "/api/cancel", {}),
    ("post", "/api/jobs/prune", {"json": {}}),
    ("put", "/api/presets/x", {"json": {}}),
    ("delete", "/api/presets/x", {}),
    ("put", "/api/settings/workers", {"json": {"concurrency": 1}}),
    ("put", "/api/settings/optimizer", {"json": {}}),
    ("put", "/api/settings/delete_source", {"json": {"enabled": False}}),
    ("get", "/api/browse", {"params": {"path": "/"}}),
]


@pytest.mark.parametrize("method,url,kw", GATED, ids=[g[1] for g in GATED])
def test_gated_routes_reject_a_missing_key(settings, store, method, url, kw):
    client = _client(settings, store, api_key="s3cret")
    assert getattr(client, method)(url, **kw).status_code == 401


@pytest.mark.parametrize("method,url,kw", GATED, ids=[g[1] for g in GATED])
def test_gated_routes_reject_a_wrong_key(settings, store, method, url, kw):
    client = _client(settings, store, api_key="s3cret")
    r = getattr(client, method)(url, headers={"X-API-Key": "nope"}, **kw)
    assert r.status_code == 401


@pytest.mark.parametrize("method,url,kw", GATED, ids=[g[1] for g in GATED])
def test_gated_routes_pass_the_key_through(settings, store, method, url, kw):
    """With the right key these must get past _auth. What they answer after
    that is the route's own business (a 404 for a job that does not exist is a
    pass here); only 401 means the gate rejected us."""
    client = _client(settings, store, api_key="s3cret")
    r = getattr(client, method)(url, headers={"X-API-Key": "s3cret"}, **kw)
    assert r.status_code != 401


@pytest.mark.parametrize("method,url,kw", GATED, ids=[g[1] for g in GATED])
def test_gated_routes_stay_open_when_no_key_is_configured(settings, store,
                                                          method, url, kw):
    """api_key defaults to empty, and that has to keep meaning "open" - this
    is the shipped configuration."""
    client = _client(settings, store, api_key="")
    assert getattr(client, method)(url, **kw).status_code != 401


def test_reads_of_app_state_stay_open(settings, store):
    """The status/job reads are deliberately not gated; only /api/browse is,
    because it reads the host filesystem rather than this app's own state."""
    client = _client(settings, store, api_key="s3cret")
    for url in ("/api/health", "/api/status", "/api/jobs", "/api/presets",
                "/api/workers", "/api/settings/optimizer", "/api/settings/safety"):
        assert client.get(url).status_code == 200, url


def test_browse_refuses_a_relative_path(settings, store):
    client = _client(settings, store)
    r = client.get("/api/browse", params={"path": "relative/dir"})
    assert r.status_code == 400


# --------------------------------------------------------------- the UI ----

def test_ui_pages_send_the_api_key():
    """Every /api call in the UI must go through the key-aware wrapper.

    Neither page used to send X-API-Key at all, which made web.api_key
    unusable: setting it locked the bundled UI out of every write rather than
    securing anything. A bare fetch("/api/...") is that regression.

    The pages' script moved out of the HTML into .js files, so every file
    the static dir serves is scanned, not a list of names: a list is exactly
    what the guard would have been left pointing at while the calls moved
    somewhere it did not look.
    """
    import re

    static = Path(__file__).resolve().parent.parent / "app" / "static"
    assert (static / "apikey.js").exists()
    for page in ("index.html", "settings.html"):
        assert "/static/apikey.js" in (static / page).read_text(), \
            f"{page} does not load the helper"
    scanned = [p for p in sorted(static.iterdir())
               if p.suffix in (".html", ".js") and p.name != "apikey.js"]
    assert {"index.html", "settings.html", "queue.js", "settings.js", "ui.js"} \
        <= {p.name for p in scanned}
    for path in scanned:
        # ui.js's own comment quotes the forbidden call to say it is
        # forbidden; comments are dropped so only code can fail this
        text = re.sub(r"/\*.*?\*/|<!--.*?-->", "", path.read_text(), flags=re.S)
        text = "\n".join(line for line in text.splitlines()
                         if not line.lstrip().startswith("//"))
        text = text.replace("afetch(", "")
        for bad in ('fetch("/api', "fetch(`/api", "fetch('/api"):
            assert bad not in text, f"{path.name} calls {bad}...) without the key"


def test_optimizer_overrides_merge_over_config_yaml(settings, store, tmp_path, monkeypatch):
    """A saved block is overrides, not a replacement: a key it does not hold
    keeps following config.yaml. A production box ran 360p probes every
    other frame for weeks because a full dump from an old page had pinned
    them."""
    from app.config import load_settings

    settings.dirs.settings_file = tmp_path / "settings.json"
    (tmp_path / "settings.json").write_text('{"optimizer": {"probe_preset": 7}}')
    fresh = load_settings()
    fresh.dirs.settings_file = tmp_path / "settings.json"
    from app.config import load_user_settings
    assert load_user_settings(fresh)["optimizer"] == {"probe_preset": 7}
    # re-run the merge the way load_settings does, against this file
    monkeypatch.setenv("AV1TC_DIRS_SETTINGS_FILE", str(tmp_path / "settings.json"))
    s2 = load_settings()
    assert s2.transcode.optimizer.probe_preset == 7
    assert s2.transcode.optimizer.probe_max_frames == s2.transcode.optimizer_defaults.probe_max_frames
    assert s2.transcode.optimizer_defaults.probe_preset != 7


def test_optimizer_defaults_and_restore(settings, store, tmp_path):
    settings.dirs.settings_file = tmp_path / "settings.json"
    client = _client(settings, store)
    client.delete("/api/settings/optimizer")          # start from config.yaml, whatever earlier tests saved
    base = settings.transcode.optimizer_defaults.probe_preset
    client.put("/api/settings/optimizer", json={"probe_preset": base + 1, "verify_shots": 4})
    r = client.get("/api/settings/optimizer/defaults")
    assert r.json()["defaults"]["probe_preset"] == base
    assert set(r.json()["overrides"]) == {"probe_preset", "verify_shots"}
    r = client.delete("/api/settings/optimizer")
    assert r.status_code == 200
    assert settings.transcode.optimizer.probe_preset == base
    assert "optimizer" not in (json.loads((tmp_path / "settings.json").read_text()))
    # and saving a value back to its default drops it from the file again
    client.put("/api/settings/optimizer", json={"probe_preset": base})
    assert "optimizer" not in json.loads((tmp_path / "settings.json").read_text())


def test_gpu_settings_persist_and_apply(settings, store, tmp_path, monkeypatch):
    from app.config import load_user_settings

    settings.dirs.settings_file = tmp_path / "settings.json"
    client = _client(settings, store)
    client.delete("/api/settings/optimizer")
    r = client.put("/api/settings/gpu", json={"vmaf_sycl_device": 0, "vmaf_sycl_min_width": 1920,
                                              "scenedetect_hwaccel": "off", "reference_hwaccel": "auto",
                                              "vulkan_device": "llvmpipe"})
    assert r.status_code == 200, r.text
    assert settings.transcode.optimizer.vmaf_sycl_device == 0
    assert settings.transcode.dovi.vulkan_device == "llvmpipe"
    saved = load_user_settings(settings)
    # only what differs from config.yaml is stored; reference_hwaccel ships
    # off, so "auto" is the override here and "off" would not be one
    assert saved["optimizer"] == {"vmaf_sycl_device": 0, "vmaf_sycl_min_width": 1920,
                                  "scenedetect_hwaccel": "off", "reference_hwaccel": "auto"}
    assert saved["dovi"] == {"vulkan_device": "llvmpipe"}
    r = client.put("/api/settings/gpu", json={"vmaf_sycl_device": 7, "scenedetect_hwaccel": "maybe"})
    assert r.status_code == 422


def test_gpu_status_and_selfcheck_routes(settings, store, monkeypatch):
    from app import gpu

    monkeypatch.setattr(gpu, "probe", lambda s, force=False: {"render_nodes": ["/dev/dri/renderD129"],
                                                               "sycl_built": True, "force": force})
    monkeypatch.setattr(gpu, "selfcheck", lambda s, dev: {"device_index": dev, "ok": True, "delta": 5e-5})
    client = _client(settings, store)
    r = client.get("/api/gpu")
    assert r.json()["status"]["render_nodes"] == ["/dev/dri/renderD129"]
    assert r.json()["settings"]["vmaf_sycl_device"] == settings.transcode.optimizer.vmaf_sycl_device
    assert client.get("/api/gpu?refresh=true").json()["status"]["force"] is True
    r = client.post("/api/gpu/selfcheck", json={"device": 1})
    assert r.json() == {"device_index": 1, "ok": True, "delta": 5e-5}
    r = client.post("/api/gpu/selfcheck", json={})
    assert r.json()["device_index"] == 0                    # -1 (off) probes device 0


def test_gpu_parsers():
    from app import gpu

    vk = ("[Vulkan @ 0x1] Supported layers:\n[Vulkan @ 0x1] \tVK_LAYER_MESA_overlay\n"
          "[Vulkan @ 0x1] GPU listing:\n"
          "[Vulkan @ 0x1]     0: Intel(R) Arc(tm) B580 Graphics (BMG G21) (discrete) (0xe20b)\n"
          "[Vulkan @ 0x1]     1: llvmpipe (LLVM 17.0.6, 256 bits) (cpu) (0x0)\n"
          "[Vulkan @ 0x1] Using device extension VK_KHR_push_descriptor\n")
    assert gpu.parse_vulkan_listing(vk) == [
        "Intel(R) Arc(tm) B580 Graphics (BMG G21) (discrete) (0xe20b)",
        "llvmpipe (LLVM 17.0.6, 256 bits) (cpu) (0x0)"]
    assert gpu.parse_sycl("libvmaf INFO SYCL: using device: Intel(R) Arc(TM) B580 Graphics\n") == {
        "device": "Intel(R) Arc(TM) B580 Graphics", "error": None}
    p = gpu.parse_sycl("libvmaf ERROR SYCL: device_index 7 out of range (1 GPUs)\n[x] vmaf_sycl_state_init(7) failed: -19.")
    assert p["count"] == 1 and p["device"] is None and "out of range" in p["error"]
    assert gpu.parse_sycl("nothing here") == {"device": None, "error": None}

