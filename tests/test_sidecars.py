"""transcode.carry_subtitles: a source's sidecar subtitles follow it to the output's name.

Plex and Jellyfin pair an external subtitle with the video whose file name it
starts with, and the AV1 output is renamed and written to <source_dir>/av1/ -
so without this every Bazarr download and aisub translation of a transcoded
episode stopped showing up on the AV1 version.
"""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("AV1TC_DIRS_INPUT", tempfile.mkdtemp(prefix="av1_in"))
os.environ.setdefault("AV1TC_DIRS_OUTPUT", tempfile.mkdtemp(prefix="av1_out"))
os.environ.setdefault("AV1TC_DIRS_RPU", tempfile.mkdtemp(prefix="av1_rpu"))
os.environ.setdefault("AV1TC_DIRS_WORK", tempfile.mkdtemp(prefix="av1_work"))
os.environ.setdefault("AV1TC_DIRS_DB", tempfile.mkdtemp(prefix="av1_db") + "/test.db")
os.environ.setdefault("AV1TC_DIRS_LOGS", tempfile.mkdtemp(prefix="av1_logs"))
os.environ.setdefault("AV1TC_DIRS_PRESETS_FILE", tempfile.mkdtemp(prefix="av1_presets") + "/presets.json")
os.environ.setdefault("AV1TC_DIRS_SETTINGS_FILE", tempfile.mkdtemp(prefix="av1_settings") + "/settings.json")

from app import db, sidecars  # noqa: E402
from app.config import load_settings, load_user_settings  # noqa: E402

SRC = "Show.S01E01.2160p.WEB-DL.DV.HEVC"
OUT = "Show.S01E01.2160p.WEB-DL.DV.AV1"


def _season(tmp_path: Path) -> Path:
    season = tmp_path / "S1"
    (season / "subs").mkdir(parents=True)
    (season / f"{SRC}.mkv").write_bytes(b"video")
    return season


def test_beside_and_subs_folder_are_renamed_in_place(tmp_path):
    season = _season(tmp_path)
    (season / f"{SRC}.en.srt").write_text("bazarr en")
    (season / f"{SRC}.zh.hi.srt").write_text("bazarr zh sdh")
    (season / "subs" / f"{SRC}.zh.srt").write_text("aisub zh")
    (season / "subs" / f"{SRC}.zh.ass").write_text("aisub zh ass")
    out = season / "av1" / f"{OUT}.mkv"
    out.parent.mkdir()
    out.write_bytes(b"av1")

    carried = sidecars.carry(season / f"{SRC}.mkv", out)

    assert {item.status for item in carried} == {sidecars.COPIED}
    assert (season / "av1" / f"{OUT}.en.srt").read_text() == "bazarr en"
    assert (season / "av1" / f"{OUT}.zh.hi.srt").read_text() == "bazarr zh sdh"
    assert (season / "av1" / "subs" / f"{OUT}.zh.srt").read_text() == "aisub zh"
    assert (season / "av1" / "subs" / f"{OUT}.zh.ass").read_text() == "aisub zh ass"
    # a copy: the source keeps its own
    assert (season / "subs" / f"{SRC}.zh.srt").read_text() == "aisub zh"
    # nothing half-written is left behind
    assert not [p for p in (season / "av1").rglob(".*.part")]


def test_only_this_videos_subtitles(tmp_path):
    """Another episode's, a longer-named video's, a non-subtitle file and a
    name that merely starts with the same letters all stay."""
    season = _season(tmp_path)
    (season / "Show.S01E02.2160p.WEB-DL.DV.HEVC.zh.srt").write_text("E02")
    (season / f"{SRC}.part2.mkv").write_bytes(b"other video")
    (season / f"{SRC}.part2.en.srt").write_text("part 2")
    (season / f"{SRC}.nfo").write_text("nfo")
    (season / f"{SRC}-thumb.jpg").write_bytes(b"jpg")
    (season / f"{SRC}X.zh.srt").write_text("not a delimiter")
    (season / f"{SRC}.zh.srt").write_text("mine")
    out = tmp_path / "out" / f"{OUT}.mkv"

    carried = sidecars.carry(season / f"{SRC}.mkv", out)

    assert [item.dest.name for item in carried] == [f"{OUT}.zh.srt"]


def test_name_match_ignores_case_like_the_players(tmp_path):
    season = _season(tmp_path)
    (season / f"{SRC.lower()}.ZH.SRT").write_text("lower case")
    (season / "Subs").mkdir()
    (season / "Subs" / f"{SRC}.en.srt").write_text("capital Subs")
    out = tmp_path / "o" / f"{OUT}.mkv"

    carried = sidecars.carry(season / f"{SRC}.mkv", out)

    assert (out.parent / f"{OUT}.ZH.SRT").read_text() == "lower case"
    assert (out.parent / "Subs" / f"{OUT}.en.srt").read_text() == "capital Subs"
    assert len(carried) == 2


def test_an_existing_file_at_the_destination_is_never_replaced(tmp_path):
    season = _season(tmp_path)
    (season / f"{SRC}.en.srt").write_text("same")
    (season / f"{SRC}.zh.srt").write_text("ours")
    out = tmp_path / "o" / f"{OUT}.mkv"
    out.parent.mkdir()
    (out.parent / f"{OUT}.en.srt").write_text("same")
    (out.parent / f"{OUT}.zh.srt").write_text("somebody else's")

    carried = {item.dest.name: item.status for item in sidecars.carry(season / f"{SRC}.mkv", out)}

    assert carried == {f"{OUT}.en.srt": sidecars.SAME, f"{OUT}.zh.srt": sidecars.KEPT}
    assert (out.parent / f"{OUT}.zh.srt").read_text() == "somebody else's"


def test_remove_sources_takes_only_what_has_a_copy(tmp_path):
    season = _season(tmp_path)
    (season / f"{SRC}.en.srt").write_text("en")
    (season / f"{SRC}.zh.srt").write_text("zh")
    out = tmp_path / "o" / f"{OUT}.mkv"
    out.parent.mkdir()
    (out.parent / f"{OUT}.zh.srt").write_text("different")

    removed = sidecars.remove_sources(sidecars.carry(season / f"{SRC}.mkv", out))

    assert removed == [season / f"{SRC}.en.srt"]
    assert (season / f"{SRC}.zh.srt").exists()       # kept: its subtitle exists nowhere else


def test_a_disc_or_a_missing_source_is_not_a_video(tmp_path):
    disc = tmp_path / "Film (2020)"
    disc.mkdir()
    assert not sidecars.applies_to(disc)
    iso = tmp_path / "Film.iso"
    iso.write_bytes(b"iso")
    assert not sidecars.applies_to(iso)
    assert not sidecars.applies_to(tmp_path / "gone.mkv")


# ------------------------------------------------------------- the queue ----

@pytest.fixture()
def settings(tmp_path):
    s = load_settings()
    s.dirs.db = tmp_path / "jobs.db"
    s.dirs.work = tmp_path / "work"
    s.dirs.work.mkdir()
    s.dirs.logs = tmp_path / "logs"
    s.dirs.logs.mkdir()
    s.dirs.presets_file = tmp_path / "presets.json"
    s.dirs.settings_file = tmp_path / "settings.json"
    return s


@pytest.fixture()
def store(settings):
    s = db.JobStore(settings)
    yield s
    s.close()


def _encode(settings, store, tmp_path, monkeypatch):
    """Run one job through TranscodeManager._process with the encoder stubbed
    to write the output: returns (source, output)."""
    from app import queue as queue_mod

    season = _season(tmp_path)
    source = season / f"{SRC}.mkv"
    (season / "subs" / f"{SRC}.zh.srt").write_text("aisub zh")
    out = season / "av1" / f"{OUT}.mkv"
    manager = queue_mod.TranscodeManager(settings, store)
    jid = manager.enqueue_file(str(source))
    info = SimpleNamespace(
        display="hevc", width=3840, height=2160, fps=24.0, video_codec="hevc", duration=60.0, size=5,
        audio_count=1, is_hdr=False, is_hlg=False, path=source,
        dovi=SimpleNamespace(present=False, profile=None))
    monkeypatch.setattr(queue_mod.analyzer_mod, "analyze", lambda *_a: info)
    monkeypatch.setattr(queue_mod.decisions, "decide_action", lambda *a, **k: SimpleNamespace(
        skip=False, skip_reason="", rpu_path=None, output_path=out))

    def run_full_transcode(_settings, _info, _plan, _src, output, *_a, **_k):
        output.write_bytes(b"av1")

    monkeypatch.setattr(queue_mod, "run_full_transcode", run_full_transcode)
    manager._process(jid)
    assert store.get(jid)["status"] == db.DONE
    return source, out, jid


def test_a_finished_job_carries_the_subtitles_and_says_so(settings, store, tmp_path, monkeypatch):
    source, out, jid = _encode(settings, store, tmp_path, monkeypatch)
    copy = out.parent / "subs" / f"{OUT}.zh.srt"
    assert copy.read_text() == "aisub zh"
    assert (source.parent / "subs" / f"{SRC}.zh.srt").exists()
    log = (settings.dirs.logs / f"job_{jid}.log").read_text()
    assert "[subtitles]" in log and f"{OUT}.zh.srt" in log


def test_with_delete_source_it_is_a_move(settings, store, tmp_path, monkeypatch):
    settings.transcode.delete_source = True
    source, out, _ = _encode(settings, store, tmp_path, monkeypatch)
    assert not source.exists()
    assert (out.parent / "subs" / f"{OUT}.zh.srt").read_text() == "aisub zh"
    assert not (source.parent / "subs" / f"{SRC}.zh.srt").exists()


def test_switched_off_nothing_is_carried(settings, store, tmp_path, monkeypatch):
    settings.transcode.carry_subtitles = False
    _, out, _ = _encode(settings, store, tmp_path, monkeypatch)
    assert not (out.parent / "subs").exists()


def test_the_switch_is_saved_and_read_back(settings, store):
    from fastapi.testclient import TestClient

    from app.api import create_app
    from app.queue import TranscodeManager

    client = TestClient(create_app(settings, store, TranscodeManager(settings, store)))
    assert client.get("/api/settings/safety").json()["carry_subtitles"] is True
    assert client.put("/api/settings/carry_subtitles", json={"enabled": False}).json() == {
        "ok": True, "carry_subtitles": False}
    assert settings.transcode.carry_subtitles is False
    assert load_user_settings(settings)["carry_subtitles"] is False
    assert client.get("/api/settings/safety").json() == {"delete_source": False, "carry_subtitles": False}
