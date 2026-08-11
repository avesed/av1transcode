import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("AV1TC_DIRS_INPUT", tempfile.mkdtemp(prefix="av1_in"))
os.environ.setdefault("AV1TC_DIRS_OUTPUT", tempfile.mkdtemp(prefix="av1_out"))
os.environ.setdefault("AV1TC_DIRS_RPU", tempfile.mkdtemp(prefix="av1_rpu"))
os.environ.setdefault("AV1TC_DIRS_WORK", tempfile.mkdtemp(prefix="av1_work"))
os.environ.setdefault("AV1TC_DIRS_DB", ":memory-slack:")
os.environ.setdefault("AV1TC_DIRS_LOGS", tempfile.mkdtemp(prefix="av1_logs"))
os.environ.setdefault("AV1TC_DIRS_PRESETS_FILE", tempfile.mkdtemp(prefix="av1_presets") + "/presets.json")
os.environ.setdefault("AV1TC_DIRS_SETTINGS_FILE", tempfile.mkdtemp(prefix="av1_settings") + "/settings.json")

from app import db  # noqa: E402
from app.analyzer import DolbyVisionInfo, MediaInfo  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.decisions import decide_action  # noqa: E402


@pytest.fixture()
def settings():
    return load_settings()


@pytest.fixture()
def store(settings):
    s = db.JobStore(settings)
    yield s
    s.close()


def test_settings_load(settings):
    assert settings.transcode.video.codec == "svt-av1"
    assert settings.transcode.default_preset in settings.transcode.presets


def test_db_roundtrip(store):
    jid = store.create(source="/tmp/x.mkv", preset="balanced")
    assert store.get(jid)["status"] == db.PENDING
    store.update(jid, status=db.DONE, progress=100)
    assert store.get(jid)["progress"] == 100
    assert store.count_by_status()["done"] >= 1


def test_skip_av1(settings):
    info = MediaInfo(path=Path("/tmp/a.mkv"))
    info.video_codec = "av1"
    info.is_av1 = True
    info.width = info.height = 3840
    plan = decide_action(settings, info)
    assert plan.skip
    assert "AV1" in plan.skip_reason


def test_dv_p5_plan(settings):
    info = MediaInfo(path=Path("/tmp/p5.mkv"))
    info.video_codec = "hevc"
    info.is_hdr = True
    info.color.transfer = "smpte2084"
    info.dovi = DolbyVisionInfo(present=True, profile=5, rpu_present=True)
    plan = decide_action(settings, info)
    assert plan.p5 is True
    assert plan.dv_profile == 5
    assert plan.output_path.name == "p5.av1.mkv"


def test_dv_p7_plan(settings):
    info = MediaInfo(path=Path("/tmp/p7.mkv"))
    info.video_codec = "hevc"
    info.is_hdr = True
    info.color.transfer = "smpte2084"
    info.dovi = DolbyVisionInfo(present=True, profile=7, rpu_present=True)
    plan = decide_action(settings, info)
    assert plan.dv_profile == 7
    assert "base layer" in plan.notes[-1]


def test_hdr_passthrough_tags(settings):
    info = MediaInfo(path=Path("/tmp/hdr.mkv"))
    info.video_codec = "hevc"
    info.width, info.height = 3840, 2160
    info.is_hdr = True
    info.color.transfer = "smpte2084"
    info.color.mastering_display = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
    info.color.max_cll = "1000,400"
    plan = decide_action(settings, info)
    assert plan.master_display
    assert plan.max_cll
    assert plan.color_primaries == "bt2020"


def test_parse_progress():
    from app.transcoder import parse_progress

    assert parse_progress("Encoding: 45% done") == 45.0
    assert parse_progress("worker 100%") == 100.0
    assert parse_progress("no percent here") is None


def test_optimizer_engine_requires_target_quality(settings):
    info = MediaInfo(path=Path("/tmp/o.mkv"))
    info.video_codec = "hevc"
    info.is_av1 = False
    info.width, info.height = 1920, 1080
    plan = decide_action(settings, info, overrides={"engine": "optimizer"})
    assert plan.skip
    assert "target_quality" in plan.skip_reason


def test_optimizer_engine_plan_note(settings):
    info = MediaInfo(path=Path("/tmp/o2.mkv"))
    info.video_codec = "hevc"
    info.is_av1 = False
    info.width, info.height = 1920, 1080
    plan = decide_action(settings, info, overrides={
        "engine": "optimizer", "target_quality": "75-85", "target_metric": "ssimulacra2",
    })
    assert not plan.skip
    assert plan.params.engine == "optimizer"
    assert any("optimizer" in n for n in plan.notes)


def test_optimizer_user_settings_roundtrip(settings):
    from app import config

    config.save_user_settings(settings, {"optimizer": {
        "probe_crfs": [22, 30],
        "probe_preset": 11,
        "probe_scale": "1280x720",
        "max_shots": 100,
    }})
    reloaded = load_settings()
    assert reloaded.transcode.optimizer.probe_crfs == [22, 30]
    assert reloaded.transcode.optimizer.probe_preset == 11
    assert reloaded.transcode.optimizer.probe_scale == "1280x720"
    assert reloaded.transcode.optimizer.max_shots == 100