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

# ---- HDR10 mastering display parsing ----
# Every spelling that reaches us in practice must land on the same real values.
MD_CASES = {
    # config default / x265 / mkvmerge: integer units of 1/50000 and 1/10000
    "integer_units": "G(13250,34500)B(7500,3000)R(34000,16000)"
                     "WP(15635,16450)L(10000000,1)",
    # ffprobe side data on an mp4 source: rationals over a fixed denominator
    "ffprobe_mp4": "G(13250/50000,34500/50000)B(7500/50000,3000/50000)"
                   "R(34000/50000,16000/50000)WP(15635/50000,16450/50000)"
                   "L(10000000/10000,1/10000)",
    # ffprobe side data on an MKV source: rationals over arbitrary denominators
    "ffprobe_mkv": "G(2222981/8388608,11576279/16777216)"
                   "B(5033165/33554432,16106127/268435456)"
                   "R(11408507/16777216,5368709/16777216)"
                   "WP(10492471/33554432,689963/2097152)"
                   "L(1000/1,209800/2098000053)",
    # SVT-AV1 / plain real numbers
    "real": "G(0.265,0.690)B(0.150,0.060)R(0.680,0.320)"
            "WP(0.3127,0.3290)L(1000,0.0001)",
}


@pytest.mark.parametrize("name", sorted(MD_CASES))
def test_parse_master_display_all_spellings(name):
    from app.transcoder import _parse_master_display

    f = _parse_master_display(MD_CASES[name])
    assert f is not None, f"{name} failed to parse"
    assert f["chromaticity-coordinates-green-x"] == pytest.approx(0.265, abs=1e-4)
    assert f["chromaticity-coordinates-green-y"] == pytest.approx(0.690, abs=1e-4)
    assert f["chromaticity-coordinates-blue-x"] == pytest.approx(0.150, abs=1e-4)
    assert f["chromaticity-coordinates-red-x"] == pytest.approx(0.680, abs=1e-4)
    assert f["white-coordinates-x"] == pytest.approx(0.3127, abs=1e-4)
    assert f["white-coordinates-y"] == pytest.approx(0.3290, abs=1e-4)
    assert f["max-luminance"] == pytest.approx(1000.0, rel=1e-4)
    assert f["min-luminance"] == pytest.approx(0.0001, abs=1e-6)


def test_parse_master_display_rejects_garbage():
    from app.transcoder import _parse_master_display

    assert _parse_master_display("not a display string") is None
    assert _parse_master_display("") is None
    # a malformed rational must not raise out of the parser
    assert _parse_master_display(
        "G(1/0,1/2)B(1/2,1/2)R(1/2,1/2)WP(1/2,1/2)L(1000,0.1)") is None


def test_md_fmt_never_uses_scientific_notation():
    """mkvpropedit is all-or-nothing: it rejects "5e-05" and then applies NONE
    of the --set values, so the output silently keeps no colour tags at all."""
    from app.transcoder import _md_fmt

    assert _md_fmt(0.0001) == "0.0001"
    assert _md_fmt(0.00005) == "0.00005"
    assert _md_fmt(9.999999747378462e-05) == "0.0001"   # float noise from an mkv
    assert _md_fmt(1000.0) == "1000.0"
    assert _md_fmt(0.26499998569488525) == "0.265"
    assert _md_fmt(0.0) == "0.0"
    for v in (0.0, 1e-7, 1e-5, 0.0001, 0.265, 1000.0, 10000.0):
        assert "e" not in _md_fmt(v).lower(), v


def test_colorpropedit_falls_back_when_display_unparseable(settings, monkeypatch, tmp_path):
    """An unreadable source string must not cost the file its mastering
    display: the configured default is better than nothing."""
    from app import transcoder
    from app.decisions import TranscodePlan

    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(transcoder.subprocess, "run", fake_run)
    monkeypatch.setattr(transcoder.Settings, "tool_path", lambda self, n: f"/usr/bin/{n}")
    out = tmp_path / "o.mkv"
    out.touch()
    plan = TranscodePlan()
    plan.color_trc = "smpte2084"
    plan.master_display = "totally unparseable"
    transcoder._colorpropedit_hdr(settings, out, plan)
    joined = " ".join(seen["cmd"])
    assert "chromaticity-coordinates-green-x=0.265" in joined
    assert "max-luminance=1000.0" in joined
