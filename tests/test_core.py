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


def test_optimizer_settings_put_merges_partial_body(settings, monkeypatch):
    """The settings page posts a subset of the optimizer fields. Validating
    that body on its own fills every absent field with its DEFAULT, so each
    save silently reset whatever the form did not carry."""
    from app.config import OptimizerSettings

    current = OptimizerSettings(ssimulacra2_frame_step=8, vmaf_4k_min_width=3000,
                                probe_preset=6)
    body = {"probe_preset": 9}                     # what a partial form sends
    merged = OptimizerSettings.model_validate({**current.model_dump(), **body})

    assert merged.probe_preset == 9                # the posted field is applied
    assert merged.ssimulacra2_frame_step == 8      # and the rest survives
    assert merged.vmaf_4k_min_width == 3000


def test_settings_page_covers_every_editable_optimizer_field():
    """Every optimizer knob is either on the settings page or explicitly
    exempt. A whitelist of names to check would pass for any field nobody
    remembered to add, which is how vmaf_sycl_device ended up settable exactly
    once: saving from the UI writes a JSON that overrides config.yaml, so a
    knob missing from the form is a knob the user cannot change afterwards.
    """
    import re
    from pathlib import Path

    from app.config import Settings

    # install paths, not tuning - they point at .so files baked into the image
    exempt = {"bestsource_plugin", "vszip_plugin"}

    fields = set(
        Settings.model_fields["transcode"].annotation
        .model_fields["optimizer"].annotation.model_fields
    )
    html = Path("app/static/settings.html").read_text()
    # the page renders its optimizer and GPU forms from one field table, and
    # posts exactly the keys in it: F("<key>", ...)
    posted = set(re.findall(r'F\("(\w+)"', html))
    missing = fields - posted - exempt
    assert not missing, f"not on the settings form and not exempt: {sorted(missing)}"


def test_settings_page_rail_lists_every_optimizer_group():
    """The left rail is hand-written HTML while the groups themselves are
    rendered from OPT_GROUPS, so a new group appears in the page and not in
    the rail unless someone remembers both. The switch a user goes looking for
    is the one that just dropped a track they wanted.
    """
    import re
    from pathlib import Path

    html = Path("app/static/settings.html").read_text()
    groups = re.findall(r'\{ id: "(grp-\w+)"', html)
    rail = re.findall(r'href="#(grp-\w+)"', html)
    assert groups and rail == groups, f"rail {rail} against groups {groups}"


def test_the_subtitle_switches_are_all_in_the_subtitle_group():
    """The rail test above says every group is reachable; this says a field is
    in the group a user would look in. A subtitle switch filed under "verify
    and debug" is one nobody finds, and all three of these are on by default -
    so the one a user goes looking for is the one that just changed a track
    they wanted left alone.
    """
    import re
    from pathlib import Path

    html = Path("app/static/settings.html").read_text()
    block = re.search(r'\{ id: "grp-subs".*?\n  \]\},', html, re.S)
    assert block, "grp-subs is not in OPT_GROUPS"
    assert set(re.findall(r'F\("(\w+)"', block.group(0))) == {
        "drop_empty_subtitles", "ass_srt_companion", "pgs_ocr_srt"}


def test_config_yaml_documents_every_optimizer_field():
    """config.yaml is the only documentation most of these knobs have. A field
    that exists in the model but not in the shipped config is one a user can
    only find by reading the source."""
    import yaml
    from pathlib import Path

    from app.config import Settings

    fields = set(
        Settings.model_fields["transcode"].annotation
        .model_fields["optimizer"].annotation.model_fields
    )
    shipped = set(yaml.safe_load(
        Path("config.yaml").read_text())["transcode"]["optimizer"])
    missing = fields - shipped
    assert not missing, f"undocumented in config.yaml: {sorted(missing)}"

def _out_info(path, duration=60.0, audio=2, subs=3, codec="av1"):
    from app.analyzer import MediaInfo

    i = MediaInfo(path=Path(path))
    i.duration = duration
    i.audio_count = audio
    i.subtitle_count = subs
    i.video_codec = codec
    return i


def _src_info(tmp_path):
    return _out_info(tmp_path / "src.mkv", duration=60.0, audio=2, subs=3, codec="hevc")


def _verify(settings, monkeypatch, tmp_path, out_info):
    from app import transcoder

    out = tmp_path / "out.av1.mkv"
    out.write_bytes(b"x" * 1024)
    monkeypatch.setattr(transcoder, "analyze", lambda _s, _p: out_info)
    transcoder._verify_output(settings, _src_info(tmp_path), out)


def test_verify_output_accepts_a_matching_encode(settings, monkeypatch, tmp_path):
    # container timestamp rounding: measured ~1ms of drift on a 46-min encode
    _verify(settings, monkeypatch, tmp_path, _out_info(tmp_path, duration=60.001))


def test_verify_output_rejects_a_truncated_encode(settings, monkeypatch, tmp_path):
    from app.transcoder import TranscodeError

    with pytest.raises(TranscodeError, match="duration"):
        _verify(settings, monkeypatch, tmp_path, _out_info(tmp_path, duration=42.0))


def test_verify_output_rejects_lost_audio(settings, monkeypatch, tmp_path):
    """A Dolby Vision job feeds the encoder a video-only intermediate; without
    a re-mux the output has no audio at all and used to ship as a success."""
    from app.transcoder import TranscodeError

    with pytest.raises(TranscodeError, match="audio"):
        _verify(settings, monkeypatch, tmp_path, _out_info(tmp_path, audio=0))


def test_verify_output_rejects_lost_subtitles(settings, monkeypatch, tmp_path):
    from app.transcoder import TranscodeError

    with pytest.raises(TranscodeError, match="subtitle"):
        _verify(settings, monkeypatch, tmp_path, _out_info(tmp_path, subs=1))


def test_verify_output_counts_what_the_mux_reported(settings, monkeypatch,
                                                    tmp_path):
    """The optimizer engine leaves the source's EMPTY subtitle tracks out and
    can add an srt companion beside an ASS one, so the output deliberately
    does not match the source. The numbers come from what that mux reported
    doing - this check also runs for the av1an engine, which does neither, and
    has to stay right when the settings are off or when detection degraded to
    keeping a track it could not read."""
    from app import transcoder
    from app.transcoder import TranscodeError

    out = tmp_path / "out.av1.mkv"
    out.write_bytes(b"x" * 1024)
    src = _src_info(tmp_path)                     # three subtitle streams

    def verify(subs, dropped=0, added=0):
        monkeypatch.setattr(transcoder, "analyze",
                            lambda _s, _p: _out_info(out, subs=subs))
        transcoder._verify_output(settings, src, out, dropped, added)

    verify(1, dropped=2)                          # two empty tracks left out
    verify(4, added=1)                            # one srt companion added
    verify(2, dropped=2, added=1)                 # both at once
    verify(3)                                     # av1an: nothing changed
    # the same output from an engine that dropped nothing is still a fault,
    # and so is a track lost on top of the ones the mux accounted for
    with pytest.raises(TranscodeError, match="subtitle"):
        verify(1)
    with pytest.raises(TranscodeError, match="subtitle"):
        verify(0, dropped=2)


def test_verify_output_rejects_an_unprobeable_file(settings, monkeypatch, tmp_path):
    from app import transcoder
    from app.transcoder import TranscodeError

    out = tmp_path / "out.av1.mkv"
    out.write_bytes(b"x" * 1024)
    monkeypatch.setattr(transcoder, "analyze", lambda _s, _p: None)
    with pytest.raises(TranscodeError, match="could not probe"):
        transcoder._verify_output(settings, _src_info(tmp_path), out)


def test_verify_output_rejects_an_empty_file(settings, tmp_path):
    from app import transcoder
    from app.transcoder import TranscodeError

    out = tmp_path / "out.av1.mkv"
    out.touch()
    with pytest.raises(TranscodeError, match="no output file"):
        transcoder._verify_output(settings, _src_info(tmp_path), out)


# ---- temp cleanup has to cope with files, not just directories ----
def test_cleanup_temp_removes_files_and_dirs(tmp_path):
    """tmp_files mixes the av1an/optimizer temp trees with plain intermediates
    (a stripped DV base layer). shutil.rmtree raises NotADirectoryError on a
    file, which ignore_errors=True swallowed, so those used to be left behind."""
    from app.transcoder import _cleanup_temp

    d = tmp_path / "tree"
    (d / "nested").mkdir(parents=True)
    (d / "nested" / "chunk.ivf").write_bytes(b"x")
    f = tmp_path / "movie.dv_bl.mkv"
    f.write_bytes(b"x")

    _cleanup_temp([d, f], keep=False)
    assert not d.exists() and not f.exists()


def test_cleanup_temp_honours_keep_temp(tmp_path):
    from app.transcoder import _cleanup_temp

    f = tmp_path / "movie.dv_bl.mkv"
    f.write_bytes(b"x")
    _cleanup_temp([f], keep=True)
    assert f.exists()


def test_cleanup_temp_tolerates_already_gone(tmp_path):
    from app.transcoder import _cleanup_temp

    _cleanup_temp([tmp_path / "vanished.mkv", tmp_path / "vanished_dir"], keep=False)


# ---- run_full_transcode: temp must go on the failure path too ----
def _full_transcode_fixture(settings, tmp_path):
    from app.analyzer import MediaInfo
    from app.decisions import TranscodePlan

    settings.dirs.work = tmp_path / "work"
    settings.dirs.work.mkdir(parents=True, exist_ok=True)
    src = tmp_path / "src.mkv"
    src.write_bytes(b"source")
    info = MediaInfo(path=src)
    info.duration, info.video_codec = 60.0, "hevc"
    info.audio_count, info.subtitle_count = 2, 3
    plan = TranscodePlan()
    plan.params = settings.transcode.video.model_copy(deep=True)
    plan.params.engine = "optimizer"
    plan.params.target_quality = "75"
    return info, plan, src, tmp_path / "out.av1.mkv"


def test_run_full_transcode_cleans_temp_on_failure(settings, monkeypatch, tmp_path):
    """A job that raises is exactly the job that gets retried twice more, so
    leaking the temp tree meant three source-sized copies per broken file."""
    from app import optimizer, transcoder
    from app.transcoder import TranscodeError

    info, plan, src, out = _full_transcode_fixture(settings, tmp_path)

    def boom(_s, _i, _p, _src, _out, tempdir, **_kw):
        (tempdir / "enc_00000.ivf").write_bytes(b"partial")
        raise TranscodeError("encoder died")

    monkeypatch.setattr(optimizer, "run_shot_transcode", boom)
    with pytest.raises(TranscodeError):
        transcoder.run_full_transcode(settings, info, plan, src, out)
    assert list(settings.dirs.work.iterdir()) == []


def test_run_full_transcode_drops_an_output_that_fails_verification(
        settings, monkeypatch, tmp_path):
    """An unverified file next to the source is what later gets mistaken for a
    finished archive - and with delete_source the source is already gone."""
    from app import optimizer, transcoder
    from app.transcoder import TranscodeError

    info, plan, src, out = _full_transcode_fixture(settings, tmp_path)

    def encode(_s, _i, _p, _src, output, _tempdir, **_kw):
        Path(output).write_bytes(b"truncated")

    monkeypatch.setattr(optimizer, "run_shot_transcode", encode)
    monkeypatch.setattr(transcoder, "analyze",
                        lambda _s, _p: _out_info(out, duration=12.0))
    with pytest.raises(TranscodeError, match="duration"):
        transcoder.run_full_transcode(settings, info, plan, src, out)
    assert not out.exists()
    assert list(settings.dirs.work.iterdir()) == []


def test_run_full_transcode_keeps_a_verified_output(settings, monkeypatch, tmp_path):
    from app import optimizer, transcoder

    info, plan, src, out = _full_transcode_fixture(settings, tmp_path)

    def encode(_s, _i, _p, _src, output, _tempdir, **_kw):
        Path(output).write_bytes(b"good output")

    monkeypatch.setattr(optimizer, "run_shot_transcode", encode)
    monkeypatch.setattr(transcoder, "analyze", lambda _s, _p: _out_info(out))
    transcoder.run_full_transcode(settings, info, plan, src, out)
    assert out.exists()
    assert list(settings.dirs.work.iterdir()) == []


def test_run_full_transcode_verifies_against_the_muxs_own_report(
        settings, monkeypatch, tmp_path):
    """The engine's mux says what it did to the subtitle streams and the check
    adds that to the source's count. A source with three of them, two empty
    and one ASS, comes out with two: one kept plus its srt companion."""
    from app import optimizer, transcoder

    info, plan, src, out = _full_transcode_fixture(settings, tmp_path)

    def encode(_s, _i, _p, _src, output, _tempdir, **_kw):
        Path(output).write_bytes(b"good output")
        return optimizer.MuxReport(dropped=2, added=1)

    monkeypatch.setattr(optimizer, "run_shot_transcode", encode)
    monkeypatch.setattr(transcoder, "analyze",
                        lambda _s, _p: _out_info(out, subs=2))
    transcoder.run_full_transcode(settings, info, plan, src, out)
    assert out.exists()


def test_analyze_records_the_video_streams_lead(settings, monkeypatch, tmp_path):
    """ffmpeg seeks relative to the container start, the optimizer counts
    frames from the first video frame; the analyzer has to expose the gap."""
    from app import analyzer

    f = tmp_path / "bcs.mkv"
    f.write_bytes(b"x")
    monkeypatch.setattr(analyzer, "_ffprobe", lambda s, p: {
        "format": {"format_name": "matroska", "duration": "2700.0",
                   "start_time": "0.000000"},
        "streams": [
            {"codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle",
             "start_time": "0.000000"},
            {"codec_type": "video", "codec_name": "hevc", "width": 3840,
             "height": 2160, "start_time": "1.955000", "r_frame_rate": "24000/1001"},
            {"codec_type": "audio", "codec_name": "dts", "start_time": "0.008000"},
        ]})
    info = analyzer.analyze(settings, str(f))
    assert info.video_start == pytest.approx(1.955)
    assert info.format_start == 0.0
    assert info.video_lead == pytest.approx(1.955)
    # "N/A" and a missing field both read as 0, never as an exception
    monkeypatch.setattr(analyzer, "_ffprobe", lambda s, p: {
        "format": {"start_time": "N/A"},
        "streams": [{"codec_type": "video", "codec_name": "hevc",
                     "r_frame_rate": "24/1"}]})
    info = analyzer.analyze(settings, str(f))
    assert info.video_lead == 0.0


def test_ffprobe_json_survives_decoder_noise_on_stderr(settings, monkeypatch, tmp_path):
    """A DTS-HD track makes ffprobe print "[dca @ ...] Residual encoded
    channels are present without core" on stderr even at -v error. Merged
    into stdout that sat in front of the JSON, and every analysis of the
    file failed - a whole Stranger Things episode was 'analysis failed'."""
    from app import analyzer
    import subprocess as sp

    def fake_run(cmd, stdout=None, stderr=None, timeout=None, text=None, errors=None):
        class P:
            returncode = 0
            def __init__(self):
                noise = "[dca @ 0x1] Residual encoded channels are present without core\n"
                body = '{"format": {"format_name": "matroska"}, "streams": [{"codec_type": "video", "codec_name": "hevc", "r_frame_rate": "24000/1001"}]}'
                self.stdout = (noise + body) if stderr == sp.STDOUT else body
        return P()

    monkeypatch.setattr(analyzer.subprocess, "run", fake_run)
    monkeypatch.setattr(type(settings), "tool_path", lambda self, name: f"/usr/bin/{name}")
    f = tmp_path / "x.mkv"; f.write_bytes(b"x")
    info = analyzer.analyze(settings, str(f))
    assert info is not None and info.video_codec == "hevc"


def test_cgroup_memory_in_use_discounts_reclaimable_page_cache(monkeypatch, tmp_path):
    """Reading a 28GB source fills the cgroup with file cache that the kernel
    would drop on demand; counting it as used starved the probe pool (10
    workers down to 5 on a 64GB container, 49.5GB of it inactive cache)."""
    from app import sysres

    current, inactive = 61_944_049_664, 53_150_220_288          # the production reading
    (tmp_path / "memory.current").write_text(str(current))
    (tmp_path / "memory.stat").write_text(f"anon 8912896000\nfile 53350220288\ninactive_file {inactive}\nactive_file 200000000\n")
    monkeypatch.setattr(sysres, "_cgroup_v2_dir", lambda: tmp_path)
    used = sysres.memory_in_use_gb()
    assert used == pytest.approx((current - inactive) / sysres._GB, rel=1e-6)
    assert used < 9                                  # ~8.2GB that cannot be reclaimed, not 58
    # no memory.stat: fall back to the raw figure rather than fail
    (tmp_path / "memory.stat").unlink()
    assert sysres.memory_in_use_gb() == pytest.approx(current / sysres._GB)



def test_rpu_extraction_never_uses_dovi_tools_matroska_reader(settings, tmp_path, monkeypatch):
    """Handing dovi_tool an .mkv silently extracted 5819 bytes of a 46-minute
    remux - about 29 frames - exited 0 and printed nothing, so the old
    `rc == 0 and size > 0` check called it a success for a whole season. The
    same file piped through ffmpeg gives 13.4MB."""
    import shutil
    from app import dovi

    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: f"/usr/bin/{name}")
    src = tmp_path / "ep.mkv"; src.write_bytes(b"x")
    dest = tmp_path / "ep.rpu.bin"
    seen = {}

    class FakePopen:
        def __init__(self, cmd, **kw):
            seen["ffmpeg"] = cmd
            self.stdout = None
        def wait(self, timeout=None): return 0

    def fake_run(cmd, **kw):
        seen["dovi"] = cmd
        dest.write_bytes(b"R" * 500_000)
        class R: returncode, stderr = 0, b""
        return R()

    monkeypatch.setattr(dovi.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(dovi.subprocess, "run", fake_run)
    monkeypatch.setattr(dovi, "_duration_seconds", lambda s, p: 100.0)
    assert dovi.extract_rpu(settings, src, dest, 7) is True
    assert "hevc_mp4toannexb" in seen["ffmpeg"]        # through ffmpeg, not the mkv reader
    assert seen["dovi"][1:] == ["extract-rpu", "-", "-o", str(dest)]


def test_a_truncated_rpu_is_a_failure_however_clean_the_exit(settings, tmp_path, monkeypatch):
    """The failure this guards against produced a valid, parseable, useless
    file: 2 bytes of RPU per second of video where a real one carries ~4800."""
    import shutil
    from app import dovi

    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: f"/usr/bin/{name}")
    src = tmp_path / "ep.mkv"; src.write_bytes(b"x")
    dest = tmp_path / "ep.rpu.bin"

    class FakePopen:
        def __init__(self, cmd, **kw): self.stdout = None
        def wait(self, timeout=None): return 0

    def fake_run(cmd, **kw):
        dest.write_bytes(b"R" * 5819)                  # what the real bug wrote
        class R: returncode, stderr = 0, b""
        return R()

    monkeypatch.setattr(dovi.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(dovi.subprocess, "run", fake_run)
    monkeypatch.setattr(dovi, "_duration_seconds", lambda s, p: 2808.0)
    assert dovi.extract_rpu(settings, src, dest, 7) is False
    assert not dest.exists()
    # and a short source with the same file is fine - the check is a rate
    monkeypatch.setattr(dovi, "_duration_seconds", lambda s, p: 10.0)
    assert dovi.extract_rpu(settings, src, dest, 7) is True
