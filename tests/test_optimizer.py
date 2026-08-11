import json
import shutil
import sys
import types
from pathlib import Path

import pytest

from app.config import Settings, VideoParams
from app.decisions import TranscodePlan
from app.analyzer import MediaInfo
from app import optimizer as opt


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    def fake_which(name, *_a, **_k):
        return f"/usr/bin/fake-{name}"

    monkeypatch.setattr(shutil, "which", fake_which)
    s = Settings()
    s.dirs.work = tmp_path / "work"
    s.dirs.logs = tmp_path / "logs"
    s.dirs.work.mkdir(parents=True, exist_ok=True)
    s.dirs.logs.mkdir(parents=True, exist_ok=True)
    return s


@pytest.fixture()
def info():
    i = MediaInfo(path=Path("/tmp/movie.mkv"))
    i.fps = 30.0
    i.duration = 60.0
    return i


@pytest.fixture()
def plan():
    p = TranscodePlan()
    p.params = VideoParams(engine="optimizer", target_quality="75", target_metric="vmaf")
    return p


def make_encoder(settings, info, plan, tmp_path):
    src = tmp_path / "src.mkv"
    src.touch()
    out = tmp_path / "out.mkv"
    enc = opt.ShotEncoder(settings, info, plan, src, out, tmp_path / "temp")
    return enc


# ---- parse_target ----
def test_parse_target_plain():
    assert opt.parse_target("75") == (75.0, None)


def test_parse_target_range():
    assert opt.parse_target("75-85") == (75.0, 85.0)


def test_parse_target_float_and_spaces():
    assert opt.parse_target("  75.5 - 80 ") == (75.5, 80.0)


def test_parse_target_invalid():
    with pytest.raises(Exception):
        opt.parse_target("")
    with pytest.raises(Exception):
        opt.parse_target("abc")
    with pytest.raises(Exception):
        opt.parse_target("85-75")


# ---- pick_crf ----
def test_pick_crf_interpolates():
    samples = [(20, 90.0), (24, 85.0), (28, 80.0), (32, 70.0)]
    assert opt.pick_crf(samples, 75.0) == pytest.approx(30.0)
    assert opt.pick_crf(samples, 78.0) == pytest.approx(28.8)


def test_pick_crf_clamps_low_when_target_unreachable():
    samples = [(20, 90.0), (24, 85.0), (28, 80.0)]
    assert opt.pick_crf(samples, 95.0) == 20.0


def test_pick_crf_clamps_high_when_target_easily_met():
    samples = [(20, 90.0), (24, 85.0), (28, 80.0)]
    assert opt.pick_crf(samples, 65.0) == 28.0


def test_pick_crf_single_point():
    assert opt.pick_crf([(30, 80.0)], 75.0) == 30.0


def test_pick_crf_empty_raises():
    with pytest.raises(Exception):
        opt.pick_crf([], 75.0)


def test_pick_crf_unsorted_input():
    samples = [(32, 70.0), (20, 90.0), (28, 80.0), (24, 85.0)]
    assert opt.pick_crf(samples, 75.0) == pytest.approx(30.0)


# ---- merge_to_max ----
def test_merge_to_max_merges_shortest():
    shots = [(0, 100), (100, 120), (120, 220)]
    merged = opt.merge_to_max(shots, 2)
    assert len(merged) == 2
    assert merged[0] == (0, 120)
    assert merged[1] == (120, 220)


def test_merge_to_max_noop_when_under():
    shots = [(0, 100), (100, 200)]
    assert opt.merge_to_max(shots, 5) == shots


# ---- smooth_crfs ----
def test_smooth_crfs_bounds_adjacent_deltas():
    out = opt.smooth_crfs([22.0, 31.0, 22.0], max_delta=4.0)
    assert all(abs(out[i] - out[i + 1]) <= 4.0 + 1e-9 for i in range(len(out) - 1))
    assert out == [22.0, 26.0, 22.0]


def test_smooth_crfs_ramp():
    out = opt.smooth_crfs([20.0, 30.0, 40.0, 50.0, 60.0], max_delta=4.0)
    assert out == [20.0, 24.0, 28.0, 32.0, 36.0]


def test_smooth_crfs_noop_when_within_bound():
    crfs = [22.0, 24.0, 26.0]
    assert opt.smooth_crfs(crfs, max_delta=4.0) == crfs


def test_smooth_crfs_disabled_and_single():
    assert opt.smooth_crfs([22.0, 31.0], max_delta=0.0) == [22.0, 31.0]
    assert opt.smooth_crfs([30.0], max_delta=4.0) == [30.0]


def test_smooth_chosen(settings, info, plan, tmp_path):
    settings.transcode.optimizer.max_crf_delta = 4.0
    enc = make_encoder(settings, info, plan, tmp_path)
    out = enc.smooth_chosen({0: 22.0, 1: 31.0, 2: 22.0})
    vals = [out[i] for i in sorted(out)]
    assert all(abs(vals[i] - vals[i + 1]) <= 4.0 + 1e-9 for i in range(len(vals) - 1))
    assert out[1] == pytest.approx(26.0, abs=0.01)


def test_smooth_chosen_disabled(settings, info, plan, tmp_path):
    settings.transcode.optimizer.max_crf_delta = 0.0
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc.smooth_chosen({0: 22.0, 1: 31.0}) == {0: 22.0, 1: 31.0}


# ---- svt params ----
def test_svt_params_dict():
    v = VideoParams(tune=1, film_grain=8, film_grain_denoise=False,
                    additional_video_params="--sharpness 1 --enable-qm 1")
    svt = opt._svt_params_dict(v)
    assert svt["tune"] == 1
    assert svt["film-grain"] == 8
    assert svt["film-grain-denoise"] == 0
    assert svt["sharpness"] == "1"
    assert svt["enable-qm"] == "1"


def test_svt_params_additional_overrides():
    v = VideoParams(tune=1, additional_video_params="--tune 2")
    assert opt._svt_params_dict(v)["tune"] == "2"


# ---- parse_score ----
def test_parse_score_vmaf(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 82.5}}}))
    assert opt.parse_score(p, "vmaf") == 82.5


def test_parse_score_ssimulacra2(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"pooled_metrics": {"ssimulacra2": {"mean": 88.0}}}))
    assert opt.parse_score(p, "ssimulacra2") == 88.0


def test_parse_score_aggregate_fallback(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"aggregateVMAF": 79.1}))
    assert opt.parse_score(p, "vmaf") == 79.1


def test_parse_score_missing_raises(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"frames": []}))
    with pytest.raises(Exception):
        opt.parse_score(p, "vmaf")


# ---- model cfg ----
def test_model_cfg_path_wrapped(settings, info, plan, tmp_path):
    settings.transcode.optimizer.vmaf_model = "/x/y.json"
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc._model_cfg() == "path=/x/y.json"


def test_model_cfg_version_passthrough(settings, info, plan, tmp_path):
    settings.transcode.optimizer.ssimulacra2_model = "version=ssimulacra2"
    plan.params.target_metric = "ssimulacra2"
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc._model_cfg() == "version=ssimulacra2"


# ---- engine config ----
def test_encoder_requires_target_quality(settings, info, tmp_path):
    plan = TranscodePlan()
    plan.params = VideoParams(engine="optimizer", target_quality="")
    with pytest.raises(Exception, match="target_quality"):
        make_encoder(settings, info, plan, tmp_path)


def test_probe_grid_respects_probes_cap(settings, info, plan, tmp_path):
    plan.params.probes = 3
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc._probe_grid() == [20, 24, 28]


def test_fmt_crf(settings, info, plan, tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)
    settings.transcode.optimizer.fractional_crf = False
    assert enc._fmt_crf(33.33) == "33"
    settings.transcode.optimizer.fractional_crf = True
    assert enc._fmt_crf(33.33) == "33.33"


def test_pick_all_crfs_clamps_to_grid(settings, info, plan, tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)
    enc.target = 75.0
    samples = {0: {20: 90.0, 24: 85.0, 28: 80.0, 32: 70.0}}
    chosen = enc.pick_all_crfs(samples, [20, 24, 28, 32])
    assert chosen[0] == pytest.approx(30.0)


# ---- detect_shots with a fake scenedetect ----
class _FrameNum:
    def __init__(self, n):
        self.frame_num = n


def _install_fake_scenedetect(scene_frames):
    mod = types.ModuleType("scenedetect")
    mod.ContentDetector = type("ContentDetector", (), {"__init__": lambda self, **k: None})
    mod.SceneManager = type("SceneManager", (), {
        "__init__": lambda self: setattr(self, "_scenes", None),
        "add_detector": lambda self, d: setattr(self, "_detector", d),
        "detect_scenes": lambda self, video, show_progress=False, callback=None: None,
        "get_scene_list": lambda self: [(_FrameNum(a), _FrameNum(b)) for a, b in scene_frames],
    })
    mod.open_video = lambda path: object()
    sys.modules["scenedetect"] = mod


def test_detect_shots(settings, info, plan, tmp_path):
    _install_fake_scenedetect([(0, 100), (100, 300), (300, 800)])
    enc = make_encoder(settings, info, plan, tmp_path)
    shots = enc.detect_shots()
    assert shots == [(0, 100), (100, 300), (300, 800)]


def test_detect_shots_merges_past_max(settings, info, plan, tmp_path):
    _install_fake_scenedetect([(0, 100), (100, 200), (200, 300)])
    settings.transcode.optimizer.max_shots = 2
    enc = make_encoder(settings, info, plan, tmp_path)
    assert len(enc.detect_shots()) == 2


def test_detect_shots_falls_back_to_single_shot(settings, info, plan, tmp_path):
    _install_fake_scenedetect([])
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc.detect_shots() == [(0, enc.total_frames)]


def test_detect_shots_reports_cuts_via_callback(settings, info, plan, tmp_path):
    mod = types.ModuleType("scenedetect")
    mod.ContentDetector = type("ContentDetector", (), {"__init__": lambda self, **k: None})

    class _SM:
        def add_detector(self, d):
            pass

        def detect_scenes(self, video, show_progress=False, callback=None):
            for pos in (100, 200, 400):
                callback(None, _FrameNum(pos))

        def get_scene_list(self):
            return [(_FrameNum(0), _FrameNum(100)), (_FrameNum(100), _FrameNum(200)),
                    (_FrameNum(200), _FrameNum(400)), (_FrameNum(400), _FrameNum(800))]

    mod.SceneManager = _SM
    mod.open_video = lambda path: object()
    sys.modules["scenedetect"] = mod
    enc = make_encoder(settings, info, plan, tmp_path)
    reports = []
    enc.progress_cb = lambda pct, stats: reports.append((pct, stats))
    shots = enc.detect_shots()
    assert shots == [(0, 100), (100, 200), (200, 400), (400, 800)]
    # scenedetect progress reports the running cut count (total=0 = indeterminate)
    sc = [s for _, s in reports if s.get("total") == 0 and s.get("done")]
    assert [s["done"] for s in sc] == [1, 2, 3]


# ---- vmaf feature config guard ----
def test_probe_ignores_av1an_style_vmaf_features(settings, info, plan, tmp_path):
    plan.params.probing_vmaf_features = "default motionless"
    enc = make_encoder(settings, info, plan, tmp_path)
    seen = {}

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        if any("libvmaf=" in a for a in args):
            seen["lavfi"] = args[args.index("-lavfi") + 1]
            log_path = seen["lavfi"].split("log_path=")[1].split(":")[0]
            Path(log_path).write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 90.0}}}))
            return ""
        return ""

    enc._run = fake_run.__get__(enc)
    ref = tmp_path / "r.y4m"
    ref.write_bytes(b"x")
    dist = tmp_path / "d.ivf"
    dist.write_bytes(b"x")
    enc._score_probe(ref, dist, 0, 28)
    assert "feature=default motionless" not in seen["lavfi"]
    assert "feature=" not in seen["lavfi"]


def test_probe_forwards_ffmpeg_style_vmaf_features(settings, info, plan, tmp_path):
    plan.params.probing_vmaf_features = "name=motion"
    enc = make_encoder(settings, info, plan, tmp_path)
    seen = {}

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        if any("libvmaf=" in a for a in args):
            seen["lavfi"] = args[args.index("-lavfi") + 1]
            log_path = seen["lavfi"].split("log_path=")[1].split(":")[0]
            Path(log_path).write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 90.0}}}))
            return ""
        return ""

    enc._run = fake_run.__get__(enc)
    ref = tmp_path / "r.y4m"
    ref.write_bytes(b"x")
    dist = tmp_path / "d.ivf"
    dist.write_bytes(b"x")
    enc._score_probe(ref, dist, 0, 28)
    assert "feature=name=motion" in seen["lavfi"]


def test_feature_warning_logged_once(settings, info, plan, tmp_path, monkeypatch):
    plan.params.probing_vmaf_features = "default motionless"
    enc = make_encoder(settings, info, plan, tmp_path)
    warnings = []
    monkeypatch.setattr(opt.logger, "warning", lambda *a, **k: warnings.append(a))

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        if any("libvmaf=" in a for a in args):
            lavfi = args[args.index("-lavfi") + 1]
            log_path = lavfi.split("log_path=")[1].split(":")[0]
            Path(log_path).write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 90.0}}}))
            return ""
        return ""

    enc._run = fake_run.__get__(enc)
    ref = tmp_path / "r.y4m"
    ref.write_bytes(b"x")
    dist = tmp_path / "d.ivf"
    dist.write_bytes(b"x")
    for crf in (20, 24, 28):
        enc._score_probe(ref, dist, 0, crf)
    assert len(warnings) == 1
    assert "ignoring probing_vmaf_features" in str(warnings[0][0])


# ---- full pipeline with a fake ffmpeg ----
def test_run_full_pipeline(settings, info, plan, tmp_path):
    _install_fake_scenedetect([(0, 300), (300, 900), (900, 1800)])
    plan.params.probes = 0
    enc = make_encoder(settings, info, plan, tmp_path)
    enc.fps = 30.0
    enc.total_frames = 1800
    score_at = {20: 95.0, 24: 89.0, 28: 83.0, 32: 77.0, 36: 71.0, 40: 65.0, 44: 59.0, 48: 53.0}
    stages = []
    progress = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        # shot extraction -> write y4m
        if "yuv4mpegpipe" in args:
            Path(args[-1]).write_bytes(b"YUV4MPEG2 dummy")
            return ""
        # probe/final encode -> write ivf
        if "-f" in args and args[args.index("-f") + 1] == "ivf" and "libsvtav1" in args:
            Path(args[-1]).write_bytes(b"ivf-dummy")
            return ""
        # vmaf score -> write score json (crf parsed from log_path filename)
        if any("libvmaf=" in a for a in args):
            lavfi = args[args.index("-lavfi") + 1]
            log_path = lavfi.split("log_path=")[1].split(":")[0]
            crf = int(Path(log_path).stem.rsplit("_", 1)[1])
            Path(log_path).write_text(json.dumps(
                {"pooled_metrics": {"vmaf": {"mean": score_at[crf]}}}))
            return ""
        # concat + mux -> create output
        Path(args[-1]).write_bytes(b"output-dummy")
        return ""

    enc._run = fake_run.__get__(enc)
    enc.stage_cb = lambda s: stages.append(s)
    progress = []
    enc.progress_cb = lambda pct, stats: progress.append((pct, stats))

    enc.run()

    assert enc.output.exists()
    assert stages[0] == "scenedetect"
    assert "probing" in stages
    assert "encoding" in stages
    assert progress[-1][0] == 100.0
    # probing progress reports the SHOT count (3 shots), stage-local pct 0->100
    probing = [s for _, s in progress if s.get("total") == 3]
    assert probing, "no probing progress reported"
    assert probing[-1]["done"] == 3
    assert probing[-1]["pct"] == pytest.approx(100.0, abs=1e-6)
    # encoding progress reports frames with a real fps and stage-local pct
    encoding = [s for _, s in progress if s.get("total") == 1800 and s.get("fps", 0) > 0]
    assert encoding, "no encoding progress with fps reported"
    assert encoding[-1]["pct"] == pytest.approx(100.0, abs=1e-6)
    # chosen crf for target 75 should interpolate between 32@77 and 36@71
    assert opt.pick_crf([(c, score_at[c]) for c in score_at], 75.0) == pytest.approx(33.33, abs=0.1)
