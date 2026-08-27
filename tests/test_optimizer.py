import json
import shutil
import sys
import time
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
    # unit tests don't spawn a real ffmpeg downscale pass; detect on source
    s.transcode.optimizer.scenedetect_scale = ""
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


def test_init_creates_output_dir(settings, info, plan, tmp_path):
    # output inside a source-relative av1/ subdir that does not exist yet
    out = tmp_path / "movie" / "av1" / "movie.av1.mkv"
    src = tmp_path / "movie" / "movie.mkv"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.touch()
    opt.ShotEncoder(settings, info, plan, src, out, tmp_path / "temp")
    assert out.parent.is_dir()


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


def test_parse_score_refuses_wrong_metric(tmp_path):
    """A default VMAF run pools adm/motion/vif alongside vmaf. Falling back to
    'the first pooled metric' for an absent metric returns integer_adm2 (~0.95)
    as if it were a 0-100 score - every shot then reads far below target and
    falls back to the lowest CRF."""
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"pooled_metrics": {
        "integer_adm2": {"mean": 0.9554}, "integer_motion2": {"mean": 1.4},
        "integer_vif_scale0": {"mean": 0.83}, "vmaf": {"mean": 85.29}}}))
    assert opt.parse_score(p, "vmaf") == 85.29
    with pytest.raises(Exception, match="ssimulacra2"):
        opt.parse_score(p, "ssimulacra2")
    # a log with a single pooled metric is still accepted by name mismatch
    p.write_text(json.dumps({"pooled_metrics": {"ssimulacra2": {"mean": 88.0}}}))
    assert opt.parse_score(p, "vmaf") == 88.0


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
    """Capping the probe count must SUBSAMPLE the grid, not truncate it.
    Truncating drops the high-CRF end, so every target cheaper than the
    surviving maximum clamps to it and the cap silently inflates output size."""
    settings.transcode.optimizer.probe_crfs = [20, 24, 28, 32, 36]
    plan.params.probes = 3
    enc = make_encoder(settings, info, plan, tmp_path)
    grid = enc._probe_grid()
    assert grid == [20, 28, 36]          # both endpoints kept
    plan.params.probes = 0
    assert make_encoder(settings, info, plan, tmp_path)._probe_grid() == [20, 24, 28, 32, 36]
    plan.params.probes = 9               # more than available -> unchanged
    assert make_encoder(settings, info, plan, tmp_path)._probe_grid() == [20, 24, 28, 32, 36]


def test_fmt_crf(settings, info, plan, tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)
    settings.transcode.optimizer.fractional_crf = False
    assert enc._fmt_crf(33.33) == "33"
    settings.transcode.optimizer.fractional_crf = True
    assert enc._fmt_crf(33.33) == "33.33"


def test_probe_window_caps_long_shots(settings, info, plan, tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)
    settings.transcode.optimizer.probe_max_frames = 120
    settings.transcode.optimizer.probing_rate = 1
    # short shot is probed whole
    assert enc._probe_window(300, 400) == (300, 400)
    # long shot -> a contiguous 120-frame window centred in the shot
    assert enc._probe_window(0, 1800) == (840, 960)
    # probing_rate widens the window so it still yields probe_max_frames frames
    settings.transcode.optimizer.probing_rate = 2
    assert enc._probe_window(0, 1800) == (780, 1020)


def test_pix_fmt_normalises_8bit(settings, info, plan, tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc._pix_fmt() == "yuv420p10le"
    # VideoParams spells 8-bit in a way ffmpeg does not know
    plan.params.pixel_format = "yuv420p8le"
    assert enc._pix_fmt() == "yuv420p"


def test_vmaf_threads_auto_sized_to_probe_pool(settings, info, plan, tmp_path, monkeypatch):
    monkeypatch.setattr(opt.os, "cpu_count", lambda: 32)
    enc = make_encoder(settings, info, plan, tmp_path)
    # ffmpeg's libvmaf default (0) is single-threaded, so never leave it unset
    enc._probe_worker_count = 4
    assert enc._vmaf_threads() == 8
    enc._probe_worker_count = 32
    assert enc._vmaf_threads() == 1
    # an explicit setting still wins
    settings.transcode.optimizer.vmaf_threads = 3
    assert enc._vmaf_threads() == 3


def test_probe_workers_ram_aware(settings, info, plan, tmp_path, monkeypatch):
    monkeypatch.setattr(opt.ShotEncoder, "_ram_gb", staticmethod(lambda: 31))
    monkeypatch.setattr(opt.os, "cpu_count", lambda: 32)
    info.width, info.height = 3840, 1920      # a 4K probe measured ~3.5GB
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc._probe_workers(100) == 5
    assert enc._probe_workers(2) == 2         # never more workers than tasks
    settings.transcode.optimizer.probe_workers = 12
    assert enc._probe_workers(100) == 12      # explicit setting wins


def test_vmaf_scale_filter(settings, info, plan, tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)
    # downscale-only and aspect preserving, so a 2:1 4K source is not squashed
    assert enc._vmaf_scale_filter() == "scale=w='min(iw,1920)':h=-2:flags=bicubic"
    settings.transcode.optimizer.vmaf_width = 0
    assert enc._vmaf_scale_filter() == ""


def test_encode_workers_ram_aware(settings, info, plan, tmp_path, monkeypatch):
    """Per-instance memory tracks frame size, so the worker count has to as
    well: a 4K instance measured 7.25GB against 2.5GB at 1080p. The old flat
    ram//8 rule gave 4 workers for both - too many to fit at 4K (it OOM-killed
    encodes) and needlessly few at 1080p."""
    monkeypatch.setattr(opt.ShotEncoder, "_ram_gb", staticmethod(lambda: 31))
    monkeypatch.setattr(opt.os, "cpu_count", lambda: 32)

    info.width, info.height = 3840, 1920
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc._encode_workers(11) == 3       # 3 x 7.4GB fits in 31GB * 0.75
    assert enc._encode_workers(1) == 1        # fewer shots than workers

    info.width, info.height = 1920, 1080
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc._encode_workers(99) == 8       # cheap instances -> cores // 4

    settings.transcode.optimizer.encode_workers = 6
    assert enc._encode_workers(11) == 6       # explicit setting wins

    # lp sizes the picture buffer pool and buys nothing above 4 (measured
    # 23.8fps/21.2GB at lp=4 vs 23.3fps/22.4GB at lp=6, 3 workers on 32 cores)
    assert enc._encode_lp(1) == 4
    assert enc._encode_lp(2) == 4
    assert enc._encode_lp(16) == 2
    # the probe path must respect SVT-AV1's own [0, 6] range, or every probe
    # gets clamped with a warning
    assert enc._svt_lp(1) == 6


def test_encode_threads_affinity(settings, info, plan, tmp_path, monkeypatch):
    enc = make_encoder(settings, info, plan, tmp_path)
    monkeypatch.setattr(opt.os, "cpu_count", lambda: 32)
    # auto: cores / workers
    assert enc._encode_threads(4) == 8
    assert enc._encode_threads(1) == 32
    # explicit setting wins and is clamped to cores
    settings.transcode.optimizer.encode_threads = 16
    assert enc._encode_threads(4) == 16
    settings.transcode.optimizer.encode_threads = 99
    assert enc._encode_threads(1) == 32
    settings.transcode.optimizer.encode_threads = 0
    # single worker spanning all cores -> no taskset wrapper
    assert enc._affinity_prefix(0, 32) == []
    # core slices are keyed on the worker slot, not the shot index: shots
    # finish out of order, so shot_idx % workers double-books a slice
    import concurrent.futures as cf
    import threading as th
    barrier = th.Barrier(4)  # hold all 4 pool threads live at once

    def slot(_):
        barrier.wait(timeout=5)
        return enc._worker_slot(4)

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        slots = list(ex.map(slot, range(4)))
    assert sorted(slots) == [0, 1, 2, 3]   # concurrent workers never collide
    assert enc._worker_slot(4) == enc._worker_slot(4)  # stable per thread
    # workers get disjoint, wrapping core ranges
    assert enc._affinity_prefix(0, 8) == ["taskset", "-c", "0-7"]
    assert enc._affinity_prefix(3, 8) == ["taskset", "-c", "24-31"]
    assert enc._affinity_prefix(4, 8) == ["taskset", "-c", "0-7"]  # wraps


def test_mkvmerge_mux_success(settings, info, plan, tmp_path, monkeypatch):
    enc = make_encoder(settings, info, plan, tmp_path)
    out = enc.output
    out.write_bytes(b"")
    video_only = tmp_path / "video_only.mkv"
    audio_subs = tmp_path / "audio_subs.mkv"
    video_only.touch()
    audio_subs.touch()

    def fake_run(cmd, capture_output=False, text=False, timeout=None):
        out.write_bytes(b"\x1aE\xdf\xa3")
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(opt.subprocess, "run", fake_run)
    assert enc._mkvmerge_mux("mkvmerge", video_only, audio_subs) is True
    assert out.stat().st_size > 0


def test_mkvmerge_mux_failure_falls_back(settings, info, plan, tmp_path, monkeypatch):
    enc = make_encoder(settings, info, plan, tmp_path)
    video_only = tmp_path / "video_only.mkv"
    audio_subs = tmp_path / "audio_subs.mkv"
    video_only.touch()
    audio_subs.touch()

    def fake_run(cmd, capture_output=False, text=False, timeout=None):
        return types.SimpleNamespace(returncode=1, stderr="boom")

    monkeypatch.setattr(opt.subprocess, "run", fake_run)
    assert enc._mkvmerge_mux("mkvmerge", video_only, audio_subs) is False


def test_pick_all_crfs_clamps_to_grid(settings, info, plan, tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)
    enc.target = 75.0
    samples = {0: {20: 90.0, 24: 85.0, 28: 80.0, 32: 70.0}}
    chosen = enc.pick_all_crfs(samples, [20, 24, 28, 32])
    assert chosen[0] == pytest.approx(30.0)


def test_pick_all_crfs_warns_and_floors_when_target_unreachable(
        settings, info, plan, tmp_path, monkeypatch):
    """A target above the whole probed curve silently pins every shot to the
    cheapest-quality/most-expensive CRF in the grid; that is what blows the
    output size up, so it must warn and honour min_crf."""
    enc = make_encoder(settings, info, plan, tmp_path)
    enc.target = 96.0
    warnings = []
    monkeypatch.setattr(opt.logger, "warning", lambda *a, **k: warnings.append(a))
    samples = {0: {20: 90.7, 24: 89.2, 28: 86.9, 32: 85.3},
               1: {20: 98.0, 24: 97.0, 28: 95.0, 32: 93.0}}

    chosen = enc.pick_all_crfs(samples, [20, 24, 28, 32])
    assert chosen[0] == 20.0            # unreachable -> bottom of the grid
    assert chosen[1] == pytest.approx(26.0)
    assert len(warnings) == 1
    assert "cannot reach" in str(warnings[0][0])

    settings.transcode.optimizer.min_crf = 26
    chosen = enc.pick_all_crfs(samples, [20, 24, 28, 32])
    assert chosen[0] == 26.0            # floored instead of running away
    assert chosen[1] == pytest.approx(26.0)


def test_pick_all_crfs_applies_probe_crf_offset(settings, info, plan, tmp_path):
    settings.transcode.optimizer.probe_crf_offset = 1.5
    enc = make_encoder(settings, info, plan, tmp_path)
    enc.target = 75.0
    samples = {0: {20: 90.0, 24: 85.0, 28: 80.0, 32: 70.0}}
    chosen = enc.pick_all_crfs(samples, [20, 24, 28, 32])
    assert chosen[0] == pytest.approx(31.5)
    # still clamped to the grid
    settings.transcode.optimizer.probe_crf_offset = 9.0
    assert enc.pick_all_crfs(samples, [20, 24, 28, 32])[0] == 32.0


def test_smooth_chosen_respects_min_crf(settings, info, plan, tmp_path):
    settings.transcode.optimizer.max_crf_delta = 4.0
    settings.transcode.optimizer.min_crf = 28
    settings.transcode.optimizer.probe_crfs = [20, 24, 28, 32, 36]
    enc = make_encoder(settings, info, plan, tmp_path)
    out = enc.smooth_chosen({0: 28.0, 1: 36.0, 2: 28.0})
    assert min(out.values()) >= 28.0


# ---- probe command construction ----
def _capture_probe(enc, tmp_path):
    """Run one probe with a fake ffmpeg, returning the two commands issued."""
    cmds = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        cmds.append(args)
        if any("libvmaf=" in a for a in args):
            lavfi = args[args.index("-lavfi") + 1]
            log_path = lavfi.split("log_path=")[1].split(":")[0]
            Path(log_path).write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 90.0}}}))
        elif "-f" in args and args[args.index("-f") + 1] == "ivf":
            Path(args[-1]).write_bytes(b"ivf-dummy")
        return ""

    enc._run = fake_run.__get__(enc)
    enc._probe_one(0, 0, 90, 28, lp=4)
    return cmds


def test_probe_encode_matches_final_encode_config(settings, info, plan, tmp_path):
    """The probe must measure the rate-distortion curve of the encode that
    actually ships: same resolution, bit depth, tune and keyint."""
    plan.params.tune = 0
    plan.params.keyint = 240
    enc = make_encoder(settings, info, plan, tmp_path)
    encode_cmd = _capture_probe(enc, tmp_path)[0]

    assert "-vf" not in encode_cmd                       # no probe downscale
    assert encode_cmd[encode_cmd.index("-pix_fmt") + 1] == "yuv420p10le"
    svt = encode_cmd[encode_cmd.index("-svtav1-params") + 1]
    assert "tune=0" in svt and "lp=4" in svt
    assert encode_cmd[encode_cmd.index("-g") + 1] == "240"
    # exactly the probe window of the source, read once (no y4m intermediate)
    assert encode_cmd.count("-i") == 1
    assert encode_cmd[encode_cmd.index("-i") - 1] == "3.000000"  # 90 frames @ 30fps


def test_probe_scores_distorted_against_reference(settings, info, plan, tmp_path):
    """ffmpeg's libvmaf takes #0 as the distorted input and #1 as the
    reference. Swapping them measures motion on the encode and lets VIF see
    detail being added, which inflates and flattens the whole CRF curve."""
    enc = make_encoder(settings, info, plan, tmp_path)
    vmaf_cmd = _capture_probe(enc, tmp_path)[1]

    inputs = [vmaf_cmd[i + 1] for i, a in enumerate(vmaf_cmd) if a == "-i"]
    assert inputs[0].endswith("probe_00000_28.ivf")      # distorted first
    assert inputs[1] == str(enc.source)                  # reference second
    lavfi = vmaf_cmd[vmaf_cmd.index("-lavfi") + 1]
    assert "[dist][ref]libvmaf=" in lavfi
    # both sides land on the model's 1080p domain, in the same pixel format
    assert lavfi.count("scale=w='min(iw,1920)':h=-2:flags=bicubic") == 2
    assert lavfi.count("format=yuv420p10le") == 2
    assert "shortest=1" in lavfi
    # mkv rounds PTS to whole ms, the ivf carries an exact frame-rate timebase;
    # framesync's default "nearest lower or equal" then slips a whole frame
    assert "ts_sync_mode=nearest" in lavfi


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


def test_detect_shots_reports_frame_progress(settings, info, plan, tmp_path):
    mod = types.ModuleType("scenedetect")
    mod.ContentDetector = type("ContentDetector", (), {"__init__": lambda self, **k: None})

    class _SM:
        def add_detector(self, d):
            pass

        def detect_scenes(self, video, show_progress=False, callback=None):
            for pos in (100, 200, 400):
                callback(None, _FrameNum(pos))
                time.sleep(0.1)

        def get_scene_list(self):
            return [(_FrameNum(0), _FrameNum(100)), (_FrameNum(100), _FrameNum(200)),
                    (_FrameNum(200), _FrameNum(400)), (_FrameNum(400), _FrameNum(800))]

    class _V:
        frame_number = 0

    mod.SceneManager = _SM
    mod.open_video = lambda path: _V()
    sys.modules["scenedetect"] = mod
    enc = make_encoder(settings, info, plan, tmp_path)
    reports = []
    enc.progress_cb = lambda pct, stats: reports.append((pct, stats))
    shots = enc.detect_shots()
    assert shots == [(0, 100), (100, 200), (200, 400), (400, 800)]
    # scene detection reports frame-level progress (done/total = frames)
    frame_reports = [s for _, s in reports if s.get("total") == enc.total_frames]
    assert frame_reports, "no frame progress reported during scene detection"


def test_make_detection_copy(settings, info, plan, tmp_path, monkeypatch):
    settings.transcode.optimizer.scenedetect_scale = "-2:540"
    enc = make_encoder(settings, info, plan, tmp_path)
    calls = {}
    monkeypatch.setattr(enc, "_run_with_progress",
                        lambda args, timeout, total_seconds, tag: calls.update(
                            {"args": args, "timeout": timeout, "total_seconds": total_seconds,
                             "tag": tag}) or (enc.probe_dir / "detect_copy.mkv").write_bytes(b"x"))
    p = enc._make_detection_copy()
    assert p is not None and p.exists()
    assert "-2:540" in " ".join(calls["args"])
    assert calls["tag"] == "downscale for detection"
    assert calls["total_seconds"] == pytest.approx(info.duration)


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
    dist = tmp_path / "d.ivf"
    dist.write_bytes(b"x")
    enc._score_probe(0, 100, dist, 0, 28)
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
    dist = tmp_path / "d.ivf"
    dist.write_bytes(b"x")
    enc._score_probe(0, 100, dist, 0, 28)
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
    dist = tmp_path / "d.ivf"
    dist.write_bytes(b"x")
    for crf in (20, 24, 28):
        enc._score_probe(0, 100, dist, 0, crf)
    assert len(warnings) == 1
    assert "ignoring probing_vmaf_features" in str(warnings[0][0])


# ---- full pipeline with a fake ffmpeg ----
def test_run_full_pipeline(settings, info, plan, tmp_path):
    _install_fake_scenedetect([(0, 300), (300, 900), (900, 1800)])
    plan.params.probes = 0
    enc = make_encoder(settings, info, plan, tmp_path)
    enc.fps = 30.0
    enc.total_frames = 1800
    score_at = {20: 95.0, 26: 86.0, 32: 77.0, 38: 68.0, 44: 59.0}
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


# ---- Dolby Vision Profile 5: per-shot shards instead of a whole-file convert ----
def _p5_encoder(settings, info, plan, tmp_path, cache_dir):
    plan.p5 = True
    info.width, info.height = 3840, 1920
    settings.transcode.dovi.p5_cache_dir = cache_dir
    return make_encoder(settings, info, plan, tmp_path)


def test_p5_probe_converts_once_per_shot(settings, info, plan, tmp_path):
    """Every CRF of a shot must share one converted shard: the probe pool runs
    one task per (shot, CRF), so converting per task would apply the RPU five
    times over, and a whole-file intermediate is ~100GB at 4K."""
    cache = tmp_path / "shm"
    cache.mkdir()
    enc = _p5_encoder(settings, info, plan, tmp_path, cache)
    cmds = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        cmds.append(args)
        if "ffv1" in args:                       # shard conversion
            Path(args[-1]).write_bytes(b"shard")
        elif any("libvmaf=" in a for a in args):
            lavfi = args[args.index("-lavfi") + 1]
            log_path = lavfi.split("log_path=")[1].split(":")[0]
            Path(log_path).write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 92.0}}}))
        elif "-f" in args and args[args.index("-f") + 1] == "ivf":
            Path(args[-1]).write_bytes(b"ivf")
        return ""

    enc._run = fake_run.__get__(enc)
    grid = [20, 32, 44]
    with __import__("concurrent.futures", fromlist=["x"]).ThreadPoolExecutor(3) as ex:
        list(ex.map(lambda c: enc._probe_one(0, 0, 90, c, 4), grid))

    shard_cmds = [c for c in cmds if "ffv1" in c]
    assert len(shard_cmds) == 1, "the shot was converted more than once"
    # the RPU is applied while building the shard, on a Vulkan device
    assert "-init_hw_device" in shard_cmds[0]
    assert any("apply_dolbyvision=1" in a for a in shard_cmds[0])
    assert shard_cmds[0][-1].startswith(str(cache))    # landed in the cache dir
    # and the shard is released once the last CRF is done
    assert not list(cache.iterdir())

    # every probe encode AND every VMAF reference now reads the shard, not the
    # raw ICtCp source - comparing against the unconverted source is meaningless
    for c in cmds:
        if "libsvtav1" in c or any("libvmaf=" in a for a in c):
            assert str(enc.source) not in c


def test_p5_shard_falls_back_to_workdir_when_cache_is_small(settings, info, plan, tmp_path,
                                                            monkeypatch):
    enc = _p5_encoder(settings, info, plan, tmp_path, tmp_path / "shm")
    (tmp_path / "shm").mkdir()
    assert enc._shard_dir(120) == tmp_path / "shm"
    # a shot far too big for the tmpfs goes to the work dir instead
    monkeypatch.setattr(opt.os, "statvfs",
                        lambda p: types.SimpleNamespace(f_bavail=1, f_frsize=4096))
    assert enc._shard_dir(120) == enc.tempdir


def test_p5_final_encode_applies_rpu_inline(settings, info, plan, tmp_path):
    """The final encode reads a shot exactly once, so there is nothing for a
    shard to amortise - the RPU goes straight into the encoder."""
    enc = _p5_encoder(settings, info, plan, tmp_path, tmp_path)
    cmds = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        cmds.append(args)
        Path(args[-1]).write_bytes(b"ivf")
        return ""

    enc._run = fake_run.__get__(enc)
    enc._encode_shot(0, 0, 90, 30.0, lp=4, threads=32, workers=1)
    cmd = cmds[0]
    assert "-init_hw_device" in cmd
    assert "apply_dolbyvision=1" in cmd[cmd.index("-vf") + 1]
    assert "ffv1" not in cmd                    # no staging file
    assert str(enc.source) in cmd


def test_non_p5_job_never_builds_a_shard(settings, info, plan, tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc._acquire_shard(0, 0, 90, []) is None
    enc._release_shard(0)                       # must be a no-op, not a crash


# ---- 4K sources get the 4K model, scored at native resolution ----
def test_4k_source_uses_4k_model_at_native_res(settings, info, plan, tmp_path):
    """vmaf_v0.6.1 is trained for 1080p at 3H. Downscaling a 4K pair to reach it
    reads optimistic (+0.80 at CRF 32, +1.63 at 38 measured), and the encoder
    spends that as lost sharpness."""
    info.width, info.height = 3840, 1920
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc._use_4k_model()
    assert enc._model_cfg() == "path=/usr/share/model/vmaf_4k_v0.6.1.json"
    assert enc._vmaf_scale_filter() == ""          # native, no downscale


def test_sub_4k_source_keeps_1080p_model_and_downscale(settings, info, plan, tmp_path):
    info.width, info.height = 1920, 1080
    enc = make_encoder(settings, info, plan, tmp_path)
    assert not enc._use_4k_model()
    assert enc._model_cfg() == "path=/usr/share/model/vmaf_v0.6.1.json"
    assert enc._vmaf_scale_filter().startswith("scale=w='min(iw,1920)'")


def test_4k_model_not_used_for_other_metrics(settings, info, plan, tmp_path):
    info.width, info.height = 3840, 1920
    plan.params.target_metric = "ssimulacra2"
    enc = make_encoder(settings, info, plan, tmp_path)
    assert not enc._use_4k_model()
    assert enc._model_cfg() == "version=ssimulacra2"


# ---- metric options: ssimulacra2 and xpsnr alongside vmaf ----
def _metric_encoder(settings, info, plan, tmp_path, metric, cache=None):
    plan.params.target_metric = metric
    info.width, info.height = 3840, 1920
    plan.colorspace, plan.color_trc, plan.color_primaries = "bt2020nc", "smpte2084", "bt2020"
    if cache is not None:
        settings.transcode.dovi.p5_cache_dir = cache
    return make_encoder(settings, info, plan, tmp_path)


def test_ssimulacra2_stages_a_reference_shard(settings, info, plan, tmp_path):
    """bestsource indexes a whole file before serving frames, so the reference
    must be a staged per-shot file rather than the multi-GB source."""
    cache = tmp_path / "shm"
    cache.mkdir()
    enc = _metric_encoder(settings, info, plan, tmp_path, "ssimulacra2", cache)
    assert enc._needs_shard()
    cmds = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        cmds.append(args)
        if "ffv1" in args:
            Path(args[-1]).write_bytes(b"shard")
            return ""
        if "app.vsmetrics" in args:
            return "93.512\n"
        Path(args[-1]).write_bytes(b"ivf")
        return ""

    enc._run = fake_run.__get__(enc)
    _, _, score = enc._probe_one(0, 0, 90, 32, lp=4)
    assert score == pytest.approx(93.512)

    shard_cmd = [c for c in cmds if "ffv1" in c][0]
    # a non-P5 shard carries no Dolby Vision filter, it is just the window
    assert "-init_hw_device" not in shard_cmd
    s2 = [c for c in cmds if "app.vsmetrics" in c][0]
    assert s2[s2.index("--step") + 1] == "4"          # scoring-side subsample
    # PQ has to be undone before a linear-light metric, not relabelled
    assert s2[s2.index("--transfer") + 1] == "st2084"
    assert s2[s2.index("--matrix") + 1] == "2020ncl"   # zimg spelling, not ffmpeg's


def test_xpsnr_parses_weighted_luma(settings, info, plan, tmp_path):
    enc = _metric_encoder(settings, info, plan, tmp_path, "xpsnr")
    assert not enc._needs_shard()      # reads the source directly, like vmaf

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        if any("xpsnr" in a for a in args):
            return ("[Parsed_xpsnr_2 @ 0x1] XPSNR  y: 43.6691  u: 49.5918  "
                    "v: 51.1706  (minimum: 43.6691)\n")
        Path(args[-1]).write_bytes(b"ivf")
        return ""

    enc._run = fake_run.__get__(enc)
    _, _, score = enc._probe_one(0, 0, 90, 32, lp=4)
    assert score == pytest.approx(43.6691)


def test_ssimulacra2_error_names_the_missing_plugins(settings, info, plan, tmp_path):
    cache = tmp_path / "shm"
    cache.mkdir()
    enc = _metric_encoder(settings, info, plan, tmp_path, "ssimulacra2", cache)

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        if "ffv1" in args:
            Path(args[-1]).write_bytes(b"shard")
            return ""
        if "app.vsmetrics" in args:
            raise opt.TranscodeError("boom")
        Path(args[-1]).write_bytes(b"ivf")
        return ""

    enc._run = fake_run.__get__(enc)
    with pytest.raises(Exception, match="vszip"):
        enc._probe_one(0, 0, 90, 32, lp=4)


# ---- subtitle codec selection for the mux ----
def _sub_args(enc, probe_out, fail=False):
    def fake_run(self, args, timeout=None):
        if fail:
            raise opt.TranscodeError("ffprobe exploded")
        return probe_out
    enc._run = fake_run.__get__(enc)
    return enc._subtitle_codec_args("/x/src.mkv")


def test_bitmap_subtitles_are_copied_not_converted(settings, info, plan, tmp_path):
    """A Blu-ray remux carries PGS, which is a BITMAP subtitle. Forcing srt on
    it fails the entire job: "Subtitle encoding currently only possible from
    text to text or bitmap to bitmap"."""
    enc = make_encoder(settings, info, plan, tmp_path)
    args = _sub_args(enc, "hdmv_pgs_subtitle\nhdmv_pgs_subtitle\n")
    assert args == ["-c:s:0", "copy", "-c:s:1", "copy"]
    assert "srt" not in args


def test_mov_text_is_still_converted(settings, info, plan, tmp_path):
    """tx3g only exists in MP4 - Matroska cannot carry it, so it must convert."""
    enc = make_encoder(settings, info, plan, tmp_path)
    assert _sub_args(enc, "mov_text\n") == ["-c:s:0", "srt"]


def test_mixed_subtitle_codecs_are_handled_per_stream(settings, info, plan, tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)
    args = _sub_args(enc, "mov_text\nhdmv_pgs_subtitle\nsubrip\ndvd_subtitle\n")
    assert args == ["-c:s:0", "srt", "-c:s:1", "copy",
                    "-c:s:2", "copy", "-c:s:3", "copy"]


def test_subtitle_probe_failure_falls_back_to_copy(settings, info, plan, tmp_path):
    """Copy is the safe default: it is right for every codec except tx3g,
    whereas srt is wrong for every bitmap one and kills the job."""
    enc = make_encoder(settings, info, plan, tmp_path)
    assert _sub_args(enc, "", fail=True) == ["-c:s", "copy"]
    assert _sub_args(enc, "\n") == ["-c:s", "copy"]
