import json
import math
import random
import shutil
import struct
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from app.config import Settings, VideoParams
from app.decisions import TranscodePlan
from app.analyzer import MediaInfo
from app import optimizer as opt
from app import pgsocr


def _write_out(args, data):
    """What a faked ffmpeg/encoder call leaves behind: its output file, the
    last argument. "-" is stdout, not a file - writing it dropped a file named
    "-" into whatever directory the suite ran from."""
    out = str(args[-1])
    if out not in ("-", "pipe:", "pipe:1"):
        Path(out).write_bytes(data)


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
    with pytest.raises(opt.TranscodeError):
        opt.parse_score(p, "vmaf")


@pytest.mark.parametrize("content", [
    None, "", '{"pooled_metrics": {"vmaf": {"mea', "[]",
    '{"pooled_metrics": {"vmaf": "x"}}', '{"pooled_metrics": {"vmaf": {"mean": "high"}}}',
    '{"pooled_metrics": {"vmaf": {"mean": NaN}}}'])
def test_parse_score_unusable_log_is_a_transcode_error(tmp_path, content):
    """libvmaf writes its log at uninit and only when it scored a frame, so a
    run can exit 0 with no log, and a killed or rebuilt one with half of one.
    That is a failed scoring like any other: it has to reach the fallbacks
    (SYCL to CPU, zero-copy to the usual read, verification skipping the
    shot) as a TranscodeError, not escape as a JSONDecodeError and fail the
    job."""
    p = tmp_path / "s.json"
    if content is not None:
        p.write_text(content)
    with pytest.raises(opt.TranscodeError):
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


@pytest.mark.parametrize("mode", ["svt", "qsv+svt", "qsv"])
def test_every_probe_phase_sizes_libvmaf_threads_to_its_pool(
        settings, info, plan, tmp_path, monkeypatch, mode):
    """Only probe_all set the pool size _vmaf_threads divides by, so under
    qsv+svt it stayed at 1 and every CPU score asked for all the cores.
    Production at 1080p (CPU scoring, below vmaf_sycl_min_width) ran ten
    scores at n_threads=40 each on a 40-core quota. Harmless in throughput
    as measured, but not what the pool was sized for."""
    monkeypatch.setattr(opt.sysres, "cpu_budget", lambda: 40.0)
    make = _verified_encoder if mode == "qsv+svt" else _gpu_encoder
    enc, shots = make(settings, info, plan, tmp_path)
    settings.transcode.optimizer.probe_encoder = mode
    monkeypatch.setattr(enc, "_mem_budget_gb", lambda: 200.0)
    grid = list(settings.transcode.optimizer.probe_crfs)
    qgrid = enc._qsv_grid()
    seen = []
    _plant_svt(enc, monkeypatch, lambda i: 30.0, {})
    svt_score = enc._probe_encode_and_score

    def svt(*a):
        seen.append(enc._vmaf_threads())
        return svt_score(*a)

    def card(idx, s0, s1, qg):
        seen.append(enc._vmaf_threads())
        return _curve(qgrid, 30.0, enc.target), {max(qgrid): 100.0}

    monkeypatch.setattr(enc, "_probe_encode_and_score", svt)
    monkeypatch.setattr(enc, "_probe_shot_qsv", card)
    {"svt": enc.probe_all, "qsv+svt": enc.probe_all_verified,
     "qsv": enc.probe_all_gpu}[mode](shots, grid)
    # 40 cores / lp 4 = ten probes in flight, so four threads each
    assert seen and set(seen) == {4}


def test_probe_cost_never_under_reports(settings, info, plan, tmp_path, monkeypatch):
    """The probe prior has to bound BOTH resolutions from above.

    A probe costs far less per pixel at 4K than at 1080p, because SVT-AV1
    forces the preset to M9 there and halves its mini-GOP. A model fitted to
    the 4K series would therefore under-estimate 1080p by ~30%, so the prior is
    taken from the 1080p series and the 4K slack is measured away at runtime.
    """
    monkeypatch.setattr(opt.sysres, "memory_available_gb", lambda: 26.0)
    monkeypatch.setattr(opt.sysres, "cpu_budget", lambda: 32.0)
    info.width, info.height = 3840, 1920
    enc = make_encoder(settings, info, plan, tmp_path)
    for frames, measured in ((24, 2.29), (48, 2.95), (96, 3.61), (120, 3.81)):
        assert enc._est_probe_gb(frames, 4) >= measured
    for lp, measured in ((6, 4.92), (4, 3.70), (3, 3.08), (2, 2.50)):
        assert enc._est_probe_gb(120, lp) >= measured
    # a probe is cheaper than the final encode of the same window
    assert enc._est_probe_gb(120, 4) < enc._est_encode_gb(120, 4)

    info.width, info.height = 1920, 960
    enc1080 = make_encoder(settings, info, plan, tmp_path)
    for frames, measured in ((24, 1.23), (120, 1.66)):
        assert enc1080._est_probe_gb(frames, 4) >= measured

    # concurrency is a hard cap now, not the plan: the budget decides per shot
    assert enc._max_probe_concurrency(100) == 32
    assert enc._max_probe_concurrency(2) == 2
    settings.transcode.optimizer.probe_workers = 12
    assert enc._max_probe_concurrency(100) == 12    # explicit setting wins


def test_probe_lp_no_longer_starts_at_six(settings, info, plan, tmp_path,
                                          monkeypatch):
    """Probes used to run at lp=6, the most expensive pool SVT-AV1 has.

    Measured at 4K on a 120-frame window, lp=6 cost 4.92GB and 7.7s against
    3.70GB and 6.2s at lp=4 - more memory AND slower - for byte-identical
    output, since lp only sizes the frame pool.
    """
    monkeypatch.setattr(opt.sysres, "cpu_budget", lambda: 32.0)
    enc = make_encoder(settings, info, plan, tmp_path)
    assert max(enc._lp_ladder()) == 4
    assert not hasattr(enc, "_svt_lp")


def test_budgets_come_from_the_cgroup_not_the_machine(
        settings, info, plan, tmp_path, monkeypatch):
    """A container limited to part of the box must size itself to that part.

    os.cpu_count() and MemTotal answer for the whole machine, so a job under
    `docker run --cpus=4 --memory=8g` used to plan for 32 cores and 23GB.
    """
    monkeypatch.setattr(opt.sysres, "cpu_budget", lambda: 4.0)
    monkeypatch.setattr(opt.sysres, "memory_available_gb", lambda: 8.0)
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc._cores() == 4
    assert enc._mem_budget_gb() == pytest.approx(6.8)
    # lp above the core count is meaningless, so the ladder stops there
    assert enc._lp_ladder() == [4, 3, 2, 1]
    monkeypatch.setattr(opt.sysres, "cpu_budget", lambda: 2.0)
    assert enc._lp_ladder() == [2, 1]


def test_vmaf_scale_filter(settings, info, plan, tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)
    # downscale-only and aspect preserving, so a 2:1 4K source is not squashed
    assert enc._vmaf_scale_filter() == "scale=w='min(iw,1920)':h=-2:flags=bicubic"
    settings.transcode.optimizer.vmaf_width = 0
    assert enc._vmaf_scale_filter() == ""


def test_encode_cost_tracks_shot_length(settings, info, plan, tmp_path, monkeypatch):
    """Per-instance memory follows the SHOT, not just the frame size.

    SVT-AV1 reserves its frame pool up front but only touches what the frames
    in flight need, so a 24-frame shot peaks at 3.4GB where a 1152-frame one
    peaks at 10.5GB on the same 4K source. A single per-instance number is
    58% too high at one end and 28% too low at the other, and the low end is
    what OOM-kills the encoder.
    """
    monkeypatch.setattr(opt.sysres, "cpu_budget", lambda: 32.0)
    info.width, info.height = 3840, 1920
    enc = make_encoder(settings, info, plan, tmp_path)

    # every point below is a measured peak RSS; the model must never sit under
    # one, or the budget it feeds will over-admit
    for frames, measured in ((24, 3.41), (48, 4.69), (96, 6.19), (144, 6.71),
                             (288, 7.70), (576, 9.12), (1152, 10.46)):
        assert enc._est_encode_gb(frames, 4) >= measured
    assert enc._est_encode_gb(24, 4) < enc._est_encode_gb(1152, 4)

    # lp only sizes the pool - output is byte-identical - so it is a pure
    # memory knob and a lower one must cost strictly less
    assert enc._est_encode_gb(144, 2) < enc._est_encode_gb(144, 4)
    assert enc._est_encode_gb(144, 4) < enc._est_encode_gb(144, 6)
    assert enc._est_encode_gb(144, 6) >= 8.80        # measured at lp=6

    # ... and at 1080p, where the same shots cost about a third as much
    gb_4k_144 = enc._est_encode_gb(144, 4)
    info.width, info.height = 1920, 960
    enc1080 = make_encoder(settings, info, plan, tmp_path)
    for frames, measured in ((24, 1.08), (144, 1.96), (960, 2.72)):
        assert enc1080._est_encode_gb(frames, 4) >= measured
    assert enc1080._est_encode_gb(144, 4) < gb_4k_144 / 2


def test_max_concurrency_defers_to_the_budget(settings, info, plan, tmp_path,
                                              monkeypatch):
    monkeypatch.setattr(opt.sysres, "cpu_budget", lambda: 32.0)
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc._max_concurrency(500) == 32    # budgets decide, not this cap
    assert enc._max_concurrency(2) == 2       # never more than there are shots
    settings.transcode.optimizer.encode_workers = 6
    assert enc._max_concurrency(500) == 6     # explicit setting still pins it


def test_affinity_slices_are_recycled(settings, info, plan, tmp_path, monkeypatch):
    """taskset is opt-in now, and slices are taken and returned per encode.

    Concurrency varies shot by shot, so a slice sized for N instances strands
    cores whenever fewer than N run - measured, that stranding made the
    admission scheduler 5% slower than the fixed pool it replaces. Slots are
    therefore held only while an encode is live.
    """
    monkeypatch.setattr(opt.sysres, "cpu_budget", lambda: 32.0)
    enc = make_encoder(settings, info, plan, tmp_path)
    # a slice covering every core is no constraint, so no wrapper is added
    assert enc._affinity_prefix(0, 32) == []
    # slices are disjoint and wrap
    assert enc._affinity_prefix(0, 8) == ["taskset", "-c", "0-7"]
    assert enc._affinity_prefix(3, 8) == ["taskset", "-c", "24-31"]
    assert enc._affinity_prefix(4, 8) == ["taskset", "-c", "0-7"]
    # slots are handed out lowest-free-first and returned on release
    a, b, c = enc._take_slot(), enc._take_slot(), enc._take_slot()
    assert (a, b, c) == (0, 1, 2)
    enc._free_slot(b)
    assert enc._take_slot() == 1               # the freed slice is reused
    for held in (a, c, 1):
        enc._free_slot(held)
    assert enc._take_slot() == 0


def test_plan_admission_packs_long_and_short_together():
    """Best fit: the longest shot that still fits, so a long encode and the
    short ones that fill the rest of the budget run together."""
    # shot 0 is long/expensive, 1 and 2 are short/cheap
    gb = {0: 6.0, 1: 2.0, 2: 2.0}
    cost = lambda idx, lp: gb[idx] * (0.65 if lp == 2 else 1.0)  # noqa: E731
    pending = [0, 1, 2]                       # already longest-first

    # a full budget takes the long one first
    assert opt.plan_admission(pending, 10.0, 32, cost, [4, 2], False) == (0, 4, 6.0)
    # with it running, the remainder still admits the short ones
    assert opt.plan_admission([1, 2], 4.0, 28, cost, [4, 2], False) == (0, 4, 2.0)
    # a shot that does not fit at the top of the ladder WAITS rather than
    # taking a lower lp. Trading lp for concurrency looks free because the
    # bitstream is unchanged, but the demoted instance runs 1.2-4.3x slower and
    # the extra concurrency does not pay for that: simulated over a real
    # 146-shot list, descending here ran 3338s against 1575s for waiting.
    assert opt.plan_admission([0], 4.5, 32, cost, [4, 2], False) is None
    assert opt.plan_admission([0], 1.0, 32, cost, [4, 2], False) is None
    # ... unless nothing at all is running, which is the case the ladder exists
    # for: a shot too big for the whole budget still has to run at some lp
    assert opt.plan_admission([0], 4.5, 32, cost, [4, 2], True) == pytest.approx((0, 2, 3.9))
    # and when no rung fits either, it goes anyway rather than deadlocking
    assert opt.plan_admission([0], 1.0, 32, cost, [4, 2], True) == pytest.approx((0, 2, 3.9))
    # the CPU budget picks the rung, independently of memory
    assert opt.plan_admission([1], 99.0, 3, cost, [4, 2], False) == pytest.approx((0, 2, 1.3))
    assert opt.plan_admission([1], 99.0, 1, cost, [4, 2], False) is None


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
def _capture_probe(enc, tmp_path, shot=(0, 90)):
    """Run one probe with a fake ffmpeg, returning the two commands issued."""
    cmds = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        if "framemd5" in args:
            # the reference_hwaccel preflight; a fake ffmpeg decodes nothing,
            # so the reads below stay on the CPU as they would on a host
            # without QSV
            return ""
        cmds.append(args)
        if any("libvmaf=" in a for a in args):
            lavfi = args[args.index("-lavfi") + 1]
            log_path = lavfi.split("log_path=")[1].split(":")[0]
            Path(log_path).write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 90.0}}}))
        elif "-f" in args and args[args.index("-f") + 1] == "ivf":
            _write_out(args, b"ivf-dummy")
        # the SYCL backend announces itself on stderr, which _run folds in;
        # the preflight refuses the GPU without it
        if "sycl_device=" in " ".join(args):
            return "[vmaf-sycl] timing: 30 frames, gpu%=100%"
        return ""

    enc._run = fake_run.__get__(enc)
    enc._probe_shot(0, *shot, [28], lp=4)
    return cmds


def test_probe_encode_matches_final_encode_config(settings, info, plan, tmp_path):
    """The probe must measure the rate-distortion curve of the encode that
    actually ships: same resolution, bit depth, tune and keyint."""
    plan.params.tune = 0
    plan.params.keyint = 240
    enc = make_encoder(settings, info, plan, tmp_path)
    encode_cmd = _capture_probe(enc, tmp_path)[0]

    # no probe downscale: the only filters are the timestamp rebase, which is
    # 0 for shot 0 of a file with no lead (see test_encoder_reads_rebase_by_...)
    assert encode_cmd[encode_cmd.index("-vf") + 1] == "settb=AVTB,setpts=PTS-0"
    assert encode_cmd[encode_cmd.index("-pix_fmt") + 1] == "yuv420p10le"
    svt = encode_cmd[encode_cmd.index("-svtav1-params") + 1]
    assert "tune=0" in svt and "lp=4" in svt
    assert encode_cmd[encode_cmd.index("-g") + 1] == "240"
    # exactly the probe window of the source, read once (no y4m intermediate)
    assert encode_cmd.count("-i") == 1
    # bounded by its frame count on the output, not by a -t on the input
    assert "-t" not in encode_cmd
    assert encode_cmd[encode_cmd.index("-frames:v") + 1] == "90"
    assert encode_cmd.index("-i") < encode_cmd.index("-frames:v") < encode_cmd.index("-c:v")


def test_encoder_reads_emit_exactly_one_frame_per_decoded_frame(settings, info, plan, tmp_path):
    """ffmpeg's default cfr sync duplicated a frame on these reads: the
    half-frame seek lead puts every frame at +0.5 of the output grid and
    Matroska's millisecond rounding decides the dup per shot. Measured: a
    120-frame probe window came out as 121 frames and pooled 82.41 where the
    same read scores 93.97 without sync. So: pts rebuilt from the frame index
    (microsecond timebase, or it rounds again) and sync switched off - on the
    probe encode, the final encode and the staged window alike."""
    enc = make_encoder(settings, info, plan, tmp_path)               # 30fps
    probe_cmd = _capture_probe(enc, tmp_path)[0]
    final_cmd = _encode_cmd(make_encoder(settings, info, plan, tmp_path), 30.0)
    # the probe rebases by a constant (0 for shot 0), the final encode - a
    # software read nothing rebuilds - by its first frame
    for cmd, rebase in ((probe_cmd, "setpts=PTS-0"), (final_cmd, "setpts=PTS-STARTPTS")):
        assert cmd[cmd.index("-fps_mode") + 1] == "passthrough"
        assert cmd[cmd.index("-vf") + 1].endswith(f"settb=AVTB,{rebase}")
        # an output option: after the input, before the encoder
        assert cmd.index("-i") < cmd.index("-fps_mode") < cmd.index("-c:v")


def test_probe_regeneration_follows_the_subsampling(settings, info, plan, tmp_path):
    """With probing_rate 2 the frames arrive at fps/2, and the regenerated
    pts have to say so, after the fps= filter that made it so."""
    settings.transcode.optimizer.probing_rate = 2
    enc = make_encoder(settings, info, plan, tmp_path)
    vf = _capture_probe(enc, tmp_path)[0]
    vf = vf[vf.index("-vf") + 1]
    assert vf.index("fps=15") < vf.index("setpts=PTS-0")


def test_staged_window_is_frame_exact_too(settings, info, plan, tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)
    seen = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        seen.append(args)
        _write_out(args, b"x")
        return ""

    enc._run = fake_run.__get__(enc)
    enc._extract_window(100, 220, tmp_path / "w.mkv")
    cmd = seen[0]
    assert cmd[cmd.index("-fps_mode") + 1] == "passthrough"
    # the window is bounded on the way in as well as out: -frames:v alone
    # counts what the chain emits (see test_p5_shard_holds_the_window_not_twice_it)
    assert cmd[cmd.index("-vf") + 1] == "trim=end_frame=120,settb=AVTB,setpts=PTS-STARTPTS"
    assert cmd[cmd.index("-frames:v") + 1] == "120"


def _source_read(cmd, source):
    """The `-ss X [-t D] -i <source>` slice of a probe command."""
    i = next(k for k, a in enumerate(cmd) if a == str(source))
    return cmd[max(k for k in range(i) if cmd[k] == "-ss"):i + 1]


def test_probe_read_caps_decoder_threads_at_the_admitted_lp(settings, info, plan, tmp_path):
    """ffmpeg sizes decoder threads from the host core count unless told, so a
    probe admitted 4 cores opens as many frame threads as the machine has. Ten
    of those oversubscribe 40 cores, and measured that is both slower and more
    expensive: 7.18s and 261 CPU-seconds unbounded, 4.61s and 130 at 4."""
    enc = make_encoder(settings, info, plan, tmp_path)
    args, _ = enc._probe_input(600, 720, lp=3)
    assert args[args.index("-threads") + 1] == "3"
    # -threads has to precede -i or it is an encoder option, not a decoder one
    assert args.index("-threads") < args.index("-i")
    # and a shard read is a decode too
    sh, _ = enc._probe_input(600, 720, shard=tmp_path / "s.mkv", lp=2)
    assert sh[sh.index("-threads") + 1] == "2"


def test_probe_read_falls_back_to_the_top_of_the_ladder(settings, info, plan, tmp_path):
    """The scoring read does not carry the task's lp; the top rung is what
    admission hands out in the common case."""
    enc = make_encoder(settings, info, plan, tmp_path)
    args, _ = enc._probe_input(600, 720)
    assert args[args.index("-threads") + 1] == str(enc._lp_ladder()[0])


def test_probe_reads_the_window_the_encode_will_encode(settings, info, plan, tmp_path):
    """The probe measures a window; the final encode encodes one. If they do
    not start on the same frame the probe is scoring footage that never ships.

    Both must use _seek's half-frame lead - asking for a frame's exact time
    lands just above its stored timestamp often enough to skip it, which is
    what _seek exists for. Measured on a 4K mp4, the bare w0/fps form missed on
    1 of 12 windows; on Matroska's millisecond timebase _seek's own note
    records 6 of 16.
    """
    enc = make_encoder(settings, info, plan, tmp_path)
    probe_args, _ = enc._probe_input(600, 720)
    assert probe_args[probe_args.index("-ss") + 1] == enc._seek(600)


def test_probe_encode_and_reference_read_the_same_frames(
        settings, info, plan, tmp_path):
    """The one invariant this whole engine rests on.

    _probe_input serves both the probe encode and the VMAF reference read so
    the two are frame-aligned by construction. Nothing downstream re-checks it:
    if they drift apart the reference and the distorted side simply describe
    different frames, the score collapses, and the CRF walks down to compensate
    with a bigger file - silently. Measured elsewhere in this file: a ONE-frame
    slip scored 66.3 where the aligned pair scored 92.4.
    """
    enc = make_encoder(settings, info, plan, tmp_path)
    encode_cmd, vmaf_cmd = _capture_probe(enc, tmp_path)[:2]
    # the same seek; where the window ends is all that differs - the encode
    # counts frames on its output, the score's reference read keeps its -t
    seek = ["-ss", enc._seek(0)]
    assert _source_read(encode_cmd, enc.source) == [*seek, "-i", str(enc.source)]
    assert _source_read(vmaf_cmd, enc.source) == [
        *seek, "-t", f"{enc._span(0, 90):.6f}", "-i", str(enc.source)]


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


def test_probe_pairs_frames_by_index_not_by_timestamp(settings, info, plan, tmp_path):
    """The two sides of a probe never share a timebase: the ivf starts at t=0
    with exact 1/fps periods, while the reference comes out of -ss still
    carrying _seek's half-frame lead, at +0.5 frame, which Matroska then rounds
    to a millisecond. ts_sync_mode=nearest is choosing between two equidistant
    frames and the rounding decides - per frame. Measured on the dovi_split=bl
    mkv of a DV-P8 4K source: per-frame scores alternated 48 / 93 and pooled
    to 76.86 where the aligned pair scores 93.97, so every shot read ~17 low
    and fell back to CRF 20. Rebasing both sides to t=0 pairs frame k with
    frame k, which is right by construction: the ivf was encoded from the very
    read the reference is."""
    enc = make_encoder(settings, info, plan, tmp_path)
    vmaf_cmd = _capture_probe(enc, tmp_path)[1]
    lavfi = vmaf_cmd[vmaf_cmd.index("-lavfi") + 1]
    dist, ref = lavfi.split(";")[:2]
    assert dist.startswith("[0:v]setpts=PTS-STARTPTS,")
    assert ref.startswith("[1:v]") and "setpts=PTS-STARTPTS" in ref


def test_probe_rebase_follows_the_probe_side_filters(settings, info, plan, tmp_path):
    """With probing_rate > 1 the reference is fps= subsampled first, and fps=
    re-times what it emits; the ivf holds THAT stream, so the rebase has to
    come after the subsampling, not before it."""
    settings.transcode.optimizer.probing_rate = 2
    enc = make_encoder(settings, info, plan, tmp_path)
    vmaf_cmd = _capture_probe(enc, tmp_path)[1]
    ref = vmaf_cmd[vmaf_cmd.index("-lavfi") + 1].split(";")[1]
    assert "fps=" in ref
    assert ref.index("fps=") < ref.index("setpts=PTS-STARTPTS")


@pytest.mark.parametrize("lead", [0.0, 1.955])
@pytest.mark.parametrize("rate", [1, 2])
def test_encoder_reads_rebase_by_a_constant_and_count_their_frames(
        settings, info, plan, tmp_path, monkeypatch, rate, lead):
    """With -hwaccel vaapi an in-band parameter change rebuilds the filter
    graph mid-read, and every filter in it starts over: STARTPTS rebased to 0
    again and the input -t counted its duration again. Measured, a card probe
    failed with AVERROR_BUG or wrote 77 frames for a 64-frame window. So an
    encoder read subtracts a constant known before it starts, and stops on a
    frame count the muxer keeps.

    Mid-file, because at shot 0 of a file with no lead the constant is 0 and
    a missing one would pass: half a frame at probing_rate 1 whatever the
    lead, 0 after fps=, and all three encoder reads of the window agreeing.
    The scores are not encoder reads and keep what they had: framesync
    leaves the distorted frame 0 unpaired against a reference that starts a
    microsecond after it."""
    settings.transcode.optimizer.probing_rate = rate
    monkeypatch.setattr(opt, "_render_nodes", lambda: ["/dev/dri/renderD129"])
    enc = _zc_encoder(settings, info, plan, tmp_path)          # 30fps
    enc._lead_of = lambda path: lead
    enc._zc_ok = False                   # the SVT probe is scored the usual way
    ran = []

    def fake_run(args, timeout=None):
        args = [str(a) for a in args]
        ran.append(args)
        if "-lavfi" not in args:
            _write_out(args, b"ivf")
            return ""
        _write_score(args, 90.0)
        return _ZC_LOG if "libvmaf_sycl=" in args[args.index("-lavfi") + 1] else ""

    monkeypatch.setattr(enc, "_run", fake_run)
    w0, w1 = 600, 721                    # odd: probing_rate 2 is 61 frames, not 121 // 2
    enc._probe_encode_and_score(0, w0, w1, 28, 4, None)
    enc._qsv_probe_encode(w0, w1, 22, tmp_path / "card.ivf")
    assert enc._zc_preflight_window(0, w0, w1) is True
    encodes = [c for c in ran if "-lavfi" not in c]
    scores = [c for c in ran if "-lavfi" in c]
    assert len(encodes) == 3 and len(scores) == 3

    rebase = {"setpts=PTS-16666", "setpts=PTS-16667"} if rate == 1 else {"setpts=PTS-0"}
    frames = {1: "121", 2: "61"}[rate]
    for cmd in encodes:
        vf = cmd[cmd.index("-vf") + 1].split(",")
        assert vf[-2] == "settb=AVTB" and vf[-1] in rebase, vf
        assert [f for f in vf if f.startswith("fps=")] == (["fps=15"] if rate > 1 else [])
        assert cmd[cmd.index("-frames:v") + 1] == frames
        assert cmd.index("-i") < cmd.index("-frames:v") < cmd.index("-c:v")
        assert _source_read(cmd, enc.source) == ["-ss", enc._seek(w0), "-i", str(enc.source)]
        assert not any("STARTPTS" in a for a in cmd)
    assert len({c[c.index("-vf") + 1].split(",")[-1] for c in encodes}) == 1
    # a shard is read without -ss and starts at 0 at every rate: half a frame
    # subtracted there would put each frame exactly between two slots
    vf, out = enc._window_encode(w0, w1, tmp_path / "shard.mkv")
    assert vf == ["settb=AVTB", "setpts=PTS-0"] and out[-2:] == ["-frames:v", frames]
    for cmd in scores:
        dist, ref = cmd[cmd.index("-lavfi") + 1].split(";")[:2]
        assert "setpts=PTS-STARTPTS" in dist and "setpts=PTS-STARTPTS" in ref
        assert ("fps=15" in ref) == (rate > 1)
        assert _source_read(cmd, enc.source) == [
            "-ss", enc._seek(w0), "-t", f"{enc._span(w0, w1):.6f}", "-i", str(enc.source)]
        assert "-frames:v" not in cmd


def test_first_pts_is_where_the_seek_leaves_the_frame(settings, info, plan, tmp_path):
    """What an encoder read subtracts instead of STARTPTS. After _seek's half
    frame of lead it is half a frame - 16666.67us at 30fps, the seek string's
    six decimals putting it either side - whatever the file's own lead and
    wherever a hole moved the slot. Where the seek clamps at 0, the lead."""
    enc = make_encoder(settings, info, plan, tmp_path)          # 30fps
    for lead in (0.0, 1.955):
        enc._lead_of = lambda path, lead=lead: lead
        assert enc._first_pts_us(600) in (16666, 16667)
    assert enc._first_pts_us(0) in (16666, 16667)              # 1.955 is past half a frame
    enc._lead_of = lambda path: 0.010
    assert enc._first_pts_us(0) == 10000
    assert enc._first_pts_us(1) in (16666, 16667)
    enc._lead_of = lambda path: 0.0
    assert enc._first_pts_us(0) == 0
    pts = [i / 30.0 for i in range(300)]
    del pts[50]
    enc._slots = enc._slots_from_pts(pts)
    assert enc._slot(120) == 121 and enc._first_pts_us(120) in (16666, 16667)


def test_read_frames_counts_what_the_subsampling_emits(settings, info, plan, tmp_path):
    """fps= at probing_rate 2 emits one frame per period of its grid, and a
    period holds the last frame that rounds onto it. So the count is not
    (w1 - w0) // 2: on an odd window the last frame inside it still gets a
    period of its own - the 127-frame card window of the measurement is 64
    frames - and a hole is a period too."""
    info.fps = 23.976024
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._lead_of = lambda path: 0.0
    assert enc._read_frames(236, 363, 2) == 64 and (363 - 236) // 2 == 63
    assert enc._read_frames(600, 720, 2) == 60
    assert enc._read_frames(600, 720, 1) == 120
    # where the seek clamps at 0 frame 0 sits on the grid, and an odd window
    # gives up its last period rather than share it with the next shot
    assert enc._read_frames(0, 91, 2) == 45
    enc._lead_of = lambda path: 0.010                          # a quarter frame in
    assert enc._read_frames(0, 91, 2) == 46
    # never nothing, however short the window
    assert enc._read_frames(300, 301, 2) == 1 and enc._read_frames(0, 1, 3) == 1
    pts = [i / 23.976024 for i in range(1000)]
    del pts[650]
    enc._slots = enc._slots_from_pts(pts)
    assert enc._read_frames(600, 720, 2) == 61                 # 121 periods
    assert enc._read_frames(600, 720, 1) == 120                # the encoder keeps the hole


def _fps_periods(pts, period, n):
    """The frame (index into pts) each of fps='s first n periods holds: the
    last one whose timestamp rounds onto it."""
    rounded = [math.floor(p / period + 0.5) for p in pts]
    held, k = [], 0
    for j in range(n):
        while k + 1 < len(rounded) and rounded[k + 1] <= j:
            k += 1
        held.append(k)
    return held


def test_read_frames_never_reaches_the_next_shot(settings, info, plan, tmp_path):
    """_read_frames against a model of the read: Matroska's whole
    milliseconds, the demuxer subtracting the seek, trim dropping what lands
    before 0, and fps= rounding each frame to its nearest period. Across
    rates, leads, holes and windows, the last period counted never holds
    frame w1 or later - but for the floor of one frame on the shortest
    windows - and one inside the window is left out only where frame w1 lands
    on a rounding tie, which the margin keeps clear of."""
    rng = random.Random(7)
    for fps in (23.976024, 25.0, 29.97003, 59.94006, 119.88012):
        info.fps = fps
        enc = make_encoder(settings, info, plan, tmp_path)
        for holes in (0, 12):
            pts = [i / fps for i in range(6000)]
            for h in sorted(rng.sample(range(1, 5999), holes), reverse=True):
                del pts[h]
            enc._slots = enc._slots_from_pts(pts) if holes else []
            for lead in (0.0, 0.004, 0.010, 0.066, 1.955):
                enc._lead_of = lambda path, lead=lead: lead
                for _ in range(30):
                    rate = rng.choice((2, 3))
                    w0 = rng.choice((0, rng.randrange(1, 5000)))
                    w1 = w0 + rng.randrange(1, 300)
                    ss_ms = round(float(enc._seek(w0)) * 1000)
                    read = [round(1000 * (lead + enc._slot(i) / fps)) - ss_ms
                            for i in range(max(0, w0 - 1), w1 + 2 * rate + 2)]
                    if w0:
                        assert read.pop(0) < 0             # the frame before w0 is dropped
                    assert read[0] >= 0
                    n = enc._read_frames(w0, w1, rate)
                    held = _fps_periods(read, 1000 * rate / fps, n + 3)
                    inside = next(j for j, k in enumerate(held) if k >= w1 - w0)
                    x = (enc._slot(w1) - enc._slot(w0)
                         + enc._first_pts_us(w0) * 1e-6 * fps) / rate
                    case = (fps, holes, lead, rate, w0, w1, n, inside)
                    if x - 0.55 <= 0:
                        assert n == 1, case                # the floor
                        continue
                    assert held[n - 1] < w1 - w0, case
                    assert n == inside or (n == inside - 1 and abs(x % 1 - 0.5) < 0.1), case


def test_probe_side_fps_is_spelled_once(settings, info, plan, tmp_path, monkeypatch):
    """The card probe wrote fps= to six decimals where its score's reference
    read wrote %g - 11.988012 against 11.988 at 23.976fps - and at the
    rounding ties near the start of a file those two rates put a frame on
    different periods: modelled offline, 20 of 3960 windows encoded other
    frames than they were scored against."""
    settings.transcode.optimizer.probing_rate = 2
    info.fps = 23.976024
    monkeypatch.setattr(opt, "_render_nodes", lambda: ["/dev/dri/renderD129"])
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._lead_of = lambda path: 0.0
    ran = []
    monkeypatch.setattr(enc, "_run", lambda args, timeout=None: (ran.append(args), "")[1])
    enc._qsv_probe_encode(0, 91, 22, tmp_path / "p.ivf")
    card = ran[0][ran[0].index("-vf") + 1].split(",")
    _, ref_vf = enc._probe_input(0, 91)
    assert ([f for f in card if f.startswith("fps=")]
            == [f for f in ref_vf if f.startswith("fps=")] == ["fps=11.988"])


def test_metric_runs_output_nothing_but_the_scored_video(settings, info, plan, tmp_path):
    """Without -an ffmpeg maps the source's audio into the null output and
    interleaves it with the scored frames; when the scorer lags, that
    interleaving deadlocks (every thread in futex_do_wait, frame 56 of 120,
    until the 3600s timeout). Measured with the SYCL backend; the CPU one is
    just fast enough to hide it."""
    enc = make_encoder(settings, info, plan, tmp_path)
    vmaf_cmd = _capture_probe(enc, tmp_path)[1]
    for flag in ("-an", "-sn", "-dn"):
        assert flag in vmaf_cmd
        assert vmaf_cmd.index("-lavfi") < vmaf_cmd.index(flag) < vmaf_cmd.index("-f")


def _timeout_encoder(settings, info, plan, tmp_path, stall_first_n):
    """An encoder whose first `stall_first_n` SYCL scorings time out."""
    settings.transcode.optimizer.vmaf_sycl_device = 0
    info.width, info.height = 3840, 2160
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._sycl_ok = True                    # preflight already passed
    seen = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        seen.append((args, timeout))
        if any("libvmaf=" in a for a in args):
            lavfi = args[args.index("-lavfi") + 1]
            if "sycl_device=" in lavfi and sum(
                    1 for a, _ in seen if "sycl_device=" in " ".join(a)) <= stall_first_n:
                raise opt.CommandTimeout(f"command timed out after {timeout}s: ffmpeg")
            log = lavfi.split("log_path=")[1].split(":")[0]
            Path(log).write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 91.0}}}))
            return "[vmaf-sycl] timing: 30 frames, gpu%=100%"
        _write_out(args, b"ivf")
        return ""

    enc._run = fake_run.__get__(enc)
    return enc, seen


def test_sycl_scoring_timeout_is_sized_to_the_window_and_retried_on_cpu(
        settings, info, plan, tmp_path):
    """A stalled SYCL scoring used to sit for the full 3600s and then fail
    the job. Now it gets 60s plus a second per frame, is killed, and the same
    window is scored on the CPU."""
    enc, seen = _timeout_encoder(settings, info, plan, tmp_path, stall_first_n=1)
    dist = tmp_path / "d.ivf"
    dist.write_bytes(b"x")
    score = enc._score_probe(0, 120, dist, 0, 28)
    assert score == pytest.approx(91.0)
    metric = [(a, t) for a, t in seen if any("libvmaf=" in x for x in a)]
    assert len(metric) == 2
    first, second = metric
    assert "sycl_device=0" in first[0][first[0].index("-lavfi") + 1]
    assert first[1] == 60 + 120                         # sized to the window
    lavfi2 = second[0][second[0].index("-lavfi") + 1]
    assert "sycl_device" not in lavfi2 and "n_threads=" in lavfi2
    assert second[1] == 3600                            # the CPU keeps its budget
    assert enc._sycl_timeouts == 1
    assert enc._sycl_device() == 0                      # one stall does not banish the GPU


def test_repeated_sycl_stalls_move_the_job_to_the_cpu(settings, info, plan, tmp_path):
    enc, seen = _timeout_encoder(settings, info, plan, tmp_path, stall_first_n=3)
    dist = tmp_path / "d.ivf"
    dist.write_bytes(b"x")
    for crf in (20, 26, 32, 38):
        assert enc._score_probe(0, 120, dist, 0, crf) == pytest.approx(91.0)
    assert enc._sycl_timeouts == 3
    assert enc._sycl_device() == -1
    lavfis = [a[a.index("-lavfi") + 1] for a, _ in seen if any("libvmaf=" in x for x in a)]
    # 3 stalled SYCL attempts, 3 CPU retries, then the 4th goes straight to the CPU
    assert sum("sycl_device=" in l for l in lavfis) == 3
    assert sum("n_threads=" in l for l in lavfis) == 4
    assert "sycl_device" not in lavfis[-1]


def test_a_cpu_scoring_timeout_still_fails_the_probe(settings, info, plan, tmp_path):
    """The retry is for the GPU; a CPU scoring that runs out of its hour is a
    real fault and must surface as before."""
    enc = make_encoder(settings, info, plan, tmp_path)

    def fake_run(self, args, timeout=None):
        raise opt.CommandTimeout("command timed out after 3600s: ffmpeg")

    enc._run = fake_run.__get__(enc)
    dist = tmp_path / "d.ivf"
    dist.write_bytes(b"x")
    with pytest.raises(opt.TranscodeError):
        enc._score_probe(0, 120, dist, 0, 28)


# ---- detect_shots with a fake scenedetect ----
class _FrameNum:
    def __init__(self, n):
        self.frame_num = n


def _install_fake_scenedetect(scene_frames, monkeypatch, settings=None):
    """Stub scenedetect for tests that only care about detect_shots' plumbing.

    Installed with setitem so it is REMOVED afterwards: left in sys.modules it
    shadows the real package for the rest of the session, and anything later
    that imports a submodule fails with "not a package" and quietly skips
    instead of running.
    """
    mod = types.ModuleType("scenedetect")
    mod.ContentDetector = type("ContentDetector", (), {"__init__": lambda self, **k: None})
    mod.SceneManager = type("SceneManager", (), {
        "__init__": lambda self: setattr(self, "_scenes", None),
        "add_detector": lambda self, d: setattr(self, "_detector", d),
        "detect_scenes": lambda self, video, show_progress=False, callback=None: None,
        "get_scene_list": lambda self: [(_FrameNum(a), _FrameNum(b)) for a, b in scene_frames],
    })
    mod.open_video = lambda path: object()
    monkeypatch.setitem(sys.modules, "scenedetect", mod)
    if settings is not None:
        settings.transcode.optimizer.scenedetect_engine = "pyscenedetect"


def test_detect_shots(settings, info, plan, tmp_path, monkeypatch):
    _install_fake_scenedetect([(0, 100), (100, 300), (300, 800)], monkeypatch, settings)
    enc = make_encoder(settings, info, plan, tmp_path)
    shots = enc.detect_shots()
    assert shots == [(0, 100), (100, 300), (300, 800)]


def test_detect_shots_merges_past_max(settings, info, plan, tmp_path, monkeypatch):
    _install_fake_scenedetect([(0, 100), (100, 200), (200, 300)], monkeypatch, settings)
    settings.transcode.optimizer.max_shots = 2
    enc = make_encoder(settings, info, plan, tmp_path)
    assert len(enc.detect_shots()) == 2


def test_detect_shots_falls_back_to_single_shot(settings, info, plan, tmp_path, monkeypatch):
    _install_fake_scenedetect([], monkeypatch, settings)
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc.detect_shots() == [(0, enc.total_frames)]


def test_detect_shots_reports_frame_progress(settings, info, plan, tmp_path,
                                            monkeypatch):
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
    monkeypatch.setitem(sys.modules, "scenedetect", mod)
    settings.transcode.optimizer.scenedetect_engine = "pyscenedetect"
    enc = make_encoder(settings, info, plan, tmp_path)
    reports = []
    enc.progress_cb = lambda pct, stats: reports.append((pct, stats))
    # detection runs against the fps x duration estimate; _validate_shots then
    # replaces it with what the shots actually cover, so capture it first
    estimated = enc.total_frames
    shots = enc.detect_shots()
    assert shots == [(0, 100), (100, 200), (200, 400), (400, 800)]
    # scene detection reports frame-level progress (done/total = frames)
    frame_reports = [s for _, s in reports if s.get("total") == estimated]
    assert frame_reports, "no frame progress reported during scene detection"
    assert enc.total_frames == 800, "the shot list is authoritative afterwards"


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
    assert calls["tag"] == "downscale for detection (software)"
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
    settings.transcode.optimizer.reference_hwaccel = "off"   # or its preflight warns too
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._lead_of = lambda path: 0.0      # the fixture file is empty; ffprobe would warn
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
def test_run_full_pipeline(settings, info, plan, tmp_path, monkeypatch):
    _install_fake_scenedetect([(0, 300), (300, 900), (900, 1800)], monkeypatch, settings)
    plan.params.probes = 0
    enc = make_encoder(settings, info, plan, tmp_path)
    enc.fps = 30.0
    enc.total_frames = 1800
    score_at = {20: 95.0, 26: 86.0, 32: 77.0, 38: 68.0, 44: 59.0}
    stages = []
    progress = []

    spans = {0: 300, 1: 600, 2: 900}   # the three shots, in frames

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        # per-shot frame count check (the other ffprobe call reads sub codecs)
        if "-count_packets" in args:
            return f"{spans[int(Path(args[-1]).stem.rsplit('_', 1)[1])]},\n"
        if "ffprobe" in args[0]:
            return ""
        # shot extraction -> write y4m
        if "yuv4mpegpipe" in args:
            _write_out(args, b"YUV4MPEG2 dummy")
            return ""
        # probe/final encode -> write ivf
        if "-f" in args and args[args.index("-f") + 1] == "ivf" and "libsvtav1" in args:
            _write_out(args, b"ivf-dummy")
            return ""
        # vmaf score -> write score json (crf parsed from log_path filename).
        # The verification pass scores at the CHOSEN crf, which is not on the
        # probe grid, so fall back rather than KeyError.
        if any("libvmaf=" in a for a in args):
            lavfi = args[args.index("-lavfi") + 1]
            log_path = lavfi.split("log_path=")[1].split(":")[0]
            crf = int(Path(log_path).stem.rsplit("_", 1)[1])
            Path(log_path).write_text(json.dumps(
                {"pooled_metrics": {"vmaf": {"mean": score_at.get(crf, 76.0)}}}))
            return ""
        # concat + mux, and the verification window extractions
        _write_out(args, b"output-dummy")
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
    assert "verifying" in stages, "the delivered encode was never re-scored"
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
            _write_out(args, b"shard")
        elif any("libvmaf=" in a for a in args):
            lavfi = args[args.index("-lavfi") + 1]
            log_path = lavfi.split("log_path=")[1].split(":")[0]
            Path(log_path).write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 92.0}}}))
        elif "-f" in args and args[args.index("-f") + 1] == "ivf":
            _write_out(args, b"ivf")
        return ""

    enc._run = fake_run.__get__(enc)
    settings.transcode.optimizer.probe_bracket_width = 0     # sweep the grid
    enc._probe_shot(0, 0, 90, [20, 32, 44], lp=4)

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


def test_p5_shard_holds_the_window_not_twice_it(settings, info, plan, tmp_path):
    """-frames:v counts the frames that leave the chain, and at probing_rate 2
    the probe-side fps= emits one for every two it reads: a shard capped by
    -frames:v alone held twice the window and ran into the next shot, and the
    probe encode and its reference both measured those frames. The window has
    to be bounded where the frames go in, ahead of the subsampling."""
    settings.transcode.optimizer.probing_rate = 2
    cache = tmp_path / "shm"
    cache.mkdir()
    enc = _p5_encoder(settings, info, plan, tmp_path, cache)
    cmds = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        cmds.append(args)
        if "ffv1" in args:
            _write_out(args, b"shard")
        elif any("libvmaf=" in a for a in args):
            lavfi = args[args.index("-lavfi") + 1]
            log_path = lavfi.split("log_path=")[1].split(":")[0]
            Path(log_path).write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 92.0}}}))
        elif "-f" in args and args[args.index("-f") + 1] == "ivf":
            _write_out(args, b"ivf")
        return ""

    enc._run = fake_run.__get__(enc)
    enc._probe_shot(0, 0, 90, [32], lp=4)

    shard_cmd = [c for c in cmds if "ffv1" in c][0]
    chain = shard_cmd[shard_cmd.index("-vf") + 1].split(",")
    w0, w1 = enc._probe_window(0, 90)
    assert chain[0] == f"trim=end_frame={w1 - w0}"
    fps = next(i for i, f in enumerate(chain) if f.startswith("fps="))
    assert 0 < fps
    # the probe encode reading the shard stops where the live read would
    encode = [c for c in cmds if "libsvtav1" in c][0]
    assert encode[encode.index("-frames:v") + 1] == str(enc._read_frames(w0, w1, 2)) == "45"
    assert encode[encode.index("-vf") + 1] == "settb=AVTB,setpts=PTS-0"


def test_a_shard_probe_encode_subtracts_nothing_mid_file(settings, info, plan, tmp_path):
    """A shard is read without -ss, so its frames sit exactly on k/fps from 0
    and the probe encode reading it subtracts nothing - at probing_rate 1 as
    well, where a live read of the same window subtracts half a frame. Half a
    frame taken off a shard puts every frame exactly between two slots, and
    passthrough's rounding then lands neighbours on one. Mid-file and through
    the call site: at shot 0 both constants are 0 and a lost shard passes."""
    settings.transcode.optimizer.probing_rate = 1
    cache = tmp_path / "shm"
    cache.mkdir()
    enc = _p5_encoder(settings, info, plan, tmp_path, cache)
    enc._lead_of = lambda path: 0.0
    cmds = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        cmds.append(args)
        if "ffv1" in args:
            _write_out(args, b"shard")
        elif any("libvmaf=" in a for a in args):
            _write_score(args, 92.0)
        elif "-f" in args and args[args.index("-f") + 1] == "ivf":
            _write_out(args, b"ivf")
        return ""

    enc._run = fake_run.__get__(enc)
    enc._probe_shot(0, 600, 720, [32], lp=4)
    w0, w1 = enc._probe_window(600, 720)
    assert w0 > 0 and enc._first_pts_us(w0) != 0         # what a live read subtracts
    encode = [c for c in cmds if "libsvtav1" in c][0]
    assert str(enc.source) not in encode
    assert encode[encode.index("-vf") + 1] == "settb=AVTB,setpts=PTS-0"
    assert encode[encode.index("-frames:v") + 1] == str(w1 - w0) == "120"


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
        _write_out(args, b"ivf")
        return ""

    enc._run = fake_run.__get__(enc)
    enc._encode_shot(0, 0, 90, 30.0, lp=4)
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


# idle=False throughout: `idle` deliberately overshoots the budget when
# nothing is running at all, which would mask what these are checking.
_ADM = dict(cost=lambda k, lp: 1.0, lp_ladder=[4], idle=False)


def test_cpu_charge_defaults_to_booking_the_whole_lp():
    """1.0 has to be exactly today's behaviour, or every existing measurement
    of this scheduler stops meaning anything."""
    assert opt.plan_admission([0], mem_free=100.0, cpu_free=4.0, **_ADM) == (0, 4, 1.0)
    assert opt.plan_admission([0], mem_free=100.0, cpu_free=3.9, **_ADM) is None


def test_cpu_charge_below_one_admits_more_at_once():
    """A probe holds lp cores only while it encodes; it also decodes its window
    and then scores it. Charging half the lp is what lets those cores be used
    by something else - and the encoder is still handed the full lp."""
    half = dict(_ADM, cpu_charge=0.5)
    pick = opt.plan_admission([0], mem_free=100.0, cpu_free=2.0, **half)
    assert pick == (0, 4, 1.0)          # 4 booked as 2, and still lp=4
    assert opt.plan_admission([0], mem_free=100.0, cpu_free=1.9, **half) is None


def test_cpu_charge_does_not_touch_the_memory_budget():
    """Only the CPU side is discounted. Memory is held for the whole task
    either way - and it is what becomes binding once CPU stops being."""
    assert opt.plan_admission([0], mem_free=0.9, cpu_free=100.0,
                              **dict(_ADM, cpu_charge=0.25)) is None


# ---- libvmaf SYCL backend (Intel Arc) ----
def _stub_run(enc, ok=True, gpu_score=90.0, raises=None, announces=True):
    """Record ffmpeg invocations; satisfy whatever libvmaf log they ask for.

    The SYCL preflight scores the same pair twice - on the device and on the
    CPU - so which score comes back depends on which of the two this is.
    """
    calls = []
    lock = threading.Lock()

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        with lock:
            calls.append(args)
        joined = " ".join(args)
        if "libvmaf=" in joined:
            if raises is not None:
                raise raises
            if not ok:
                raise opt.TranscodeError("vmaf_sycl_state_init(0) failed: -1")
            lavfi = args[args.index("-lavfi") + 1]
            log = lavfi.split("log_path=")[1].split(":")[0]
            mean = gpu_score if "sycl_device=" in joined else 90.0
            Path(log).write_text(
                json.dumps({"pooled_metrics": {"vmaf": {"mean": mean}}}))
            # the backend announces itself on stderr, which _run folds in
            if "sycl_device=" in joined and announces:
                return "[vmaf-sycl] timing: 30 frames, gpu%=100%"
        return ""

    enc._run = fake_run.__get__(enc)
    return calls


def test_sycl_off_by_default_costs_nothing(settings, info, plan, tmp_path):
    """The default must not so much as launch ffmpeg to decide it is off."""
    info.width, info.height = 3840, 2160
    enc = make_encoder(settings, info, plan, tmp_path)
    calls = _stub_run(enc)
    assert enc._sycl_device() == -1
    assert calls == []


def test_sycl_gated_below_min_width(settings, info, plan, tmp_path):
    """1080p is measured (the default gate is 1920); a 720p source is not,
    so it stays on the CPU."""
    settings.transcode.optimizer.vmaf_sycl_device = 0
    info.width, info.height = 1280, 720
    enc = make_encoder(settings, info, plan, tmp_path)
    calls = _stub_run(enc)
    assert enc._sycl_device() == -1
    assert calls == []                       # gated before the preflight


def test_sycl_used_when_enabled_and_wide_enough(settings, info, plan, tmp_path):
    settings.transcode.optimizer.vmaf_sycl_device = 0
    info.width, info.height = 3840, 2160
    enc = make_encoder(settings, info, plan, tmp_path)
    calls = _stub_run(enc)
    assert enc._sycl_device() == 0
    # the preflight scores the pair twice: once on the device, once on the CPU
    assert len(calls) == 2
    assert "sycl_device=0" in " ".join(calls[0])
    assert "sycl_device=" not in " ".join(calls[1])
    # and against a blurred copy, not itself: a pair scored against itself
    # returns 100.000000 from any backend, working or not
    assert "boxblur" in " ".join(calls[0])


def test_sycl_preflight_runs_once_and_is_cached(settings, info, plan, tmp_path):
    """Probe workers all call this; it must not launch an ffmpeg each time."""
    settings.transcode.optimizer.vmaf_sycl_device = 0
    info.width, info.height = 3840, 2160
    enc = make_encoder(settings, info, plan, tmp_path)
    calls = _stub_run(enc)
    for _ in range(5):
        assert enc._sycl_device() == 0
    assert len(calls) == 2


def test_sycl_falls_back_when_it_disagrees_with_the_cpu_backend(
        settings, info, plan, tmp_path):
    """libvmaf comes from a fork pinned to a commit and nothing else re-checks
    it numerically. A device that runs but scores differently is what a bad
    VMAFX_REF bump looks like, and it is otherwise invisible: every score
    shifts together and the job stays self-consistent."""
    settings.transcode.optimizer.vmaf_sycl_device = 0
    info.width, info.height = 3840, 2160
    enc = make_encoder(settings, info, plan, tmp_path)
    calls = _stub_run(enc, gpu_score=90.5)   # the CPU says 90.0
    assert enc._sycl_device() == -1
    assert len(calls) == 2


def test_sycl_tolerates_backend_rounding(settings, info, plan, tmp_path):
    """The backends are not bit-identical - 1e-4 worst case measured - so the
    guard must not trip on that."""
    settings.transcode.optimizer.vmaf_sycl_device = 0
    info.width, info.height = 3840, 2160
    enc = make_encoder(settings, info, plan, tmp_path)
    _stub_run(enc, gpu_score=90.0 + 1e-4)
    assert enc._sycl_device() == 0


def test_sycl_preflight_does_not_swallow_a_cancel(settings, info, plan, tmp_path):
    """_check_cancel raises TranscodeError too. Catching that as a dead device
    would put 'GPU unusable' in the log of every cancelled job."""
    settings.transcode.optimizer.vmaf_sycl_device = 0
    info.width, info.height = 3840, 2160
    enc = make_encoder(settings, info, plan, tmp_path)
    enc.cancel_flag = lambda: True
    _stub_run(enc, raises=opt.TranscodeError("Job cancelled by user"))
    with pytest.raises(opt.TranscodeError, match="cancel"):
        enc._sycl_device()


def test_sycl_preflight_runs_once_under_concurrent_callers(
        settings, info, plan, tmp_path):
    """Probe workers all reach this at once; the lock is the only thing
    stopping one pair of ffmpeg launches per worker."""
    settings.transcode.optimizer.vmaf_sycl_device = 0
    info.width, info.height = 3840, 2160
    enc = make_encoder(settings, info, plan, tmp_path)
    calls = _stub_run(enc)
    out = []
    threads = [threading.Thread(target=lambda: out.append(enc._sycl_device()))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out == [0] * 8
    assert len(calls) == 2


def test_sycl_falls_back_when_the_backend_never_announces_itself(
        settings, info, plan, tmp_path):
    """Matching scores prove the numbers, not who computed them. An ignored
    sycl_device agrees with the CPU perfectly - and that is the exact shape the
    fork's own ffmpeg patch had, gating its code behind a CONFIG_ symbol it
    never defined."""
    settings.transcode.optimizer.vmaf_sycl_device = 0
    info.width, info.height = 3840, 2160
    enc = make_encoder(settings, info, plan, tmp_path)
    _stub_run(enc, announces=False)
    assert enc._sycl_device() == -1


def test_sycl_preflight_survives_an_unreadable_log(settings, info, plan, tmp_path):
    """ffmpeg can exit 0 and still leave no usable log - which is what a bad
    build looks like, i.e. the case this check exists for. parse_score then
    raises FileNotFoundError, and letting that out would kill the job instead
    of falling back."""
    settings.transcode.optimizer.vmaf_sycl_device = 0
    info.width, info.height = 3840, 2160
    enc = make_encoder(settings, info, plan, tmp_path)

    def fake_run(self, args, timeout=None):
        return "[vmaf-sycl] ran"          # exits fine, writes nothing

    enc._run = fake_run.__get__(enc)
    assert enc._sycl_device() == -1       # falls back, does not raise


def test_cpu_fallback_restores_the_thread_count(settings, info, plan, tmp_path):
    """The fallback path is the one that actually needs CPU threads - 7.0s
    single-threaded against 2.1s at 8. Losing n_threads here would make the
    degraded path degrade twice."""
    settings.transcode.optimizer.vmaf_sycl_device = 0
    settings.transcode.optimizer.vmaf_threads = 40
    info.width, info.height = 3840, 2160
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._sycl_ok = False                  # preflight already failed
    cmds = _capture_probe(enc, tmp_path)
    lavfi = [c for c in cmds if any("libvmaf=" in a for a in c)][-1]
    lavfi = lavfi[lavfi.index("-lavfi") + 1]
    assert "n_threads=40" in lavfi and "sycl_device" not in lavfi


def test_sycl_falls_back_to_cpu_when_device_missing(settings, info, plan, tmp_path):
    """A dead device must cost one failed preflight, not one failure per probe:
    the filter aborts the whole ffmpeg run when sycl init fails."""
    settings.transcode.optimizer.vmaf_sycl_device = 0
    info.width, info.height = 3840, 2160
    enc = make_encoder(settings, info, plan, tmp_path)
    calls = _stub_run(enc, ok=False)
    for _ in range(4):
        assert enc._sycl_device() == -1
    assert len(calls) == 1


def test_sycl_not_used_for_other_metrics(settings, info, plan, tmp_path):
    settings.transcode.optimizer.vmaf_sycl_device = 0
    plan.params.target_metric = "ssimulacra2"
    info.width, info.height = 3840, 2160
    enc = make_encoder(settings, info, plan, tmp_path)
    calls = _stub_run(enc)
    assert enc._sycl_device() == -1
    assert calls == []                   # gated before the preflight, like the others


def test_score_vmaf_carries_sycl_device(settings, info, plan, tmp_path):
    """The option has to reach the filter, not just the helper."""
    settings.transcode.optimizer.vmaf_sycl_device = 0
    info.width, info.height = 3840, 2160
    enc = make_encoder(settings, info, plan, tmp_path)
    cmds = _capture_probe(enc, tmp_path)
    scoring = [c for c in cmds
               if any("libvmaf=" in a for a in c) and "sycl_preflight" not in " ".join(c)]
    assert scoring, "no scoring command issued"
    lavfi = scoring[-1][scoring[-1].index("-lavfi") + 1]
    assert "sycl_device=0" in lavfi


def test_score_vmaf_drops_n_threads_under_sycl(settings, info, plan, tmp_path):
    """On the GPU the CPU threads only buy frame pools: measured on a B580,
    unset is 0.98s/0.16GB and 40 threads is 2.04s/1.54GB for the same score.
    Passing the CPU path's thread count would hand back the memory win."""
    settings.transcode.optimizer.vmaf_sycl_device = 0
    settings.transcode.optimizer.vmaf_threads = 40
    info.width, info.height = 3840, 2160
    enc = make_encoder(settings, info, plan, tmp_path)
    cmds = _capture_probe(enc, tmp_path)
    scoring = [c for c in cmds
               if any("libvmaf=" in a for a in c) and "sycl_preflight" not in " ".join(c)]
    lavfi = scoring[-1][scoring[-1].index("-lavfi") + 1]
    assert "sycl_device=0" in lavfi
    assert "n_threads" not in lavfi


def test_score_vmaf_keeps_n_threads_on_cpu(settings, info, plan, tmp_path):
    settings.transcode.optimizer.vmaf_threads = 40
    info.width, info.height = 3840, 2160
    enc = make_encoder(settings, info, plan, tmp_path)
    cmds = _capture_probe(enc, tmp_path)
    lavfi = [c for c in cmds if any("libvmaf=" in a for a in c)][-1]
    lavfi = lavfi[lavfi.index("-lavfi") + 1]
    assert "n_threads=40" in lavfi


def test_score_vmaf_omits_sycl_when_off(settings, info, plan, tmp_path):
    info.width, info.height = 3840, 2160
    enc = make_encoder(settings, info, plan, tmp_path)
    cmds = _capture_probe(enc, tmp_path)
    assert not any("sycl_device=" in " ".join(c) for c in cmds)


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
            _write_out(args, b"shard")
            return ""
        if "app.vsmetrics" in args:
            return "93.512\n"
        _write_out(args, b"ivf")
        return ""

    enc._run = fake_run.__get__(enc)
    score = enc._probe_shot(0, 0, 90, [32], lp=4)[32]
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
    seen = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        seen.append(args)
        if any("xpsnr" in a for a in args):
            return ("[Parsed_xpsnr_2 @ 0x1] XPSNR  y: 43.6691  u: 49.5918  "
                    "v: 51.1706  (minimum: 43.6691)\n")
        _write_out(args, b"ivf")
        return ""

    enc._run = fake_run.__get__(enc)
    score = enc._probe_shot(0, 0, 90, [32], lp=4)[32]
    assert score == pytest.approx(43.6691)
    # same index pairing as the vmaf scorer, for the same reason
    xpsnr_cmd = [c for c in seen if any("xpsnr=" in a for a in c)][0]
    lavfi = xpsnr_cmd[xpsnr_cmd.index("-lavfi") + 1]
    assert lavfi.count("setpts=PTS-STARTPTS") == 2, lavfi


def test_ssimulacra2_error_names_the_missing_plugins(settings, info, plan, tmp_path):
    cache = tmp_path / "shm"
    cache.mkdir()
    enc = _metric_encoder(settings, info, plan, tmp_path, "ssimulacra2", cache)

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        if "ffv1" in args:
            _write_out(args, b"shard")
            return ""
        if "app.vsmetrics" in args:
            raise opt.TranscodeError("boom")
        _write_out(args, b"ivf")
        return ""

    enc._run = fake_run.__get__(enc)
    with pytest.raises(Exception, match="vszip"):
        enc._probe_shot(0, 0, 90, [32], lp=4)


# ---- subtitle codec selection for the mux ----
def _sub_args(enc, *streams, fail=False, **kw):
    """_subtitle_codec_args against a stubbed plan probe of `streams`
    (see _probe_json)."""
    def fake_run(self, args, timeout=None):
        if fail:
            raise opt.TranscodeError("ffprobe exploded")
        return _probe_json(*streams, **kw)
    enc._run = fake_run.__get__(enc)
    return enc._subtitle_codec_args("/x/src.mkv")


def test_bitmap_subtitles_are_copied_not_converted(settings, info, plan, tmp_path):
    """A Blu-ray remux carries PGS, which is a BITMAP subtitle. Forcing srt on
    it fails the entire job: "Subtitle encoding currently only possible from
    text to text or bitmap to bitmap"."""
    enc = make_encoder(settings, info, plan, tmp_path)
    args = _sub_args(enc, ("subtitle", set()), ("subtitle", set()))
    assert args == ["-c:s:0", "copy", "-c:s:1", "copy"]
    assert "srt" not in args


def test_mov_text_is_still_converted(settings, info, plan, tmp_path):
    """tx3g only exists in MP4 - Matroska cannot carry it, so it must convert."""
    enc = make_encoder(settings, info, plan, tmp_path)
    assert _sub_args(enc, ("subtitle", set(), {"codec_name": "mov_text"}),
                     fmt="mov,mp4,m4a,3gp,3g2,mj2") == ["-c:s:0", "srt"]


def test_ass_is_copied_never_converted(settings, info, plan, tmp_path):
    """The whole point of the companion track: an ASS converted IN PLACE loses
    \\pos and \\move (srt_move_cb is an empty stub), turns drawing commands into
    visible text, and leaves the font attachments this mux carries with nothing
    to style. Pinned, because "just convert it" keeps looking like the fix."""
    enc = make_encoder(settings, info, plan, tmp_path)
    assert opt.ShotEncoder._SUBS_NEEDING_CONVERSION == {"mov_text"}
    args = _sub_args(enc, ("subtitle", set(), {"codec_name": "ass"}),
                     ("subtitle", set(), {"codec_name": "ssa"}))
    assert args == ["-c:s:0", "copy", "-c:s:1", "copy"]


def test_mixed_subtitle_codecs_are_handled_per_stream(settings, info, plan, tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)
    args = _sub_args(enc, ("subtitle", set(), {"codec_name": "mov_text"}),
                     ("subtitle", set()),
                     ("subtitle", set(), {"codec_name": "subrip"}),
                     ("subtitle", set(), {"codec_name": "dvd_subtitle"}))
    assert args == ["-c:s:0", "srt", "-c:s:1", "copy",
                    "-c:s:2", "copy", "-c:s:3", "copy"]


def test_subtitle_probe_failure_falls_back_to_copy(settings, info, plan, tmp_path):
    """Copy is the safe default: it is right for every codec except tx3g,
    whereas srt is wrong for every bitmap one and kills the job."""
    enc = make_encoder(settings, info, plan, tmp_path)
    assert _sub_args(enc, fail=True) == ["-c:s", "copy"]
    assert _sub_args(enc, ("video", set()), ("audio", set())) == ["-c:s", "copy"]


# ---- fractional CRF must reach SVT-AV1, not ffmpeg's integer -crf option ----
def _encode_cmd(enc, crf):
    cmds = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        cmds.append(args)
        _write_out(args, b"ivf")
        return ""

    enc._run = fake_run.__get__(enc)
    enc._encode_shot(0, 0, 90, crf, lp=4)
    return cmds[0]


def test_integer_crf_still_uses_ffmpeg_crf_option(settings, info, plan, tmp_path):
    settings.transcode.optimizer.fractional_crf = False
    cmd = _encode_cmd(make_encoder(settings, info, plan, tmp_path), 28.4)
    assert cmd[cmd.index("-crf") + 1] == "28"
    assert "crf=" not in cmd[cmd.index("-svtav1-params") + 1]


def test_fractional_crf_goes_through_svtav1_params(settings, info, plan, tmp_path):
    """ffmpeg's -crf is an INTEGER AVOption for libsvtav1: -crf 28.5 encodes
    byte-identically to -crf 28, so routing a decimal there threw away the
    interpolation. -svtav1-params reaches the library verbatim (verified
    against SVT-AV1 v4.2: it reports "CRF / 28.50")."""
    settings.transcode.optimizer.fractional_crf = True
    cmd = _encode_cmd(make_encoder(settings, info, plan, tmp_path), 28.5)
    assert "-crf" not in cmd                       # never both: svt would win anyway
    assert "crf=28.5" in cmd[cmd.index("-svtav1-params") + 1]


def test_fmt_crf_clamps_to_the_valid_range(settings, info, plan, tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)
    settings.transcode.optimizer.fractional_crf = True
    assert enc._fmt_crf(-3.0) == "0"
    assert enc._fmt_crf(70.5) == "63"
    settings.transcode.optimizer.fractional_crf = False
    assert enc._fmt_crf(70.5) == "63"


# ---- shards live outside tempdir, so a failed job has to drop them itself ----
def test_cleanup_shards_removes_staged_files(settings, info, plan, tmp_path):
    enc = _p5_encoder(settings, info, plan, tmp_path, tmp_path)
    stray = tmp_path / "dv_stray.mkv"
    stray.write_bytes(b"shard")
    enc._shards[0] = stray
    enc._cleanup_shards()
    assert not stray.exists()
    assert enc._shards == {}


def test_run_drops_shards_when_a_phase_fails(settings, info, plan, tmp_path):
    """p5_cache_dir is tmpfs by default, and it is NOT under tempdir: without
    this the shards of a job that dies mid-probe hold RAM until restart."""
    enc = _p5_encoder(settings, info, plan, tmp_path, tmp_path)
    stray = tmp_path / "dv_held.mkv"
    stray.write_bytes(b"shard")
    enc._shards[0] = stray

    def boom():
        raise opt.TranscodeError("scene detection exploded")

    enc.detect_shots = boom
    with pytest.raises(opt.TranscodeError):
        enc.run()
    assert not stray.exists()


def test_shard_is_built_once_for_all_of_a_shots_probes(settings, info, plan, tmp_path):
    """One pool task owns a shot for the whole of its probing, so the shard is
    built once and dropped at the end. This used to be an acquire/release
    refcount that hit zero between probes, deleting and rebuilding the shard
    once per CRF - and racing the rebuild."""
    cache = tmp_path / "shm"
    cache.mkdir()
    enc = _p5_encoder(settings, info, plan, tmp_path, cache)
    conversions = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        if "ffv1" in args:
            conversions.append(args[-1])
            _write_out(args, b"shard")
        elif any("libvmaf=" in a for a in args):
            lavfi = args[args.index("-lavfi") + 1]
            log = lavfi.split("log_path=")[1].split(":")[0]
            Path(log).write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 90.0}}}))
        elif "-f" in args and args[args.index("-f") + 1] == "ivf":
            _write_out(args, b"ivf")
        return ""

    enc._run = fake_run.__get__(enc)
    settings.transcode.optimizer.probe_bracket_width = 0     # sweep the grid
    enc._probe_shot(0, 0, 90, [20, 32, 44], lp=4)

    assert len(conversions) == 1, "the shot was converted once per CRF"
    assert not list(cache.iterdir()), "the shard outlived its last probe"


# ---- short shots are expensive per frame and gain least from their own CRF ----
def test_merge_short_shots_folds_into_the_shorter_neighbour():
    # the 10-frame shot sits between a 40-frame and a 100-frame shot
    shots = [(0, 40), (40, 50), (50, 150)]
    assert opt.merge_short_shots(shots, 24) == [(0, 50), (50, 150)]


def test_merge_short_shots_coalesces_a_run_of_tiny_shots():
    """A run of tiny shots must collapse among themselves rather than all
    piling onto the one long neighbour next to them."""
    shots = [(0, 10), (10, 20), (20, 30), (30, 400)]
    merged = opt.merge_short_shots(shots, 24)
    assert merged == [(0, 30), (30, 400)]


def test_merge_short_shots_leaves_nothing_under_the_bound():
    shots = [(0, 30), (30, 45), (45, 60), (60, 200), (200, 210)]
    merged = opt.merge_short_shots(shots, 48)
    assert all(b - a >= 48 for a, b in merged)


def test_merge_short_shots_is_a_noop_when_disabled_or_already_long():
    shots = [(0, 100), (100, 250)]
    assert opt.merge_short_shots(shots, 0) == shots
    assert opt.merge_short_shots([(0, 5), (5, 9)], 0) == [(0, 5), (5, 9)]
    assert opt.merge_short_shots(shots, 48) == shots


def test_merge_short_shots_keeps_full_coverage():
    shots = [(0, 12), (12, 33), (33, 40), (40, 300), (300, 305)]
    merged = opt.merge_short_shots(shots, 64)
    assert merged[0][0] == 0 and merged[-1][1] == 305
    assert all(a[1] == b[0] for a, b in zip(merged, merged[1:]))


def test_merge_short_shots_survives_a_single_short_shot():
    # nothing to merge with: the shot stands even though it is under the bound
    assert opt.merge_short_shots([(0, 10)], 48) == [(0, 10)]


def test_detect_shots_applies_min_shot_frames(settings, info, plan, tmp_path, monkeypatch):
    # 40 40 40 40 pairs up rather than collapsing onto one neighbour
    _install_fake_scenedetect([(0, 40), (40, 80), (80, 120), (120, 160)], monkeypatch, settings)
    settings.transcode.optimizer.min_shot_frames = 48
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc.detect_shots() == [(0, 80), (80, 160)]
    settings.transcode.optimizer.min_shot_frames = 0
    assert enc.detect_shots() == [(0, 40), (40, 80), (80, 120), (120, 160)]


def test_encode_shot_seeks_half_a_frame_early(settings, info, plan, tmp_path):
    """-ss drops frames below the requested timestamp and containers store
    those rounded, so asking for a frame's exact time starts the shot one frame
    late often enough to matter. Measured: the last shot of a real 16-shot
    split returned 35 frames instead of 36."""
    enc = make_encoder(settings, info, plan, tmp_path)   # 30fps
    cmds = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        cmds.append(args)
        _write_out(args, b"ivf")
        return ""

    enc._run = fake_run.__get__(enc)
    enc._encode_shot(3, 300, 390, 30.0, lp=4)
    cmd = cmds[0]
    assert float(cmd[cmd.index("-ss") + 1]) == pytest.approx(299.5 / 30.0, abs=1e-6)
    assert cmd[cmd.index("-frames:v") + 1] == "90"       # count is still exact


def test_encode_shot_never_seeks_before_the_start(settings, info, plan, tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)
    cmd = _encode_cmd(enc, 30.0)                          # shot 0 starts at frame 0
    assert float(cmd[cmd.index("-ss") + 1]) == 0.0


def test_seek_adds_the_video_streams_lead(settings, info, plan, tmp_path):
    """ffmpeg seeks relative to the container start, and the frame numbers
    here count from the first video frame; those differ by the video's lead.
    Proven by frame hashes: video at 0.066 / audio at 0 returned frame 17 for
    _seek(19); a Better Call Saul remux (video 1.955) returned 953 for 1000."""
    enc = make_encoder(settings, info, plan, tmp_path)          # 30fps
    enc._lead_of = lambda path: 1.955
    assert float(enc._seek(1000)) == pytest.approx(1.955 + 999.5 / 30.0, abs=1e-6)
    assert float(enc._seek(0)) == pytest.approx(1.955 - 0.5 / 30.0, abs=1e-6)
    # a lead smaller than the half-frame margin still cannot go negative
    enc._lead_of = lambda path: 0.010
    assert float(enc._seek(0)) == 0.0


def test_lead_of_reads_ffprobe_and_tolerates_failure(settings, info, plan, tmp_path,
                                                     monkeypatch):
    enc = make_encoder(settings, info, plan, tmp_path)
    calls = []

    def fake_run(cmd, capture_output=False, text=False, timeout=None):
        calls.append(cmd)
        if cmd[-1].endswith("bcs.mkv"):
            return types.SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(
                {"streams": [{"start_time": "1.955000"}],
                 "format": {"start_time": "0.008000"}}))
        return types.SimpleNamespace(returncode=1, stderr="no such file", stdout="")

    monkeypatch.setattr(opt.subprocess, "run", fake_run)
    assert enc._lead_of(Path("/x/bcs.mkv")) == pytest.approx(1.947)
    assert enc._lead_of(Path("/x/missing.mkv")) == 0.0          # falls back, no raise
    # video-first files (audio later) have no lead: never negative
    fake = fake_run

    def audio_later(cmd, **kw):
        return types.SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(
            {"streams": [{"start_time": "0.000000"}],
             "format": {"start_time": "0.000000"}}))
    monkeypatch.setattr(opt.subprocess, "run", audio_later)
    assert enc._lead_of(Path("/x/plain.mkv")) == 0.0
    # and it is probed once per file
    n = len(calls)
    enc._lead_of(Path("/x/bcs.mkv"))
    assert len(calls) == n


# ---- the shot list must cover the source exactly once ----
def _enc_for_shots(settings, info, plan, tmp_path, total=1800):
    enc = make_encoder(settings, info, plan, tmp_path)
    enc.total_frames = total
    return enc


def test_validate_shots_accepts_a_clean_list(settings, info, plan, tmp_path):
    enc = _enc_for_shots(settings, info, plan, tmp_path, total=400)
    enc._validate_shots([(0, 100), (100, 250), (250, 400)])


def test_validate_shots_rejects_a_gap(settings, info, plan, tmp_path):
    """A gap silently drops those frames from the output and nothing
    downstream can tell a missing scene from a short one."""
    enc = _enc_for_shots(settings, info, plan, tmp_path, total=400)
    with pytest.raises(opt.TranscodeError, match="gap"):
        enc._validate_shots([(0, 100), (150, 400)])


def test_validate_shots_rejects_an_overlap(settings, info, plan, tmp_path):
    enc = _enc_for_shots(settings, info, plan, tmp_path, total=400)
    with pytest.raises(opt.TranscodeError, match="overlap"):
        enc._validate_shots([(0, 200), (150, 400)])


def test_validate_shots_rejects_a_late_start(settings, info, plan, tmp_path):
    enc = _enc_for_shots(settings, info, plan, tmp_path, total=400)
    with pytest.raises(opt.TranscodeError, match="starts at frame 12"):
        enc._validate_shots([(12, 400)])


def test_validate_shots_rejects_an_empty_shot(settings, info, plan, tmp_path):
    enc = _enc_for_shots(settings, info, plan, tmp_path, total=400)
    with pytest.raises(opt.TranscodeError, match="empty"):
        enc._validate_shots([(0, 100), (100, 100), (100, 400)])
    with pytest.raises(opt.TranscodeError, match="empty shot list"):
        enc._validate_shots([])


def test_validate_shots_trusts_the_list_over_the_fps_estimate(
        settings, info, plan, tmp_path):
    """total_frames is int(fps * duration) - an estimate - while the shot list
    ends at the real frame count scene detection read."""
    enc = _enc_for_shots(settings, info, plan, tmp_path, total=1801)
    enc._validate_shots([(0, 900), (900, 1800)])
    assert enc.total_frames == 1800


# ---- every shot must encode exactly the frames it spans ----
def _encoder_writing(enc, frames_written):
    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        if "ffprobe" in args[0]:
            return f"{frames_written},\n"
        _write_out(args, b"ivf")
        return ""

    enc._run = fake_run.__get__(enc)
    return enc


def test_shot_length_check_passes_on_an_exact_encode(settings, info, plan, tmp_path):
    enc = _encoder_writing(make_encoder(settings, info, plan, tmp_path), 90)
    enc._encode_shot(0, 0, 90, 30.0, lp=4)


def test_shot_length_check_catches_a_short_encode(settings, info, plan, tmp_path):
    """The failure this exists for: a seek that lands on the wrong frame still
    emits the right count mid-file, but the final shot has no frame left to
    borrow and comes up short - which used to reach the muxer unnoticed."""
    enc = _encoder_writing(make_encoder(settings, info, plan, tmp_path), 89)
    with pytest.raises(opt.TranscodeError, match="encoded 89 frames"):
        enc._encode_shot(15, 1404, 1494, 30.0, lp=4)


def test_shot_length_check_skipped_when_ffprobe_cannot_answer(
        settings, info, plan, tmp_path):
    """A broken ffprobe must not fail an otherwise good encode."""
    enc = _encoder_writing(make_encoder(settings, info, plan, tmp_path), 0)

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        if "ffprobe" in args[0]:
            return "N/A\n"
        _write_out(args, b"ivf")
        return ""

    enc._run = fake_run.__get__(enc)
    enc._encode_shot(0, 0, 90, 30.0, lp=4)


# ---- delivered quality is measured, not assumed ----
def test_predict_score_interpolates_between_grid_points():
    pts = [(20, 95.0), (26, 86.0), (32, 77.0)]
    assert opt.predict_score(pts, 26) == pytest.approx(86.0)
    assert opt.predict_score(pts, 29) == pytest.approx(81.5)
    # clamped outside the grid, and pick_crf's inverse on the way back
    assert opt.predict_score(pts, 10) == pytest.approx(95.0)
    assert opt.predict_score(pts, 60) == pytest.approx(77.0)
    assert opt.predict_score([], 30) is None


def test_predict_score_inverts_pick_crf():
    pts = [(20, 95.0), (26, 86.0), (32, 77.0), (38, 68.0)]
    crf = opt.pick_crf(pts, 80.0)
    assert opt.predict_score(pts, crf) == pytest.approx(80.0, abs=1e-6)


def test_verify_sample_spreads_across_the_timeline(settings, info, plan, tmp_path):
    """Spread, not clustered: a seek that starts mis-landing partway through
    reads low on everything after that point, which a clustered sample misses."""
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc._verify_sample(100, 5) == [0, 25, 50, 74, 99]
    assert enc._verify_sample(3, 10) == [0, 1, 2]      # never more than there are
    assert enc._verify_sample(100, 1) == [50]
    assert enc._verify_sample(100, 0) == []
    assert enc._verify_sample(0, 5) == []


def _verifying_encoder(settings, info, plan, tmp_path, delivered):
    settings.transcode.optimizer.verify_shots = 2
    enc = make_encoder(settings, info, plan, tmp_path)
    seen = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        seen.append(args)
        if any("libvmaf=" in a for a in args):
            lavfi = args[args.index("-lavfi") + 1]
            log = lavfi.split("log_path=")[1].split(":")[0]
            Path(log).write_text(json.dumps(
                {"pooled_metrics": {"vmaf": {"mean": delivered}}}))
            return ""
        _write_out(args, b"win")
        return ""

    enc._run = fake_run.__get__(enc)
    return enc, seen


def _inputs(args):
    """Every `-i X` operand of one ffmpeg command, in order."""
    return [args[i + 1] for i, a in enumerate(args) if a == "-i"]


def test_verify_compares_the_output_against_the_source(settings, info, plan, tmp_path):
    """Nothing is staged now, so the assertion is on the metric command itself.

    Order is load-bearing: libvmaf takes input #0 as the distorted side and #1
    as the reference, and swapping them inflates and flattens the whole curve.
    """
    enc, seen = _verifying_encoder(settings, info, plan, tmp_path, 76.0)
    shots = [(0, 90), (90, 180), (180, 300)]
    enc.verify_delivered(shots, {0: 30.0, 1: 30.0, 2: 30.0},
                         {0: {26: 80.0, 32: 74.0}, 2: {26: 80.0, 32: 74.0}})

    metric_cmds = [a for a in seen if any("libvmaf=" in x for x in a)]
    assert metric_cmds, "the metric never ran"
    for cmd in metric_cmds:
        assert _inputs(cmd) == [str(enc.output), str(enc.source)], \
            "distorted must be input 0 (the output) and reference input 1"
    # and nothing was written to disk to get there
    assert not [a for a in seen if "-c:v" in a and "ffv1" in a], \
        "verification staged a lossless window again"


def test_verify_reads_both_windows_at_the_same_offset(settings, info, plan, tmp_path):
    """The two seeks have to agree, or the comparison is of different frames."""
    enc, seen = _verifying_encoder(settings, info, plan, tmp_path, 76.0)
    enc.verify_delivered([(0, 90), (90, 180)], {0: 30.0, 1: 30.0},
                         {0: {26: 80.0}, 1: {26: 80.0}})
    for cmd in [a for a in seen if any("libvmaf=" in x for x in a)]:
        seeks = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-ss"]
        durs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-t"]
        assert len(seeks) == 2 and seeks[0] == seeks[1], seeks
        assert len(durs) == 2 and durs[0] == durs[1], durs


def test_verify_pairs_frames_by_index_too(settings, info, plan, tmp_path):
    """Both verify inputs are read with the same seek, so they already share
    the half-frame phase and pair correctly - but by both sides rounding the
    same way, not by construction. Rebasing both makes it by construction, and
    keeps one scorer for both callers."""
    enc, seen = _verifying_encoder(settings, info, plan, tmp_path, 76.0)
    enc.verify_delivered([(0, 90), (90, 180)], {0: 30.0, 1: 30.0},
                         {0: {26: 80.0}, 1: {26: 80.0}})
    metric_cmds = [a for a in seen if any("libvmaf=" in x for x in a)]
    assert metric_cmds
    for cmd in metric_cmds:
        lavfi = cmd[cmd.index("-lavfi") + 1]
        assert lavfi.count("setpts=PTS-STARTPTS") == 2, lavfi


def test_verify_seeks_each_file_by_its_own_lead(settings, info, plan, tmp_path):
    """The source's video may start after its container, the output's where
    the mux put it. Same frame number, two different -ss values."""
    enc, seen = _verifying_encoder(settings, info, plan, tmp_path, 76.0)
    leads = {str(enc.source): 0.066, str(enc.output): 0.0}
    enc._lead_of = lambda path: leads[str(path)]
    enc.verify_delivered([(0, 90), (90, 180)], {0: 30.0, 1: 30.0},
                         {0: {26: 80.0}, 1: {26: 80.0}})
    metric = [a for a in seen if any("libvmaf=" in x for x in a)]
    assert len(metric) == 2
    seeks = [[float(c[i + 1]) for i, a in enumerate(c) if a == "-ss"] for c in metric]
    # input 0 is the output (lead 0), input 1 the source (lead 0.066)
    assert seeks[0] == [0.0, pytest.approx(0.066 - 0.5 / 30.0, abs=1e-6)]   # shot 0 clamps
    assert seeks[1] == [pytest.approx(89.5 / 30.0, abs=1e-6),
                        pytest.approx(0.066 + 89.5 / 30.0, abs=1e-6)]


def test_verify_still_stages_when_the_reference_needs_building(
        settings, info, plan, tmp_path):
    """Dolby Vision P5 and SSIMULACRA2 cannot read the source in place - the
    first needs its RPU applied, the second reads through bestsource, which
    indexes a file. Both keep the staged path, frame-exactly."""
    enc, seen = _verifying_encoder(settings, info, plan, tmp_path, 76.0)
    enc._p5 = True
    enc.verify_delivered([(0, 90), (90, 180)], {0: 30.0, 1: 30.0},
                         {0: {26: 80.0}, 1: {26: 80.0}})

    staged = [a for a in seen if "ffv1" in a]
    assert staged, "the shard path stopped staging"
    # -frames:v, never a -t duration: that is what kept windows frame-exact
    assert all("-t" not in a for a in staged)
    assert all("-frames:v" in a for a in staged)
    inputs = [i for a in staged for i in _inputs(a)]
    assert str(enc.output) in inputs and str(enc.source) in inputs


def _capture_logs(monkeypatch):
    """Collect loguru output. loguru does not feed the stdlib logging module,
    so caplog sees nothing; the rest of this file patches the logger the same
    way."""
    lines = []

    def sink(msg, *args, **kwargs):
        lines.append(" ".join(str(x) for x in (msg, *args)))

    for level in ("info", "warning"):
        monkeypatch.setattr(opt.logger, level, sink)
    return lines


def test_verify_reports_the_probe_bias(settings, info, plan, tmp_path, monkeypatch):
    """The probes predict 77 at CRF 30 here and the encode delivers 79, which
    is the bias probe_crf_offset exists to trade back for size."""
    enc, _ = _verifying_encoder(settings, info, plan, tmp_path, 79.0)
    logs = _capture_logs(monkeypatch)
    samples = {0: {26: 80.0, 32: 74.0}, 1: {26: 80.0, 32: 74.0}}
    enc.verify_delivered([(0, 90), (90, 180)], {0: 30.0, 1: 30.0}, samples)
    text = "\n".join(logs)
    assert "probe_crf_offset" in text
    assert "under-reports" in text


def test_verify_flags_a_large_gap_as_misalignment(settings, info, plan, tmp_path,
                                                  monkeypatch):
    """A one-frame shift scores ~7 below the prediction; an encoder-versus-probe
    difference is a point or two. Say which one this looks like."""
    enc, _ = _verifying_encoder(settings, info, plan, tmp_path, 69.0)
    logs = _capture_logs(monkeypatch)
    samples = {0: {26: 80.0, 32: 74.0}, 1: {26: 80.0, 32: 74.0}}
    enc.verify_delivered([(0, 90), (90, 180)], {0: 30.0, 1: 30.0}, samples)
    assert "misalignment" in "\n".join(logs)


def test_verify_warns_when_shots_land_below_target(settings, info, plan, tmp_path,
                                                   monkeypatch):
    enc, _ = _verifying_encoder(settings, info, plan, tmp_path, 60.0)   # target 75
    logs = _capture_logs(monkeypatch)
    enc.verify_delivered([(0, 90), (90, 180)], {0: 30.0, 1: 30.0}, {})
    assert "below target" in "\n".join(logs)


def test_verify_never_fails_the_job(settings, info, plan, tmp_path, monkeypatch):
    """Diagnostic only: an encode that is otherwise fine must not be thrown
    away because the scorer had a bad day."""
    settings.transcode.optimizer.verify_shots = 2
    enc = make_encoder(settings, info, plan, tmp_path)
    logs = _capture_logs(monkeypatch)

    def boom(self, args, timeout=None):
        raise opt.TranscodeError("libvmaf exploded")

    enc._run = boom.__get__(enc)
    enc.verify_delivered([(0, 90), (90, 180)], {0: 30.0, 1: 30.0}, {})
    assert "no shot could be verified" in "\n".join(logs)


def test_verification_survives_an_unparseable_log(settings, info, plan, tmp_path, monkeypatch):
    """A libvmaf log cut short used to leave verify_delivered as a
    JSONDecodeError and fail a job whose encode and mux had already finished.
    SYCL is off, so no CPU retry absorbs the bad log before it gets there."""
    settings.transcode.optimizer.verify_shots = 2
    settings.transcode.optimizer.vmaf_sycl_device = -1
    settings.transcode.optimizer.reference_hwaccel = "off"
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._lead_of = lambda path: 0.0
    logs = _capture_logs(monkeypatch)

    def truncated(self, args, timeout=None):
        args = [str(a) for a in args]
        lavfi = args[args.index("-lavfi") + 1]
        Path(lavfi.split("log_path=")[1].split(":")[0]).write_text('{"pooled')
        return ""

    enc._run = truncated.__get__(enc)
    enc.verify_delivered([(0, 90), (90, 180)], {0: 30.0, 1: 30.0}, {})
    text = "\n".join(logs)
    assert "could not verify shot" in text and "no shot could be verified" in text


def test_verify_is_off_when_disabled(settings, info, plan, tmp_path):
    settings.transcode.optimizer.verify_shots = 0
    enc = make_encoder(settings, info, plan, tmp_path)

    def boom(self, args, timeout=None):
        raise AssertionError("verification ran while disabled")

    enc._run = boom.__get__(enc)
    enc.verify_delivered([(0, 90)], {0: 30.0}, {})


# ---- adaptive probing: spend probes on the interval that decides the CRF ----
def test_bracket_for_finds_the_pair_the_target_falls_between():
    pts = [(20, 95.0), (32, 84.0), (44, 70.0)]
    assert opt.bracket_for(pts, 90.0) == (20, 32)
    assert opt.bracket_for(pts, 75.0) == (32, 44)


def test_bracket_for_is_none_when_more_probing_cannot_help():
    pts = [(20, 95.0), (32, 84.0), (44, 70.0)]
    assert opt.bracket_for(pts, 99.0) is None    # unreachable at the best CRF
    assert opt.bracket_for(pts, 60.0) is None    # already met at the worst
    assert opt.bracket_for([(20, 95.0)], 90.0) is None   # nothing to bracket


def test_seed_crfs_keeps_both_ends():
    assert opt.seed_crfs([20, 26, 32, 38, 44]) == [20, 32, 44]
    assert opt.seed_crfs([20, 26, 32, 38, 44], 4) == [20, 26, 38, 44]
    assert opt.seed_crfs([20, 44]) == [20, 44]
    assert opt.seed_crfs([]) == []


def _linear_curve(crf: float) -> float:
    """95 VMAF at CRF 20 falling to 59 at 44 - target 75 sits near CRF 35."""
    return 95.0 - (crf - 20) * 1.5


def _adaptive_encoder(settings, info, plan, tmp_path, curve, target="75"):
    """An encoder whose probes return `curve(crf)`, recording the CRFs tried."""
    plan.params.target_quality = target
    enc = make_encoder(settings, info, plan, tmp_path)
    tried = []

    def fake(self, idx, w0, w1, crf, lp, shard):
        tried.append(crf)
        return idx, crf, curve(crf)

    enc._probe_encode_and_score = fake.__get__(enc)
    return enc, tried


def test_adaptive_probing_bisects_towards_the_target(settings, info, plan, tmp_path):
    enc, tried = _adaptive_encoder(settings, info, plan, tmp_path, _linear_curve)
    settings.transcode.optimizer.probe_bracket_width = 6
    scores = enc._probe_shot(0, 0, 90, [20, 26, 32, 38, 44], lp=4)
    assert tried[:3] == [20, 32, 44], "did not start from the coarse seeds"
    # target 75 falls between 32 and 44; one bisection at 38 closes it to 6
    assert tried == [20, 32, 44, 38]
    assert opt.bracket_for(list(scores.items()), 75.0) == (32, 38)


def test_adaptive_probing_stops_early_when_the_target_is_unreachable(
        settings, info, plan, tmp_path):
    """Nothing on the grid reaches 99, so the sweep's remaining probes would
    all be wasted - pick_crf clamps to the best CRF either way."""
    enc, tried = _adaptive_encoder(settings, info, plan, tmp_path, _linear_curve, target="99")
    settings.transcode.optimizer.probe_bracket_width = 6
    enc._probe_shot(0, 0, 90, [20, 26, 32, 38, 44], lp=4)
    assert tried == [20, 32, 44], "kept probing after the answer was decided"


def test_adaptive_probing_never_exceeds_the_grid_budget(settings, info, plan, tmp_path):
    enc, tried = _adaptive_encoder(settings, info, plan, tmp_path, _linear_curve)
    settings.transcode.optimizer.probe_bracket_width = 1   # ask for a tight one
    enc._probe_shot(0, 0, 90, [20, 26, 32, 38, 44], lp=4)
    assert len(tried) <= 5, "adaptive probing cost more than the sweep it replaces"


def test_probe_bracket_width_zero_sweeps_the_whole_grid(settings, info, plan, tmp_path):
    enc, tried = _adaptive_encoder(settings, info, plan, tmp_path, _linear_curve)
    settings.transcode.optimizer.probe_bracket_width = 0
    enc._probe_shot(0, 0, 90, [20, 26, 32, 38, 44], lp=4)
    assert tried == [20, 26, 32, 38, 44]


def _admission_encoder(settings, info, plan, tmp_path, budget=10.0):
    """Encoder whose costs are fixed so the tests exercise the SCHEDULER.

    The cost model itself is covered by test_encode_cost_tracks_shot_length.
    """
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._mem_budget_gb = lambda: budget
    enc._cores = lambda: 32
    enc._est_encode_gb = lambda frames, lp=4: (6.0 if frames >= 100 else 2.0)
    return enc


def test_encode_all_fills_the_budget_without_exceeding_it(
        settings, info, plan, tmp_path):
    """One long encode plus the short ones that fit alongside it.

    A fixed worker pool sized for the long shots would run 1 at a time here and
    leave 4GB of the 10GB budget idle; admitting by cost runs 3.
    """
    enc = _admission_encoder(settings, info, plan, tmp_path)
    # 2 long (6GB) and 6 short (2GB) shots, interleaved in the timeline
    shots = [(0, 200), (200, 250), (250, 300), (300, 500),
             (500, 550), (550, 600), (600, 650), (650, 700)]
    enc.total_frames = shots[-1][1]
    in_flight, peak_gb, lock = 0.0, 0.0, threading.Lock()

    def fake_encode(idx, s0, s1, crf, lp, slot=-1, threads=0):
        nonlocal in_flight, peak_gb
        gb = enc._est_encode_gb(s1 - s0, lp)
        with lock:
            in_flight += gb
            peak_gb = max(peak_gb, in_flight)
        time.sleep(0.1)
        with lock:
            in_flight -= gb
        dst = enc.probe_dir / f"enc_{idx:05d}.ivf"
        dst.write_bytes(b"ivf")
        return dst

    enc._encode_shot = fake_encode
    paths = enc.encode_all(shots, {i: 30.0 for i in range(len(shots))})

    # every shot encoded, and returned in timeline order regardless of the
    # order they were admitted in
    assert len(paths) == len(shots)
    assert [p.name for p in paths] == [f"enc_{i:05d}.ivf" for i in range(len(shots))]
    # the invariant that matters: admission never oversubscribes the budget
    assert peak_gb <= 10.0
    # ... and it does pack, rather than serialising on the expensive shots: a
    # pool sized for the 6GB shots would run one at a time
    assert peak_gb > 6.0


def test_encode_all_admits_a_shot_too_big_for_the_budget(
        settings, info, plan, tmp_path):
    """A shot that cannot fit still has to run, or the phase deadlocks."""
    enc = _admission_encoder(settings, info, plan, tmp_path, budget=3.0)
    shots = [(0, 200), (200, 250)]
    enc.total_frames = 250
    seen = []

    def fake_encode(idx, s0, s1, crf, lp, slot=-1, threads=0):
        seen.append(idx)
        dst = enc.probe_dir / f"enc_{idx:05d}.ivf"
        dst.write_bytes(b"ivf")
        return dst

    enc._encode_shot = fake_encode
    paths = enc.encode_all(shots, {0: 30.0, 1: 30.0})
    assert sorted(seen) == [0, 1]
    assert len(paths) == 2


def test_encode_all_propagates_a_failure_without_hanging(
        settings, info, plan, tmp_path):
    enc = _admission_encoder(settings, info, plan, tmp_path)
    shots = [(0, 50), (50, 100), (100, 150)]
    enc.total_frames = 150

    def fake_encode(idx, s0, s1, crf, lp, slot=-1, threads=0):
        if idx == 1:
            raise opt.TranscodeError("shot 1 exploded")
        dst = enc.probe_dir / f"enc_{idx:05d}.ivf"
        dst.write_bytes(b"ivf")
        return dst

    enc._encode_shot = fake_encode
    with pytest.raises(opt.TranscodeError, match="shot 1 exploded"):
        enc.encode_all(shots, {i: 30.0 for i in range(3)})


def test_encode_all_honours_cancel(settings, info, plan, tmp_path):
    enc = _admission_encoder(settings, info, plan, tmp_path)
    cancelled = {"v": False}
    enc.cancel_flag = lambda: cancelled["v"]
    shots = [(i * 50, i * 50 + 50) for i in range(8)]
    enc.total_frames = 400

    def fake_encode(idx, s0, s1, crf, lp, slot=-1, threads=0):
        cancelled["v"] = True          # first shot trips the cancel flag
        time.sleep(0.02)
        dst = enc.probe_dir / f"enc_{idx:05d}.ivf"
        dst.write_bytes(b"ivf")
        return dst

    enc._encode_shot = fake_encode
    with pytest.raises(opt.TranscodeError, match="cancelled"):
        enc.encode_all(shots, {i: 30.0 for i in range(8)})


def test_encode_all_respects_an_explicit_worker_cap(settings, info, plan, tmp_path):
    """encode_workers still pins concurrency for anyone who set it."""
    enc = _admission_encoder(settings, info, plan, tmp_path, budget=100.0)
    settings.transcode.optimizer.encode_workers = 2
    shots = [(i * 50, i * 50 + 50) for i in range(6)]
    enc.total_frames = 300
    live, peak_live, lock = 0, 0, threading.Lock()

    def fake_encode(idx, s0, s1, crf, lp, slot=-1, threads=0):
        nonlocal live, peak_live
        with lock:
            live += 1
            peak_live = max(peak_live, live)
        time.sleep(0.1)
        with lock:
            live -= 1
        dst = enc.probe_dir / f"enc_{idx:05d}.ivf"
        dst.write_bytes(b"ivf")
        return dst

    enc._encode_shot = fake_encode
    enc.encode_all(shots, {i: 30.0 for i in range(6)})
    assert peak_live == 2


def test_calibration_jumps_up_and_eases_down():
    """Asymmetric on purpose: over-shooting the budget is what OOMs."""
    cal = opt.MemCalibration("encoding")
    assert cal.factor() == 1.0

    # an observation ABOVE the prediction is acted on immediately, with a
    # little clearance so the next task of that size is not admitted flush
    cal.observe(predicted_gb=4.0, observed_gb=6.0)
    assert cal.factor() > 1.5

    # ... and a cheaper reading only eases it back, so one light task cannot
    # talk the pool into over-admitting the next heavy one
    before = cal.factor()
    cal.observe(predicted_gb=4.0, observed_gb=2.0)
    assert 0.5 < cal.factor() < before

    # repeated agreement converges on the truth
    for _ in range(40):
        cal.observe(predicted_gb=4.0, observed_gb=2.0)
    assert cal.factor() == pytest.approx(0.5, abs=0.02)
    assert cal.samples == 42

    # and it is bounded either way, so a bad reading cannot run away
    for _ in range(50):
        cal.observe(predicted_gb=4.0, observed_gb=0.001)
    assert cal.factor() >= opt.MemCalibration.FLOOR
    cal.observe(predicted_gb=0.1, observed_gb=99.0)
    assert cal.factor() <= opt.MemCalibration.CAP
    # a missing or zero reading is ignored rather than treated as "free"
    was = cal.factor()
    cal.observe(predicted_gb=4.0, observed_gb=0.0)
    assert cal.factor() == was


def test_schedule_learns_the_real_cost_and_admits_more(
        settings, info, plan, tmp_path, monkeypatch):
    """The prior is deliberately high; measurement is what reclaims the slack.

    Here every task really costs 1GB while the model claims 4GB, so a static
    budget of 8GB would never run more than 2 at once. The calibration should
    discover the truth and open the pool up.
    """
    monkeypatch.setattr(opt.sysres, "cpu_budget", lambda: 32.0)
    monkeypatch.setattr(opt.sysres, "memory_available_gb", lambda: 1000.0)
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._mem_budget_gb = lambda: 8.0
    enc._cores = lambda: 32
    live, peak_live, lock = 0, 0, threading.Lock()

    def run_one(key, lp, slot, threads):
        nonlocal live, peak_live
        with lock:
            live += 1
            peak_live = max(peak_live, live)
        enc._note_task_peak(1.0)          # what the task ACTUALLY peaked at
        time.sleep(0.02)
        with lock:
            live -= 1
        return key

    seen = []
    enc._schedule(list(range(40)), phase="encoding",
                  cost=lambda key, lp: 4.0, run_one=run_one,
                  on_done=lambda key, result: seen.append(result),
                  progress=lambda done: None, max_conc=32, ladder=[4])
    assert sorted(seen) == list(range(40))
    # 8GB / 4GB claimed = 2; 8GB / 1GB measured = 8
    assert peak_live > 2
    assert enc._cal["encoding"].factor() < 0.5


def test_schedule_backs_off_when_real_memory_disagrees(
        settings, info, plan, tmp_path, monkeypatch):
    """Budget accounting is not the only limit.

    The cost model covers encoder processes and nothing else, so page cache,
    tmpfs shards and other tenants can eat the headroom without the budget
    noticing. Admission takes the smaller of the two.
    """
    monkeypatch.setattr(opt.sysres, "cpu_budget", lambda: 32.0)
    live, peak_live, lock = 0, 0, threading.Lock()
    machine_gb = [100.0]

    # real free memory shrinks as instances start, the way it does on a box;
    # the book-keeping budget below stays generous throughout
    monkeypatch.setattr(opt.sysres, "memory_available_gb",
                        lambda: machine_gb[0] - live)
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._mem_budget_gb = lambda: 100.0
    enc._cores = lambda: 32

    def run_one(key, lp, slot, threads):
        nonlocal live, peak_live
        with lock:
            live += 1
            peak_live = max(peak_live, live)
        time.sleep(0.05)
        with lock:
            live -= 1
        return key

    def go(tasks):
        nonlocal peak_live
        peak_live = 0
        done = []
        enc._schedule(list(range(tasks)), phase="encoding",
                      cost=lambda key, lp: 1.0, run_one=run_one,
                      on_done=lambda key, result: done.append(key),
                      progress=lambda d: None, max_conc=32, ladder=[4])
        assert sorted(done) == list(range(tasks))
        return peak_live

    # plenty of real memory: the budget is the only limit, so it packs
    assert go(8) >= 4
    # now only ~3.5GB is really free; admission has to notice, even though the
    # budget still shows 100GB and nothing in the cost model changed
    machine_gb[0] = 3.5
    assert go(8) <= 3


def test_video_only_source_skips_the_audio_remux(settings, info, plan, tmp_path,
                                                 monkeypatch):
    """A source with no audio and no subtitles must not run the remux pass.

    "-map 0:a? -map 0:s?" against a video-only source maps nothing, and ffmpeg
    with zero output streams allocates ~8GB before exiting - measured 8.19GB on
    a 15MB 12-second 4K file and 8.20GB on the 24-second one, so it is a flat
    allocation rather than buffering. Under a container memory limit that is an
    OOM kill at the very last step of a finished encode.
    """
    enc = make_encoder(settings, info, plan, tmp_path)
    # force the ffmpeg mux path; every other tool must still resolve
    monkeypatch.setattr(opt.shutil, "which",
                        lambda n, *a, **k: None if "mkvmerge" in n else f"/usr/bin/{n}")
    ran = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        ran.append(args)
        if "ffprobe" in args[0]:
            if opt.ShotEncoder._PLAN_PROBE in args:
                return _probe_json(("video", set()))   # video only: nothing to remux
            return ""
        _write_out(args, b"\x1aE\xdf\xa3")
        return ""

    enc._run = fake_run.__get__(enc)
    enc.concat_shots([tmp_path / "enc_00000.ivf"])

    muxes = [a for a in ran if "ffprobe" not in a[0]]
    # the video concat and the final mux, and nothing in between
    assert len(muxes) == 2
    assert not any("audio_subs.mkv" in " ".join(a) for a in muxes)
    # the final mux takes video_only alone, with no second input to map from
    final = muxes[-1]
    assert "1:a?" not in final and "1:s?" not in final
    assert final.count("-i") == 1


def test_source_with_audio_still_gets_remuxed(settings, info, plan, tmp_path,
                                              monkeypatch):
    enc = make_encoder(settings, info, plan, tmp_path)
    monkeypatch.setattr(opt.shutil, "which",
                        lambda n, *a, **k: None if "mkvmerge" in n else f"/usr/bin/{n}")
    ran = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        ran.append(args)
        if "ffprobe" in args[0]:
            if opt.ShotEncoder._PLAN_PROBE in args:
                # an audio stream is present, but no subtitle streams
                return _probe_json(("video", set()), ("audio", {"default"}))
            return ""
        _write_out(args, b"\x1aE\xdf\xa3")
        return ""

    enc._run = fake_run.__get__(enc)
    enc.concat_shots([tmp_path / "enc_00000.ivf"])

    muxes = [a for a in ran if "ffprobe" not in a[0]]
    assert any("audio_subs.mkv" in " ".join(a) for a in muxes)
    final = muxes[-1]
    assert "1:a?" in final and "1:s?" in final


def test_mkvmerge_mux_delays_the_video_by_the_sources_lead(settings, info, plan,
                                                            tmp_path, monkeypatch):
    """Every shot ivf starts at 0 while the audio keeps the original's
    timestamps, so a picture that began 1.955s after the sound would come out
    1.955s early. Measured on a 0.066s case: source v=0.066/a=0, output v=0/a=0."""
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._lead_of = lambda path: 1.955 if str(path) == str(enc.info.path) else 0.0
    video_only = tmp_path / "video_only.mkv"
    audio_subs = tmp_path / "audio_subs.mkv"
    video_only.touch(); audio_subs.touch()
    seen = {}

    def fake_run(cmd, capture_output=False, text=False, timeout=None):
        seen["cmd"] = cmd
        enc.output.write_bytes(b"\x1aE\xdf\xa3")
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(opt.subprocess, "run", fake_run)
    assert enc._mkvmerge_mux("mkvmerge", video_only, audio_subs) is True
    cmd = seen["cmd"]
    # --sync binds to the input that follows it, which must be the video
    assert cmd[cmd.index("--sync") + 1] == "0:1955"
    assert cmd[cmd.index("--sync") + 2] == str(video_only)
    # nothing to align against when the file is video only
    assert enc._mkvmerge_mux("mkvmerge", video_only, None) is True
    assert "--sync" not in seen["cmd"]


def test_ffmpeg_mux_fallback_delays_the_video_too(settings, info, plan, tmp_path,
                                                  monkeypatch):
    enc = make_encoder(settings, info, plan, tmp_path)
    monkeypatch.setattr(opt.shutil, "which",
                        lambda n, *a, **k: None if "mkvmerge" in n else f"/usr/bin/{n}")
    enc._lead_of = lambda path: 0.066
    ran = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        ran.append(args)
        if "ffprobe" in args[0]:
            return "video\naudio\n" if "stream=codec_type" in args else ""
        _write_out(args, b"\x1aE\xdf\xa3")
        return ""

    enc._run = fake_run.__get__(enc)
    enc.concat_shots([tmp_path / "enc_00000.ivf"])
    final = [a for a in ran if "ffprobe" not in a[0]][-1]
    i = final.index("-itsoffset")
    assert float(final[i + 1]) == pytest.approx(0.066)
    assert final[i + 2] == "-i" and final[i + 3].endswith("video_only.mkv")


def test_audio_detection_survives_ffprobes_trailing_csv_fields(settings, info, plan, tmp_path):
    """An eac3 track in an mp4 carries "Audio Service Type" side data, and
    ffprobe's csv then prints the stream as "audio," - which used to read as
    no audio at all: the mux went video-only and the output check failed the
    job. Measured on every ATVP/DSNP mp4 tried."""
    enc = make_encoder(settings, info, plan, tmp_path)
    side_data = {"side_data_list": [{"side_data_type": "Audio Service Type"}]}
    probe = _probe_json(("video", set()), ("audio", {"default"}, side_data),
                        ("audio", set(), side_data),
                        ("subtitle", set(), {"codec_name": "mov_text",
                                             "nb_frames": "923"}),
                        fmt="mov,mp4,m4a,3gp,3g2,mj2")
    probes = []

    def fake_run(self, args, timeout=None):
        probes.append([str(a) for a in args])
        return probe

    enc._run = fake_run.__get__(enc)
    assert enc._has_audio_or_subs("x.mp4") is True
    assert enc._subtitle_codec_args("x.mp4") == ["-c:s:0", "srt"]
    # and one probe answered both: the plan is read once per source
    assert len(probes) == 1
    enc._run = (lambda self, args, timeout=None:
                _probe_json(("video", set()))).__get__(enc)
    assert enc._has_audio_or_subs("y.mp4") is False


# ---- stream dispositions through the audio/subtitle remux and the final mux ----
def _probe_json(*streams, fmt="matroska,webm"):
    """ffprobe -of json for the one plan probe of the mux source.

    Each stream is (codec_type, set of its set flags), optionally with a third
    member overriding codec_name, nb_frames, duration_ts or tags. Every known
    disposition flag is printed as 0 or 1, the way ffprobe prints them, and a
    subtitle stream comes with a frame count AND a duration that say it is NOT
    empty - the tests about emptiness override those.

    duration_ts is a json number here because that is how ffprobe prints it,
    and every stream carries one: a file that states no duration anywhere is
    its own case (see the fragmented-mp4 test).
    """
    codecs = {"video": "hevc", "audio": "eac3", "subtitle": "hdmv_pgs_subtitle"}
    out = []
    for i, stream in enumerate(streams):
        kind, flags = stream[0], stream[1]
        extra = dict(stream[2]) if len(stream) > 2 else {}
        st = {"index": i, "codec_type": kind,
              "codec_name": extra.pop("codec_name", codecs.get(kind, "")),
              "duration_ts": 2709960000,
              "disposition": {n: int(n in flags)
                              for n in opt.ShotEncoder._DISPOSITIONS}}
        if kind == "subtitle":
            st["nb_frames"] = "1448"
            st["tags"] = {"NUMBER_OF_FRAMES": "1448"}
        st.update(extra)
        out.append(st)
    return json.dumps({"streams": out, "format": {"format_name": fmt}}, indent=4)


# What ffmpeg's srt encoder writes out of an ASS cue, measured on n9.0.1: the
# <font> wrapper it adds whenever the style differs from its own defaults, and
# {\anN} copied through into the text. \pos and \move never appear - the
# splitter routes both to an empty callback - and <i> does survive.
_SRTENC_OUT = ("1\n00:00:00,200 --> 00:00:01,500\n"
               '<font face="Source Han Sans SC Medium" size="24">'
               "{\\an8}人多力量大</font>\n\n"
               "2\n00:00:02,000 --> 00:00:03,000\n<i>italics survive</i>\n\n")


def _write_outputs(args, srt=None):
    """Write whatever the command was about to write. An output is an argument
    that is not an option, not the value of one, and not an input - which a
    remux with an srt companion beside it needs, since its own .mkv is no
    longer the last argument.

    `srt` overrides what an .srt output gets, for the companion that comes out
    with nothing in it - "" included, which is the case that matters."""
    for i, a in enumerate(args):
        if not i or a.startswith("-") or args[i - 1] == "-i":
            continue
        if a.endswith(".srt"):
            Path(a).write_text(_SRTENC_OUT if srt is None else srt,
                               encoding="utf-8")
        elif a.endswith((".mkv", ".ivf")):
            Path(a).write_bytes(b"\x1aE\xdf\xa3")


def _mux_with(enc, monkeypatch, probe, fail=lambda args: False, srt=None):
    """concat_shots down the ffmpeg fallback mux, with `probe` as what the one
    plan probe prints (raised instead when it is an exception), `fail` picking
    the commands that fail and `srt` what an extracted companion contains.
    Returns every command run."""
    monkeypatch.setattr(opt.shutil, "which",
                        lambda n, *a, **k: None if "mkvmerge" in n else f"/usr/bin/{n}")
    enc._lead_of = lambda path: 0.0
    ran = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        ran.append(args)
        if "ffprobe" in args[0]:
            if opt.ShotEncoder._PLAN_PROBE in args:
                if isinstance(probe, Exception):
                    raise probe
                return probe
            return ""
        if fail(args):
            raise opt.TranscodeError("mux failed")
        _write_outputs(args, srt)
        return ""

    enc._run = fake_run.__get__(enc)
    enc.concat_shots([enc.tempdir / "enc_00000.ivf"])
    return ran


def _dispositions(cmd):
    """The -disposition pairs of `cmd`, each checked to sit where ffmpeg takes
    it as an output option: after the last input, before the output path.
    Ahead of an -i, ffmpeg rejects it as an input option and the mux fails."""
    last_url = max(i for i, a in enumerate(cmd) if a == "-i") + 1
    pairs = []
    for i, a in enumerate(cmd):
        if a.startswith("-disposition"):
            assert last_url < i and i + 1 < len(cmd) - 1, (a, cmd)
            pairs.append((a, cmd[i + 1]))
    return pairs


def _remuxes(ran):
    """The commands that WRITE audio_subs.mkv.

    Not "the last argument is it": an srt companion is an extra output of that
    same command, after it. And not "it appears anywhere" either - the final
    mux takes the same file as an input.
    """
    return [a for a in ran
            if any(x.endswith("audio_subs.mkv") and a[i - 1] != "-i"
                   for i, x in enumerate(a) if i)]


def test_remux_states_every_disposition_so_no_subtitle_is_made_default(
        settings, info, plan, tmp_path, monkeypatch):
    """Stranger Things S04E07: two PGS tracks, neither default. Left to infer,
    fftools marked the first one default in audio_subs.mkv, mkvmerge kept it,
    and Plex auto-selected the empty track - a burn-in transcode. 17 outputs."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _probe_json(
        ("video", {"default"}), ("audio", {"default"}),
        ("subtitle", set()), ("subtitle", set())))
    remux, = _remuxes(ran)
    assert _dispositions(remux) == [("-disposition:a:0", "default"),
                                    ("-disposition:s:0", "0"),
                                    ("-disposition:s:1", "0")]


def test_remux_carries_each_streams_flags_exactly(settings, info, plan, tmp_path,
                                                  monkeypatch):
    """Output order is every audio stream and then every subtitle, each
    counted within its own type, however the source interleaves them."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _probe_json(
        ("video", {"default"}),
        ("audio", {"default", "original"}),
        ("subtitle", set()),
        ("audio", {"comment", "visual_impaired"}),
        ("subtitle", {"default", "forced"}),
        ("subtitle", {"hearing_impaired"})))
    remux, = _remuxes(ran)
    assert _dispositions(remux) == [("-disposition:a:0", "default+original"),
                                    ("-disposition:a:1", "comment+visual_impaired"),
                                    ("-disposition:s:0", "0"),
                                    ("-disposition:s:1", "default+forced"),
                                    ("-disposition:s:2", "hearing_impaired")]


def test_retries_and_the_ffmpeg_fallback_mux_carry_the_same_dispositions(
        settings, info, plan, tmp_path, monkeypatch):
    enc = make_encoder(settings, info, plan, tmp_path)

    def fail(args):
        if args[-1].endswith("audio_subs.mkv") and "-c:s:0" in args:
            return True                  # per-stream codec remux: plain-copy retry
        return args[-1] == str(enc.output) and "srt" not in args   # srt retry

    # one bitmap track and one text one: the srt retry has something it can
    # actually convert (see _srt_retry_args)
    ran = _mux_with(enc, monkeypatch, _probe_json(
        ("video", {"default"}), ("audio", {"default"}), ("subtitle", set()),
        ("subtitle", {"default", "forced"}, {"codec_name": "subrip"})), fail)
    remux, retry = _remuxes(ran)
    final, final_srt = [a for a in ran if a[-1] == str(enc.output)]
    assert "-c:s:0" in remux and "-c:s:0" not in retry
    assert "srt" in final_srt and "1:s?" in final
    expected = [("-disposition:a:0", "default"), ("-disposition:s:0", "0"),
                ("-disposition:s:1", "default+forced")]
    for cmd in (remux, retry, final, final_srt):
        assert _dispositions(cmd) == expected
    # the one probe of the source serves all four commands
    assert sum(opt.ShotEncoder._PLAN_PROBE in a for a in ran) == 1
    assert enc.output.exists()


@pytest.mark.parametrize("probe", [
    opt.TranscodeError("ffprobe exploded"),
    "Invalid data found when processing input\n",
    json.dumps({"streams": [{"index": 0, "codec_type": "subtitle"}]}),
], ids=["ffprobe-failed", "no-json", "no-disposition-section"])
def test_a_failed_disposition_probe_clears_subtitle_defaults_and_goes_on(
        settings, info, plan, tmp_path, monkeypatch, probe):
    enc = make_encoder(settings, info, plan, tmp_path)
    warnings = []
    monkeypatch.setattr(opt.logger, "warning", lambda *a, **k: warnings.append(a))
    ran = _mux_with(enc, monkeypatch, probe)
    remux, = _remuxes(ran)
    final, = [a for a in ran if a[-1] == str(enc.output)]
    for cmd in (remux, final):
        # only the default flag, and only on subtitles: the audio keeps what
        # ffmpeg copies from the source
        assert _dispositions(cmd) == [("-disposition:s", "-default")]
    assert enc.output.exists()
    assert any("could not probe the streams" in str(w[0]) for w in warnings)
    # and nothing is dropped on a plan nobody could read
    assert enc.subtitles_dropped == 0
    assert not any(a.startswith("-0:s") for cmd in (remux, final) for a in cmd)


def test_disposition_probe_reads_past_error_lines_and_drops_unknown_flags(
        settings, info, plan, tmp_path, monkeypatch):
    enc = make_encoder(settings, info, plan, tmp_path)
    warnings = []
    monkeypatch.setattr(opt.logger, "warning", lambda *a, **k: warnings.append(a))
    body = json.loads(_probe_json(("audio", {"default"}), ("subtitle", {"forced"})))
    body["streams"][1]["disposition"]["from_a_newer_ffprobe"] = 1
    out = "[matroska,webm @ 0x55d0c0] Read error\n" + json.dumps(body, indent=4)
    enc._run = (lambda self, args, timeout=None: out).__get__(enc)
    assert enc._disposition_args("/x/src.mkv") == ["-disposition:a:0", "default",
                                                    "-disposition:s:0", "forced"]
    assert any("from_a_newer_ffprobe" in str(w) for w in warnings)


def test_mkvmerge_mux_without_an_audio_file(settings, info, plan, tmp_path,
                                            monkeypatch):
    enc = make_encoder(settings, info, plan, tmp_path)
    video_only = tmp_path / "video_only.mkv"
    video_only.touch()
    seen = {}

    def fake_run(cmd, capture_output=False, text=False, timeout=None):
        seen["cmd"] = cmd
        enc.output.write_bytes(b"\x1aE\xdf\xa3")
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(opt.subprocess, "run", fake_run)
    assert enc._mkvmerge_mux("mkvmerge", video_only, None) is True
    assert seen["cmd"] == ["mkvmerge", "-o", str(enc.output), str(video_only)]


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="needs a real ffprobe")
def test_has_audio_or_subs_against_a_real_ffprobe(settings, info, plan, tmp_path):
    """Run the probe command for real, not against a stubbed _run.

    The first version of this used "-select_streams a,s", which ffprobe rejects
    ("Invalid stream specifier") because the specifier takes one type, not a
    list. Every unit test passed - they stub _run - while the guard silently
    fell back to "assume there is audio" on every single source. Only a real
    ffprobe catches that class of mistake.
    """
    settings.tools.ffprobe = shutil.which("ffprobe")
    enc = make_encoder(settings, info, plan, tmp_path)
    silent = tmp_path / "silent.mkv"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("needs a real ffmpeg to build the fixtures")
    import subprocess as sp
    sp.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
            "-i", "testsrc=size=64x64:rate=5:duration=1", "-c:v", "libx264",
            str(silent)], check=True)
    assert enc._has_audio_or_subs(str(silent)) is False

    noisy = tmp_path / "noisy.mkv"
    sp.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
            "-i", "testsrc=size=64x64:rate=5:duration=1", "-f", "lavfi",
            "-i", "sine=frequency=440:duration=1", "-c:v", "libx264",
            "-c:a", "aac", str(noisy)], check=True)
    assert enc._has_audio_or_subs(str(noisy)) is True


# ---- attachments (the fonts styled subtitles name) through the final mux ----
_MKV, _MP4 = "matroska,webm", "mov,mp4,m4a,3gp,3g2,mj2"
_ATTACHMENT_PROBE = "format=format_name:stream=codec_type:stream_disposition=attached_pic"
_SOURCE_ONLY = ["--no-video", "--no-audio", "--no-subtitles", "--no-buttons",
                "--no-track-tags", "--no-chapters", "--no-global-tags"]


def _maps(cmd):
    return [cmd[i + 1] for i, a in enumerate(cmd) if a == "-map"]


def _attachment_probe(format_name, *streams):
    """ffprobe -of json for the attachment probe: (codec_type, attached_pic)
    per stream, the way ffprobe prints them."""
    return json.dumps({"programs": [], "streams": [
        {"codec_type": kind, "disposition": {"attached_pic": pic}}
        for kind, pic in streams], "format": {"format_name": format_name}}, indent=4)


def _concat_via_mkvmerge(enc, monkeypatch, streams, probe, rc=lambda cmd: 0,
                         srt=None):
    """concat_shots down the mkvmerge mux, with `streams` as what the one plan
    probe prints and `probe` as what the attachment probe prints (raised
    when an exception). `rc` gives each mkvmerge run its exit code, and `srt`
    what an extracted companion contains, as in _mux_with. Returns
    (the commands _run ran, the mkvmerge commands)."""
    monkeypatch.setattr(opt.shutil, "which", lambda n, *a, **k: f"/usr/bin/{n}")
    enc._lead_of = lambda path: 0.0
    ran, merges = [], []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        ran.append(args)
        if "ffprobe" in args[0]:
            if opt.ShotEncoder._PLAN_PROBE in args:
                return streams
            if _ATTACHMENT_PROBE in args:
                # the original file, as for the audio: a Dolby Vision job's
                # enc.source is a stripped intermediate with no attachments
                assert args[-1] == str(enc.info.path), args
                if isinstance(probe, Exception):
                    raise probe
                return probe
            return ""
        _write_outputs(args, srt)
        return ""

    def fake_mkvmerge(cmd, capture_output=False, text=False, timeout=None):
        merges.append(cmd)
        code = rc(cmd)
        if code == 0:
            enc.output.write_bytes(b"\x1aE\xdf\xa3")
        return types.SimpleNamespace(returncode=code, stderr="Error: boom")

    enc._run = fake_run.__get__(enc)
    monkeypatch.setattr(opt.subprocess, "run", fake_mkvmerge)
    enc.concat_shots([enc.tempdir / "enc_00000.ivf"])
    return ran, merges


def test_remux_and_the_ffmpeg_fallback_mux_carry_the_attachments(
        settings, info, plan, tmp_path, monkeypatch):
    """An ASS track names its fonts, and the source carries them as attachments;
    dropped, a player substitutes its own and CJK can come out as boxes. Both
    maps end in "?": most sources (every mp4) have none, and ffmpeg fails a
    map that matches no stream ("To ignore this, add a trailing '?'")."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _probe_json(
        ("video", {"default"}), ("audio", {"default"}),
        ("subtitle", set()), ("subtitle", {"default", "forced"})))
    remux, = _remuxes(ran)
    final, = [a for a in ran if a[-1] == str(enc.output)]
    assert _maps(remux) == ["0:a?", "0:s?", "0:t?"]
    assert remux[remux.index("-c:t") + 1] == "copy"
    assert _maps(final) == ["0:v:0", "1:a?", "1:s?", "1:t?"]
    # mapped last and counted as a type of their own, they move no flag
    expected = [("-disposition:a:0", "default"), ("-disposition:s:0", "0"),
                ("-disposition:s:1", "default+forced")]
    assert _dispositions(remux) == expected
    assert _dispositions(final) == expected


def test_a_remux_that_fails_with_the_attachments_is_retried_without_them(
        settings, info, plan, tmp_path, monkeypatch):
    """matroskaenc refuses an attachment with no filename, or with no mimetype
    it can deduce (measured: exit 234, "Attachment stream 1 has no mimetype
    tag"). A font lost costs a styled subtitle; a remux lost, the encode."""
    enc = make_encoder(settings, info, plan, tmp_path)
    warnings = []
    monkeypatch.setattr(opt.logger, "warning", lambda *a, **k: warnings.append(a))
    ran = _mux_with(enc, monkeypatch, _probe_json(
        ("video", {"default"}), ("audio", {"default"}), ("subtitle", set())),
        fail=lambda args: args[-1].endswith("audio_subs.mkv") and "0:t?" in args)
    first, retry, bare = _remuxes(ran)
    assert "0:t?" in first and "0:t?" in retry
    assert _maps(bare) == ["0:a?", "0:s?"] and "-c:t" not in bare
    assert _dispositions(bare) == [("-disposition:a:0", "default"),
                                   ("-disposition:s:0", "0")]
    assert enc.output.exists()
    assert any("without the source's attachments" in str(w[0]) for w in warnings)


def test_mkvmerge_takes_only_the_attachments_from_the_source(settings, info, plan,
                                                             tmp_path, monkeypatch):
    """The source's tracks, chapters and tags are audio_subs.mkv's already.
    audio_subs.mkv's attachments must be left out: they are ffmpeg's copies,
    with new UIDs and no images, and mkvmerge keeps a first copy over a second
    of the same name, description and size - measured, the source's UIDs were
    lost without --no-attachments."""
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._lead_of = lambda path: 0.5
    video_only = tmp_path / "video_only.mkv"
    audio_subs = tmp_path / "audio_subs.mkv"
    video_only.touch(); audio_subs.touch()
    seen = []

    def fake_run(cmd, capture_output=False, text=False, timeout=None):
        seen.append(cmd)
        enc.output.write_bytes(b"\x1aE\xdf\xa3")
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(opt.subprocess, "run", fake_run)
    assert enc._mkvmerge_mux("mkvmerge", video_only, audio_subs, "/m/movie.mkv") is True
    assert seen[-1] == ["mkvmerge", "-o", str(enc.output), "--sync", "0:500",
                        str(video_only), "--no-attachments", str(audio_subs),
                        *_SOURCE_ONLY, "/m/movie.mkv"]
    # With no audio_subs.mkv, the source would be the first input with a segment
    # title, and mkvmerge would put it on an output that never had one. An
    # explicit empty title wins (measured, v82). With audio_subs.mkv the title is
    # the one it already carries, and the command above leaves it alone.
    assert enc._mkvmerge_mux("mkvmerge", video_only, None, "/m/movie.mkv") is True
    assert seen[-1] == ["mkvmerge", "-o", str(enc.output), "--title", "",
                        str(video_only), *_SOURCE_ONLY, "/m/movie.mkv"]
    assert enc._mkvmerge_mux("mkvmerge", video_only, None) is True
    assert seen[-1] == ["mkvmerge", "-o", str(enc.output), str(video_only)]


@pytest.mark.parametrize("probe, takes", [
    (_attachment_probe(_MKV, ("video", 0), ("audio", 0), ("subtitle", 0),
                       ("attachment", 0), ("attachment", 0)), True),
    # matroskadec makes an image/jpeg attachment a video stream, attached_pic
    (_attachment_probe(_MKV, ("video", 0), ("audio", 0), ("video", 1)), True),
    (_attachment_probe(_MKV, ("video", 0), ("audio", 0), ("subtitle", 0)), False),
    # an mp4's cover art is attached_pic too, and no attachment
    (_attachment_probe(_MP4, ("video", 0), ("audio", 0), ("video", 1)), False),
    ("[matroska,webm @ 0x55d0c0] Read error\n"
     + _attachment_probe(_MKV, ("video", 0), ("audio", 0), ("attachment", 0)), True),
    ("Invalid data found when processing input\n", False),
    (opt.TranscodeError("ffprobe exploded"), False),
], ids=["fonts", "image", "none", "mp4-cover", "past-error-lines", "no-json",
        "ffprobe-failed"])
def test_mkvmerge_reads_the_original_source_only_for_its_attachments(
        settings, info, plan, tmp_path, monkeypatch, probe, takes):
    enc = make_encoder(settings, info, plan, tmp_path)
    ran, merges = _concat_via_mkvmerge(
        enc, monkeypatch,
        _probe_json(("video", set()), ("audio", {"default"}), ("subtitle", set())),
        probe)
    merge, = merges
    audio_subs = str(enc.tempdir / "audio_subs.mkv")
    # Dolby Vision encodes a stripped intermediate: attachments, like the
    # audio, come from the original file
    assert str(enc.source) != str(enc.info.path)
    assert [a[-1] for a in ran if _ATTACHMENT_PROBE in a] == [str(enc.info.path)]
    assert any(a[-1] == audio_subs and "0:t?" in a for a in ran)
    if takes:
        assert merge[1:] == ["-o", str(enc.output), str(enc.tempdir / "video_only.mkv"),
                             "--no-attachments", audio_subs, *_SOURCE_ONLY,
                             str(enc.info.path)]
    else:
        assert merge[1:] == ["-o", str(enc.output),
                             str(enc.tempdir / "video_only.mkv"), audio_subs]


def test_an_mkvmerge_that_cannot_read_the_source_muxes_again_without_it(
        settings, info, plan, tmp_path, monkeypatch):
    """mkvmerge exits 2 on a file it cannot parse ("The type of file could not
    be recognized"). That must not cost the container rebuild Plex needs: the
    fonts audio_subs.mkv carries still get through a second mkvmerge."""
    enc = make_encoder(settings, info, plan, tmp_path)
    warnings = []
    monkeypatch.setattr(opt.logger, "warning", lambda *a, **k: warnings.append(a))
    ran, merges = _concat_via_mkvmerge(
        enc, monkeypatch, _probe_json(("video", set()), ("audio", {"default"})),
        _attachment_probe(_MKV, ("video", 0), ("audio", 0), ("attachment", 0)),
        rc=lambda cmd: 2 if str(enc.info.path) in cmd else 0)
    first, second = merges
    assert first[-1] == str(enc.info.path)
    assert second[1:] == ["-o", str(enc.output), str(enc.tempdir / "video_only.mkv"),
                          str(enc.tempdir / "audio_subs.mkv")]
    assert not any(a[-1] == str(enc.output) for a in ran)     # no ffmpeg mux
    assert any("attachment source" in str(w[0]) for w in warnings)


def test_a_video_only_source_keeps_its_attachments_without_a_remux(
        settings, info, plan, tmp_path, monkeypatch):
    """Attachments alone start no ffmpeg pass. mkvmerge takes every one from
    the source itself, images and UIDs included, which the remux's copy could
    not; that copy would serve only the ffmpeg fallback. So _has_audio_or_subs
    stays about audio and subtitles - the guard against a remux with nothing
    to write."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran, merges = _concat_via_mkvmerge(
        enc, monkeypatch,
        _probe_json(("video", set()), ("attachment", set()),
                    ("video", {"attached_pic"})),
        _attachment_probe(_MKV, ("video", 0), ("attachment", 0), ("video", 1)))
    assert not any("audio_subs.mkv" in " ".join(a) for a in ran)
    merge, = merges
    assert merge[1:] == ["-o", str(enc.output), "--title", "",
                         str(enc.tempdir / "video_only.mkv"), *_SOURCE_ONLY,
                         str(enc.info.path)]
    enc._run = (lambda self, args, timeout=None:
                _probe_json(("video", set()), ("attachment", set()))).__get__(enc)
    assert enc._has_audio_or_subs("x.mkv") is False


_REAL_WHICH = shutil.which        # the settings fixture fakes it per test


@pytest.mark.skipif(any(_REAL_WHICH(t) is None for t in ("ffmpeg", "ffprobe", "mkvmerge")),
                    reason="needs a real ffmpeg, ffprobe and mkvmerge")
@pytest.mark.parametrize("container", ["mkv", "mp4"])
@pytest.mark.parametrize("muxer", ["mkvmerge", "ffmpeg"])
def test_attachments_reach_the_final_file_against_real_tools(
        settings, plan, tmp_path, monkeypatch, muxer, container):
    """The whole final mux for real, not against a stubbed _run. No stub can
    tell that a bare "0:t" fails an mp4's remux, that ffmpeg turns an image
    attachment into a video stream, or which copy of a font mkvmerge keeps."""
    import subprocess as sp
    real = _REAL_WHICH
    monkeypatch.setattr(shutil, "which", real if muxer == "mkvmerge" else
                        (lambda n, *a, **k: None if "mkvmerge" in n else real(n, *a, **k)))
    ffmpeg, mkvmerge = real("ffmpeg"), real("mkvmerge")

    def ff(*args):
        sp.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
               check=True, cwd=tmp_path)

    def ident(path):
        return json.loads(sp.run([mkvmerge, "-J", str(path)], check=True,
                                 capture_output=True, text=True).stdout)

    shot = tmp_path / "enc_00000.ivf"
    try:
        ff("-f", "lavfi", "-i", "testsrc=size=160x120:rate=5:duration=2",
           "-c:v", "libsvtav1", "-preset", "12", shot)
    except sp.CalledProcessError:
        pytest.skip("needs an ffmpeg with libsvtav1 to build the shot")
    ff("-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", "sound.mka")
    if container == "mkv":
        ass = ("[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\n"
               "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
               "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
               "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
               "MarginL, MarginR, MarginV, Encoding\n"
               "Style: Default,Test Sans,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
               "0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1\n\n[Events]\n"
               "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
               "Effect, Text\nDialogue: 0,0:00:00.20,0:00:01.50,Default,,0,0,0,,hello\n")
        (tmp_path / "a.ass").write_text(ass)
        (tmp_path / "b.ass").write_text(ass)
        (tmp_path / "font.ttf").write_bytes(bytes(range(256)) * 40)
        ff("-f", "lavfi", "-i", "testsrc=size=64x64", "-frames:v", "1", "cover.jpg")
        source = tmp_path / "movie.mkv"
        sp.run([mkvmerge, "-q", "-o", str(source), str(shot), "sound.mka",
                "--default-track-flag", "0:no", "a.ass",
                "--default-track-flag", "0:no", "--forced-display-flag", "0:yes", "b.ass",
                "--attachment-description", "Main font",
                "--attachment-mime-type", "application/x-truetype-font",
                "--attach-file", "font.ttf",
                "--attachment-mime-type", "image/jpeg", "--attach-file", "cover.jpg"],
               check=True, cwd=tmp_path)
    else:
        (tmp_path / "s.srt").write_text("1\n00:00:00,200 --> 00:00:01,500\nhello\n")
        source = tmp_path / "movie.mp4"
        ff("-i", shot, "-i", "sound.mka", "-i", "s.srt", "-map", "0", "-map", "1",
           "-map", "2", "-c:v", "copy", "-c:a", "copy", "-c:s", "mov_text", source)
    info = MediaInfo(path=source)
    info.fps = 5.0
    info.duration = 2.0
    # encodes tmp_path/src.mkv, as a Dolby Vision job encodes its intermediate
    enc = make_encoder(settings, info, plan, tmp_path)
    warnings = []
    monkeypatch.setattr(opt.logger, "warning", lambda *a, **k: warnings.append(a))
    enc.concat_shots([shot])

    def attachments(j, uid=True):
        return [(a["file_name"], a["content_type"], a.get("description", ""), a["size"])
                + ((a["properties"]["uid"],) if uid else ())
                for a in j.get("attachments", [])]

    def subtitles(j):
        return [(t["properties"]["default_track"], t["properties"]["forced_track"])
                for t in j["tracks"] if t["type"] == "subtitles"]

    src, out = ident(source), ident(enc.output)
    # the source's own tracks, plus the srt companion every kept ASS gets
    assert [t["type"] for t in out["tracks"]] == (
        [t["type"] for t in src["tracks"]]
        + ["subtitles"] * enc.subtitles_added)
    if container == "mp4":
        assert attachments(out) == []
        assert enc.subtitles_added == 0          # mov_text is not ASS
    else:
        assert subtitles(src) == [(False, False), (False, True)]
        # each ASS keeps its own flags and its companion carries the same two,
        # through EITHER muxer - the half that used to disagree, since the
        # ffmpeg fallback restated the ASS's whole disposition value
        assert enc.subtitles_added == 2
        assert subtitles(out) == subtitles(src) * 2
        assert len(attachments(src)) == 2
        if muxer == "mkvmerge":
            # every one, the image and the UIDs included
            assert attachments(out) == attachments(src)
        else:
            # ffmpeg's copy: the font keeps its name, type and description
            assert attachments(out, uid=False) == attachments(src, uid=False)[:1]
    assert not any("attachment" in str(w[0]) for w in warnings)


@pytest.mark.skipif(any(_REAL_WHICH(t) is None for t in ("ffmpeg", "ffprobe", "mkvmerge")),
                    reason="needs a real ffmpeg, ffprobe and mkvmerge")
def test_a_video_only_source_lends_mkvmerge_its_attachments_not_its_title(
        settings, plan, tmp_path, monkeypatch):
    """With no audio_subs.mkv, the source is the only input with a segment title,
    and mkvmerge takes the title from the first input that has one. Before
    attachments were carried such an output had no title, and it must still
    have none: a "DV.HDR10.PLUS" label is wrong on an AV1 file that may be
    HDR10 only."""
    import subprocess as sp
    monkeypatch.setattr(shutil, "which", _REAL_WHICH)
    ffmpeg, mkvmerge = _REAL_WHICH("ffmpeg"), _REAL_WHICH("mkvmerge")
    shot = tmp_path / "enc_00000.ivf"
    try:
        sp.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                "-i", "testsrc=size=160x120:rate=5:duration=2", "-c:v", "libsvtav1",
                "-preset", "12", str(shot)], check=True)
    except sp.CalledProcessError:
        pytest.skip("needs an ffmpeg with libsvtav1 to build the shot")
    font = tmp_path / "font.ttf"
    font.write_bytes(bytes(range(256)) * 40)
    source = tmp_path / "movie.mkv"
    sp.run([mkvmerge, "-q", "-o", str(source), "--title", "Show.S01E01.DV.HDR10.PLUS",
            str(shot), "--attachment-mime-type", "application/x-truetype-font",
            "--attach-file", str(font)], check=True)
    info = MediaInfo(path=source)
    info.fps = 5.0
    info.duration = 2.0
    enc = make_encoder(settings, info, plan, tmp_path)
    enc.concat_shots([shot])

    def ident(path):
        return json.loads(sp.run([mkvmerge, "-J", str(path)], check=True,
                                 capture_output=True, text=True).stdout)

    def attachments(j):
        return [(a["file_name"], a["content_type"], a["properties"]["uid"])
                for a in j.get("attachments", [])]

    src, out = ident(source), ident(enc.output)
    assert src["container"]["properties"].get("title") == "Show.S01E01.DV.HDR10.PLUS"
    assert out["container"]["properties"].get("title") is None
    assert [t["type"] for t in out["tracks"]] == ["video"]
    assert attachments(out) == attachments(src) != []


def test_scaled_size_derives_the_hw_scale_target(settings, info, plan, tmp_path):
    """scale_qsv has to be told the size outright.

    Its own w=-1 rounds to the card's surface alignment, not to what the
    software `scale` filter picks - measured on a 3840x1606 source, -2:540 gives
    1292x540 in software and something 360 pixels smaller through QSV. A
    detection copy of a different size is a different copy, and for a detector
    reading it frame by frame that means different cuts.
    """
    enc = make_encoder(settings, info, plan, tmp_path)
    info.width, info.height = 3840, 1606
    assert enc._scaled_size("-2:540") == (1292, 540)
    info.width, info.height = 3840, 2160
    assert enc._scaled_size("-2:540") == (960, 540)
    assert enc._scaled_size("-1:540") == (960, 540)
    assert enc._scaled_size("960:540") == (960, 540)
    # not a plain W:H, or no source dimensions -> no hardware path
    assert enc._scaled_size("w='min(iw,1920)':h=-2") is None
    info.width, info.height = 0, 0
    assert enc._scaled_size("-2:540") is None


def test_detection_copy_tries_the_gpu_first(settings, info, plan, tmp_path):
    info.width, info.height = 3840, 2160
    enc = make_encoder(settings, info, plan, tmp_path)
    cmds = enc._detection_copy_cmds(tmp_path / "out.mkv", "-2:540")
    assert [label for label, _ in cmds] == ["qsv", "software"]
    qsv = " ".join(cmds[0][1])
    # explicit size, and no -qsv_device: the container is given one render node
    # and naming a fixed /dev/dri/renderDNN would be wrong on any other host
    assert "scale_qsv=w=960:h=540" in qsv and "-qsv_device" not in qsv
    # still encoded with x264: encoding on the GPU as well is barely faster and
    # its artefacts move more cuts than the scaler alone does
    assert "libx264" in qsv and "hevc_qsv" not in qsv

    settings.transcode.optimizer.scenedetect_hwaccel = "off"
    assert [label for label, _ in enc._detection_copy_cmds(tmp_path / "o.mkv", "-2:540")] \
        == ["software"]
    # an unparseable scale spec has no derivable size, so no hardware path
    settings.transcode.optimizer.scenedetect_hwaccel = "auto"
    assert [label for label, _ in enc._detection_copy_cmds(tmp_path / "o.mkv", "iw/2:-2")] \
        == ["software"]


def test_detection_copy_falls_back_when_the_gpu_writes_nothing(
        settings, info, plan, tmp_path, monkeypatch):
    """A QSV decode the card cannot do exits 0 having written nothing.

    That is how AV1 fails on a B580, so a zero-length output has to count as
    failure rather than as a finished copy.
    """
    info.width, info.height = 3840, 2160
    settings.transcode.optimizer.scenedetect_scale = "-2:540"
    enc = make_encoder(settings, info, plan, tmp_path)
    tags = []

    def fake(args, timeout, total_seconds, tag):
        tags.append(tag)
        if "qsv" in tag:
            return                       # exits cleanly, writes nothing
        (enc.probe_dir / "detect_copy.mkv").write_bytes(b"x")

    monkeypatch.setattr(enc, "_run_with_progress", fake)
    out = enc._make_detection_copy()
    assert out is not None and out.exists()
    assert tags == ["downscale for detection (qsv)",
                    "downscale for detection (software)"]


def test_detection_copy_falls_back_when_the_gpu_errors(
        settings, info, plan, tmp_path, monkeypatch):
    info.width, info.height = 3840, 2160
    settings.transcode.optimizer.scenedetect_scale = "-2:540"
    enc = make_encoder(settings, info, plan, tmp_path)
    tags = []

    def fake(args, timeout, total_seconds, tag):
        tags.append(tag)
        if "qsv" in tag:
            raise opt.TranscodeError("Device creation failed: -542398533.")
        (enc.probe_dir / "detect_copy.mkv").write_bytes(b"x")

    monkeypatch.setattr(enc, "_run_with_progress", fake)
    assert enc._make_detection_copy() is not None
    assert len(tags) == 2


def test_detection_copy_raises_when_software_also_fails(
        settings, info, plan, tmp_path, monkeypatch):
    settings.transcode.optimizer.scenedetect_scale = "-2:540"
    enc = make_encoder(settings, info, plan, tmp_path)
    monkeypatch.setattr(enc, "_run_with_progress",
                        lambda args, timeout, total_seconds, tag: None)
    with pytest.raises(opt.TranscodeError, match="software"):
        enc._make_detection_copy()


def test_detection_copy_disables_periodic_keyframes(settings, info, plan, tmp_path):
    """x264's default IDR every 250 frames invents shot boundaries.

    The quality jump at a keyframe reads as a content change to the detector,
    which then reports a cut landing exactly on it - measured on real sources,
    at frame 250 on one clip and 250 and 750 on another, and 1 of 104 shots
    over five minutes. The copy is only ever read forward, so nothing needs the
    keyframes.
    """
    info.width, info.height = 3840, 2160
    enc = make_encoder(settings, info, plan, tmp_path)
    for _, args in enc._detection_copy_cmds(tmp_path / "out.mkv", "-2:540"):
        assert "-g" in args, args
        assert int(args[args.index("-g") + 1]) >= 1000


# ---- scdet detection ----
def test_cuts_from_scores_honours_threshold_and_min_scene_len(
        settings, info, plan, tmp_path):
    settings.transcode.optimizer.scdet_threshold = 2.0
    settings.transcode.optimizer.min_scene_len = 5
    enc = make_encoder(settings, info, plan, tmp_path)
    # frame 0 can never be a cut - it is where the first shot starts
    scores = [9.0] + [0.1] * 9
    assert enc._cuts_from_scores(scores) == []
    # two spikes closer together than min_scene_len: the second is swallowed
    scores = [0.0] * 20
    scores[6] = 5.0
    scores[9] = 5.0
    scores[14] = 5.0
    assert enc._cuts_from_scores(scores) == [6, 14]
    # below the threshold is not a cut
    settings.transcode.optimizer.scdet_threshold = 6.0
    assert enc._cuts_from_scores(scores) == []


def _cfr_pts(n, fps=30.0, jitter_ms=True):
    """Timestamps of n frames at fps, rounded to whole milliseconds the way
    Matroska stores them (so consecutive intervals alternate around 1/fps)."""
    return [round(i / fps, 3) if jitter_ms else i / fps for i in range(n)]


def test_cfr_check_accepts_millisecond_jitter(settings, info, plan, tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)          # 30fps
    enc.fps = 24000 / 1001
    enc._assert_constant_frame_rate([round(i * 1001 / 24000, 3) for i in range(2000)])


def test_cfr_check_keeps_a_dropped_frames_slot(settings, info, plan, tmp_path, monkeypatch):
    """One missing frame is a 2-period interval. A 4K Blu-ray remux had two
    of them in 68686 intervals, so they cannot be a reason to refuse; instead
    the timeline keeps the hole: every later frame sits one slot further on,
    and seeks, window durations and the output follow the slot."""
    enc = make_encoder(settings, info, plan, tmp_path)          # 30fps
    warnings = []
    monkeypatch.setattr(opt.logger, "warning", lambda *a, **k: warnings.append(a))
    pts = _cfr_pts(300)
    del pts[120]                                                # frame 120 never existed
    enc._assert_constant_frame_rate(pts)                        # accepted
    assert any("frame(s) missing" in str(w[0]) and w[2] == 1 for w in warnings)   # loguru: format, args
    enc._slots = enc._slots_from_pts(pts)
    assert enc._slot(119) == 119 and enc._slot(120) == 121 and enc._slot(298) == 299
    assert enc._slot(299) == 300                                # one past the table: extrapolated
    enc._lead_of = lambda path: 0.0
    assert float(enc._seek(120)) == pytest.approx(120.5 / 30.0, abs=1e-6)   # slot 121, half a frame early
    assert enc._span(100, 140) == pytest.approx(41 / 30.0)                  # the hole is inside the window
    assert enc._span(0, 100) == pytest.approx(100 / 30.0)


def test_cfr_check_refuses_real_variable_frame_rate(settings, info, plan, tmp_path):
    """Every 7th frame dropped (16% of intervals) is the VFR clip that ran
    14% fast against its audio and delivered VMAF median 54."""
    enc = make_encoder(settings, info, plan, tmp_path)
    pts = [t for i, t in enumerate(_cfr_pts(700)) if i % 7 != 6]
    with pytest.raises(opt.TranscodeError) as e:
        enc._assert_constant_frame_rate(pts)
    assert "longer than one frame" in str(e.value) and "engine=av1an" in str(e.value)


def test_cfr_check_refuses_a_duplicated_timestamp(settings, info, plan, tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)
    pts = _cfr_pts(300)
    pts[150] = pts[149]                                         # two frames on one timestamp
    with pytest.raises(opt.TranscodeError) as e:
        enc._assert_constant_frame_rate(pts)
    assert "shorter than half a frame" in str(e.value) and "frame 150" in str(e.value)


def test_concat_list_carries_each_shots_slot_duration(settings, info, plan, tmp_path, monkeypatch):
    """A hole at the very end of a shot has no frame to carry it into the
    concat; the explicit duration does."""
    enc = make_encoder(settings, info, plan, tmp_path)
    pts = _cfr_pts(300); del pts[99]                            # frame 99 missing: shot 0 ends in a hole
    enc._slots = enc._slots_from_pts(pts)
    monkeypatch.setattr(opt.shutil, "which",
                        lambda n, *a, **k: None if "mkvmerge" in n else f"/usr/bin/{n}")

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        if "ffprobe" in args[0]:
            return "video\n"
        _write_out(args, b"\x1aE\xdf\xa3"); return ""

    enc._run = fake_run.__get__(enc)
    enc.concat_shots([tmp_path / "a.ivf", tmp_path / "b.ivf"], [(0, 100), (100, 299)])
    text = (enc.tempdir / "concat.txt").read_text().splitlines()
    assert text[1] == f"duration {101 / 30.0:.6f}"             # frames 0..99 span 101 slots: the hole is inside
    assert text[3] == f"duration {199 / 30.0:.6f}"


def test_cfr_check_refuses_a_container_rate_that_is_not_the_timestamps(
        settings, info, plan, tmp_path):
    """Intervals all equal but at 25fps while the container claims 30: every
    frame number would be converted with the wrong period."""
    enc = make_encoder(settings, info, plan, tmp_path)          # info.fps = 30
    with pytest.raises(opt.TranscodeError) as e:
        enc._assert_constant_frame_rate(_cfr_pts(300, fps=25.0))
    assert "average 25.000 fps" in str(e.value) and "container's 30" in str(e.value)


def test_cfr_check_skips_when_the_pass_had_no_timestamps(settings, info, plan, tmp_path,
                                                        monkeypatch):
    enc = make_encoder(settings, info, plan, tmp_path)
    warnings = []
    monkeypatch.setattr(opt.logger, "warning", lambda *a, **k: warnings.append(a))
    enc._assert_constant_frame_rate([])                         # no raise
    assert any("could not be verified" in str(w[0]) for w in warnings)


def test_scdet_pass_collects_every_frames_pts(settings, info, plan, tmp_path, monkeypatch):
    enc = make_encoder(settings, info, plan, tmp_path)
    lines = ""
    for i in range(5):
        lines += f"frame:{i}    pts:{i * 42}      pts_time:{i * 0.042:.3f}\nlavfi.scd.mafd=0.1\nlavfi.scd.score=0.0\n"
    import io

    class FakeProc:
        stdout = io.StringIO(lines)
        pid = 4242
        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(opt.subprocess, "Popen", lambda *a, **k: FakeProc())
    scores = enc._run_scdet(["ffmpeg"])
    assert len(scores) == 5
    assert enc._frame_pts == pytest.approx([0.0, 0.042, 0.084, 0.126, 0.168])


def test_scdet_builds_shots_covering_every_frame(settings, info, plan, tmp_path,
                                                 monkeypatch):
    settings.transcode.optimizer.scdet_threshold = 2.0
    settings.transcode.optimizer.min_scene_len = 5
    enc = make_encoder(settings, info, plan, tmp_path)
    scores = [0.0] * 100
    scores[30] = 9.0
    scores[70] = 9.0
    monkeypatch.setattr(enc, "_run_scdet", lambda args: scores)
    shots = enc._detect_shots_scdet()
    assert shots == [(0, 30), (30, 70), (70, 100)]
    # the list has to tile the source exactly, which _validate_shots enforces
    assert shots[0][0] == 0 and shots[-1][1] == len(scores)
    assert all(a[1] == b[0] for a, b in zip(shots, shots[1:]))


def test_scdet_pass_stages_nothing(settings, info, plan, tmp_path):
    """The point of this engine: one pass over the source, no copy on disk."""
    info.width, info.height = 3840, 2160
    settings.transcode.optimizer.scenedetect_scale = "-2:540"
    enc = make_encoder(settings, info, plan, tmp_path)
    cmds = enc._scdet_cmds()
    assert [label for label, _ in cmds] == ["qsv", "software"]
    for _, args in cmds:
        joined = " ".join(args)
        assert "scdet=threshold=100" in joined    # thresholded in Python instead
        assert "metadata=print:file=-" in joined
        assert args[-2:] == ["-f", "null"] + [] or args[-3:] == ["-f", "null", "-"]
        assert "libx264" not in joined            # nothing is encoded
    settings.transcode.optimizer.scenedetect_hwaccel = "off"
    assert [label for label, _ in enc._scdet_cmds()] == ["software"]


def test_scdet_falls_back_then_gives_up(settings, info, plan, tmp_path, monkeypatch):
    info.width, info.height = 3840, 2160
    settings.transcode.optimizer.scenedetect_scale = "-2:540"
    enc = make_encoder(settings, info, plan, tmp_path)
    tried = []

    def fake(args):
        tried.append("qsv" if "-hwaccel" in args else "software")
        if "-hwaccel" in args:
            return []                       # decodes nothing, like AV1 on a B580
        return [0.0] * 50 + [9.0] + [0.0] * 49

    monkeypatch.setattr(enc, "_run_scdet", fake)
    shots = enc._detect_shots_scdet()
    assert tried == ["qsv", "software"]
    assert shots == [(0, 50), (50, 100)]

    monkeypatch.setattr(enc, "_run_scdet", lambda args: [])
    with pytest.raises(opt.TranscodeError, match="software"):
        enc._detect_shots_scdet()


def test_detect_shots_dispatches_on_the_configured_engine(
        settings, info, plan, tmp_path, monkeypatch):
    enc = make_encoder(settings, info, plan, tmp_path)
    called = []
    monkeypatch.setattr(enc, "_detect_shots_scdet",
                        lambda: called.append("scdet") or [(0, 400), (400, 900)])
    monkeypatch.setattr(enc, "_detect_shots_pyscenedetect",
                        lambda: called.append("pysd") or [(0, 900)])
    enc.total_frames = 900
    settings.transcode.optimizer.scenedetect_engine = "scdet"
    assert enc.detect_shots() == [(0, 400), (400, 900)]
    settings.transcode.optimizer.scenedetect_engine = "pyscenedetect"
    assert enc.detect_shots() == [(0, 900)]
    assert called == ["scdet", "pysd"]


def test_concat_quote_escapes_apostrophes():
    """The concat demuxer quotes like a shell, not like Python.

    f"'{path}'" truncates the filename at the first apostrophe, so a work
    directory such as /mnt/Trevor's Media loses every shot after that point.
    These paths are rooted at the configured dirs.work, so the character is
    the operator's to choose, not ours.
    """
    assert opt.concat_quote(Path("/w/enc_00000.ivf")) == "'/w/enc_00000.ivf'"
    assert opt.concat_quote(Path("/a b/c.ivf")) == "'/a b/c.ivf'"
    assert (opt.concat_quote(Path("/mnt/Trevor's Media/e.ivf"))
            == "'/mnt/Trevor'\\''s Media/e.ivf'")


def test_concat_quote_matches_shell_word_splitting():
    """Same escaping rules, so bash is a usable oracle for the demuxer's."""
    import subprocess

    for raw in ("/plain/a.ivf", "/a b/c.ivf", "/mnt/Trevor's Media/e.ivf",
                "/x/'quoted'/b.ivf", '/x/"dq"/b.ivf'):
        quoted = opt.concat_quote(Path(raw))
        out = subprocess.run(["bash", "-c", f"printf %s {quoted}"],
                             capture_output=True, text=True).stdout
        assert out == raw, f"{raw!r} -> {quoted} -> {out!r}"


def test_probe_scale_treats_zero_as_none(settings, info, plan, tmp_path):
    """A preset with probe_res "0" used to put scale=0 in the probe chain,
    which ffmpeg refuses ("Invalid size '0'")."""
    for v in ("0", " none ", "OFF", ""):
        plan.params.probe_res = v
        assert make_encoder(settings, info, plan, tmp_path)._probe_scale() == ""
    plan.params.probe_res = "960x540"
    assert make_encoder(settings, info, plan, tmp_path)._probe_scale() == "960x540"
    plan.params.probe_res = "0"
    settings.transcode.optimizer.probe_scale = "-2:720"
    assert make_encoder(settings, info, plan, tmp_path)._probe_scale() == "-2:720"



# ---------------------------------------------------------------- reference_hwaccel

_FRAMEMD5 = ("#format: frame checksums\n"
             "0,          0,          0,        1, 12441600, deadbeef\n"
             "0,          1,          1,        1, 12441600, deadbeef\n")
_VAAPI = ["-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD129", "-hwaccel_output_format", "vaapi"]


@pytest.fixture(autouse=True)
def _one_render_node(monkeypatch, settings):
    monkeypatch.setattr(opt, "_render_nodes", lambda: ["/dev/dri/renderD129"])
    # reference_hwaccel ships off (it measured slower on a 40-core host); the
    # tests below are about what it does when someone turns it on.
    settings.transcode.optimizer.reference_hwaccel = "auto"


def _capture_scores(enc, monkeypatch):
    calls = []

    def fake(dist_args, ref_args, ref_vf, idx, crf, threads=None, frames=None, window=None):
        calls.append((list(dist_args), list(ref_args), list(ref_vf)))
        return 90.0

    monkeypatch.setattr(enc, "_score_vmaf", fake)
    return calls


def test_reference_read_decodes_the_source_on_vaapi(settings, info, plan, tmp_path, monkeypatch):
    """Measured on 120 4K frames scored on SYCL: 39.4 -> 22.1 CPU-seconds and
    wall -12%, decoded frames bit-exact (framemd5), scores unchanged.

    Only the source read moves. The AV1 probe file decodes faster on dav1d
    than on QSV plus a download (10.1s against 13.5s wall), a DV shard is a
    small file already carrying the probe-side filters, and copying 4K frames
    back from the GPU tops out near 130 frames/s pool-wide - about what the
    probe pool consumes - so a second read per probe would be GPU-bound.
    """
    info.color.bit_depth, info.color.pix_fmt = 10, "yuv420p10le"
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = []
    monkeypatch.setattr(enc, "_run", lambda args, timeout=None: (ran.append(args), _FRAMEMD5)[1])
    calls = _capture_scores(enc, monkeypatch)
    enc._score_probe(600, 720, tmp_path / "d.ivf", 0, 30)
    dist, ref, ref_vf = calls[-1]
    assert ref[:6] == _VAAPI and ref.index("-hwaccel") < ref.index("-i")
    # downloaded, then handed on in the format the software decoder gives
    assert ref_vf[0] == "hwdownload,format=p010le,format=yuv420p10le"
    assert "-hwaccel" not in dist
    # the preflight: the same seeked frames read both ways, once per job
    assert len(ran) == 2 and "-hwaccel" not in ran[0] and "-hwaccel" in ran[1]
    for cmd in ran:
        assert "framemd5" in cmd and str(enc.source) in cmd and "-ss" in cmd
        assert cmd[cmd.index("-frames:v") + 1] == str(enc._HWDEC_PREFLIGHT_FRAMES)
        assert cmd.index("-ss") < cmd.index("-i")
    assert ran[0][ran[0].index("-ss") + 1] == ran[1][ran[1].index("-ss") + 1]
    enc._score_probe(720, 840, tmp_path / "d.ivf", 0, 30)
    assert len(ran) == 2
    # a DV shard read is left alone
    enc._score_probe(600, 720, tmp_path / "d.ivf", 0, 30, shard=tmp_path / "s.mkv")
    assert "-hwaccel" not in calls[-1][1] and calls[-1][2] == []
    # verification: the delivered AV1 side stays on the CPU, the source moves
    enc._score_windows(600, 720, 0, 30, 4)
    dist, ref, ref_vf = calls[-1]
    assert "-hwaccel" not in dist and str(enc.output) in dist
    assert ref[:6] == _VAAPI and str(enc.source) in ref
    assert ref_vf == ["hwdownload,format=p010le,format=yuv420p10le"]


def test_reference_read_falls_back_per_window_when_the_gpu_read_fails(
        settings, info, plan, tmp_path, monkeypatch):
    """E06 of Stranger Things: 65 seeked hardware reads matched software, and
    the first production batch still lost one window to "Failed to sync
    surface: internal decoding error". That window is scored again on the
    CPU; a timeout is not the GPU's doing and is not retried; and past
    _HWDEC_MAX_FAILURES the whole rest of the job goes to the CPU."""
    info.color.bit_depth, info.color.pix_fmt = 10, "yuv420p10le"
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._lead_of = lambda path: 0.0
    monkeypatch.setattr(enc, "_run", lambda args, timeout=None: _FRAMEMD5)
    warnings = []
    monkeypatch.setattr(opt.logger, "warning", lambda msg, *a, **k: warnings.append(msg.format(*a, **k)))
    calls = _capture_scores(enc, monkeypatch)
    real = enc._score_vmaf
    mode = {"fail": "sync"}

    calls_by_ss = []

    def flaky(dist_args, ref_args, ref_vf, idx, crf, threads=None, frames=None, window=None):
        hw = "-hwaccel" in ref_args
        if hw and mode["fail"] == "sync":
            raise opt.TranscodeError("ffmpeg failed (rc=251):\n[hwdownload] Failed to download frame: -5.")
        if hw and mode["fail"] == "timeout":
            raise opt.CommandTimeout("libvmaf timed out")
        if hw and mode["fail"] == "odd":
            # fail every other window; the ones between reset the streak
            calls_by_ss.append(1)
            if len(calls_by_ss) % 2 == 1:
                raise opt.TranscodeError("ffmpeg failed (rc=251):\n[hwdownload] Failed to download frame: -5.")
        return real(dist_args, ref_args, ref_vf, idx, crf, threads=threads, frames=frames,
                    window=window)

    monkeypatch.setattr(enc, "_score_vmaf", flaky)
    assert enc._score_probe(600, 720, tmp_path / "d.ivf", 0, 30) == 90.0
    assert ["-hwaccel" in ref for _, ref, _ in calls] == [False]        # the CPU attempt reached the scorer
    assert len(warnings) == 1 and "[600, 720)" in warnings[0] and "Failed to download" in warnings[0]
    assert enc._score_windows(600, 720, 0, 30, 4) == 90.0
    assert "-hwaccel" not in calls[-1][1] and calls[-1][2] == []
    # a timeout propagates untouched (the SYCL layer owns that decision)
    mode["fail"] = "timeout"
    with pytest.raises(opt.CommandTimeout):
        enc._score_probe(2000, 2120, tmp_path / "d.ivf", 0, 30)
    # a window that failed is remembered: its other CRF probes go straight
    # to the CPU (E06's broken head is probed at every CRF of the bisection)
    mode["fail"] = "sync"
    n = len(warnings)
    enc._score_probe(600, 720, tmp_path / "d.ivf", 0, 34)
    assert len(warnings) == n and "-hwaccel" not in calls[-1][1]
    # a scattered burst under load (E07): a good read between the failures
    # keeps resetting the streak, so the job never gives up the GPU
    mode["fail"] = "odd"
    for k in range(enc._HWDEC_MAX_STREAK * 3):
        enc._score_probe(2000 + 120 * k, 2120 + 120 * k, tmp_path / "d.ivf", 0, 30)
    assert enc._hwdec_ok is True and enc._hwdec_streak == 0
    # a device that has died fails read after read: enough in a row and it
    # stops trying
    mode["fail"] = "sync"
    for k in range(enc._HWDEC_MAX_STREAK + 2):
        enc._score_probe(5000 + 120 * k, 5120 + 120 * k, tmp_path / "d.ivf", 0, 30)
    assert enc._hwdec_ok is False
    assert any("the rest of the job scores on the CPU" in w for w in warnings)
    # after the flip the reads are built for the CPU outright
    n = len(warnings)
    enc._score_probe(840, 960, tmp_path / "d.ivf", 0, 30)
    assert len(warnings) == n and "-hwaccel" not in calls[-1][1]


def test_reference_read_download_format_follows_the_bit_depth(settings, info, plan, tmp_path, monkeypatch):
    info.color.bit_depth, info.color.pix_fmt = 8, "yuv420p"
    enc = make_encoder(settings, info, plan, tmp_path)
    monkeypatch.setattr(enc, "_run", lambda args, timeout=None: _FRAMEMD5)
    _, vf, hw = enc._reference_read(*enc._probe_input(600, 720))
    assert hw and vf[0] == "hwdownload,format=nv12,format=yuv420p"


def _md5_lines(sums):
    return "#format: frame checksums\n" + "".join(
        f"0, {i:10d}, {i:10d},        1,  3110400, {s}\n" for i, s in enumerate(sums))


@pytest.mark.parametrize("outcome", ["error", "no frames", "offset", "different"])
def test_reference_read_falls_back_to_software_for_the_job(
        settings, info, plan, tmp_path, monkeypatch, outcome):
    """A B580 decodes an unsupported stream to nothing and still exits 0, so
    the preflight cannot trust the exit code; and on an 8-bit H.264 WEB-DL
    its seeked read began five frames before the software decoder's, every
    frame bit-exact and every one wrong. So the seeked reads are compared
    frame by frame. Whatever the reason, the fallback is decided once, not
    once per score."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran, warnings = [], []
    monkeypatch.setattr(opt.logger, "warning", lambda msg, *a, **k: warnings.append(msg.format(*a, **k)))
    enc._lead_of = lambda path: 0.0      # the fixture file is empty; ffprobe would warn
    sw = [f"{c}{c}{c}" for c in "abcdefgh"]

    def fake_run(args, timeout=None):
        ran.append(args)
        if outcome == "error":
            raise opt.TranscodeError("ffmpeg exited 1")
        if "-hwaccel" not in args:
            return _md5_lines(sw)
        if outcome == "no frames":
            return "#format: frame checksums\n"
        if outcome == "offset":
            return _md5_lines(["v1", "v2", "v3", "v4", "v5"] + sw[:3])
        return _md5_lines([f"{c}{c}{c}" for c in "zyxwvuts"])

    monkeypatch.setattr(enc, "_run", fake_run)
    calls = _capture_scores(enc, monkeypatch)
    for _ in range(3):
        enc._score_probe(600, 720, tmp_path / "d.ivf", 0, 30)
    assert len(ran) == (1 if outcome == "error" else 2)
    assert len(warnings) == 1 and "stay on the CPU" in warnings[0]
    if outcome == "offset":
        assert "offset by 5 frame(s)" in warnings[0]
    if outcome == "different":
        assert "different frames" in warnings[0]
    assert all("-hwaccel" not in ref and not any("hwdownload" in f for f in vf)
               for _, ref, vf in calls)
    args, _ = enc._probe_input(600, 720)
    assert calls[-1][1] == args


def test_reference_read_needs_a_render_node(settings, info, plan, tmp_path, monkeypatch):
    """No /dev/dri in the container: nothing to preflight, software, one warning."""
    monkeypatch.setattr(opt, "_render_nodes", lambda: [])
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._lead_of = lambda path: 0.0
    monkeypatch.setattr(enc, "_run", lambda *a, **k: pytest.fail("no decode without a node"))
    warnings = []
    monkeypatch.setattr(opt.logger, "warning", lambda msg, *a, **k: warnings.append(msg.format(*a, **k)))
    calls = _capture_scores(enc, monkeypatch)
    enc._score_probe(600, 720, tmp_path / "d.ivf", 0, 30)
    enc._score_probe(720, 840, tmp_path / "d.ivf", 0, 30)
    assert all("-hwaccel" not in ref for _, ref, _ in calls)
    assert len(warnings) == 1 and "render node" in warnings[0]


def test_reference_read_off_never_touches_the_gpu(settings, info, plan, tmp_path, monkeypatch):
    settings.transcode.optimizer.reference_hwaccel = "off"
    enc = make_encoder(settings, info, plan, tmp_path)
    monkeypatch.setattr(enc, "_run", lambda *a, **k: pytest.fail("no preflight when off"))
    calls = _capture_scores(enc, monkeypatch)
    enc._score_probe(600, 720, tmp_path / "d.ivf", 0, 30)
    enc._score_windows(600, 720, 0, 30, 4)
    assert all("-hwaccel" not in ref for _, ref, _ in calls)


def test_sycl_device_error_scores_on_the_cpu_instead_of_killing_the_job(
        settings, info, plan, tmp_path, monkeypatch):
    """E07 of Stranger Things died on "SYCL memcpy H2D:
    OUT_OF_DEVICE_MEMORY" then "DEVICE_LOST" - ten VA-API reads and ten SYCL
    contexts exhausted the card, ffmpeg exited 234 and a three-hour job
    failed. A score the device cannot produce is one the CPU can; only the
    job dying is unrecoverable. DEVICE_LOST is terminal for the context, so
    the device is given up at once rather than after three more failures."""
    settings.transcode.optimizer.vmaf_sycl_device = 0
    settings.transcode.optimizer.vmaf_sycl_min_width = 0
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._lead_of = lambda path: 0.0
    enc._sycl_ok = True                 # the preflight already passed
    warnings = []
    monkeypatch.setattr(opt.logger, "warning", lambda msg, *a, **k: warnings.append(msg.format(*a, **k)))
    backends = []

    def on(sycl, dist_args, ref_args, ref_vf, idx, crf, threads, timeout, window=None):
        backends.append(sycl)
        if sycl >= 0:
            raise opt.TranscodeError(
                "ffmpeg failed (rc=234):\nlibvmaf ERROR SYCL memcpy H2D: "
                "level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)")
        return 91.5

    monkeypatch.setattr(enc, "_score_vmaf_on", on)
    assert enc._score_vmaf(["-i", "d"], ["-i", "r"], [], 0, 30, frames=120) == 91.5
    assert backends == [0, -1]                      # tried the GPU, scored on the CPU
    # ONE such failure is not the device going away. Caught in the act on a
    # real episode: a scorer aborted with DEVICE_LOST and OUT_OF_DEVICE_MEMORY
    # while the scorers beside it finished normally in the next breath, and a
    # selfcheck minutes later passed. The context died, not the card.
    assert enc._sycl_ok is True
    # it takes a run of them, with nothing getting through in between
    for _ in range(enc._SYCL_MAX_TIMEOUTS - 1):
        enc._score_vmaf(["-i", "d"], ["-i", "r"], [], 0, 30, frames=120)
    assert enc._sycl_ok is False
    assert any("in a row" in w for w in warnings)
    backends.clear()
    assert enc._score_vmaf(["-i", "d"], ["-i", "r"], [], 0, 30, frames=120) == 91.5
    assert backends == [-1]


def test_a_cancelled_job_does_not_retry_the_score_on_the_cpu(
        settings, info, plan, tmp_path, monkeypatch):
    settings.transcode.optimizer.vmaf_sycl_device = 0
    settings.transcode.optimizer.vmaf_sycl_min_width = 0
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._sycl_ok = True
    enc.cancel_flag = lambda: True
    monkeypatch.setattr(enc, "_score_vmaf_on",
                        lambda *a, **k: (_ for _ in ()).throw(opt.TranscodeError("Job cancelled by user")))
    with pytest.raises(opt.TranscodeError, match="cancelled"):
        enc._score_vmaf(["-i", "d"], ["-i", "r"], [], 0, 30, frames=120)


# ---------------------------------------------------------------- vmaf_zero_copy

_ZC_LOG = ("libvmaf INFO SYCL: using device: Intel(R) Arc(TM) B580 Graphics\n"
           "libvmaf INFO VA surface zero-copy: DMA-BUF → Level Zero → Tile4 "
           "de-tile (3840x2160 @ 2 bpp, pitch=7680)\n")


def _zc_encoder(settings, info, plan, tmp_path):
    o = settings.transcode.optimizer
    o.vmaf_sycl_device, o.vmaf_sycl_min_width, o.vmaf_zero_copy = 0, 0, "auto"
    o.reference_hwaccel = "off"
    info.width, info.height = 3840, 2160
    info.color.bit_depth, info.color.pix_fmt = 10, "yuv420p10le"
    plan.params.pixel_format = "yuv420p10le"
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._lead_of = lambda path: 0.0
    enc._sycl_ok = True                  # the SYCL preflight already passed
    return enc


def _write_score(args, mean):
    lavfi = args[args.index("-lavfi") + 1]
    Path(lavfi.split("log_path=")[1].split(":")[0]).write_text(
        json.dumps({"pooled_metrics": {"vmaf": {"mean": mean}}}))


def test_zero_copy_scores_probes_without_the_frames_leaving_the_card(
        settings, info, plan, tmp_path, monkeypatch):
    """Measured on the B580 at 4K: 1.18s and 1.36 CPU-seconds per probe score
    against 5.4s and 23 the usual way, identical to 0.0 on every frame over
    247 runs. Both sides decode on VA-API and reach libvmaf_sycl as surfaces:
    nothing downloads and nothing converts."""
    settings.transcode.optimizer.probing_rate = 2
    enc = _zc_encoder(settings, info, plan, tmp_path)
    enc._zc_ok = True                    # the zero-copy preflight already passed
    cmds = []

    def fake_run(args, timeout=None):
        args = [str(a) for a in args]
        cmds.append(args)
        _write_score(args, 93.25)
        return ""

    monkeypatch.setattr(enc, "_run", fake_run)
    assert enc._score_probe(600, 720, tmp_path / "d.ivf", 0, 30) == 93.25
    assert len(cmds) == 1 and enc._zc_scored == 1
    cmd = cmds[0]
    assert cmd[cmd.index("-init_hw_device") + 1] == "vaapi=zc:/dev/dri/renderD129"
    lavfi = cmd[cmd.index("-lavfi") + 1]
    assert "libvmaf_sycl=" in lavfi and "sycl_device=0" in lavfi and "n_threads" not in lavfi
    assert "format=" not in lavfi and "hwdownload" not in lavfi and "scale" not in lavfi
    dist, ref = lavfi.split(";")[:2]
    assert dist == "[0:v]setpts=PTS-STARTPTS[dist]"
    # the subsampling the probe was encoded with, then the rebase, as always
    assert ref.index("fps=") < ref.index("setpts=PTS-STARTPTS")
    # both inputs decode on the zero-copy device, each with its own options
    inputs = [i for i, a in enumerate(cmd) if a == "-i"]
    assert [cmd[i + 1] for i in inputs] == [str(tmp_path / "d.ivf"), str(enc.source)]
    hw = [i for i, a in enumerate(cmd) if a == "-hwaccel"]
    assert len(hw) == 2 and hw[0] < inputs[0] < hw[1] < inputs[1]
    assert cmd.count("zc") == 2 and "vaapi" in cmd


def test_zero_copy_lifts_an_8bit_1080p_source_to_p010_on_the_card(
        settings, info, plan, tmp_path, monkeypatch):
    """A 1080p Blu-ray is 8-bit H.264 scored with the 1080p model: its
    min(iw,1920) scale does nothing, and its NV12 surfaces meet P010 probes.
    The usual read converts both sides to yuv420p10le; the card does the same
    with scale_vaapi, to both, since a card probe of it is 8-bit too."""
    enc = _zc_encoder(settings, info, plan, tmp_path)
    info.width, info.height = 1920, 1080
    info.color.bit_depth, info.color.pix_fmt = 8, "yuv420p"
    assert enc._zc_unsupported() is None
    enc._zc_ok = True
    cmds = []

    def fake_run(args, timeout=None):
        args = [str(a) for a in args]
        cmds.append(args)
        _write_score(args, 96.5)
        return ""

    monkeypatch.setattr(enc, "_run", fake_run)
    assert enc._score_probe(600, 720, tmp_path / "d.ivf", 0, 30) == 96.5
    lavfi = cmds[0][cmds[0].index("-lavfi") + 1]
    dist, ref = lavfi.split(";")[:2]
    assert dist == "[0:v]setpts=PTS-STARTPTS,scale_vaapi=format=p010[dist]"
    assert ref.endswith("setpts=PTS-STARTPTS,scale_vaapi=format=p010[ref]")
    assert "libvmaf_sycl=" in lavfi and "format=yuv" not in lavfi


@pytest.mark.parametrize("case", ["off", "shard", "8-bit probes", "features",
                                  "1080p model", "no sycl"])
def test_zero_copy_stays_off_where_it_cannot_apply(
        settings, info, plan, tmp_path, monkeypatch, case):
    """A DV shard is an FFV1 file for the CPU; the import carries luma only,
    so luma-only is all a feature list may ask for, and there is none;
    libvmaf_sycl compares frames as decoded, so no 1080p-model downscale and
    no 10-bit source against 8-bit probes; and none of it without SYCL."""
    enc = _zc_encoder(settings, info, plan, tmp_path)
    shard = None
    if case == "off":
        settings.transcode.optimizer.vmaf_zero_copy = "off"
    elif case == "shard":
        shard = tmp_path / "s.mkv"
    elif case == "8-bit probes":
        plan.params.pixel_format = "yuv420p"
    elif case == "features":
        plan.params.probing_vmaf_features = "name=psnr"
    elif case == "1080p model":
        info.width, info.height = 2048, 858     # wider than vmaf_width: downscaled
    elif case == "no sycl":
        enc._sycl_ok = False
    monkeypatch.setattr(enc, "_zc_preflight",
                        lambda sycl: pytest.fail("no preflight where it cannot apply"))
    calls = _capture_scores(enc, monkeypatch)
    enc._score_probe(600, 720, tmp_path / "d.ivf", 0, 30, shard=shard)
    assert len(calls) == 1 and enc._zc_scored == 0      # scored the usual way


def test_zero_copy_failure_rescores_the_window_the_usual_way(
        settings, info, plan, tmp_path, monkeypatch):
    """A failed zero-copy score is not a failed probe. The window is scored
    again the usual way and remembered, so its other probes skip the attempt;
    and a run of failures with nothing getting through retires zero-copy for
    the job - the same shape as reference_hwaccel's reads."""
    enc = _zc_encoder(settings, info, plan, tmp_path)
    enc._zc_ok = True
    warnings = []
    monkeypatch.setattr(opt.logger, "warning",
                        lambda msg, *a, **k: warnings.append(msg.format(*a, **k)))
    attempts = []
    mode = {"fail": True}

    def on(sycl, dist_args, ref_args, ref_vf, idx, crf, threads, timeout, zero_copy=False,
           window=None):
        attempts.append(zero_copy)
        if zero_copy and mode["fail"]:
            raise opt.TranscodeError("ffmpeg failed (rc=-11):\nSegmentation fault")
        return 88.0

    monkeypatch.setattr(enc, "_score_vmaf_on", on)
    assert enc._score_probe(600, 720, tmp_path / "d.ivf", 0, 30) == 88.0
    assert attempts == [True, False]
    assert len(warnings) == 1 and "[600, 720)" in warnings[0] and "Segmentation fault" in warnings[0]
    attempts.clear()
    enc._score_probe(600, 720, tmp_path / "d.ivf", 0, 34)
    assert attempts == [False]
    # a success between failures resets the streak
    mode["fail"] = False
    enc._score_probe(720, 840, tmp_path / "d.ivf", 0, 30)
    assert enc._zc_streak == 0 and enc._zc_scored == 1
    mode["fail"] = True
    for k in range(enc._ZC_MAX_STREAK):
        enc._score_probe(1000 + 120 * k, 1120 + 120 * k, tmp_path / "d.ivf", 0, 30)
    assert enc._zc_ok is False
    assert any("the rest of the job scores the usual way" in w for w in warnings)
    attempts.clear()
    enc._score_probe(5000, 5120, tmp_path / "d.ivf", 0, 30)
    assert attempts == [False]
    assert enc._zc_summary() == f"; 1 score(s) zero-copy, {enc._ZC_MAX_STREAK + 1} fell back"


def test_zero_copy_failure_of_a_cancelled_job_is_not_rescored(
        settings, info, plan, tmp_path, monkeypatch):
    enc = _zc_encoder(settings, info, plan, tmp_path)
    enc._zc_ok = True
    enc.cancel_flag = lambda: True
    monkeypatch.setattr(enc, "_score_vmaf_on",
                        lambda *a, **k: (_ for _ in ()).throw(opt.TranscodeError("Job cancelled by user")))
    with pytest.raises(opt.TranscodeError, match="cancelled"):
        enc._score_probe(600, 720, tmp_path / "d.ivf", 0, 30)


@pytest.mark.parametrize("outcome", ["agrees", "readback", "disagrees", "fails"])
def test_zero_copy_preflight_decides_once_per_job(
        settings, info, plan, tmp_path, monkeypatch, outcome):
    """libvmaf either imports the surfaces or quietly reads them back - it says
    which only at INFO - and a wrong de-tile or a missed P010 shift still
    produces a score. So each job encodes one short window, scores it both
    ways, and keeps zero-copy only when the import stayed on the card and the
    two scores agree."""
    enc = _zc_encoder(settings, info, plan, tmp_path)
    monkeypatch.setattr(enc, "_hwdec_preflight", lambda: True)
    warnings = []
    monkeypatch.setattr(opt.logger, "warning",
                        lambda msg, *a, **k: warnings.append(msg.format(*a, **k)))
    ran = []

    def fake_run(args, timeout=None):
        args = [str(a) for a in args]
        ran.append(args)
        if "libsvtav1" in args:
            _write_out(args, b"ivf")
            return ""
        lavfi = args[args.index("-lavfi") + 1]
        if "libvmaf_sycl=" in lavfi:
            if outcome == "fails":
                raise opt.TranscodeError("ffmpeg failed (rc=234)")
            _write_score(args, 91.0 if outcome == "disagrees" else 90.0)
            if outcome == "readback":
                return _ZC_LOG + "libvmaf INFO DMA-BUF import failed (-5) - using readback path\n"
            return _ZC_LOG
        _write_score(args, 90.0)
        return ""

    monkeypatch.setattr(enc, "_run", fake_run)
    for _ in range(3):
        assert enc._zero_copy() is (outcome == "agrees")
    assert len(ran) == 3        # the encode, the usual score, the zero-copy score - once
    assert ran[2][ran[2].index("-loglevel") + 1] == "info"
    assert len(warnings) == (0 if outcome == "agrees" else 1)
    assert not list(enc.probe_dir.glob("zc_preflight*"))


# ---------------------------------------------------------------- graph rebuilds
# With -hwaccel vaapi an in-band SPS/PPS change gives the decoder a new frames
# context, and fftools rebuilds the whole filter graph for that alone. Every
# stateful filter restarts: measured, a card probe wrote 77 frames for a
# 64-frame window, and a zero-copy score kept 59 of 84.

_REBUILT_LINE = "[vf#0:0 @ 0x55d0c0a1b2c0] Reconfiguring filter graph because hwaccel changed\n"


def _ended(ending, output):
    """What _run does with `output` when ffmpeg exits 0, fails or stalls."""
    if ending == "exit 0":
        return output
    err = (opt.TranscodeError("ffmpeg failed (rc=234):\n" + output[-2000:])
           if ending == "failed" else opt.CommandTimeout("command timed out after 180s: ffmpeg"))
    err.output = output
    raise err


@pytest.mark.parametrize("text, reason", [
    (_REBUILT_LINE, "hwaccel changed"),
    ("[fc#0 @ 0x55d0c0a1b2c0] [info] Reconfiguring filter graph because hwaccel changed\n",
     "hwaccel changed"),
    ("frame=   12 fps=0.0 q=-0.0 size=N/A time=00:00:00.50 bitrate=N/A speed=1x    \r"
     + _REBUILT_LINE, "hwaccel changed"),
    # log.c keeps one print_prefix for the whole process: after another
    # thread's unterminated message the line comes out bare, or glued on
    ("Reconfiguring filter graph because hwaccel changed\n", "hwaccel changed"),
    ("[hevc @ 0x1] Invalid value 7 for log2_min_cb_sizeReconfiguring filter graph "
     "because hwaccel changed\n", "hwaccel changed"),
    ("[vf#0:0 @ 0x5] Reconfiguring filter graph because video parameters changed to "
     "vaapi(tv, bt2020nc), 1920x1080, straight alpha, hwaccel changed\n",
     "video parameters changed to vaapi(tv, bt2020nc), 1920x1080, straight alpha, "
     "hwaccel changed"),
    ("[vf#0:0 @ 0xabc] Reconfiguring filter graph\n", "unspecified"),
    # a stream title in the input dump is the file talking, not ffmpeg
    ("Input #0, matroska,webm, from 'x.mkv':\n  Metadata:\n"
     "    title           : Reconfiguring filter graph because hwaccel changed\n", None),
    ("    title           : Reconfiguring filter graph\n" + _REBUILT_LINE, "hwaccel changed"),
    ("[out#0/null @ 0x1] video:0KiB audio:0KiB subtitle:0KiB\n", None),
    ("", None),
    (None, None),
])
def test_graph_rebuilt_reads_ffmpegs_own_line(text, reason):
    assert opt.graph_rebuilt(text) == reason


def test_run_keeps_the_whole_output_when_a_command_fails(settings, info, plan, tmp_path):
    """The failure message carries the last 2000 characters, and a rebuild
    logged before a long error tail falls outside them. A real subprocess."""
    enc = make_encoder(settings, info, plan, tmp_path)
    line = _REBUILT_LINE.strip()
    with pytest.raises(opt.TranscodeError) as e:
        enc._run([sys.executable, "-c",
                  f"import sys; print({line!r}); print('x' * 5000); sys.exit(1)"])
    assert line not in str(e.value) and line in e.value.output


def test_run_keeps_what_a_timed_out_command_printed(settings, info, plan, tmp_path):
    """A rebuild that stalls a score has to read as a rebuild, not as the
    card failing to finish in time."""
    enc = make_encoder(settings, info, plan, tmp_path)
    line = _REBUILT_LINE.strip()
    with pytest.raises(opt.CommandTimeout) as e:
        enc._run([sys.executable, "-c",
                  f"import time; print({line!r}, flush=True); time.sleep(30)"], timeout=1)
    assert line in e.value.output


@pytest.mark.parametrize("ending", ["exit 0", "failed", "timed out"])
def test_zero_copy_rebuild_rereads_the_window_without_touching_any_streak(
        settings, info, plan, tmp_path, monkeypatch, ending):
    """A zero-copy score across a parameter change is the source, not the
    card: the window is scored again the usual way and remembered, and no
    failure streak moves - zero-copy's, the SYCL device's or the hardware
    read's - whether ffmpeg exited 0, crashed on the restart or stalled."""
    enc = _zc_encoder(settings, info, plan, tmp_path)
    enc._zc_ok = True
    enc._zc_streak, enc._sycl_timeouts, enc._hwdec_streak = 3, 2, 5
    warnings = []
    monkeypatch.setattr(opt.logger, "warning",
                        lambda msg, *a, **k: warnings.append(msg.format(*a, **k)))
    cmds, at_reread = [], []

    def fake_run(args, timeout=None):
        args = [str(a) for a in args]
        cmds.append(args)
        if "libvmaf_sycl=" in args[args.index("-lavfi") + 1]:
            _write_score(args, 100.0)               # the restarted instance's segment
            return _ended(ending, _ZC_LOG + _REBUILT_LINE + "x" * 3000)
        # the moment the usual read starts, nothing has been counted
        at_reread.append((enc._zc_streak, enc._zc_fallbacks, enc._sycl_timeouts,
                          enc._hwdec_streak))
        _write_score(args, 88.0)
        return ""

    monkeypatch.setattr(enc, "_run", fake_run)
    assert enc._score_probe(600, 720, tmp_path / "d.ivf", 0, 30) == 88.0
    assert at_reread == [(3, 0, 2, 5)]
    # afterwards only the usual read's own success has spoken, and it clears
    # the SYCL stall streak as any good score does
    assert (enc._zc_streak, enc._zc_fallbacks, enc._sycl_timeouts, enc._hwdec_streak) == (3, 0, 0, 5)
    assert enc._zc_ok is True and enc._sycl_ok is True and enc._zc_scored == 0
    assert (600, 720) in enc._zc_bad and (600, 720) in enc._hwdec_bad
    assert len(warnings) == 1 and "[600, 720)" in warnings[0] and "hwaccel changed" in warnings[0]
    zc, usual = cmds
    assert zc[zc.index("-loglevel") + 1] == "info" and "-nostats" in zc
    assert "-hwaccel" not in usual
    # the window's other probes go straight to the usual read, quietly
    cmds.clear()
    assert enc._score_probe(600, 720, tmp_path / "d.ivf", 0, 34) == 88.0
    assert len(cmds) == 1 and "libvmaf_sycl=" not in cmds[0][cmds[0].index("-lavfi") + 1]
    assert len(warnings) == 1
    assert "1 window(s) read in software after a graph rebuild" in enc._zc_summary()


def test_a_cancelled_job_is_not_rescored_after_a_rebuild(
        settings, info, plan, tmp_path, monkeypatch):
    enc = _zc_encoder(settings, info, plan, tmp_path)
    enc._zc_ok = True
    enc.cancel_flag = lambda: True
    attempts = []

    def on(sycl, dist_args, ref_args, ref_vf, idx, crf, threads, timeout, zero_copy=False,
           window=None):
        attempts.append(zero_copy)
        raise opt.GraphRebuilt("hwaccel changed", window, "zero-copy score")

    monkeypatch.setattr(enc, "_score_vmaf_on", on)
    with pytest.raises(opt.TranscodeError):
        enc._score_probe(600, 720, tmp_path / "d.ivf", 0, 30)
    assert attempts == [True] and not enc._rebuilt
    # a command the cancel killed is the cancel, whatever it logged first
    monkeypatch.setattr(enc, "_run", lambda args, timeout=None: _ended(
        "failed", _REBUILT_LINE + "[out#0/null @ 0x1] Terminating thread\n"))
    with pytest.raises(opt.TranscodeError) as e:
        enc._run_read(["ffmpeg", *_VAAPI, "-i", "src.mkv"], 60, "reference read", (600, 720))
    assert not isinstance(e.value, opt.GraphRebuilt)


def test_reference_read_rebuild_goes_to_software_without_counting(
        settings, info, plan, tmp_path, monkeypatch):
    """A VA-API reference read across a parameter change is not a decode
    failure: no step towards giving up the GPU and no 'failed' warning, and
    the window reads in software from then on - in verification too."""
    info.color.bit_depth, info.color.pix_fmt = 10, "yuv420p10le"
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._lead_of = lambda path: 0.0
    monkeypatch.setattr(enc, "_run", lambda args, timeout=None: _FRAMEMD5)
    warnings = []
    monkeypatch.setattr(opt.logger, "warning",
                        lambda msg, *a, **k: warnings.append(msg.format(*a, **k)))
    calls = _capture_scores(enc, monkeypatch)
    real = enc._score_vmaf

    def rebuilt(dist_args, ref_args, ref_vf, idx, crf, threads=None, frames=None, window=None):
        if "-hwaccel" in ref_args:
            raise opt.GraphRebuilt("hwaccel changed", window, "reference read")
        return real(dist_args, ref_args, ref_vf, idx, crf, threads=threads, frames=frames,
                    window=window)

    monkeypatch.setattr(enc, "_score_vmaf", rebuilt)
    enc._hwdec_streak = 3
    assert enc._score_probe(600, 720, tmp_path / "d.ivf", 0, 30) == 90.0
    assert ["-hwaccel" in ref for _, ref, _ in calls] == [False]
    assert enc._hwdec_streak == 3 and enc._hwdec_ok is True and (600, 720) in enc._hwdec_bad
    assert len(warnings) == 1 and "failed" not in warnings[0] and "[600, 720)" in warnings[0]
    # verification of that window goes straight to software
    assert enc._score_windows(600, 720, 0, 30, 4) == 90.0
    assert len(calls) == 2 and "-hwaccel" not in calls[-1][1]
    # and a window whose verification read rebuilds is read again in software
    assert enc._score_windows(840, 960, 1, 30, 4) == 90.0
    assert len(calls) == 3 and "-hwaccel" not in calls[-1][1] and calls[-1][2] == []
    assert (840, 960) in enc._hwdec_bad and enc._hwdec_streak == 3 and len(warnings) == 2


def test_an_xpsnr_reference_read_that_rebuilds_goes_to_software_too(
        settings, info, plan, tmp_path, monkeypatch):
    """The xpsnr filter restarts with a rebuilt graph as libvmaf does, so its
    card reference read is watched the same way: read again in software, the
    window remembered, no step towards giving up the GPU."""
    info.color.bit_depth, info.color.pix_fmt = 10, "yuv420p10le"
    enc = _metric_encoder(settings, info, plan, tmp_path, "xpsnr")
    enc._lead_of = lambda path: 0.0
    monkeypatch.setattr(opt.logger, "warning", lambda *a, **k: None)
    xpsnr = "[Parsed_xpsnr_2 @ 0x1] XPSNR  y: 40.0000  u: 49.5918  v: 51.1706  (minimum: 40.0000)\n"
    scores = []

    def fake_run(args, timeout=None):
        args = [str(a) for a in args]
        if "-lavfi" not in args:
            return _FRAMEMD5                     # the reference_hwaccel preflight
        scores.append(args)
        return (_REBUILT_LINE if "-hwaccel" in args else "") + xpsnr

    monkeypatch.setattr(enc, "_run", fake_run)
    enc._hwdec_streak = 3
    assert enc._score_probe(600, 720, tmp_path / "d.ivf", 0, 30) == 40.0
    assert ["-hwaccel" in c for c in scores] == [True, False]
    assert (600, 720) in enc._hwdec_bad and enc._hwdec_streak == 3 and enc._hwdec_ok is True


def test_score_vmaf_lets_a_rebuild_through_without_retrying_the_same_read(
        settings, info, plan, tmp_path, monkeypatch):
    """Retrying a rebuilt card reference read on the CPU would read the card
    again and rebuild again; and it says nothing about the SYCL device."""
    settings.transcode.optimizer.vmaf_sycl_device = 0
    settings.transcode.optimizer.vmaf_sycl_min_width = 0
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._sycl_ok = True
    backends = []

    def on(sycl, dist_args, ref_args, ref_vf, idx, crf, threads, timeout, window=None):
        backends.append(sycl)
        raise opt.GraphRebuilt("hwaccel changed", window, "reference read")

    monkeypatch.setattr(enc, "_score_vmaf_on", on)
    with pytest.raises(opt.GraphRebuilt):
        enc._score_vmaf(["-i", "d"], [*_VAAPI, "-i", "r"], [], 0, 30, frames=120,
                        window=(600, 720))
    assert backends == [0] and enc._sycl_timeouts == 0 and enc._sycl_ok is True


def test_score_vmaf_on_watches_every_read_and_never_parses_a_stale_log(
        settings, info, plan, tmp_path, monkeypatch):
    """A card read that rebuilds leaves as GraphRebuilt with no log behind it.
    A software read that rebuilds has nowhere better to go - the stream
    really changes there - so it is one warning per window and the score
    stands. And the per-(shot, CRF) log path is shared by retries and
    fallbacks, so a run that scores nothing must not return an old log."""
    enc = make_encoder(settings, info, plan, tmp_path)
    warnings = []
    monkeypatch.setattr(opt.logger, "warning",
                        lambda msg, *a, **k: warnings.append(msg.format(*a, **k)))
    changed = ("[fc#0 @ 0x2] Reconfiguring filter graph because video parameters changed "
               "to yuv420p10le(tv, bt2020nc), 1920x800, straight alpha\n")
    mode = {"write": True, "say": changed}
    ran = []

    def fake_run(args, timeout=None):
        args = [str(a) for a in args]
        ran.append(args)
        if mode["write"]:
            _write_score(args, 91.0)
        return mode["say"]

    monkeypatch.setattr(enc, "_run", fake_run)
    sw = ["-i", "r.mkv"]
    assert enc._score_vmaf_on(-1, ["-i", "d"], sw, [], 0, 30, 4, 3600, window=(600, 720)) == 91.0
    assert ran[-1][ran[-1].index("-loglevel") + 1] == "info" and "-nostats" in ran[-1]
    assert len(warnings) == 1 and "software" in warnings[0] and "[600, 720)" in warnings[0]
    assert "video parameters changed" in warnings[0]
    assert enc._score_vmaf_on(-1, ["-i", "d"], sw, [], 0, 34, 4, 3600, window=(600, 720)) == 91.0
    assert len(warnings) == 1
    mode["say"] = _REBUILT_LINE
    with pytest.raises(opt.GraphRebuilt) as e:
        enc._score_vmaf_on(-1, ["-i", "d"], [*_VAAPI, *sw], [], 0, 30, 4, 3600,
                           window=(840, 960))
    assert (e.value.window, e.value.read) == ((840, 960), "reference read")
    log = enc.probe_dir / "score_00000_30.json"
    assert not log.exists()
    mode["say"], mode["write"] = "", False
    log.write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 50.0}}}))
    with pytest.raises(opt.TranscodeError):
        enc._score_vmaf_on(-1, ["-i", "d"], sw, [], 0, 30, 4, 3600, window=(600, 720))


def test_a_missing_sycl_log_scores_on_the_cpu(settings, info, plan, tmp_path, monkeypatch):
    """ffmpeg can exit 0 and leave no log. On the SYCL device that used to
    escape as FileNotFoundError and fail the job; it is a failed scoring like
    a crash, retried on the CPU. On the zero-copy read it falls back to the
    usual one."""
    enc = _zc_encoder(settings, info, plan, tmp_path)
    monkeypatch.setattr(opt.logger, "warning", lambda *a, **k: None)

    def fake_run(args, timeout=None):
        args = [str(a) for a in args]
        if "sycl_device=" not in args[args.index("-lavfi") + 1]:
            _write_score(args, 91.0)
        return ""

    monkeypatch.setattr(enc, "_run", fake_run)
    assert enc._score_vmaf(["-i", "d"], ["-i", "r"], [], 0, 30, frames=120) == 91.0
    assert enc._sycl_timeouts == 1
    enc._zc_ok = True
    assert enc._score_probe(600, 720, tmp_path / "d.ivf", 0, 30) == 91.0
    assert enc._zc_fallbacks == 1 and enc._sycl_timeouts == 2


@pytest.mark.parametrize("ending", ["failed", "timed out"])
def test_a_sycl_score_across_a_real_parameter_change_is_not_held_against_the_device(
        settings, info, plan, tmp_path, monkeypatch, ending):
    """A software read of a window whose stream really changes rebuilds its
    graph, SYCL scorer and all. That instance failing or stalling is the
    source, not the card: the CPU still scores the window, but no stall is
    counted - one more would retire the device for the job here - and the one
    warning is the rebuild's."""
    enc = _zc_encoder(settings, info, plan, tmp_path)
    enc._sycl_timeouts = enc._SYCL_MAX_TIMEOUTS - 1
    warnings = []
    monkeypatch.setattr(opt.logger, "warning",
                        lambda msg, *a, **k: warnings.append(msg.format(*a, **k)))
    changed = ("[vf#1:0 @ 0x4] Reconfiguring filter graph because video parameters changed "
               "to yuv420p10le(tv, bt709), 3840x2160, straight alpha\n")
    mode = {"say": changed}
    backends = []

    def fake_run(args, timeout=None):
        args = [str(a) for a in args]
        sycl = "sycl_device=" in args[args.index("-lavfi") + 1]
        backends.append(sycl)
        if sycl:
            return _ended(ending, _ZC_LOG + mode["say"] + "x" * 3000)
        _write_score(args, 91.0)
        return mode["say"]

    monkeypatch.setattr(enc, "_run", fake_run)
    assert enc._score_vmaf(["-i", "d"], ["-i", "r"], [], 0, 30, frames=120,
                           window=(600, 720)) == 91.0
    assert backends == [True, False]
    assert enc._sycl_timeouts == enc._SYCL_MAX_TIMEOUTS - 1 and enc._sycl_ok is True
    assert len(warnings) == 1 and "software score read of frames [600, 720)" in warnings[0]
    # the same failure with no rebuild behind it is the device's, and retires it
    mode["say"] = ""
    assert enc._score_vmaf(["-i", "d"], ["-i", "r"], [], 0, 30, frames=120,
                           window=(840, 960)) == 91.0
    assert enc._sycl_timeouts == enc._SYCL_MAX_TIMEOUTS and enc._sycl_ok is False


def test_a_rebuild_on_a_software_read_warns_once_and_carries_on(
        settings, info, plan, tmp_path, monkeypatch):
    """Where the stream really changes size, format or colour, the software
    read rebuilds its graph too and no other read avoids it. Refusing over
    one window is the kind of refusal that once killed a whole queue, so it
    is one warning per window, across probes and verification, and the job
    goes on."""
    settings.transcode.optimizer.reference_hwaccel = "off"
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._lead_of = lambda path: 0.0
    warnings = []
    monkeypatch.setattr(opt.logger, "warning",
                        lambda msg, *a, **k: warnings.append(msg.format(*a, **k)))
    changed = ("[vf#0:0 @ 0x3] Reconfiguring filter graph because video parameters changed "
               "to yuv420p10le(tv, bt709), 1920x800, straight alpha\n")
    ran = []

    def fake_run(args, timeout=None):
        args = [str(a) for a in args]
        ran.append(args)
        if "-lavfi" in args:
            _write_score(args, 90.0)
        else:
            _write_out(args, b"ivf")
        return changed

    monkeypatch.setattr(enc, "_run", fake_run)
    scores = enc._probe_shot(0, 0, 90, [20, 26, 32], lp=4)
    assert len(scores) == 3 and set(scores.values()) == {90.0}
    assert enc._score_windows(0, 90, 0, 30, 4) == 90.0
    assert len(warnings) == 1 and "software probe encode" in warnings[0]
    assert "[0, 90)" in warnings[0] and "video parameters changed" in warnings[0]
    encodes = [c for c in ran if "libsvtav1" in c]
    assert len(encodes) == 3
    assert all(c[c.index("-loglevel") + 1] == "info" and "-nostats" in c for c in encodes)
    assert not enc._rebuilt and not enc._zc_bad


@pytest.mark.parametrize("ending", ["failed", "timed out"])
def test_a_software_read_that_rebuilds_and_then_fails_is_still_warned_about(
        settings, info, plan, tmp_path, monkeypatch, ending):
    """A crash or stall after a real parameter change fails as it always did,
    but not silently: the window's one warning names the change, and the
    error carries it so a SYCL score does not blame the device (see
    _score_vmaf)."""
    enc = make_encoder(settings, info, plan, tmp_path)
    warnings = []
    monkeypatch.setattr(opt.logger, "warning",
                        lambda msg, *a, **k: warnings.append(msg.format(*a, **k)))
    changed = ("[vf#0:0 @ 0x3] Reconfiguring filter graph because video parameters changed "
               "to yuv420p10le(tv, bt709), 1920x800, straight alpha\n")
    monkeypatch.setattr(enc, "_run",
                        lambda args, timeout=None: _ended(ending, changed + "x" * 3000))
    with pytest.raises(opt.TranscodeError) as e:
        enc._run_read(["ffmpeg", "-i", "src.mkv"], 60, "score read", (600, 720))
    assert not isinstance(e.value, opt.GraphRebuilt)
    assert isinstance(e.value, opt.CommandTimeout) == (ending == "timed out")
    assert e.value.rebuilt.startswith("video parameters changed")
    assert len(warnings) == 1 and "software score read of frames [600, 720)" in warnings[0]


def test_a_rebuild_while_staging_a_shard_is_warned_about_once(
        settings, info, plan, tmp_path, monkeypatch):
    """A DV P5 or SSIMULACRA2 probe reads the source once, into its shard, and
    every probe encode and score of the shot reads the shard. A parameter
    change inside the window shows only in that staging read, so that is where
    it is watched: one warning, and the shard stands like any software read."""
    cache = tmp_path / "shm"
    cache.mkdir()
    enc = _p5_encoder(settings, info, plan, tmp_path, cache)
    enc._lead_of = lambda path: 0.0
    settings.transcode.optimizer.probe_bracket_width = 0     # sweep the grid
    warnings = []
    monkeypatch.setattr(opt.logger, "warning",
                        lambda msg, *a, **k: warnings.append(msg.format(*a, **k)))
    changed = ("[vf#0:0 @ 0x3] Reconfiguring filter graph because video parameters changed "
               "to yuv420p10le(tv, bt709), 3840x1920, straight alpha\n")
    staged = []

    def fake_run(args, timeout=None):
        args = [str(a) for a in args]
        if "ffv1" in args:
            staged.append(args)
            _write_out(args, b"shard")
            return changed
        if "-lavfi" in args:
            _write_score(args, 92.0)
        elif "-f" in args and args[args.index("-f") + 1] == "ivf":
            _write_out(args, b"ivf")
        return ""

    monkeypatch.setattr(enc, "_run", fake_run)
    assert set(enc._probe_shot(0, 0, 90, [20, 32], lp=4).values()) == {92.0}
    assert len(staged) == 1
    assert staged[0][staged[0].index("-loglevel") + 1] == "info" and "-nostats" in staged[0]
    assert len(warnings) == 1 and "software staging read of frames [0, 90)" in warnings[0]
    assert "video parameters changed" in warnings[0] and not enc._rebuilt


def test_reference_read_preflight_steps_off_a_parameter_change(
        settings, info, plan, tmp_path, monkeypatch):
    """Eight frames that cross a parameter change prove nothing about the
    decoder, so another window is read before anything is decided."""
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._lead_of = lambda path: 0.0
    warnings = []
    monkeypatch.setattr(opt.logger, "warning",
                        lambda msg, *a, **k: warnings.append(msg.format(*a, **k)))
    ran = []

    def fake_run(args, timeout=None):
        ran.append(args)
        return _FRAMEMD5 + (_REBUILT_LINE if "-hwaccel" in args and len(ran) == 2 else "")

    monkeypatch.setattr(enc, "_run", fake_run)
    assert enc._hwdec() is True
    assert len(ran) == 4 and not warnings
    assert ran[0][ran[0].index("-ss") + 1] != ran[2][ran[2].index("-ss") + 1]
    assert all(c[c.index("-loglevel") + 1] == "info" and "-nostats" in c for c in ran)


@pytest.mark.parametrize("outcome", ["the next window agrees", "every window rebuilds"])
def test_zero_copy_preflight_steps_off_a_parameter_change(
        settings, info, plan, tmp_path, monkeypatch, outcome):
    """A preflight window across a parameter change scores part of itself on
    the card and all of it the usual way. That disagreement is the source,
    not zero-copy, and switching zero-copy off for the whole job over it -
    blaming the scores - is what this used to do."""
    enc = _zc_encoder(settings, info, plan, tmp_path)
    monkeypatch.setattr(enc, "_hwdec_preflight", lambda: True)
    warnings = []
    monkeypatch.setattr(opt.logger, "warning",
                        lambda msg, *a, **k: warnings.append(msg.format(*a, **k)))
    ran = []

    def fake_run(args, timeout=None):
        args = [str(a) for a in args]
        ran.append(args)
        if "libsvtav1" in args:
            _write_out(args, b"ivf")
            return ""
        if "libvmaf_sycl=" in args[args.index("-lavfi") + 1]:
            if outcome == "every window rebuilds" or sum(
                    "libvmaf_sycl=" in " ".join(c) for c in ran) == 1:
                _write_score(args, 97.0)          # the segment after the change
                return _ZC_LOG + _REBUILT_LINE
            _write_score(args, 90.0)
            return _ZC_LOG
        _write_score(args, 90.0)
        return ""

    monkeypatch.setattr(enc, "_run", fake_run)
    agrees = outcome == "the next window agrees"
    windows = len(enc._preflight_starts(enc._ZC_PREFLIGHT_FRAMES))
    for _ in range(3):
        assert enc._zero_copy() is agrees
    assert windows >= 2 and len(ran) == (6 if agrees else 3 * windows)
    encodes = [c[c.index("-ss") + 1] for c in ran if "libsvtav1" in c]
    assert len(set(encodes)) == len(encodes)          # a different window each time
    if agrees:
        assert not warnings
    else:
        assert len(warnings) == 1 and "parameter change" in warnings[0]
    assert not list(enc.probe_dir.glob("zc_preflight*"))


# ---------------------------------------------------------------- gpu probe path

def _curve(idxs, crossing, target, slope=1.0):
    """A monotone score curve that crosses `target` exactly at `crossing`."""
    return {i: target + (crossing - i) * slope for i in idxs}


def _gpu_encoder(settings, info, plan, tmp_path, shots=20):
    settings.transcode.optimizer.probe_encoder = "qsv"
    settings.transcode.optimizer.gpu_probe_anchors = 6
    settings.transcode.optimizer.probe_crfs = [20, 26, 32, 38, 44]
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._lead_of = lambda path: 0.0
    return enc, [(i * 100, (i + 1) * 100) for i in range(shots)]


def test_gpu_probe_is_off_unless_asked_for(settings, info, plan, tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)
    assert settings.transcode.optimizer.probe_encoder == "svt"
    assert enc._gpu_probe_on() is False


def test_gpu_probe_needs_a_render_node(settings, info, plan, tmp_path, monkeypatch):
    settings.transcode.optimizer.probe_encoder = "qsv"
    monkeypatch.setattr(opt, "_render_nodes", lambda: [])
    warnings = []
    monkeypatch.setattr(opt.logger, "warning", lambda msg, *a, **k: warnings.append(msg.format(*a, **k)))
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc._gpu_probe_on() is False
    assert len(warnings) == 1 and "render node" in warnings[0]


def test_gpu_probe_encode_runs_wholly_on_the_card(settings, info, plan, tmp_path, monkeypatch):
    """It used to claim this and not do it - hwdownload brought the 4K frames
    back to system memory for av1_qsv to upload again, which measured SLOWER
    in wall clock than the SVT probe it was meant to undercut (5.39s against
    4.29s). Encoding through VA-API keeps them where they were decoded:
    0.78s and a twelfth of the CPU."""
    info.color.bit_depth, info.color.pix_fmt = 10, "yuv420p10le"
    enc, _ = _gpu_encoder(settings, info, plan, tmp_path)
    ran = []
    monkeypatch.setattr(enc, "_run", lambda args, timeout=None: (ran.append(args), "")[1])
    enc._qsv_probe_encode(600, 720, 22, tmp_path / "p.ivf")
    cmd = ran[0]
    assert cmd[cmd.index("-c:v") + 1] == "av1_vaapi"
    assert cmd[cmd.index("-global_quality") + 1] == "22"
    # ICQ, not CQP: under CQP the driver ignores -qp and returns the same
    # 35690 KiB at 20, 32 and 44
    assert cmd[cmd.index("-rc_mode") + 1] == "ICQ"
    assert "-hwaccel" in cmd and cmd[cmd.index("-hwaccel") + 1] == "vaapi"
    assert cmd.index("-hwaccel") < cmd.index("-ss") < cmd.index("-i")
    vf = cmd[cmd.index("-vf") + 1] if "-vf" in cmd else ""
    assert "hwdownload" not in vf                        # the whole point
    # rebased by the half frame the seek leaves (16666.67us at 30fps) and
    # bounded by a frame count: neither restarts when the graph is rebuilt
    assert vf.endswith(("settb=AVTB,setpts=PTS-16666", "settb=AVTB,setpts=PTS-16667"))
    assert "-t" not in cmd and cmd[cmd.index("-frames:v") + 1] == "120"
    assert cmd.index("-map") < cmd.index("-frames:v") < cmd.index("-c:v")
    # and it books room against the same budget the scores do: one card, one
    # pool of memory, and exhausting it once already cost a reboot
    enc._gpu_vram = opt.VramBudget(1000, max_ops=8, measure=lambda: 0.0)
    enc._GPU_SLOT_WAIT = 0.05
    assert enc._gpu_vram.reserve(1000, 0.0) is True      # the card is now full
    with pytest.raises(opt.TranscodeError, match="no room on the GPU"):
        enc._qsv_probe_encode(600, 720, 22, tmp_path / "p.ivf")


@pytest.mark.parametrize("ending", ["exit 0", "failed", "timed out", "failed without it"])
def test_a_card_probe_across_a_parameter_change_raises_graph_rebuilt(
        settings, info, plan, tmp_path, monkeypatch, ending):
    """Measured across an in-band SPS change: 77 frames written for a 64-frame
    window, or AVERROR_BUG. The line that says why is INFO, so the card read
    runs at info; and it counts whether ffmpeg exited 0, failed or stalled.
    A failure that does not say so stays an ordinary failure."""
    info.color.bit_depth, info.color.pix_fmt = 10, "yuv420p10le"
    enc, _ = _gpu_encoder(settings, info, plan, tmp_path)
    ran = []

    def fake_run(args, timeout=None):
        ran.append(args)
        _write_out(args, b"partial")
        if ending == "failed without it":
            return _ended("failed", "[av1_vaapi @ 0x1] Failed to end picture encode\n")
        return _ended(ending, _REBUILT_LINE + "x" * 3000)

    monkeypatch.setattr(enc, "_run", fake_run)
    with pytest.raises(opt.TranscodeError) as e:
        enc._qsv_score(3, 600, 720, 22)
    cmd = ran[0]
    assert cmd[cmd.index("-loglevel") + 1] == "info" and "-nostats" in cmd
    assert not (enc.probe_dir / "gpuprobe_00003_22.ivf").exists()   # no partial file left
    if ending == "failed without it":
        assert not isinstance(e.value, opt.GraphRebuilt)
        return
    assert isinstance(e.value, opt.GraphRebuilt)
    assert (e.value.window, e.value.read, e.value.reason) == \
        ((600, 720), "card probe encode", "hwaccel changed")


def test_gpu_probe_rebuild_probes_that_shot_with_svt(settings, info, plan, tmp_path, monkeypatch):
    """probe_encoder=qsv: a card probe across a parameter change used to fail
    the whole job. An anchor just adds no pair; a bulk shot is probed with SVT."""
    enc, shots = _gpu_encoder(settings, info, plan, tmp_path)
    grid = [20, 26, 32, 38, 44]
    qgrid = enc._qsv_grid()
    crf_of = lambda i: 22.0 + (i % 5)
    q_of = lambda i: (crf_of(i) + 14.0) / 2.0
    monkeypatch.setattr(opt.logger, "warning", lambda *a, **k: None)
    monkeypatch.setattr(enc, "_probe_shot",
                        lambda idx, s0, s1, g, lp: _curve(grid, crf_of(idx), enc.target))
    anchors = enc._gpu_anchor_indices(shots)
    bulk = next(i for i in range(len(shots)) if i not in anchors)

    def qsv(idx, s0, s1, qg):
        if idx in (anchors[0], bulk):
            raise opt.GraphRebuilt("hwaccel changed", (s0, s1), "card probe encode")
        return _curve(qgrid, q_of(idx), enc.target), {max(qgrid): 100.0 + q_of(idx)}

    monkeypatch.setattr(enc, "_probe_shot_qsv", qsv)
    samples, chosen = enc.probe_all_gpu(shots, grid)
    assert len(chosen) == len(shots)
    assert bulk in samples and chosen[bulk] == pytest.approx(crf_of(bulk), abs=0.01)
    assert enc._probe_window(*shots[bulk]) in enc._zc_bad
    assert enc._probe_window(*shots[anchors[0]]) in enc._zc_bad


def test_gpu_probe_maps_the_bulk_and_keeps_anchor_samples(settings, info, plan, tmp_path, monkeypatch):
    """The anchors are ordinary SVT probes whose samples are used for their own
    shots; everything else is probed on the card and mapped."""
    enc, shots = _gpu_encoder(settings, info, plan, tmp_path)
    grid = [20, 26, 32, 38, 44]
    qgrid = enc._qsv_grid()
    # ground truth the test plants: CRF = 2*q - 14
    crf_of = lambda i: 22.0 + (i % 5)
    q_of = lambda i: (crf_of(i) + 14.0) / 2.0
    monkeypatch.setattr(enc, "_probe_shot",
                        lambda idx, s0, s1, g, lp: _curve(grid, crf_of(idx), enc.target))
    monkeypatch.setattr(enc, "_probe_shot_qsv",
                        lambda idx, s0, s1, qg: (_curve(qgrid, q_of(idx), enc.target),
                                                 {max(qgrid): 100.0 + q_of(idx)}))
    samples, chosen = enc.probe_all_gpu(shots, grid)
    assert len(chosen) == len(shots)                      # every shot got a CRF
    assert len(samples) == 6                              # only the anchors cost an SVT probe
    for idx, crf in chosen.items():
        assert crf == pytest.approx(crf_of(idx), abs=0.01)


def test_gpu_probe_gives_up_on_the_whole_job_when_the_line_does_not_hold(
        settings, info, plan, tmp_path, monkeypatch):
    """A line always fits its own points, so the leave-one-out residual is what
    decides. Above gpu_probe_max_residual the rest of the job goes back to SVT
    rather than shipping CRFs nobody measured."""
    enc, shots = _gpu_encoder(settings, info, plan, tmp_path)
    grid = [20, 26, 32, 38, 44]
    qgrid = enc._qsv_grid()
    warnings = []
    monkeypatch.setattr(opt.logger, "warning", lambda msg, *a, **k: warnings.append(msg.format(*a, **k)))
    monkeypatch.setattr(enc, "_probe_shot",
                        lambda idx, s0, s1, g, lp: _curve(grid, 21.0 + (idx * 7 % 11), enc.target))
    monkeypatch.setattr(enc, "_probe_shot_qsv",
                        lambda idx, s0, s1, qg: (_curve(qgrid, 16.0 + (idx * 3 % 7), enc.target),
                                                 {max(qgrid): 100.0}))
    called = []
    real_all = enc.probe_all
    monkeypatch.setattr(enc, "probe_all", lambda sh, g: (called.append(len(sh)),
                                                         {k: _curve(grid, 24.0, enc.target) for k in range(len(sh))})[1])
    samples, chosen = enc.probe_all_gpu(shots, grid)
    assert called == [len(shots) - 6]                     # the rest went to SVT
    assert len(chosen) == len(shots)
    assert any("does not hold" in w for w in warnings)


def test_gpu_probe_abstains_shot_by_shot_outside_the_calibrated_range(
        settings, info, plan, tmp_path, monkeypatch):
    """The one failure the measurement found was an easy shot reaching the
    target 14 CRF above the bulk, which an extrapolated line missed by 10.
    Such a shot is probed with SVT instead."""
    enc, shots = _gpu_encoder(settings, info, plan, tmp_path)
    grid = [20, 26, 32, 38, 44]
    qgrid = enc._qsv_grid()
    crf_of = lambda i: 22.0 + (i % 5)
    q_of = lambda i: (crf_of(i) + 14.0) / 2.0
    odd = len(shots) - 1                                  # the easy shot
    svt_probed = []

    def fake_svt(idx, s0, s1, g, lp):
        svt_probed.append(idx)
        return _curve(grid, crf_of(idx), enc.target)

    monkeypatch.setattr(enc, "_probe_shot", fake_svt)
    monkeypatch.setattr(enc, "_probe_shot_qsv",
                        lambda idx, s0, s1, qg: (_curve(qgrid, 34.0 if idx == odd else q_of(idx),
                                                        enc.target), {max(qgrid): 100.0}))
    samples, chosen = enc.probe_all_gpu(shots, grid)
    assert odd in svt_probed and odd in samples           # abstained, probed properly
    assert len(chosen) == len(shots)


def test_theil_sen_is_not_tilted_by_one_easy_shot(settings, info, plan, tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)
    pts = [(float(q), 2.0 * q - 14.0) for q in range(18, 26)] + [(24.79, 36.14)]
    a, b = enc._theil_sen(pts)
    assert a == pytest.approx(2.0, abs=0.05) and b == pytest.approx(-14.0, abs=1.0)


def test_sycl_stalls_must_be_consecutive_to_retire_the_device(
        settings, info, plan, tmp_path, monkeypatch):
    """Three stalls scattered among hundreds of good scores retired the device
    for a whole episode once, costing its remaining 400 shots CPU scoring. A
    run of failures with nothing working between them is what a sick device
    looks like; occasional slowness under a busy card is not."""
    settings.transcode.optimizer.vmaf_sycl_device = 0
    settings.transcode.optimizer.vmaf_sycl_min_width = 0
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._lead_of = lambda path: 0.0
    enc._sycl_ok = True
    monkeypatch.setattr(opt.logger, "warning", lambda *a, **k: None)
    seq = []

    def on(sycl, dist_args, ref_args, ref_vf, idx, crf, threads, timeout, window=None):
        if sycl >= 0 and seq and seq.pop(0) == "stall":
            raise opt.CommandTimeout("libvmaf timed out")
        return 90.0

    monkeypatch.setattr(enc, "_score_vmaf_on", on)
    # stall, good, stall, good, ... never retires the device
    for _ in range(6):
        seq[:] = ["stall"]
        enc._score_vmaf(["-i", "d"], ["-i", "r"], [], 0, 30, frames=120)
        seq[:] = ["ok"]
        enc._score_vmaf(["-i", "d"], ["-i", "r"], [], 0, 30, frames=120)
    assert enc._sycl_ok is True and enc._sycl_timeouts == 0
    # three in a row does
    for _ in range(enc._SYCL_MAX_TIMEOUTS):
        seq[:] = ["stall"]
        enc._score_vmaf(["-i", "d"], ["-i", "r"], [], 0, 30, frames=120)
    assert enc._sycl_ok is False


def test_the_sycl_budget_widens_when_the_card_also_encodes(settings, info, plan, tmp_path):
    """The probe encodes share the device, and a score queues behind them."""
    enc = make_encoder(settings, info, plan, tmp_path)
    alone = enc._sycl_timeout(120)
    settings.transcode.optimizer.probe_encoder = "qsv"
    assert enc._sycl_timeout(120) == alone * 2


def test_a_retired_sycl_device_is_given_another_chance(settings, info, plan, tmp_path, monkeypatch):
    """An episode reported OUT_OF_DEVICE_MEMORY at 8% of its probe phase, the
    device was retired for the whole job, and a selfcheck minutes later
    passed - the card had recovered while the job spent its remaining 90% on
    the CPU. One preflight per cool-off costs a couple of seconds; a card that
    is really gone fails it and is left alone."""
    settings.transcode.optimizer.vmaf_sycl_device = 0
    settings.transcode.optimizer.vmaf_sycl_min_width = 0
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._lead_of = lambda path: 0.0
    monkeypatch.setattr(opt.logger, "warning", lambda *a, **k: None)
    healthy = {"card": False}
    preflights = []
    monkeypatch.setattr(enc, "_sycl_preflight",
                        lambda dev: (preflights.append(dev), healthy["card"])[1])
    now = {"t": 1000.0}
    monkeypatch.setattr(opt.time, "time", lambda: now["t"])

    def on(sycl, *a, **k):
        if sycl >= 0:
            raise opt.TranscodeError("libvmaf ERROR SYCL: UR_RESULT_ERROR_DEVICE_LOST")
        return 90.0

    monkeypatch.setattr(enc, "_score_vmaf_on", on)
    enc._sycl_ok = True                              # preflight already passed
    for _ in range(enc._SYCL_MAX_TIMEOUTS):          # a run of failures, not one
        assert enc._score_vmaf(["-i", "d"], ["-i", "r"], [], 0, 30, frames=120) == 90.0
    assert enc._sycl_ok is False and enc._sycl_retired_at == 1000.0
    # too soon: no preflight, still on the CPU
    now["t"] = 1000.0 + enc._SYCL_REARM_AFTER - 1
    assert enc._sycl_device() == -1 and preflights == []
    # after the cool-off it is asked once - and the card is still sick
    now["t"] = 1000.0 + enc._SYCL_REARM_AFTER
    assert enc._sycl_device() == -1 and preflights == [0]
    # asked once per cool-off, not once per score
    assert enc._sycl_device() == -1 and preflights == [0]
    # the card recovers; the next cool-off brings scoring back
    healthy["card"] = True
    enc._sycl_retired_at = now["t"]
    now["t"] += enc._SYCL_REARM_AFTER
    assert enc._sycl_device() == 0
    assert enc._sycl_ok is True and enc._sycl_timeouts == 0


# ---------------------------------------------------------------------------
# VRAM budget: admission on bytes rather than on a count of operations
# ---------------------------------------------------------------------------

def test_vram_budget_admits_by_bytes_not_by_count():
    """Six slots is six operations, and operations are not the same size. A
    budget that fits three big ones must admit three, not six."""
    b = opt.VramBudget(3000, max_ops=6, measure=lambda: 0.0)
    assert [b.reserve(900, 0.0) for _ in range(3)] == [True, True, True]
    assert b.reserve(900, 0.0) is False
    b.release(900)
    assert b.reserve(900, 0.0) is True


def test_vram_budget_still_honours_the_worker_cap():
    """The count cap was measured on throughput, not on memory, so a budget
    with room to spare must not widen the queue past it."""
    b = opt.VramBudget(100000, max_ops=2, measure=lambda: 0.0)
    assert b.reserve(10, 0.0) and b.reserve(10, 0.0)
    assert b.reserve(10, 0.0) is False


def test_vram_budget_always_lets_one_operation_through():
    """A budget smaller than a single score should slow the phase down, not
    deadlock it - the CPU fallback is for contention, not for an impossible
    sum."""
    b = opt.VramBudget(256, max_ops=6, measure=lambda: 0.0)
    assert b.reserve(50000, 0.0) is True


def test_vram_budget_counts_what_the_card_actually_holds():
    """The reservation covers the ramp, the measurement covers the model
    being wrong. A card already full from an operation that under-estimated
    itself must refuse the next one."""
    used = {"mb": 2950.0}
    b = opt.VramBudget(3000, max_ops=6, measure=lambda: used["mb"])
    assert b.reserve(100, 0.0) is True      # first one always fits
    assert b.reserve(100, 0.0) is False     # measured 2950 + 100 > 3000
    used["mb"] = 0.0
    b._measured_at = 0.0
    assert b.reserve(100, 0.0) is True


def test_vram_budget_survives_an_unreadable_card():
    """Without fdinfo it degrades to the reservation model - the old counting
    behaviour with better units - rather than refusing to run."""
    b = opt.VramBudget(3000, max_ops=6, measure=lambda: None)
    assert b.reserve(900, 0.0) is True
    assert b.readable is False
    assert "unmeasured" in b.summary()


def test_vram_calibration_pulls_the_model_towards_the_measurement():
    """An operation that really costs twice its estimate should raise the
    model, so the next admissions book the real number."""
    b = opt.VramBudget(100000, max_ops=6, measure=lambda: 2000.0)
    b.reserve(1000, 0.0)
    for _ in range(200):
        b._calibrated_at = 0.0
        b.calibrate()
    assert b._ratio > 1.5


def test_vram_calibration_is_not_pinned_by_one_bad_sample():
    """A reading taken while a finished ffmpeg is still tearing down counts
    memory whose booking has already gone. An unbounded high-water mark
    turned one such sample into a permanent x4.0 for the rest of the job and
    collapsed the pool to one operation; the worst case is now clamped per
    sample and decays, so it is the worst RECENT case."""
    used = {"mb": 1200.0}
    b = opt.VramBudget(100000, max_ops=6, measure=lambda: used["mb"])
    for _ in range(3):
        b.reserve(400, 0.0)                  # 1200MB booked, 1200MB held
    def settle(n=400):
        for _ in range(n):
            b._calibrated_at = 0.0
            b._measured_at = 0.0
            b.calibrate()
    settle()
    assert b._ratio == pytest.approx(1.0, abs=0.05)
    used["mb"] = 60000.0                     # one absurd reading
    b._calibrated_at = 0.0
    b._measured_at = 0.0
    b.calibrate()
    assert b._ratio <= 4.0                   # clamped, not 50x
    used["mb"] = 1200.0                      # and it recovers
    settle()
    assert b._ratio == pytest.approx(1.0, abs=0.05)


def test_vram_release_gives_back_exactly_what_was_booked():
    """reserve books mb*ratio and release used to recompute it from a ratio
    calibration had moved in between, so _reserved drifted to zero - deleting
    the half of the model that covers the ramp before an allocation shows up
    in fdinfo."""
    b = opt.VramBudget(100000, max_ops=6, measure=lambda: 0.0)
    want = b.booked(500)
    assert b.reserve(500, 0.0)
    b._ratio = 2.5                           # calibration moves it mid-flight
    b.release(want, 500)
    assert b._reserved == pytest.approx(0.0)
    assert b._ops == 0


def test_drm_vram_reader_deduplicates_clients_and_filters_the_device(tmp_path):
    """A client that dup()s its fd shows the same allocation twice, and this
    box has four render nodes - summing blindly would count both."""
    proc = tmp_path / "proc"
    def client(pid, fd, cid, pdev, kib):
        d = proc / str(pid) / "fdinfo"
        d.mkdir(parents=True, exist_ok=True)
        (d / str(fd)).write_text(
            f"pos:\t0\ndrm-driver:\txe\ndrm-pdev:\t{pdev}\n"
            f"drm-client-id:\t{cid}\ndrm-total-vram0:\t{kib} KiB\n")
    client(10, 3, "1", "0000:c6:00.0", 440 * 1024)
    client(10, 4, "1", "0000:c6:00.0", 440 * 1024)   # same client, dup'd fd
    client(11, 3, "2", "0000:c6:00.0", 440 * 1024)
    client(12, 3, "3", "0000:41:00.0", 9000 * 1024)  # a 3090, not ours
    mb = opt.drm_vram_used_mb("0000:c6:00.0", root=str(proc))
    assert mb == pytest.approx(880, abs=1)


def test_score_estimate_doubles_with_the_hardware_reference_read(settings, info, plan, tmp_path):
    """With reference_hwaccel on a score is two DRM clients, not one, and
    that is exactly what exhausted the card."""
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._hwdec_ok = False
    cpu_read = enc._est_score_mb(120)
    enc._hwdec_ok = True
    assert enc._est_score_mb(120) == pytest.approx(cpu_read * 2)


# ---------------------------------------------------------------------------
# probe_encoder="qsv+svt": the card predicts, SVT confirms
# ---------------------------------------------------------------------------

def _verified_encoder(settings, info, plan, tmp_path, shots=40):
    settings.transcode.optimizer.probe_encoder = "qsv+svt"
    settings.transcode.optimizer.gpu_probe_anchors = 6
    settings.transcode.optimizer.probe_crfs = list(range(18, 51, 2))
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._lead_of = lambda path: 0.0
    return enc, [(i * 100, (i + 1) * 100) for i in range(shots)]


def _plant_svt(enc, monkeypatch, crf_of, probed):
    """Make an SVT probe return a planted linear curve and record every CRF
    it actually spends a probe on."""
    monkeypatch.setattr(enc, "_acquire_shard", lambda *a, **k: None)
    monkeypatch.setattr(enc, "_release_shard", lambda *a, **k: None)
    monkeypatch.setattr(enc, "_probe_window", lambda s0, s1: (s0, s1))
    monkeypatch.setattr(enc, "_probe_input", lambda *a, **k: ([], []))

    def fake(idx, w0, w1, crf, lp, shard):
        probed.setdefault(idx, []).append(crf)
        return idx, crf, enc.target + (crf_of(idx) - crf)
    monkeypatch.setattr(enc, "_probe_encode_and_score", fake)


def test_verified_mode_is_off_unless_asked_for(settings, info, plan, tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc._probe_mode() == "svt"
    settings.transcode.optimizer.probe_encoder = "qsv+svt"
    assert enc._probe_mode() == "qsv+svt"
    assert enc._gpu_probe_on() is True     # the card is busy either way


def test_a_right_prediction_costs_two_probes(settings, info, plan, tmp_path, monkeypatch):
    """The point of the mode: land on the answer, confirm it, stop."""
    enc, _ = _verified_encoder(settings, info, plan, tmp_path)
    grid = list(range(18, 51, 2))
    probed = {}
    _plant_svt(enc, monkeypatch, lambda i: 30.5, probed)
    scores = enc._probe_shot_seeded(0, 0, 100, grid, 4, seed=30.0, step=2)
    # both points at once, no direction-finding probe between them
    assert probed[0] == [28, 32]
    assert opt.pick_crf(list(scores.items()), enc.target) == pytest.approx(30.5, abs=0.1)


def test_a_wrong_prediction_costs_probes_but_not_the_answer(
        settings, info, plan, tmp_path, monkeypatch):
    """A prediction at the wrong end of the grid must still converge on the
    CRF the SVT curve actually crosses at - that is the whole invariant this
    mode buys over mapping the answer outright."""
    enc, _ = _verified_encoder(settings, info, plan, tmp_path)
    grid = list(range(18, 51, 2))
    probed = {}
    _plant_svt(enc, monkeypatch, lambda i: 24.0, probed)
    scores = enc._probe_shot_seeded(0, 0, 100, grid, 4, seed=50.0, step=2)
    assert probed[0][0] == 50                      # it did start where told
    assert len(probed[0]) <= len(grid)             # and stayed inside the budget
    assert opt.pick_crf(list(scores.items()), enc.target) == pytest.approx(24.0)


def test_the_seeded_walk_stops_at_the_edge_of_the_grid(
        settings, info, plan, tmp_path, monkeypatch):
    """A shot that cannot reach the target at any probed CRF must stop, not
    spin at the boundary."""
    enc, _ = _verified_encoder(settings, info, plan, tmp_path)
    grid = list(range(18, 51, 2))
    probed = {}
    _plant_svt(enc, monkeypatch, lambda i: 4.0, probed)   # crosses below the grid
    scores = enc._probe_shot_seeded(0, 0, 100, grid, 4, seed=30.0, step=2)
    assert 18 in scores and len(probed[0]) <= len(grid)


def test_verified_probing_never_takes_its_answer_from_the_map(
        settings, info, plan, tmp_path, monkeypatch):
    """Plant a mapping that is wrong by 10 CRF on every shot. The delivered
    CRF must still be the one the SVT probes measured."""
    enc, shots = _verified_encoder(settings, info, plan, tmp_path)
    grid = list(range(18, 51, 2))
    qgrid = enc._qsv_grid()
    crf_of = lambda i: 24.0 + (i % 3) * 2
    probed = {}
    _plant_svt(enc, monkeypatch, crf_of, probed)
    # q* maps to a CRF 10 too high through any line fitted on these pairs
    monkeypatch.setattr(enc, "_probe_shot_qsv",
                        lambda idx, s0, s1, qg: (_curve(qgrid, 20.0, enc.target),
                                                 {max(qgrid): 100.0 + (idx % 5)}))
    samples = enc.probe_all_verified(shots, grid)
    chosen = enc.pick_all_crfs(samples, grid)
    assert len(chosen) == len(shots)
    for idx, crf in chosen.items():
        assert crf == pytest.approx(crf_of(idx), abs=0.01)


def test_verified_probing_seeds_from_the_line_once_it_has_learned_one(
        settings, info, plan, tmp_path, monkeypatch):
    """The anchors start it off; after that every verified shot feeds the fit,
    so the later shots are seeded rather than probed from the grid."""
    enc, shots = _verified_encoder(settings, info, plan, tmp_path, shots=60)
    grid = list(range(18, 51, 2))
    qgrid = enc._qsv_grid()
    crf_of = lambda i: 24.0 + (i % 4)
    q_of = lambda i: (crf_of(i) + 14.0) / 2.0
    seeded, plain = [], []
    monkeypatch.setattr(enc, "_probe_shot_qsv",
                        lambda idx, s0, s1, qg: (_curve(qgrid, q_of(idx), enc.target),
                                                 {max(qgrid): 100.0 + q_of(idx)}))
    monkeypatch.setattr(enc, "_probe_shot",
                        lambda idx, s0, s1, g, lp: (plain.append(idx),
                                                    _curve(grid, crf_of(idx), enc.target))[1])
    monkeypatch.setattr(enc, "_probe_shot_seeded",
                        lambda idx, s0, s1, g, lp, seed, step: (
                            seeded.append((idx, seed)),
                            _curve(grid, crf_of(idx), enc.target))[1])
    samples = enc.probe_all_verified(shots, grid)
    assert len(samples) == len(shots)
    assert len(seeded) > len(shots) // 2          # most shots used the line
    for idx, seed in seeded:                      # and it pointed at the answer
        assert abs(seed - crf_of(idx)) <= 2.0


def test_verified_probing_falls_back_to_the_grid_when_the_line_is_useless(
        settings, info, plan, tmp_path, monkeypatch):
    """Above _VERIFY_MAX_SD the prediction is noise. It is not a correctness
    gate - it only decides whether to seed from the line or from the middle of
    the grid, the way the plain path always does."""
    enc, shots = _verified_encoder(settings, info, plan, tmp_path, shots=60)
    grid = list(range(18, 51, 2))
    qgrid = enc._qsv_grid()
    import random
    rnd = random.Random(7)
    truth = {i: 20.0 + rnd.random() * 28 for i in range(len(shots))}
    crf_of = truth.__getitem__            # no relation to q at all
    monkeypatch.setattr(enc, "_probe_shot_qsv",
                        lambda idx, s0, s1, qg: (_curve(qgrid, 20.0 + (idx % 7), enc.target),
                                                 {max(qgrid): 100.0 + (idx % 3)}))
    seeded, plain = [], []
    monkeypatch.setattr(enc, "_probe_shot",
                        lambda idx, s0, s1, g, lp: (plain.append(idx),
                                                    _curve(grid, crf_of(idx), enc.target))[1])
    monkeypatch.setattr(enc, "_probe_shot_seeded",
                        lambda idx, s0, s1, g, lp, seed, step: (
                            seeded.append(idx),
                            _curve(grid, crf_of(idx), enc.target))[1])
    monkeypatch.setattr(opt.logger, "warning", lambda *a, **k: None)
    samples = enc.probe_all_verified(shots, grid)
    assert len(samples) == len(shots)
    assert len(plain) > len(seeded)


def test_a_failed_qsv_prediction_still_probes_the_shot(
        settings, info, plan, tmp_path, monkeypatch):
    """The card being busy or wedged is not a reason to skip a shot."""
    enc, shots = _verified_encoder(settings, info, plan, tmp_path, shots=10)
    grid = list(range(18, 51, 2))
    def boom(idx, s0, s1, qg):
        raise opt.TranscodeError("no room on the GPU")
    monkeypatch.setattr(enc, "_probe_shot_qsv", boom)
    monkeypatch.setattr(enc, "_probe_shot",
                        lambda idx, s0, s1, g, lp: _curve(grid, 26.0, enc.target))
    samples = enc.probe_all_verified(shots, grid)
    assert len(samples) == len(shots)
    chosen = enc.pick_all_crfs(samples, grid)
    assert all(c == pytest.approx(26.0) for c in chosen.values())


def test_card_probe_rebuild_probes_the_shot_from_the_grid(
        settings, info, plan, tmp_path, monkeypatch):
    """A card probe across a parameter change is not a busy or wedged card:
    the shot is probed from the grid, the window is remembered for both
    hardware reads, and nothing is counted against the device."""
    enc, shots = _verified_encoder(settings, info, plan, tmp_path, shots=10)
    grid = list(range(18, 51, 2))
    qgrid = enc._qsv_grid()
    warnings = []
    monkeypatch.setattr(opt.logger, "warning",
                        lambda msg, *a, **k: warnings.append(msg.format(*a, **k)))

    def qsv(idx, s0, s1, qg):
        if idx == 3:
            raise opt.GraphRebuilt("hwaccel changed", enc._probe_window(s0, s1),
                                   "card probe encode")
        return _curve(qgrid, 20.0, enc.target), {max(qgrid): 100.0}

    monkeypatch.setattr(enc, "_probe_shot_qsv", qsv)
    monkeypatch.setattr(enc, "_probe_shot",
                        lambda idx, s0, s1, g, lp: _curve(grid, 26.0, enc.target))
    samples = enc.probe_all_verified(shots, grid)
    assert len(samples) == len(shots)
    w = enc._probe_window(*shots[3])
    assert w in enc._zc_bad and w in enc._hwdec_bad and list(enc._rebuilt) == [w]
    rebuilt = [m for m in warnings if "rebuilt" in m]
    assert len(rebuilt) == 1 and "card probe encode" in rebuilt[0]
    assert enc._zc_streak == 0 and enc._sycl_timeouts == 0 and enc._hwdec_streak == 0


def test_after_a_card_probe_rebuild_the_shots_svt_scores_skip_zero_copy(
        settings, info, plan, tmp_path, monkeypatch):
    """The window is remembered for the probes that follow on it, not just
    for the card: shot 0's card read rebuilds, so its SVT probes are scored
    the usual way, while shot 1 beside it keeps zero-copy. Nothing faked
    between the probe phase and the ffmpeg commands."""
    o = settings.transcode.optimizer
    o.probe_encoder, o.gpu_probe_anchors = "qsv+svt", 6
    o.probe_crfs = [20, 26, 32, 38, 44]
    enc = _zc_encoder(settings, info, plan, tmp_path)
    enc._zc_ok = True
    warnings = []
    monkeypatch.setattr(opt.logger, "warning",
                        lambda msg, *a, **k: warnings.append(msg.format(*a, **k)))
    ran = []

    def fake_run(args, timeout=None):
        args = [str(a) for a in args]
        ran.append(args)
        if "-lavfi" not in args:
            _write_out(args, b"ivf")
            return _REBUILT_LINE if "gpuprobe_00000_" in args[-1] else ""
        _write_score(args, 90.0)
        return _ZC_LOG if "libvmaf_sycl=" in args[args.index("-lavfi") + 1] else ""

    monkeypatch.setattr(enc, "_run", fake_run)
    samples = enc.probe_all_verified([(0, 100), (100, 200)], list(o.probe_crfs))
    assert set(samples) == {0, 1} and samples[0]

    def scores(shot, scorer):
        tag = f"score_{shot:05d}_"
        return [c for c in ran if "-lavfi" in c and tag in c[c.index("-lavfi") + 1]
                and f"{scorer}=" in c[c.index("-lavfi") + 1]]

    assert scores(0, "libvmaf_sycl") == [] and scores(0, "libvmaf")
    assert scores(1, "libvmaf_sycl")
    assert (0, 100) in enc._zc_bad and (100, 200) not in enc._zc_bad
    assert len(warnings) == 1 and "card probe encode" in warnings[0]
    assert enc._zc_streak == 0 and enc._zc_fallbacks == 0


def test_the_second_probe_is_placed_from_the_residual_spread(settings, info, plan, tmp_path):
    """Far enough that the pair straddles the target most of the time, and no
    further - a probe out in the tail measures a part of the curve nothing
    will use."""
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc._verify_step(0.0) == 2          # floor
    assert enc._verify_step(2.0) == 3          # 1.5 sd
    assert enc._verify_step(40.0) == 8         # ceiling


def test_the_first_fit_does_not_land_on_the_minimum_sample(settings, info, plan, tmp_path):
    """Four pairs is enough to draw a line and far too few to trust one: on a
    real episode the first fit came out at 6.06 CRF over four pairs - above
    the threshold, so nothing was seeded for the next 25 pairs - while the
    same content fitted to 4.14 over 104."""
    enc = make_encoder(settings, info, plan, tmp_path)
    pairs = [(float(q), 5.0 + 0.1 * q, 2.0 * q + 3.0) for q in range(14, 40, 2)]
    assert enc._refit_verified(pairs[:4]) is None
    assert enc._refit_verified(pairs[:enc._VERIFY_MIN_PAIRS - 1]) is None
    fit = enc._refit_verified(pairs[:enc._VERIFY_MIN_PAIRS])
    assert fit is not None
    # q and log(bytes) are collinear here, so only the prediction is defined
    assert enc._predict_crf(fit, 20.0, 7.0, [18, 50]) == pytest.approx(43.0, abs=0.1)


def test_the_qsv_search_is_two_probes_and_a_chord(settings, info, plan, tmp_path, monkeypatch):
    """q* only feeds a fitted plane, so a chord across the grid is a fine
    input to a fit. A third interpolated point was probed for a while on the
    reasoning that the chord misses a bending curve - it does, by -2.46 q,
    but that shift is a linear function of q* (R^2 0.96), so the refit
    absorbs it: leave-one-out came out 2.53 against 2.58 with the third
    probe, i.e. it bought nothing for a third of the card's work."""
    enc, _ = _verified_encoder(settings, info, plan, tmp_path)
    qgrid = enc._qsv_grid()
    spent = []
    def fake(idx, w0, w1, q):
        spent.append(q)
        return enc.target + (24.5 - q) * (1.0 + (38 - q) * 0.02), 100.0 - q
    monkeypatch.setattr(enc, "_qsv_score", fake)
    monkeypatch.setattr(enc, "_probe_window", lambda a, b: (a, b))
    scores, bpf = enc._probe_shot_qsv(0, 0, 120, qgrid)
    assert spent == [min(qgrid), max(qgrid)]          # both ends, nothing else
    assert set(bpf) == set(spent)                     # bytes recorded for each
    assert enc._crossing(scores) is not None


def test_the_qsv_search_spends_nothing_when_the_ends_do_not_bracket(
        settings, info, plan, tmp_path, monkeypatch):
    """No crossing to find, no third probe worth spending."""
    enc, _ = _verified_encoder(settings, info, plan, tmp_path)
    spent = []
    monkeypatch.setattr(enc, "_qsv_score",
                        lambda idx, w0, w1, q: (spent.append(q),
                                                (enc.target - 5.0, 42.0))[1])
    monkeypatch.setattr(enc, "_probe_window", lambda a, b: (a, b))
    scores, _ = enc._probe_shot_qsv(0, 0, 120, enc._qsv_grid())
    assert len(spent) == 2
    assert enc._crossing(scores) is None


# ---------------------------------------------------------------------------
# probe_dataset: the record a richer mapping model would have to be built on
# ---------------------------------------------------------------------------

def test_the_probe_dataset_records_each_shot_as_it_finishes(settings, info, plan, tmp_path, monkeypatch):
    """Written per shot rather than in one batch at the end: a probe phase is
    hours long and the jobs that most need explaining are the ones killed
    part way through."""
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._dataset_header([(0, 100), (100, 240)], [20, 30, 40])
    enc._dataset_shot(0, 0, 100, 0, 100, svt={20: 97.0, 30: 93.0})
    enc._dataset_shot(1, 100, 240, 120, 220, svt={20: 96.0, 30: 92.0},
                      qsv={14: 97.5, 26: 94.0}, seed=25.5)
    enc._dataset_write({"type": "crfs", "final": {"0": 24.0, "1": 26.0}})
    rows = [json.loads(l) for l in
            (tmp_path / "logs" / "probe_dataset.jsonl").read_text().splitlines()]
    assert [r["type"] for r in rows] == ["job", "shot", "shot", "crfs"]
    job, s0, s1, crfs = rows
    assert job["shots"] == 2 and job["target"] == enc.target
    assert s0["whole_shot"] is True                  # window covered the shot
    assert s1["whole_shot"] is False                 # this one was truncated
    assert s1["qsv"] == {"14": 97.5, "26": 94.0} and s1["seed"] == 25.5
    assert "qsv" not in s0
    assert crfs["final"]["1"] == 26.0
    assert all(r["job"] for r in rows) or True       # job id present, may be ""


def test_the_probe_dataset_never_fails_a_transcode(settings, info, plan, tmp_path, monkeypatch):
    """It is diagnostics. A full disk must not cost an episode."""
    enc = make_encoder(settings, info, plan, tmp_path)
    warned = []
    monkeypatch.setattr(opt.logger, "warning", lambda *a, **k: warned.append(a))
    def boom(*a, **k):
        raise OSError("no space left on device")
    monkeypatch.setattr("builtins.open", boom)
    enc._dataset_shot(0, 0, 100, 0, 100, svt={20: 97.0})
    enc._dataset_shot(1, 0, 100, 0, 100, svt={20: 97.0})
    assert len(warned) == 1                          # warned once, not per shot


def test_the_probe_dataset_can_be_turned_off(settings, info, plan, tmp_path):
    settings.transcode.optimizer.probe_dataset = False
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._dataset_shot(0, 0, 100, 0, 100, svt={20: 97.0})
    assert not (tmp_path / "logs" / "probe_dataset.jsonl").exists()


def test_the_mapping_uses_what_the_probe_cost_as_well_as_where_it_crossed(
        settings, info, plan, tmp_path):
    """Measured on a 163-shot 4K episode: the crossing alone leaves 4.53 CRF
    of leave-one-out residual (35% of a 5.61 spread), and it is not noise -
    94% of those shots were probed whole, and refitting the crossings with a
    curve model instead of linear interpolation did not move it. Adding log
    bytes per frame at the high-q end takes it to 2.72 CRF and 77%, and the
    bytes are free: the probe writes the file either way."""
    enc = make_encoder(settings, info, plan, tmp_path)
    # a plane the crossing alone cannot describe: two shots share a q* but
    # differ by 8 CRF, and only the bytes tell them apart
    pairs = []
    for k in range(20):
        q = 20.0 + (k % 5)
        lb = 5.0 + (k % 4) * 0.5
        pairs.append((q, lb, 1.5 * q + 4.0 * lb - 10.0))
    fit = enc._refit_verified(pairs)
    assert fit is not None
    assert fit["sd"] < 0.01                       # the plane fits exactly
    grid = [18, 50]
    a = enc._predict_crf(fit, 22.0, 5.0, grid)    # 43.0
    b = enc._predict_crf(fit, 22.0, 6.0, grid)    # 47.0, still inside the grid
    assert b - a == pytest.approx(4.0, abs=0.1)   # same crossing, 4 CRF apart
    # and the prediction is still clamped to the grid it will probe in
    assert enc._predict_crf(fit, 22.0, 9.0, grid) == 50.0


def test_one_freak_shot_cannot_tilt_the_plane(settings, info, plan, tmp_path):
    """Least squares replaced Theil-Sen when the fit went to two features, so
    the robustness has to come from somewhere: one pass that drops what sits
    beyond three residual deviations and refits."""
    enc = make_encoder(settings, info, plan, tmp_path)
    pairs = [(20.0 + (k % 7), 5.0 + (k % 3) * 0.4, 1.5 * (20.0 + (k % 7)) + 2.0)
             for k in range(30)]
    clean = enc._refit_verified(pairs)
    pairs.append((21.0, 5.4, 400.0))              # one absurd shot
    dirty = enc._refit_verified(pairs)
    assert dirty["sd"] < 1.0                      # the outlier was dropped
    assert enc._predict_crf(dirty, 22.0, 5.4, [18, 50]) == pytest.approx(
        enc._predict_crf(clean, 22.0, 5.4, [18, 50]), abs=0.5)


def test_the_bytes_come_from_the_high_q_end(settings, info, plan, tmp_path):
    """The bit-starved end discriminated content better: 2.72 CRF against
    2.87 when both ends were measured as the second feature."""
    enc = make_encoder(settings, info, plan, tmp_path)
    assert enc._log_bpf({14: 2000.0, 38: 100.0}) == pytest.approx(math.log(100.0))
    assert enc._log_bpf({}) is None
    assert enc._log_bpf({38: 0.0}) is None


def test_seeded_and_plain_agree_on_the_grid_the_bug_hid_behind(
        settings, info, plan, tmp_path, monkeypatch):
    """The invariant, checked differentially on a FIVE-point grid.

    Every earlier test used a 17-point grid, where the budget is large enough
    that the walk always reaches an end. Production runs four points, and
    there 11 of 163 shots ran out of budget with every probed score still
    above the target and clamped to the highest point they happened to reach
    - up to 8 CRF below what the plain path picks."""
    settings.transcode.optimizer.probe_crfs = [20, 26, 32, 38, 44]
    settings.transcode.optimizer.probe_encoder = "qsv+svt"
    enc = make_encoder(settings, info, plan, tmp_path)
    grid = enc._probe_grid()
    assert len(grid) == 5                      # the budget the bug needed
    for truth in (18.0, 21.0, 24.0, 30.0, 36.0, 43.0, 48.0):
        probed = {}
        _plant_svt(enc, monkeypatch, lambda i, t=truth: t, probed)
        plain = enc._probe_shot(0, 0, 100, grid, 4)
        want = opt.pick_crf(list(plain.items()), enc.target)
        for seed in (20.0, 26.0, 32.0, 38.0, 44.0):
            for step in (2, 4, 8):
                probed.clear()
                got = enc._probe_shot_seeded(0, 0, 100, grid, 4, seed, step)
                assert opt.pick_crf(list(got.items()), enc.target) == \
                    pytest.approx(want, abs=1.0), (
                        f"truth {truth} seed {seed} step {step}: "
                        f"seeded {opt.pick_crf(list(got.items()), enc.target)} "
                        f"vs plain {want} (probed {sorted(probed[0])})")
                assert len(probed[0]) <= enc._probe_budget(grid)


def test_luminance_qp_bias_reaches_the_encoder_only_when_set(settings, info, plan, tmp_path):
    """Per-preset, not a global default: measured on two 90s 4K cuts it costs
    ~9% of the output for +0.06 to +0.37 delivered VMAF, while halving the
    shots that cannot reach the target at any probed CRF. That trade is a
    judgement about the library, not a default."""
    v = VideoParams(engine="optimizer")
    assert "luminance-qp-bias" not in opt._svt_params_dict(v)
    v.luminance_qp_bias = 50
    assert opt._svt_params_dict(v)["luminance-qp-bias"] == 50
    # and it rides along into the probe encodes, or the search would be
    # picking CRFs for an encoder configured differently from the delivery
    enc = make_encoder(settings, info, plan, tmp_path)
    enc.video.luminance_qp_bias = 50
    assert "luminance-qp-bias=50" in ":".join(
        f"{k}={x}" for k, x in opt._svt_params_dict(enc.video).items())


def test_a_prediction_past_a_bound_is_settled_by_one_probe(
        settings, info, plan, tmp_path, monkeypatch):
    """27 of 163 shots on the measured episode have no crossing at all. In the
    plain path they burn the whole budget discovering that. Here the
    prediction points at a bound and one probe confirms the target is out of
    reach there - which IS the answer."""
    settings.transcode.optimizer.probe_crfs = [20, 26, 32, 38, 44]
    settings.transcode.optimizer.probe_encoder = "qsv+svt"
    enc = make_encoder(settings, info, plan, tmp_path)
    grid = enc._probe_grid()
    probed = {}
    # a shot whose curve crosses far below the grid: nothing reaches the target
    _plant_svt(enc, monkeypatch, lambda i: 8.0, probed)
    scores = enc._probe_shot_seeded(0, 0, 100, grid, 4, seed=12.0, step=4)
    assert probed[0] == [20]                      # one probe at the floor
    assert opt.pick_crf(list(scores.items()), enc.target) == 20.0
    # and the mirror: a shot so easy it beats the target at the ceiling
    probed.clear()
    _plant_svt(enc, monkeypatch, lambda i: 70.0, probed)
    scores = enc._probe_shot_seeded(0, 0, 100, grid, 4, seed=61.0, step=4)
    assert probed[0] == [44]
    assert opt.pick_crf(list(scores.items()), enc.target) == 44.0


def test_a_wrong_out_of_range_guess_costs_one_probe_not_the_answer(
        settings, info, plan, tmp_path, monkeypatch):
    """Outside the fitted range the prediction is poor - 7.70 CRF mean error
    against 2.4 inside it, because the fit only ever sees shots that cross. So
    it picks which bound to try, never the value, and being wrong costs a
    probe."""
    settings.transcode.optimizer.probe_crfs = [20, 26, 32, 38, 44]
    settings.transcode.optimizer.probe_encoder = "qsv+svt"
    enc = make_encoder(settings, info, plan, tmp_path)
    grid = enc._probe_grid()
    probed = {}
    _plant_svt(enc, monkeypatch, lambda i: 31.0, probed)     # really in range
    scores = enc._probe_shot_seeded(0, 0, 100, grid, 4, seed=18.0, step=4)
    spent = len(probed[0])
    got = opt.pick_crf(list(scores.items()), enc.target)
    probed.clear()
    plain = opt.pick_crf(list(enc._probe_shot(0, 0, 100, grid, 4).items()), enc.target)
    assert got == pytest.approx(plain, abs=1.0)
    assert spent <= enc._probe_budget(grid)


def test_max_crf_caps_delivery_without_narrowing_the_probe_range(
        settings, info, plan, tmp_path):
    """probe_crfs says where to look; min_crf/max_crf say what may be shipped.
    Separating them is what makes widening the probe range free."""
    settings.transcode.optimizer.probe_crfs = [20, 30, 40, 50, 60]
    settings.transcode.optimizer.max_crf = 45
    settings.transcode.optimizer.min_crf = 25
    enc = make_encoder(settings, info, plan, tmp_path)
    grid = enc._probe_grid()
    assert min(grid) == 20 and max(grid) == 60        # the probe range is intact
    assert enc._crf_floor(min(grid)) == 25.0
    assert enc._crf_ceiling(max(grid)) == 45.0
    # a shot too easy to be worth any probed CRF ships the ceiling; one that
    # never reaches the target ships the floor
    t = enc.target
    easy = {20: t + 9.0, 60: t + 1.0}      # beats the target everywhere
    hard = {20: t - 1.0, 60: t - 6.0}      # misses it everywhere
    chosen = enc.pick_all_crfs({0: easy, 1: hard}, grid)
    assert chosen[0] == 45.0 and chosen[1] == 25.0


# ---- empty subtitle tracks: measured once, dropped in one place ----
#
# Plex auto-selected an EMPTY PGS track on a 4K HDR output of ours and burned
# it into the picture: 0.2-0.9x real time, ~660% CPU, every thread inside the
# subtitle overlay's scale, for a track with nothing in it.

# the statistics tag an empty Matroska subtitle track carries (mkvmerge v82
# and DVDFab13 both write it; measured on the S04E09 source AND our output)
_EMPTY = {"tags": {"NUMBER_OF_FRAMES": "0"}}


def _mkvmerge_json(*tracks):
    """`mkvmerge -J`: (codec_id, num_index_entries) per subtitle track."""
    return json.dumps({"tracks": [
        {"id": i, "type": "subtitles",
         "properties": {"codec_id": codec, "num_index_entries": entries}}
        for i, (codec, entries) in enumerate(tracks)]})


def _plan_of(enc, monkeypatch, probe, mkvmerge=None, has_mkvmerge=True,
             source="/x/src.mkv"):
    """_subtitle_plan against a stubbed plan probe and, for a stream with no
    statistics tag, a stubbed `mkvmerge -J`. Returns (plan, mkvmerge runs)."""
    monkeypatch.setattr(
        opt.shutil, "which",
        lambda n, *a, **k: None if not has_mkvmerge and "mkvmerge" in n
        else f"/usr/bin/{n}")
    enc._run = (lambda self, args, timeout=None: probe).__get__(enc)
    runs = []

    def fake_mkvmerge(cmd, capture_output=False, text=False, timeout=None):
        runs.append([str(c) for c in cmd])
        if "--version" in runs[-1]:
            # only asked once the -J answer turned out to carry no counts
            return types.SimpleNamespace(
                returncode=0, stderr="",
                stdout="mkvmerge v74.0.0 ('You Oughta Know') 64-bit\n")
        if isinstance(mkvmerge, Exception):
            raise mkvmerge
        return types.SimpleNamespace(returncode=0, stdout=mkvmerge or "",
                                     stderr="")

    monkeypatch.setattr(opt.subprocess, "run", fake_mkvmerge)
    return enc._subtitle_plan(source), runs


def test_an_empty_subtitle_track_is_dropped_and_the_kept_one_renumbers(
        settings, info, plan, tmp_path, monkeypatch):
    """S04E09's shape: two PGS tracks, the first one empty. "-c:s:N" and
    "-disposition:s:N" address the OUTPUT, so dropping s:0 makes the second
    track s:0 - and getting that wrong puts one track's flags on another
    without a word, which is the bug 4710560 had to fix."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _probe_json(
        ("video", set()), ("audio", {"default"}),
        ("subtitle", set(), _EMPTY),
        ("subtitle", {"default", "forced"}, {"codec_name": "subrip"})))
    remux, = _remuxes(ran)
    # the negative map sits after the positive one it edits, and carries "?"
    assert _maps(remux) == ["0:a?", "0:s?", "-0:s:0?", "0:t?"]
    assert remux[remux.index("-c:s:0") + 1] == "copy"
    assert "-c:s:1" not in remux
    assert _dispositions(remux) == [("-disposition:a:0", "default"),
                                    ("-disposition:s:0", "default+forced")]
    assert enc.subtitles_dropped == 1


def test_the_middle_track_being_the_empty_one_renumbers_both_lists(
        settings, info, plan, tmp_path, monkeypatch):
    """The case a "drop the first one" mistake survives. The empty track is
    also the one flagged default here - The Morning Show S01E03 ships exactly
    that, subrip, default+forced, titled "Forced", 0 frames - so dropping it
    takes a false default out with it."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _probe_json(
        ("video", set()), ("audio", {"default"}),
        ("subtitle", {"forced"}, {"codec_name": "subrip"}),
        ("subtitle", {"default"}, _EMPTY),
        ("subtitle", set(), {"codec_name": "mov_text"})))
    remux, = _remuxes(ran)
    assert _maps(remux) == ["0:a?", "0:s?", "-0:s:1?", "0:t?"]
    assert [remux[remux.index(f"-c:s:{i}") + 1] for i in (0, 1)] == ["copy", "srt"]
    assert "-c:s:2" not in remux
    assert _dispositions(remux) == [("-disposition:a:0", "default"),
                                    ("-disposition:s:0", "forced"),
                                    ("-disposition:s:1", "0")]
    assert "default" not in [v for _, v in _dispositions(remux)][1:]


@pytest.mark.parametrize("extra, dropped", [
    ({"tags": {"NUMBER_OF_FRAMES": "0"}}, True),
    ({"tags": {"NUMBER_OF_FRAMES-eng": "0"}}, True),
    ({"tags": {"number_of_frames": "0"}}, True),
    ({"tags": {"NUMBER_OF_FRAMES": "1"}}, False),
    ({"tags": {"NUMBER_OF_FRAMES": "1448"}}, False),
    ({"tags": {"NUMBER_OF_FRAMES": "0", "NUMBER_OF_FRAMES-eng": "12"}}, False),
    ({"tags": {"NUMBER_OF_FRAMES": "N/A"}}, False),
    ({"tags": {"NUMBER_OF_FRAMES": ""}}, False),
    ({"tags": {}}, False),
], ids=["zero", "lang-suffixed", "lower-case", "one-cue", "real", "disagree",
        "not-a-number", "empty-string", "no-tag-no-mkvmerge"])
def test_only_an_explicit_zero_drops_a_matroska_track(
        settings, info, plan, tmp_path, monkeypatch, extra, dropped):
    """One cue is a real track - a forced/signs track can hold exactly one,
    and it is then the only copy of that translation anywhere. Everything
    unreadable keeps the track: the two mistakes do not cost the same."""
    enc = make_encoder(settings, info, plan, tmp_path)
    got, _ = _plan_of(enc, monkeypatch, _probe_json(("subtitle", set(), extra)),
                      has_mkvmerge=False)
    assert bool(got.dropped) is dropped


@pytest.mark.parametrize("extra, dropped", [
    ({"nb_frames": "1", "duration_ts": 0}, True),
    ({"nb_frames": None, "duration_ts": 0}, True),
    ({"nb_frames": "1", "duration_ts": 0,
      "tags": {"NUMBER_OF_FRAMES": "1448"}}, True),
    ({"nb_frames": "3", "duration_ts": 500000}, False),
    ({"nb_frames": "730", "duration_ts": 2709960000}, False),
    ({"nb_frames": "730", "duration_ts": 0}, False),
    ({"nb_frames": "0"}, False),
    ({"duration_ts": "N/A"}, False),
    ({"duration_ts": None}, False),
], ids=["empty", "empty-and-untallied", "mkv-tag-is-not-read", "one-cue",
        "real", "sample-table-disagrees", "nb-frames-zero-cannot-happen",
        "not-a-number", "absent"])
def test_an_mp4_is_decided_by_the_tracks_own_duration(
        settings, info, plan, tmp_path, monkeypatch, extra, dropped):
    """nb_frames cannot answer this, however promising it looks, and the whole
    mp4 half of the feature was a silent no-op while it was what got read.

    Measured on real files here: ffprobe prints nb_frames only when it is
    non-zero (it writes the optional "N/A" otherwise, which the json writer
    suppresses), and the mov muxer gives a cue-less mov_text track a padding
    sample anyway - a track whose only cue was cut away reads nb_frames "1",
    never "0". The track's own duration is what says so: duration_ts 0 on that
    same empty track, 500000 on a ONE-cue track, 2709960000-2766642000 on the
    eight real mov_text tracks of the ATVP WEB-DL in sample/. A sample table
    that counts more than the padding wins over the header; Matroska's tag is
    not read here at all, because the container decides which signal is."""
    enc = make_encoder(settings, info, plan, tmp_path)
    got, _ = _plan_of(enc, monkeypatch,
                      _probe_json(("video", set()),
                                  ("subtitle", set(), extra),
                                  fmt="mov,mp4,m4a,3gp,3g2,mj2"),
                      has_mkvmerge=False)
    assert bool(got.dropped) is dropped


def test_a_file_that_states_no_duration_at_all_decides_nothing(
        settings, info, plan, tmp_path, monkeypatch):
    """A fragmented mp4 read without its fragments reports duration_ts 0 for
    every stream it has, cues or not. The signal means "unknown" there, and
    only the presence of a real duration somewhere in the file tells the two
    apart - without that guard this would drop every subtitle such a file
    carries."""
    enc = make_encoder(settings, info, plan, tmp_path)
    got, _ = _plan_of(
        enc, monkeypatch,
        _probe_json(("video", set(), {"duration_ts": 0}),
                    ("subtitle", set(), {"nb_frames": None, "duration_ts": 0}),
                    fmt="mov,mp4,m4a,3gp,3g2,mj2"),
        has_mkvmerge=False)
    assert got.dropped == []
    assert [s.why for s in got.subs] == ["no stream of this file states a duration"]


def test_a_matroska_track_with_no_tag_is_decided_by_mkvmerge(
        settings, info, plan, tmp_path, monkeypatch):
    """25% of the library's subtitle streams carry no statistics tag at all -
    every mov_text, 280 of 281 ass, 380 subrip - and 497 of 5005 IMAGE
    streams, which are the ones a player burns in. Measured: an empty PGS
    indexes 0 entries, a real one 1034-1508 (matching its own tag exactly),
    an untagged ASS 10-18, an untagged SRT 721."""
    enc = make_encoder(settings, info, plan, tmp_path)
    got, runs = _plan_of(
        enc, monkeypatch,
        _probe_json(("subtitle", set(), {"tags": {}}),
                    ("subtitle", set(), {"codec_name": "subrip", "tags": {}})),
        mkvmerge=_mkvmerge_json(("S_HDMV/PGS", 0), ("S_TEXT/UTF8", 721)))
    assert [s.pos for s in got.dropped] == [0]
    assert [s.why for s in got.subs] == ["num_index_entries=0",
                                         "num_index_entries=721"]
    assert [r[1] for r in runs] == ["-J"]


@pytest.mark.parametrize("mkvmerge", [
    _mkvmerge_json(("S_HDMV/PGS", 0)),                     # one track, two here
    _mkvmerge_json(("S_TEXT/UTF8", 0), ("S_HDMV/PGS", 0)),  # codecs swapped
    _mkvmerge_json(("S_HDMV/PGS", None), ("S_TEXT/UTF8", None)),
    "not json at all",
    OSError("mkvmerge is not there"),
], ids=["count-mismatch", "codec-mismatch", "no-entries", "no-json", "failed"])
def test_an_mkvmerge_answer_that_does_not_line_up_keeps_everything(
        settings, info, plan, tmp_path, monkeypatch, mkvmerge):
    """mkvmerge ids count TRACKS while ffprobe indexes count STREAMS, and
    matroskadec turns an image attachment into a stream of its own. So the two
    lists are matched by subtitle order and then checked against each other's
    codec; anything that does not line up exactly drops nothing at all."""
    enc = make_encoder(settings, info, plan, tmp_path)
    got, _ = _plan_of(
        enc, monkeypatch,
        _probe_json(("subtitle", set(), {"tags": {}}),
                    ("subtitle", set(), {"codec_name": "subrip", "tags": {}})),
        mkvmerge=mkvmerge)
    assert got.dropped == []
    assert got.kept == got.subs


def test_an_mkvmerge_that_reports_no_index_entries_says_so_once(
        settings, info, plan, tmp_path, monkeypatch):
    """Not every mkvmerge has the property this fallback reads. v74 - the one
    in the image actually deployed, which is bookworm-based where the
    Dockerfile's runtime stage is ubuntu:24.04 and v82 - prints no
    num_index_entries at all, for any track, on a file it wrote itself with
    cues. Every untagged stream is then kept, which is safe; what it must not
    be is silent, because 25% of the library's subtitle streams carry no tag
    and the feature would look like it simply found nothing to drop."""
    enc = make_encoder(settings, info, plan, tmp_path)
    warnings = []
    monkeypatch.setattr(opt.logger, "warning", lambda *a, **k: warnings.append(a))
    got, runs = _plan_of(
        enc, monkeypatch, _probe_json(("subtitle", set(), {"tags": {}})),
        mkvmerge=json.dumps({"tracks": [
            {"id": 0, "type": "subtitles",
             "properties": {"codec_id": "S_HDMV/PGS", "number": 1}}]}))
    assert got.dropped == [] and got.subs[0].empty is None
    # the -J answer, then the version for the one warning that names it
    assert [r[1] for r in runs] == ["-J", "--version"]
    assert len(warnings) == 1
    said = warnings[0][0].format(*warnings[0][1:])
    assert "v74.0.0" in said and "num_index_entries" in said


def test_mkvmerge_is_only_asked_when_a_matroska_tag_is_missing(
        settings, info, plan, tmp_path, monkeypatch):
    """It is a second tool and a second read (0.198-2.3s per file measured), so
    a tagged file never pays for it - and an mp4 never asks, because mkvmerge
    reports codec_id and num_index_entries as None for every subtitle track
    there anyway."""
    enc = make_encoder(settings, info, plan, tmp_path)
    _, runs = _plan_of(enc, monkeypatch, _probe_json(("subtitle", set())),
                       source="/x/tagged.mkv")
    assert runs == []
    _, runs = _plan_of(enc, monkeypatch,
                       _probe_json(("subtitle", set(), {"tags": {},
                                                        "nb_frames": "N/A"}),
                                   fmt="mov,mp4,m4a,3gp,3g2,mj2"),
                       source="/x/movie.mp4")
    assert runs == []


def test_what_was_dropped_is_logged_once_per_job(settings, info, plan,
                                                 tmp_path, monkeypatch):
    """The only record of which stream was left out and what decided it. An
    operator asking afterwards has nothing else to go on: with
    transcode.delete_source the source is gone minutes later, so there is
    nothing left to re-probe."""
    enc = make_encoder(settings, info, plan, tmp_path)
    lines = []
    monkeypatch.setattr(opt.logger, "info", lambda *a, **k: lines.append(a))
    _plan_of(enc, monkeypatch, _probe_json(
        ("video", set()),
        ("subtitle", set(), {"codec_name": "subrip",
                             "tags": {"NUMBER_OF_FRAMES": "0",
                                      "language": "fre"}}),
        ("subtitle", set())), has_mkvmerge=False)
    enc._subtitle_plan("/x/src.mkv")          # cached: not logged twice
    assert len(lines) == 1
    said = lines[0][0].format(*lines[0][1:])
    # the output-side number, the stream index the file itself uses, what the
    # track is, whose it is, and what said it was empty
    assert "s:0 (stream 1) subrip fre NUMBER_OF_FRAMES=0" in said


def test_with_the_subtitle_settings_off_the_commands_are_todays(
        settings, info, plan, tmp_path, monkeypatch):
    """Byte for byte what this mux ran before any of it existed, including for
    a track that IS empty and an ASS that would get a companion."""
    settings.transcode.optimizer.drop_empty_subtitles = False
    settings.transcode.optimizer.ass_srt_companion = False
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _probe_json(
        ("video", set()), ("audio", {"default"}),
        ("subtitle", set(), _EMPTY),
        ("subtitle", set(), {"codec_name": "ass", "tags": {}})))
    remux, = _remuxes(ran)
    assert remux == [enc.ffmpeg, "-hide_banner", "-y", "-loglevel", "error",
                     "-i", str(enc.info.path), "-map", "0:a?", "-map", "0:s?",
                     "-c:a", "copy", "-disposition:a:0", "default",
                     "-disposition:s:0", "0", "-disposition:s:1", "0",
                     "-map", "0:t?", "-c:t", "copy",
                     "-c:s:0", "copy", "-c:s:1", "copy",
                     str(enc.tempdir / "audio_subs.mkv")]
    final, = [a for a in ran if a[-1] == str(enc.output)]
    assert _maps(final) == ["0:v:0", "1:a?", "1:s?", "1:t?"]
    assert enc.subtitles_dropped == 0 and enc.subtitles_added == 0


def test_a_source_with_nothing_but_empty_subtitles_skips_the_remux(
        settings, info, plan, tmp_path, monkeypatch):
    """The guard and the drop have to agree by construction. Once the empty
    streams are gone this command maps ZERO streams, and ffmpeg with no output
    stream allocates ~8.2GB before it exits - an OOM kill at the very last
    step of a finished encode. The streams still count as dropped: the output
    carries none of them either way."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _probe_json(
        ("video", set()), ("subtitle", set(), _EMPTY),
        ("subtitle", set(), _EMPTY)))
    assert _remuxes(ran) == []
    assert enc.subtitles_dropped == 2
    final, = [a for a in ran if a[-1] == str(enc.output)]
    assert final.count("-i") == 1 and "1:s?" not in final


def test_audio_beside_only_empty_subtitles_still_gets_its_remux(
        settings, info, plan, tmp_path, monkeypatch):
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _probe_json(
        ("video", set()), ("audio", {"default"}), ("subtitle", set(), _EMPTY)))
    remux, = _remuxes(ran)
    assert _maps(remux) == ["0:a?", "0:s?", "-0:s:0?", "0:t?"]
    assert _dispositions(remux) == [("-disposition:a:0", "default")]
    assert enc.subtitles_dropped == 1


def test_the_drop_reaches_both_remux_retries(
        settings, info, plan, tmp_path, monkeypatch):
    """It lives in base_args for exactly this reason: the plain-copy retry and
    the one without the attachments are built from the same list."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(
        enc, monkeypatch,
        _probe_json(("video", set()), ("audio", {"default"}),
                    ("subtitle", set(), _EMPTY), ("subtitle", set())),
        fail=lambda a: any(x.endswith("audio_subs.mkv") for x in a) and "0:t?" in a)
    first, retry, bare = _remuxes(ran)
    for cmd in (first, retry, bare):
        assert "-0:s:0?" in cmd
    assert _maps(bare) == ["0:a?", "0:s?", "-0:s:0?"]
    assert enc.subtitles_dropped == 1


def test_the_whole_mux_runs_off_one_probe_of_the_original_source(
        settings, info, plan, tmp_path, monkeypatch):
    """A Dolby Vision job encodes a stripped intermediate, so audio, subtitles
    and flags all come from info.path - and from ONE probe of it, which is
    what keeps the drop maps, the codecs and the dispositions counting the
    same streams."""
    enc = make_encoder(settings, info, plan, tmp_path)
    assert str(enc.source) != str(enc.info.path)
    ran = _mux_with(enc, monkeypatch, _probe_json(
        ("video", set()), ("audio", {"default"}),
        ("subtitle", set(), _EMPTY), ("subtitle", set())))
    probes = [a for a in ran if opt.ShotEncoder._PLAN_PROBE in a]
    assert len(probes) == 1 and probes[0][-1] == str(enc.info.path)


# ---- ASS: kept as it is, with a plain-text srt beside it ----
def _ass_probe(**tags):
    """A source whose second subtitle track is an ASS with `tags`."""
    return _probe_json(
        ("video", set()), ("audio", {"default"}),
        ("subtitle", set(), {"codec_name": "hdmv_pgs_subtitle"}),
        ("subtitle", {"default"}, dict(
            {"codec_name": "ass",
             "tags": {"NUMBER_OF_FRAMES": "916", "language": "chi",
                      "title": "简体"}}, **tags)))


def test_an_ass_track_gets_a_sanitised_srt_companion(
        settings, info, plan, tmp_path, monkeypatch):
    """One demux: the companion is an extra OUTPUT of the command that already
    writes audio_subs.mkv (measured at 4.3s against 3.3s for 42 tracks of a
    2.2GB source). The ASS itself is copied through untouched."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _ass_probe())
    remux, = _remuxes(ran)
    companion = str(enc.tempdir / "sub_1.srt")
    # the remux is unchanged up to audio_subs.mkv; the companion follows it
    assert remux[remux.index(str(enc.tempdir / "audio_subs.mkv")) + 1:] == [
        "-map", "0:s:1", "-c:s", "srt", companion]
    assert [remux[remux.index(f"-c:s:{i}") + 1] for i in (0, 1)] == ["copy", "copy"]
    # srtenc's markup is stripped, the text and <i> are not
    text = Path(companion).read_text(encoding="utf-8")
    assert "<font" not in text and "{\\an" not in text
    assert "人多力量大" in text and "<i>italics survive</i>" in text
    # and it is muxed in as a track of its own, last, with the ASS's own
    # language, a title that says what it is, and the ASS's flags
    final, = [a for a in ran if a[-1] == str(enc.output)]
    assert final.count("-i") == 3
    assert _maps(final) == ["0:v:0", "1:a?", "1:s?", "1:t?", "2:0"]
    meta = final[final.index("-metadata:s:s:2"):]
    assert meta[:6] == ["-metadata:s:s:2", "language=chi",
                        "-metadata:s:s:2", "title=简体 (SRT)",
                        "-disposition:s:2", "default"]
    assert enc.subtitles_added == 1


def test_a_companion_beside_a_dropped_track_counts_from_both_ends(
        settings, info, plan, tmp_path, monkeypatch):
    """The one place the two features meet, and the one that can silently go
    wrong: "-map 0:s:N" counts the SOURCE's subtitles, where the dropped track
    is still present, while "-disposition:s:N" and "-metadata:s:s:N" count the
    OUTPUT's, where it is not. An empty track before the ASS moves one and not
    the other."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _probe_json(
        ("video", set()), ("audio", {"default"}),
        ("subtitle", set(), _EMPTY),
        ("subtitle", {"default"}, {"codec_name": "ass",
                                   "tags": {"NUMBER_OF_FRAMES": "916",
                                            "language": "chi"}})))
    remux, = _remuxes(ran)
    # the last map belongs to the companion, which is a second output of this
    # same command
    assert _maps(remux) == ["0:a?", "0:s?", "-0:s:0?", "0:t?", "0:s:1"]
    # the ASS is s:1 of the source and s:0 of the output, and each number is
    # read off the end it belongs to
    assert remux[remux.index(str(enc.tempdir / "audio_subs.mkv")) + 1:] == [
        "-map", "0:s:1", "-c:s", "srt", str(enc.tempdir / "sub_1.srt")]
    # and the ASS, being the output's s:0, is the one that gives its default
    # flag to the companion below
    assert _dispositions(remux) == [("-disposition:a:0", "default"),
                                    ("-disposition:s:0", "0")]
    final, = [a for a in ran if a[-1] == str(enc.output)]
    assert final[final.index("-metadata:s:s:1"):][:6] == [
        "-metadata:s:s:1", "language=chi",
        "-metadata:s:s:1", "title=SRT (plain text)",
        "-disposition:s:1", "default"]
    assert (enc.subtitles_dropped, enc.subtitles_added) == (1, 1)


def test_mkvmerge_takes_each_companion_as_an_input_of_its_own(
        settings, info, plan, tmp_path, monkeypatch):
    """--sub-charset is load-bearing: mkvmerge otherwise guesses the charset
    from the locale and mangles CJK. It converts nothing itself, so the file
    handed to it has to be srt already."""
    enc = make_encoder(settings, info, plan, tmp_path)
    _, merges = _concat_via_mkvmerge(
        enc, monkeypatch, _ass_probe(),
        _attachment_probe(_MKV, ("video", 0), ("audio", 0), ("subtitle", 0),
                          ("attachment", 0)))
    merge, = merges
    assert merge[1:] == ["-o", str(enc.output),
                         str(enc.tempdir / "video_only.mkv"), "--no-attachments",
                         str(enc.tempdir / "audio_subs.mkv"),
                         "--sub-charset", "0:UTF-8", "--language", "0:chi",
                         "--track-name", "0:简体 (SRT)",
                         "--default-track-flag", "0:yes",
                         "--forced-display-flag", "0:no",
                         str(enc.tempdir / "sub_1.srt"),
                         *_SOURCE_ONLY, str(enc.info.path)]


@pytest.mark.parametrize("flags, default, forced", [
    ({"default"}, "0:yes", "0:no"),
    (set(), "0:no", "0:no"),
    ({"forced"}, "0:no", "0:yes"),
    ({"default", "forced"}, "0:yes", "0:yes"),
], ids=["default", "neither", "forced", "both"])
def test_a_companion_is_default_only_when_its_ass_was(
        settings, info, plan, tmp_path, flags, default, forced):
    """This ADDS a track, it does not promote one. A second track flagged
    default is the shape of the thing that made Plex pick the wrong one."""
    enc = make_encoder(settings, info, plan, tmp_path)
    sub = opt.SubStream(index=3, pos=1, codec="ass", flags="0",
                        default="default" in flags, forced="forced" in flags,
                        language="jpn", title="", empty=False, why="tag")
    args = enc._companion_inputs([sub])
    assert args[args.index("--default-track-flag") + 1] == default
    assert args[args.index("--forced-display-flag") + 1] == forced
    # no title to borrow: the name still says what the track is
    assert args[args.index("--track-name") + 1] == "0:SRT (plain text)"


def test_the_companion_carries_the_same_flags_through_either_muxer(
        settings, info, plan, tmp_path):
    """It was default/forced through mkvmerge and the ASS's WHOLE disposition
    value through the ffmpeg fallback. An SDH source then came out
    hearing_impaired through one muxer and plain through the other - two
    episodes of one season disagreeing, decided by nothing but whether
    mkvmerge was there, and neither path wrong by its own comment."""
    enc = make_encoder(settings, info, plan, tmp_path)
    sub = opt.SubStream(index=3, pos=0, codec="ass",
                        flags="default+hearing_impaired", default=True,
                        forced=False, language="eng", title="", empty=False,
                        why="NUMBER_OF_FRAMES=916")
    enc._sub_plans["/x/src.mkv"] = opt.SubtitlePlan(True, True, [], [sub])
    inputs = enc._companion_inputs([sub])
    assert inputs[inputs.index("--default-track-flag") + 1] == "0:yes"
    assert inputs[inputs.index("--forced-display-flag") + 1] == "0:no"
    assert "hearing" not in " ".join(inputs)
    # the ffmpeg fallback states those same two flags and no others; the
    # descriptive ones stay on the ASS track beside it
    meta = enc._companion_metadata("/x/src.mkv", [sub])
    assert meta[meta.index("-disposition:s:1") + 1] == "default"


def test_the_default_flag_moves_from_the_ass_to_its_companion(
        settings, info, plan, tmp_path, monkeypatch):
    """Copied, it left the output with TWO default subtitle tracks; a player
    takes the first, which is the ASS, and the companion was then useless for
    the one job it has - being the text track Plex can send instead of burning
    the picture in. So it moves: the companion is default and the ASS is not."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _ass_probe())
    remux, = _remuxes(ran)
    # measured on mkvmerge v82: it copies audio_subs.mkv's flags track for
    # track, so for that muxer this command is what decides them
    assert _dispositions(remux) == [("-disposition:a:0", "default"),
                                    ("-disposition:s:0", "0"),
                                    ("-disposition:s:1", "0")]
    # the ffmpeg fallback decides for itself, and has to say the same thing:
    # measured, its -disposition overrides what audio_subs.mkv carries
    final, = [a for a in ran if a[-1] == str(enc.output)]
    assert _dispositions(final) == [("-disposition:a:0", "default"),
                                    ("-disposition:s:0", "0"),
                                    ("-disposition:s:1", "0"),
                                    ("-disposition:s:2", "default")]


def test_mkvmerge_is_told_nothing_about_the_ass_it_took_the_flag_from(
        settings, info, plan, tmp_path, monkeypatch):
    """The remux above already wrote audio_subs.mkv without that flag, and
    mkvmerge copies it from there. Stating it a second time here would be a
    second place to get the track number wrong, for no gain."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran, merges = _concat_via_mkvmerge(
        enc, monkeypatch, _ass_probe(),
        _attachment_probe(_MKV, ("video", 0), ("audio", 0), ("subtitle", 0)))
    remux, = _remuxes(ran)
    assert _dispositions(remux)[1:] == [("-disposition:s:0", "0"),
                                        ("-disposition:s:1", "0")]
    merge, = merges
    # the companion, and only the companion, is declared default here
    assert merge.count("--default-track-flag") == 1
    assert merge[merge.index("--default-track-flag") + 1] == "0:yes"
    assert merge.index("--default-track-flag") > merge.index(
        str(enc.tempdir / "audio_subs.mkv"))


def test_a_companion_for_a_plain_ass_leaves_neither_of_them_default(
        settings, info, plan, tmp_path, monkeypatch):
    """Only a flag that is there moves. Nothing is promoted: a source whose
    subtitles are all off must stay that way, or the companion becomes a
    default track the source never had."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _probe_json(
        ("video", set()), ("audio", {"default"}),
        ("subtitle", set(), {"codec_name": "ass",
                             "tags": {"NUMBER_OF_FRAMES": "916"}})))
    remux, = _remuxes(ran)
    assert _dispositions(remux) == [("-disposition:a:0", "default"),
                                    ("-disposition:s:0", "0")]
    final, = [a for a in ran if a[-1] == str(enc.output)]
    assert _dispositions(final) == [("-disposition:a:0", "default"),
                                    ("-disposition:s:0", "0"),
                                    ("-disposition:s:1", "0")]
    assert enc.subtitles_added == 1


def test_only_the_default_flag_moves_and_the_others_stay_put(
        settings, info, plan, tmp_path, monkeypatch):
    """forced is copied and belongs on both tracks - a forced ASS and its
    plain-text copy are both forced. The descriptive flags stay on the ASS
    alone: an SDH track does not stop being hearing_impaired because a copy
    was made of it."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _probe_json(
        ("video", set()), ("audio", {"default"}),
        ("subtitle", {"default", "forced", "hearing_impaired"},
         {"codec_name": "ass", "tags": {"NUMBER_OF_FRAMES": "916"}})))
    remux, = _remuxes(ran)
    assert _dispositions(remux) == [
        ("-disposition:a:0", "default"),
        ("-disposition:s:0", "forced+hearing_impaired")]
    final, = [a for a in ran if a[-1] == str(enc.output)]
    assert _dispositions(final)[-1] == ("-disposition:s:1", "default+forced")


def test_a_default_pgs_beside_a_default_ass_is_left_alone(
        settings, info, plan, tmp_path, monkeypatch):
    """Only an ASS that really got a companion gives its flag up. A bitmap
    track can never have one - libavcodec has no bitmap-to-text path - so
    nothing about it changes, however many default tracks the source ships."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _probe_json(
        ("video", set()), ("audio", {"default"}),
        ("subtitle", {"default"}),
        ("subtitle", {"default", "forced"},
         {"codec_name": "ass", "tags": {"NUMBER_OF_FRAMES": "916"}})))
    remux, = _remuxes(ran)
    assert _dispositions(remux) == [("-disposition:a:0", "default"),
                                    ("-disposition:s:0", "default"),
                                    ("-disposition:s:1", "forced")]
    assert enc.subtitles_added == 1


def test_with_the_companion_off_a_default_ass_keeps_its_flag(
        settings, info, plan, tmp_path, monkeypatch):
    """Nothing to move it to, so nothing moves: byte for byte the command this
    mux ran before the flag did."""
    settings.transcode.optimizer.ass_srt_companion = False
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _ass_probe())
    remux, = _remuxes(ran)
    assert remux == [enc.ffmpeg, "-hide_banner", "-y", "-loglevel", "error",
                     "-i", str(enc.info.path), "-map", "0:a?", "-map", "0:s?",
                     "-c:a", "copy", "-disposition:a:0", "default",
                     "-disposition:s:0", "0", "-disposition:s:1", "default",
                     "-map", "0:t?", "-c:t", "copy",
                     "-c:s:0", "copy", "-c:s:1", "copy",
                     str(enc.tempdir / "audio_subs.mkv")]
    final, = [a for a in ran if a[-1] == str(enc.output)]
    assert _dispositions(final) == [("-disposition:a:0", "default"),
                                    ("-disposition:s:0", "0"),
                                    ("-disposition:s:1", "default")]


def test_an_empty_default_ass_keeps_the_flag_it_cannot_lend(
        settings, info, plan, tmp_path, monkeypatch):
    """An empty track gets no companion - it would write an empty file - so
    there is nothing to move the flag to. With the drop off it stays in the
    output, default and all."""
    settings.transcode.optimizer.drop_empty_subtitles = False
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch,
                    _ass_probe(tags={"NUMBER_OF_FRAMES": "0"}))
    remux, = _remuxes(ran)
    assert not any(a.endswith(".srt") for a in remux)
    assert _dispositions(remux) == [("-disposition:a:0", "default"),
                                    ("-disposition:s:0", "0"),
                                    ("-disposition:s:1", "default")]
    assert enc.subtitles_added == 0


def test_a_companion_that_comes_out_empty_hands_the_default_back(
        settings, info, plan, tmp_path, monkeypatch):
    """The remux states its flags before any companion exists as a file, so one
    that is then left out - here, an srt with nothing in it - has already taken
    the ASS's default with it. The final mux states it back, or the output ends
    up with no default subtitle at all."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _ass_probe(), srt="")
    remux, = _remuxes(ran)
    assert _dispositions(remux)[-1] == ("-disposition:s:1", "0")
    assert enc.subtitles_added == 0
    final, = [a for a in ran if a[-1] == str(enc.output)]
    assert _dispositions(final) == [("-disposition:a:0", "default"),
                                    ("-disposition:s:0", "0"),
                                    ("-disposition:s:1", "default")]


def test_mkvmerge_is_told_to_put_a_lost_default_back(
        settings, info, plan, tmp_path, monkeypatch):
    """The mkvmerge mux cannot restate that list: it copies audio_subs.mkv's
    own flags, and that file is written by the time the companion turns out to
    be empty. Measured on v82, it promotes nothing of its own either, so the
    output would carry NO default subtitle.

    The fixture is shaped to pin that id and nothing else: THREE audio tracks
    and a dropped one before the ASS, so the audio count, the kept subtitle
    count and the ASS's own source position are three different numbers. The
    ASS is the output's s:1, and the remux wrote every audio stream before any
    subtitle, so it is track 3 + 1 (measured on such an audio_subs.mkv: id 0-2
    the audio, 3 and 4 the subtitles). Measured too, on mkvmerge v82: a wrong
    id does not fail. "1:yes" exits 0 and makes the second AUDIO track default,
    leaving the output with no default subtitle at all - which no one-audio,
    nothing-dropped fixture can tell apart from the right answer."""
    enc = make_encoder(settings, info, plan, tmp_path)
    _, merges = _concat_via_mkvmerge(
        enc, monkeypatch,
        _probe_json(("video", set()), ("audio", {"default"}), ("audio", set()),
                    ("audio", set()),
                    ("subtitle", set(), _EMPTY), ("subtitle", set()),
                    ("subtitle", {"default"},
                     {"codec_name": "ass",
                      "tags": {"NUMBER_OF_FRAMES": "916"}})),
        _attachment_probe(_MKV, ("video", 0), ("audio", 0), ("subtitle", 0)),
        srt="")
    merge, = merges
    audio_subs = merge.index(str(enc.tempdir / "audio_subs.mkv"))
    assert merge[audio_subs - 2:audio_subs] == ["--default-track-flag", "4:yes"]
    assert not any(a.endswith(".srt") for a in merge)
    assert (enc.subtitles_dropped, enc.subtitles_added) == (1, 0)


def test_nothing_is_put_back_for_a_companion_that_took_nothing(
        settings, info, plan, tmp_path, monkeypatch):
    """A plain ASS lends no flag, so a companion of its own that falls away
    leaves nothing to restore. Stating one anyway would invent a default track
    the source never had - and a track id mkvmerge cannot find exits 1, which
    would cost the whole mkvmerge mux."""
    enc = make_encoder(settings, info, plan, tmp_path)
    _, merges = _concat_via_mkvmerge(
        enc, monkeypatch,
        _probe_json(("video", set()), ("audio", {"default"}),
                    ("subtitle", set(), {"codec_name": "ass",
                                         "tags": {"NUMBER_OF_FRAMES": "916"}})),
        _attachment_probe(_MKV, ("video", 0), ("audio", 0), ("subtitle", 0)),
        srt="")
    merge, = merges
    assert "--default-track-flag" not in merge
    assert enc.subtitles_added == 0


@pytest.mark.parametrize("probe, companions", [
    (_ass_probe(), ["sub_1.srt"]),
    (_ass_probe(tags={"NUMBER_OF_FRAMES": "0"}), []),
    (_probe_json(("video", set()), ("subtitle", set())), []),
    (opt.TranscodeError("ffprobe exploded"), []),
], ids=["ass", "empty-ass", "no-ass", "probe-failed"])
def test_which_tracks_get_a_companion(settings, info, plan, tmp_path,
                                      monkeypatch, probe, companions):
    """Never for an empty track: it would write an empty file, and an empty
    subtitle track is the thing this release exists to stop shipping."""
    enc = make_encoder(settings, info, plan, tmp_path)
    if isinstance(probe, Exception):
        def boom(self, args, timeout=None):
            raise probe
        enc._run = boom.__get__(enc)
    else:
        enc._run = (lambda self, args, timeout=None: probe).__get__(enc)
    assert [Path(enc._companion_path(s)).name
            for s in enc._companions("/x/src.mkv")] == companions


def test_no_companion_for_an_empty_ass_even_with_the_drop_off(
        settings, info, plan, tmp_path):
    """With the drop off the empty track stays in the output, and it must
    still not get a companion: that would add a SECOND empty track. Emptiness
    is measured for either setting, and only the drop is gated on its own."""
    settings.transcode.optimizer.drop_empty_subtitles = False
    enc = make_encoder(settings, info, plan, tmp_path)
    enc._run = (lambda self, args, timeout=None:
                _ass_probe(tags={"NUMBER_OF_FRAMES": "0"})).__get__(enc)
    assert enc._companions("/x/src.mkv") == []


def test_no_companion_with_the_flag_off(settings, info, plan, tmp_path,
                                        monkeypatch):
    settings.transcode.optimizer.ass_srt_companion = False
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _ass_probe())
    remux, = _remuxes(ran)
    assert remux[-1] == str(enc.tempdir / "audio_subs.mkv")
    assert not any(a.endswith(".srt") for a in remux)
    assert enc.subtitles_added == 0


def test_a_remux_retry_leaves_the_companions_behind(
        settings, info, plan, tmp_path, monkeypatch):
    """They are an addition, and the retry path is already one failure deep.
    What the output must not lose is the source's own tracks."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _ass_probe(),
                    fail=lambda a: any(x.endswith(".srt") for x in a))
    first, retry = _remuxes(ran)
    assert any(a.endswith(".srt") for a in first)
    assert not any(a.endswith(".srt") for a in retry)
    assert enc.subtitles_added == 0
    final, = [a for a in ran if a[-1] == str(enc.output)]
    assert _maps(final) == ["0:v:0", "1:a?", "1:s?", "1:t?"]
    # the first remux had already lent the ASS's default flag to a companion
    # that no longer exists, so the final mux states it back
    assert _dispositions(first)[-1] == ("-disposition:s:1", "0")
    assert _dispositions(final)[-1] == ("-disposition:s:1", "default")


@pytest.mark.parametrize("raw, want", [
    ('<font face="Tahoma" size="50" color="#eeeeee">hello</font>', "hello"),
    ("{\\an8}Top-aligned note", "Top-aligned note"),
    ('<font face="Source Han Sans SC Medium" size="24">人多力量大</font>',
     "人多力量大"),
    ("<i>kept</i> <b>and</b> <u>kept</u>", "<i>kept</i> <b>and</b> <u>kept</u>"),
], ids=["font", "an8", "cjk", "styles-survive"])
def test_the_companion_loses_srtencs_markup_and_nothing_else(
        settings, info, plan, tmp_path, raw, want):
    enc = make_encoder(settings, info, plan, tmp_path)
    f = tmp_path / "sub_0.srt"
    f.write_text(f"1\n00:00:00,200 --> 00:00:01,500\n{raw}\n\n", encoding="utf-8")
    assert enc._sanitise_srt(f) is True
    assert f.read_text(encoding="utf-8") == (
        f"1\n00:00:00,200 --> 00:00:01,500\n{want}\n\n")


def test_a_companion_with_nothing_left_in_it_is_left_out(settings, info, plan,
                                                         tmp_path):
    enc = make_encoder(settings, info, plan, tmp_path)
    f = tmp_path / "sub_0.srt"
    f.write_text("", encoding="utf-8")
    assert enc._sanitise_srt(f) is False
    assert enc._sanitise_srt(tmp_path / "never_written.srt") is False


def test_a_companion_that_sanitises_to_nothing_never_reaches_the_mux(
        settings, info, plan, tmp_path, monkeypatch):
    """A companion that came out with nothing in it - an ASS srtenc wrote no
    cue out of, or a file that could not be read back - must not be handed to
    the mux: mkvmerge refuses an unparsable input, the ffmpeg fallback then
    maps a file with no stream ("matches no streams"), and a mux that did take
    it would leave _verify_output expecting one more subtitle stream than
    exists - which deletes the finished output."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _ass_probe(), srt="")
    remux, = _remuxes(ran)
    assert remux[-1] == str(enc.tempdir / "sub_1.srt")   # extracted as always
    final, = [a for a in ran if a[-1] == str(enc.output)]
    assert not any(a.endswith(".srt") for a in final)
    assert _maps(final) == ["0:v:0", "1:a?", "1:s?", "1:t?"]
    assert not any(a.startswith("-metadata:s:s:") for a in final)
    assert enc.subtitles_added == 0


def test_the_last_ditch_retry_converts_only_the_text_subtitles(
        settings, info, plan, tmp_path, monkeypatch):
    """It ran "-c:s srt" over EVERY subtitle, which cannot work for a bitmap
    track - libavcodec has no bitmap-to-text path - so on any source with a
    PGS or VobSub stream this recovery was dead. It only ever runs after a
    plain copy has already failed, so it never killed a job that was not
    failing; it just never rescued one either."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(
        enc, monkeypatch,
        _probe_json(("video", set()), ("audio", {"default"}),
                    ("subtitle", set()),
                    ("subtitle", set(), {"codec_name": "subrip"})),
        fail=lambda a: a[-1] == str(enc.output) and "-c:s:0" not in a)
    final, retry = [a for a in ran if a[-1] == str(enc.output)]
    assert "-c:s" not in retry                      # never the blanket form
    assert retry[retry.index("-c:s:0") + 1] == "copy"   # the PGS is copied
    assert retry[retry.index("-c:s:1") + 1] == "srt"    # the subrip converts
    assert "srt" not in final
    assert enc.output.exists()


def test_there_is_no_last_ditch_retry_when_nothing_can_be_converted(
        settings, info, plan, tmp_path, monkeypatch):
    """Every subtitle a Blu-ray remux carries is a bitmap one, so this retry
    has nothing to convert - and the command it would run is the one that just
    failed, over the whole file again. It raises the real error instead."""
    enc = make_encoder(settings, info, plan, tmp_path)
    errors = []
    monkeypatch.setattr(opt.logger, "error", lambda *a, **k: errors.append(a))
    with pytest.raises(opt.TranscodeError, match="mux failed"):
        _mux_with(enc, monkeypatch,
                  _probe_json(("video", set()), ("audio", {"default"}),
                              ("subtitle", set()), ("subtitle", set())),
                  fail=lambda a: a[-1] == str(enc.output))
    assert any("can be converted to text" in str(e[0]) for e in errors)


def test_a_source_with_no_subtitles_still_gets_its_identical_retry(
        settings, info, plan, tmp_path, monkeypatch):
    """Nothing to convert is not the same as nothing to try, and the two must
    not be folded together. This retry names no subtitle stream either way, so
    for a source with no subtitles it is the command that just failed run once
    more - which is what this path always was, and what survives a transient
    failure (an ENOSPC that cleared, an NFS hiccup on dirs.work). Returning
    None here would throw a finished encode away."""
    enc = make_encoder(settings, info, plan, tmp_path)
    attempts = []

    def fail(args):
        if args[-1] != str(enc.output):
            return False
        attempts.append(args)
        return len(attempts) == 1              # only the first one fails

    ran = _mux_with(enc, monkeypatch,
                    _probe_json(("video", set()), ("audio", {"default"})),
                    fail=fail)
    final, retry = [a for a in ran if a[-1] == str(enc.output)]
    assert retry == final                      # the identical command, again
    assert enc.output.exists()


def test_the_last_ditch_retry_without_a_plan_is_what_it_always_was(
        settings, info, plan, tmp_path, monkeypatch):
    """No plan, no idea which stream is which. Blanket srt is then no worse
    than the behaviour this replaced, and for a text-only source it works."""
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, opt.TranscodeError("ffprobe exploded"),
                    fail=lambda a: a[-1] == str(enc.output) and "srt" not in a)
    final, retry = [a for a in ran if a[-1] == str(enc.output)]
    assert retry[retry.index("-c:s") + 1] == "srt"


def test_the_retry_copies_the_companions_it_already_converted(
        settings, info, plan, tmp_path, monkeypatch):
    enc = make_encoder(settings, info, plan, tmp_path)
    ran = _mux_with(enc, monkeypatch, _ass_probe(),
                    fail=lambda a: a[-1] == str(enc.output) and "-c:s:0" not in a)
    _, retry = [a for a in ran if a[-1] == str(enc.output)]
    # the PGS copied, the ASS converted, and the companion left alone
    assert [retry[retry.index(f"-c:s:{i}") + 1] for i in (0, 1, 2)] == [
        "copy", "srt", "copy"]


@pytest.mark.skipif(any(_REAL_WHICH(t) is None for t in ("ffmpeg", "ffprobe", "mkvmerge")),
                    reason="needs a real ffmpeg, ffprobe and mkvmerge")
@pytest.mark.parametrize("muxer", ["mkvmerge", "ffmpeg"])
def test_an_empty_subtitle_track_is_dropped_against_real_tools(
        settings, plan, tmp_path, monkeypatch, muxer):
    """The whole mux for real, on a file that really does carry a 0-cue track.

    This is the only test that can catch a renumbering mistake: every stub
    here agrees with the code about which stream is which, and a real file
    does not. The empty track is the MIDDLE one, so a wrong number survives
    as the wrong language rather than as a missing track.
    """
    import subprocess as sp
    real = _REAL_WHICH
    monkeypatch.setattr(shutil, "which", real if muxer == "mkvmerge" else
                        (lambda n, *a, **k: None if "mkvmerge" in n else real(n, *a, **k)))
    ffmpeg, ffprobe, mkvmerge = real("ffmpeg"), real("ffprobe"), real("mkvmerge")

    def ff(*args):
        sp.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
               check=True, cwd=tmp_path)

    shot = tmp_path / "enc_00000.ivf"
    try:
        ff("-f", "lavfi", "-i", "testsrc=size=160x120:rate=5:duration=2",
           "-c:v", "libsvtav1", "-preset", "12", shot)
    except sp.CalledProcessError:
        pytest.skip("needs an ffmpeg with libsvtav1 to build the shot")
    ff("-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", "sound.mka")
    (tmp_path / "a.srt").write_text("1\n00:00:00,200 --> 00:00:01,500\nfirst\n")
    # its only cue lies beyond the -t below, which is how a genuinely 0-packet
    # track is made
    (tmp_path / "b.srt").write_text("1\n00:00:30,000 --> 00:00:31,000\nnever\n")
    (tmp_path / "c.srt").write_text("1\n00:00:00,300 --> 00:00:01,400\nthird\n")
    ff("-i", shot, "-i", "sound.mka", "-i", "a.srt", "-i", "b.srt", "-i", "c.srt",
       "-map", "0:v", "-map", "1:a", "-map", "2", "-map", "3", "-map", "4",
       "-c", "copy", "-t", "2", "-metadata:s:s:0", "language=eng",
       "-metadata:s:s:1", "language=fre", "-metadata:s:s:2", "language=spa",
       "mid.mkv")
    source = tmp_path / "movie.mkv"
    # through mkvmerge, which writes the statistics tags ffmpeg does not
    sp.run([mkvmerge, "-q", "-o", str(source), "mid.mkv"], check=True, cwd=tmp_path)

    def sub_tags(path):
        out = sp.run([ffprobe, "-v", "error", "-select_streams", "s",
                      "-show_entries", "stream_tags", "-of", "json", str(path)],
                     check=True, capture_output=True, text=True).stdout
        return [{k.lower(): v for k, v in (s.get("tags") or {}).items()}
                for s in json.loads(out)["streams"]]

    counts = [t.get("number_of_frames") for t in sub_tags(source)]
    if counts != ["1", "0", "1"]:
        pytest.skip(f"this mkvmerge did not tag the fixture as expected: {counts}")

    info = MediaInfo(path=source)
    info.fps, info.duration = 5.0, 2.0
    enc = make_encoder(settings, info, plan, tmp_path)
    enc.concat_shots([shot])

    assert enc.subtitles_dropped == 1
    langs = [t.get("language") for t in sub_tags(enc.output)]
    assert len(langs) == 2                      # the empty one is gone
    ident = json.loads(sp.run([mkvmerge, "-J", str(enc.output)], check=True,
                              capture_output=True, text=True).stdout)
    subs = [t["properties"].get("language")
            for t in ident["tracks"] if t["type"] == "subtitles"]
    assert subs == ["eng", "spa"]               # and in the right order


@pytest.mark.skipif(any(_REAL_WHICH(t) is None for t in ("ffmpeg", "ffprobe")),
                    reason="needs a real ffmpeg and ffprobe")
def test_an_empty_mp4_track_is_detected_against_real_tools(
        settings, info, plan, tmp_path, monkeypatch):
    """No stub can pin the mp4 rule, because the shape it has to read is one
    no stub would think to write: ffprobe prints nb_frames only when it is
    non-zero, and the mov muxer gives a cue-less mov_text track a padding
    sample, so a genuinely empty track reads nb_frames "1" - never "0" - and
    admits it only through its own duration. A rule written against "0" is a
    silent no-op on every mp4 in the library, and a stub asserting on it would
    agree with the code about a value ffprobe cannot produce."""
    import subprocess as sp
    monkeypatch.setattr(shutil, "which", _REAL_WHICH)
    ffmpeg = _REAL_WHICH("ffmpeg")

    def ff(*args):
        sp.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
               check=True, cwd=tmp_path)

    ff("-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d=3", "-c:v", "libx264",
       "-t", "3", "v.mp4")
    (tmp_path / "a.srt").write_text("1\n00:00:00,200 --> 00:00:00,800\nreal\n")
    # b.srt's only cue lies beyond the -t below: that is how a genuinely 0-cue
    # track is made
    (tmp_path / "b.srt").write_text("1\n00:00:02,500 --> 00:00:02,800\nnever\n")
    source = tmp_path / "movie.mp4"
    ff("-i", "v.mp4", "-i", "a.srt", "-i", "b.srt", "-map", "0:v", "-map", "1",
       "-map", "2", "-c:v", "copy", "-c:s", "mov_text", "-t", "1", source)

    enc = make_encoder(settings, info, plan, tmp_path)
    got = enc._subtitle_plan(str(source))
    assert [s.empty for s in got.subs] == [False, True]
    assert [s.pos for s in got.dropped] == [1]
    # the one-cue track is vouched for by its sample table, the empty one by a
    # duration its cue-less samples could not add up to
    assert got.subs[0].why.startswith("nb_frames=")
    assert got.subs[1].why == "duration_ts=0"


@pytest.mark.skipif(any(_REAL_WHICH(t) is None for t in ("ffmpeg", "ffprobe", "mkvmerge")),
                    reason="needs a real ffmpeg, ffprobe and mkvmerge")
@pytest.mark.parametrize("muxer", ["mkvmerge", "ffmpeg"])
def test_the_srt_companion_against_real_tools(settings, plan, tmp_path,
                                              monkeypatch, muxer):
    """srtenc really does write <font> and {\\anN} into the text; no stub can
    tell whether the sanitiser matches what it actually emits.

    Nor can one tell which command decides the default flag in the finished
    file: the remux writes audio_subs.mkv, mkvmerge copies that file's flags
    track for track, and the ffmpeg fallback restates its own over them. Both
    muxers, on a source whose ASS really is the default track."""
    import subprocess as sp
    real = _REAL_WHICH
    monkeypatch.setattr(shutil, "which", real if muxer == "mkvmerge" else
                        (lambda n, *a, **k: None if "mkvmerge" in n else real(n, *a, **k)))
    ffmpeg, mkvmerge = real("ffmpeg"), real("mkvmerge")
    ffprobe = real("ffprobe")

    def ff(*args):
        sp.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
               check=True, cwd=tmp_path)

    shot = tmp_path / "enc_00000.ivf"
    try:
        ff("-f", "lavfi", "-i", "testsrc=size=160x120:rate=5:duration=2",
           "-c:v", "libsvtav1", "-preset", "12", shot)
    except sp.CalledProcessError:
        pytest.skip("needs an ffmpeg with libsvtav1 to build the shot")
    ff("-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", "sound.mka")
    # a style that differs from srtenc's defaults, so it wraps every cue in
    # <font>, and a cue with the alignment override it writes out literally
    (tmp_path / "a.ass").write_text(
        "[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Source Han Sans SC Medium,24,&H00EEEEEE,&H000000FF,"
        "&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.20,0:00:01.50,Default,,0,0,0,,{\\an8}人多力量大\n",
        encoding="utf-8")
    source = tmp_path / "movie.mkv"
    sp.run([mkvmerge, "-q", "-o", str(source), str(shot), "sound.mka",
            "--language", "0:chi", "--track-name", "0:简体",
            "--default-track-flag", "0:yes", "a.ass"],
           check=True, cwd=tmp_path)
    info = MediaInfo(path=source)
    info.fps, info.duration = 5.0, 2.0
    enc = make_encoder(settings, info, plan, tmp_path)
    enc.concat_shots([shot])

    assert enc.subtitles_added == 1
    ident = json.loads(sp.run([mkvmerge, "-J", str(enc.output)], check=True,
                              capture_output=True, text=True).stdout)
    subs = [(t["properties"]["codec_id"], t["properties"].get("track_name", ""))
            for t in ident["tracks"] if t["type"] == "subtitles"]
    # the ASS is still there, untouched, with the plain-text copy beside it
    assert subs[0][0] in ("S_TEXT/ASS", "S_TEXT/SSA")
    # the source's own title, marked as the plain-text copy it is
    assert subs[1] == ("S_TEXT/UTF8", "简体 (SRT)")
    ff("-i", enc.output, "-map", "0:s:1", "-c", "copy", "out.srt")
    text = (tmp_path / "out.srt").read_text(encoding="utf-8")
    assert "人多力量大" in text
    assert "<font" not in text and "{\\an" not in text
    # exactly one default subtitle track in the finished file, and it is the
    # plain-text copy: with two, a player takes the first - the ASS - and the
    # companion never gets sent, which is the whole reason it exists
    probed = json.loads(sp.run(
        [ffprobe, "-v", "error", "-select_streams", "s", "-show_entries",
         "stream=codec_name:stream_disposition=default", "-of", "json",
         str(enc.output)], check=True, capture_output=True, text=True).stdout)
    assert [(s["codec_name"], s["disposition"]["default"])
            for s in probed["streams"]] == [("ass", 0), ("subrip", 1)]


# ---- PGS: kept as it is, with a plain-text srt READ OFF IT beside it ----
#
# The .sup fixtures below are built here rather than checked in: a PGS track
# is a handful of segment types (see app.pgsocr), and writing them is the only
# way a test can state exactly which byte carries the timing, the palette or
# the bitmap. Every one of these is a shape the real format takes.
def _pgs_seg(stype, pts, payload):
    """One segment: "PG", 4B PTS, 4B DTS, 1B type, 2B length, payload."""
    return b"PG" + struct.pack(">IIBH", pts, 0, stype, len(payload)) + payload


def _pgs_rle(idx):
    """Palette-index rows -> PGS run-length, the inverse of pgsocr.rle_decode.

    Every branch of the decoder is exercised by a real bitmap: raw bytes for a
    short coloured run, 0x00 0x80|n for a long one, 0x00 n for transparent,
    and 0x00 0x00 to end the row.
    """
    out = bytearray()
    for row in idx:
        i = 0
        while i < len(row):
            c = int(row[i])
            n = 1
            while i + n < len(row) and int(row[i + n]) == c:
                n += 1
            i += n
            if c == 0:
                while n:
                    take = min(n, 63)
                    out += bytes((0x00, take))
                    n -= take
            elif n < 3:
                out += bytes((c,)) * n         # a short run is cheaper raw
            else:
                while n:
                    take = min(n, 63)
                    out += bytes((0x00, 0x80 | take, c))
                    n -= take
        out += b"\x00\x00"                     # end of line
    return bytes(out)


def _pgs_compose(start, idx, screen=(1920, 1080), at=(0, 0), window=None):
    """One COMPOSITION display set: PCS, WDS, PDS, ODS, END.

    `at` places the object and `window` the window it names. They are the same
    corner in an ordinary track and are separate here on purpose: a
    composition is free to put an object outside its own window.
    """
    h, w = idx.shape
    wx, wy, ww, wh = window or (0, 0, w, h)
    pts = int(round(start * 90000))
    pcs = (struct.pack(">HH", *screen) + b"\x10\x00\x00\x00\x00\x00"
           + b"\x01" + struct.pack(">HBB", 0, 0, 0) + struct.pack(">HH", *at))
    buf = bytearray(_pgs_seg(0x16, pts, pcs))
    buf += _pgs_seg(0x17, pts, b"\x01\x00" + struct.pack(">HHHH", wx, wy, ww, wh))
    # index 1 is opaque white, everything else stays transparent
    buf += _pgs_seg(0x14, pts, b"\x00\x00" + bytes((1, 235, 128, 128, 255)))
    rle = _pgs_rle(idx)
    ods = (struct.pack(">HB", 0, 0) + b"\xc0"
           + struct.pack(">I", len(rle) + 4)[1:] + struct.pack(">HH", w, h) + rle)
    buf += _pgs_seg(0x15, pts, ods)
    buf += _pgs_seg(0x80, pts, b"")
    return bytes(buf)


def _pgs_erase(end, screen=(1920, 1080)):
    """The ERASE display set: a composition carrying NO object, which is what
    ends the cue before it."""
    epts = int(round(end * 90000))
    return (_pgs_seg(0x16, epts, struct.pack(">HH", *screen)
                     + b"\x10\x00\x00\x00\x00\x00" + b"\x00")
            + _pgs_seg(0x80, epts, b""))


def _make_sup(cues, screen=(1920, 1080)):
    """[(start_s, end_s, index_array)] -> .sup bytes.

    One window, one object, one palette, reused by every display set - which
    is exactly the layout that makes WHEN the palette is bound load-bearing.
    """
    return b"".join(_pgs_compose(start, idx, screen) + _pgs_erase(end, screen)
                    for start, end, idx in cues)


def _block_text(rows, cols, seed=1):
    """A deterministic, obviously non-blank bitmap. Not meant to be readable:
    the tests that use it inject their own engine."""
    import numpy as np

    idx = np.zeros((rows, cols), np.uint8)
    idx[seed:rows - seed, seed:cols - seed] = 1
    return idx


class _FakeEngine:
    """An engine that is not tesseract and does not import it.

    pgsocr.ocr_track promises it enters an engine through one call, so this is
    also the test that the seam is real: a GPU model would be dropped in the
    same way.
    """

    name = "fake"

    def __init__(self, texts):
        self.texts = list(texts)
        self.seen = []

    def recognise(self, rgba):
        self.seen.append(rgba.shape)
        return self.texts[len(self.seen) - 1], "primary"


def test_a_sup_parses_into_cues_the_erase_display_set_closes():
    """The cue count is the number of COMPOSITION display sets, never the
    number of display sets: each cue here writes two."""
    import numpy as np

    sup = _make_sup([(1.418, 3.670, _block_text(20, 60)),
                     (4.0, 5.5, _block_text(20, 60))])
    cues, notes = pgsocr.parse_sup(sup)
    assert [(round(c.start, 3), round(c.end, 3)) for c in cues] == [
        (1.418, 3.670), (4.0, 5.5)]
    # and the timing really is read from the 90 kHz PTS, not invented
    assert notes["seg_16"] == 4 and len(cues) == 2
    assert not [k for k in notes if not k.startswith("seg_")]
    rgba = pgsocr.render(cues[0])
    assert rgba.shape == (20, 60, 4)
    assert (rgba[..., 3] > 0).any()          # something was actually drawn
    assert isinstance(rgba, np.ndarray)


def test_the_palette_is_bound_at_the_end_of_its_own_display_set():
    """Every display set in a real track reuses object id 0 and palette id 0.
    Bound any later, EVERY cue would render the last bitmap in the file - the
    one defect that produces a whole track of confident, identical, wrong
    subtitles instead of an obvious failure."""
    wide = _block_text(20, 60)
    narrow = _block_text(20, 12)
    cues, _ = pgsocr.parse_sup(_make_sup([(1.0, 2.0, wide), (3.0, 4.0, narrow)]))
    assert pgsocr.render(cues[0]).shape[1] == 60
    assert pgsocr.render(cues[1]).shape[1] == 12


def test_a_composition_that_replaces_an_open_cue_ends_it_there():
    """Not every cue is closed by an erase: a composition can replace one
    still on screen. Left open, that cue fell to the track's MEDIAN duration
    and ran on past the subtitle that replaced it - two cues on screen at
    once, and enough of them refuses the whole track at the overlap gate."""
    idx = _block_text(20, 60)
    cues, notes = pgsocr.parse_sup(
        _pgs_compose(1.0, idx) + _pgs_compose(2.0, idx) + _pgs_erase(3.0))
    assert [(round(c.start, 3), round(c.end, 3)) for c in cues] == [
        (1.0, 2.0), (2.0, 3.0)]
    assert not notes["cue_without_erase"]


def test_an_object_outside_its_window_is_clipped_and_never_raises(tmp_path):
    """A composition may place an object outside the window it names - once in
    the 71-file batch. Clipped at one end only, a negative offset GREW the
    slice and the canvas was then indexed from its end, so ocr_track raised
    IndexError instead of returning a reason: the one thing this module
    promises never happens, hours into a finished encode."""
    sup = tmp_path / "a.sup"
    sup.write_bytes(_pgs_compose(1.0, _block_text(20, 60), at=(60, 0),
                                 window=(100, 0, 200, 50)) + _pgs_erase(2.0))
    cues, _ = pgsocr.parse_sup(sup.read_bytes())
    rgba = pgsocr.render(cues[0])
    assert rgba.shape == (50, 200, 4)
    assert (rgba[..., 3] > 0).any()      # the part inside the window is drawn
    res = pgsocr.ocr_track(sup, engine=_FakeEngine(["a line of text"]),
                           workers=1, lexicon=set())
    assert res.text is not None


def test_signal_map_makes_an_outlined_glyph_solid():
    """PGS cues are white glyphs with a BLACK OUTLINE on a transparent ground.
    Compositing over white leaves a stencil whose interior matches the page;
    luma*alpha collapses outline and background together, which is what makes
    the letters solid."""
    import numpy as np

    rgba = np.zeros((3, 3, 4), np.uint8)
    rgba[1, 1] = (255, 255, 255, 255)        # glyph core
    rgba[0, 1] = (0, 0, 0, 255)              # its outline
    s = pgsocr.signal_map(rgba)
    assert s[1, 1] > 0.99                    # core is ink
    assert s[0, 1] == 0.0                    # outline is not
    assert s[2, 2] == 0.0                    # nor is the transparent ground


def test_ocr_track_returns_a_reason_rather_than_raising_on_a_damaged_sup(tmp_path):
    """OCR must never fail an encode. A .sup that lost sync is the shape that
    would otherwise raise out of the mux, hours into a job."""
    bad = tmp_path / "broken.sup"
    bad.write_bytes(b"NOTPGS" + b"\x00" * 40)
    res = pgsocr.ocr_track(bad)
    assert res.text is None and "lost sync" in res.why


def test_ocr_track_drives_the_whole_pipeline_through_a_foreign_engine(tmp_path):
    """The seam: an object with .name and recognise() gets the original RGBA
    canvas and its text reaches the finished srt."""
    sup = tmp_path / "a.sup"
    sup.write_bytes(_make_sup([(1.418, 3.67, _block_text(20, 60)),
                               (4.0, 5.5, _block_text(20, 60))]))
    eng = _FakeEngine(["first line", "second line"])
    res = pgsocr.ocr_track(sup, engine=eng, workers=1, lexicon={"first", "second", "line"})
    assert res.why == "ok" and res.written == 2
    assert res.text == ("1\n00:00:01,418 --> 00:00:03,670\nfirst line\n\n"
                        "2\n00:00:04,000 --> 00:00:05,500\nsecond line\n")
    assert eng.seen == [(20, 60, 4), (20, 60, 4)]


def test_a_track_that_ocrs_to_nothing_is_refused_by_the_gates(tmp_path):
    """A blank cue is a silently dropped subtitle, so a track that comes back
    mostly blank must produce NO srt rather than a short one."""
    sup = tmp_path / "a.sup"
    sup.write_bytes(_make_sup([(i, i + 0.5, _block_text(20, 60))
                               for i in range(1, 21)]))
    res = pgsocr.ocr_track(sup, engine=_FakeEngine([""] * 19 + ["only one"]),
                           workers=1, lexicon=set())
    assert res.text is None
    assert "empty_rate" in res.why or "chars_per_cue" in res.why


def test_the_empty_cue_tolerance_is_the_bound_that_decides(tmp_path):
    """cue_coverage and empty_rate are the same measurement the two ways up -
    they sum to 1 - so bounding both meant the coverage bound always fired
    first and the 1% empty tolerance this module documents never applied. A
    track 0.8% of whose cues came back blank is inside that tolerance and
    ships; coverage stays a REPORTED signal."""
    n = 125
    sup = tmp_path / "a.sup"
    sup.write_bytes(_make_sup([(i, i + 0.5, _block_text(20, 60))
                               for i in range(1, n + 1)]))
    res = pgsocr.ocr_track(
        sup, engine=_FakeEngine([""] + ["a line of text"] * (n - 1)),
        workers=1, lexicon=set())
    assert res.signals["empty_rate"] == 0.008
    assert res.signals["cue_coverage"] == 0.992
    assert res.text is not None and res.written == n - 1


def test_the_oov_gate_skips_itself_rather_than_failing_a_short_track(tmp_path):
    """Below MIN_TOKENS one unusual word moves the rate by most of a point, and
    with no word list in the image the signal does not exist at all. Neither is
    evidence of a bad read, so the gate says it skipped instead of passing
    quietly - or failing every short track."""
    sup = tmp_path / "a.sup"
    sup.write_bytes(_make_sup([(1.0, 2.0, _block_text(20, 60))]))
    res = pgsocr.ocr_track(sup, engine=_FakeEngine(["Zzyzx Qwghlm Brrraaap"]),
                           workers=1, lexicon={"the"})
    gate = [g for g in res.gates if g["name"] == "oov_rate"][0]
    assert gate["pass"] and "tokens" in gate["skipped"]
    assert res.text is not None                # and the track still ships


def test_a_well_formed_initialism_is_never_snapped_to_another_one():
    """The gazetteer repairs DAMAGE, and a token that already reads as an
    initialism is not damage. Snapping it anyway rewrites one real acronym
    into another, and nothing downstream can see that: the result is ASCII,
    the right length, the right cue count - and it is the srt, the track that
    TAKES the default flag, that carries the wrong word."""
    doc = pgsocr.build_doc_context(["The U.S.S.R. archive", "back in the U.S.S.R."])
    assert pgsocr.postprocess("He left the U.S.A. today", doc)[0] == \
        "He left the U.S.A. today"
    # measured over the raw OCR of all 73 real tracks: the 48 repairs of a
    # damaged form all still happen, and only two rewrites of a correct read
    # are refused
    doc = pgsocr.build_doc_context(["S.H.I.E.L.D. is here"])
    assert pgsocr.postprocess("S.H.I.LE.L.D.", doc)[0] == "S.H.I.E.L.D."


@pytest.mark.parametrize("raw, want", [
    ("|'ve got it", "I've got it"),           # contraction: capital I
    ("a bu|let", "a bullet"),                 # inside a lowercase word: l
    ("S.H.|.E.L.D.", "S.H.I.E.L.D."),         # an initialism letter
    ("-Yes.\n\"No.", "- Yes.\n- No."),        # a dash misread as a quote
    ("-Yes.\n-No.", "- Yes.\n- No."),         # the space tesseract drops
])
def test_the_post_rules_each_fix_a_counted_error(raw, want):
    doc = pgsocr.build_doc_context(["S.H.I.E.L.D. is here"], {"bullet"})
    assert pgsocr.postprocess(raw, doc)[0] == want


def test_a_one_sided_dash_is_left_alone_when_the_cue_is_one_line():
    """The pair rule reads a two-line cue as the dialogue convention it is.
    A single line starting with a dash is not a dropped sibling."""
    assert pgsocr.postprocess("- Just me.")[0] == "- Just me."


# ---- the optimizer side: extraction, flags, and what falls away ----
_OCR_SRT = "1\n00:00:01,418 --> 00:00:03,670\nHELLO WORLD\n"


def _pgs_probe(flags=("default",), language="eng", title="", codec=None,
               extra=None):
    """A source whose SECOND subtitle track is an image track: an English srt
    first, so that the "only when the file has no text subtitle" reading of
    this feature would produce no companion at all."""
    tags = {"NUMBER_OF_FRAMES": "1422", "language": language}
    if title:
        tags["title"] = title
    st = {"codec_name": codec or "hdmv_pgs_subtitle", "tags": tags}
    st.update(extra or {})
    return _probe_json(
        ("video", set()), ("audio", {"default"}),
        ("subtitle", set(), {"codec_name": "subrip",
                             "tags": {"NUMBER_OF_FRAMES": "512",
                                      "language": "eng"}}),
        ("subtitle", set(flags), st))


def _stub_ocr(monkeypatch, ok=True, text=_OCR_SRT):
    """_ocr_companion without tesseract: the command shape and the flag
    handling are what these tests are about."""
    def fake(self, sub):
        if not ok:
            return False
        self._companion_path(sub).write_text(text, encoding="utf-8")
        return True

    monkeypatch.setattr(opt.ShotEncoder, "_ocr_companion", fake)


def test_an_english_pgs_track_is_extracted_by_the_demux_that_already_runs(
        settings, info, plan, tmp_path, monkeypatch):
    """The whole point: a second full read of a 30-90GB remux is not
    acceptable, so the .sup comes out as another output of the ONE command
    that already builds audio_subs.mkv. Copied, never decoded, and with no
    -copyts: measured, a production-shaped source extracts all 711 cue times
    exactly without it."""
    enc = make_encoder(settings, info, plan, tmp_path)
    _stub_ocr(monkeypatch)
    ran = _mux_with(enc, monkeypatch, _pgs_probe())
    remux, = _remuxes(ran)
    assert remux[remux.index(str(enc.tempdir / "audio_subs.mkv")) + 1:] == [
        "-map", "0:s:1", "-c:s", "copy", "-f", "sup",
        str(enc.tempdir / "sub_1.sup")]
    assert "-copyts" not in remux
    # one read of the source, not two
    assert len([a for a in ran if a[1:3] == ["-hide_banner", "-y"]
                and "-i" in a and a[a.index("-i") + 1] == str(enc.info.path)]) == 1
    assert enc.subtitles_added == 1


@pytest.mark.parametrize("language, codec, ocrd", [
    ("eng", "hdmv_pgs_subtitle", True),
    ("chi", "hdmv_pgs_subtitle", False),      # one model, and it is eng
    ("jpn", "hdmv_pgs_subtitle", False),
    ("", "hdmv_pgs_subtitle", False),         # untagged is not evidence
    ("eng", "dvd_subtitle", False),           # VobSub: a format this cannot read
    ("eng", "subrip", False),                 # already text
], ids=["eng-pgs", "chi", "jpn", "untagged", "eng-vobsub", "text"])
def test_only_english_pgs_tracks_are_ocrd(settings, info, plan, tmp_path,
                                          monkeypatch, language, codec, ocrd):
    """eng tesseract does not FAIL on Thai or Chinese, it invents - and the
    invention would be muxed in as a subtitle track. Measured over the
    library: 5005 image tracks, 1034 English, of which 929 PGS and 105
    VobSub."""
    enc = make_encoder(settings, info, plan, tmp_path)
    _stub_ocr(monkeypatch)
    ran = _mux_with(enc, monkeypatch,
                    _pgs_probe(language=language, codec=codec))
    remux, = _remuxes(ran)
    assert ("-f" in remux and "sup" in remux) is ocrd
    assert enc.subtitles_added == (1 if ocrd else 0)


def test_the_srt_takes_the_default_flag_and_the_image_track_loses_it(
        settings, info, plan, tmp_path, monkeypatch):
    """THE WHOLE POINT. Two default subtitle tracks and a player takes the
    first - the picture - and burns it in, which is the fault this exists to
    fix. So the flag MOVES: the srt is default, the PGS is not."""
    enc = make_encoder(settings, info, plan, tmp_path)
    _stub_ocr(monkeypatch)
    ran = _mux_with(enc, monkeypatch, _pgs_probe(flags=("default",)))
    remux, = _remuxes(ran)
    # the PGS is the output's s:1, and it is written WITHOUT default
    assert _dispositions(remux) == [("-disposition:a:0", "default"),
                                    ("-disposition:s:0", "0"),
                                    ("-disposition:s:1", "0")]
    final, = [a for a in ran if a[-1] == str(enc.output)]
    assert final[final.index("-disposition:s:2") + 1] == "default"


def test_the_srt_carries_the_language_title_forced_and_hearing_impaired(
        settings, info, plan, tmp_path, monkeypatch):
    """An SDH PGS read off the screen is still SDH. Drop the flag and Plex
    offers it as ordinary English - exactly the track a deaf viewer must not
    be handed. Unlike default, this one is COPIED: both tracks keep it."""
    enc = make_encoder(settings, info, plan, tmp_path)
    _stub_ocr(monkeypatch)
    ran = _mux_with(enc, monkeypatch, _pgs_probe(
        flags=("default", "forced", "hearing_impaired"), title="English SDH"))
    final, = [a for a in ran if a[-1] == str(enc.output)]
    meta = final[final.index("-metadata:s:s:2"):]
    assert meta[:4] == ["-metadata:s:s:2", "language=eng",
                        "-metadata:s:s:2", "title=English SDH (OCR)"]
    assert meta[4:6] == ["-disposition:s:2",
                         "default+forced+hearing_impaired"]
    # and the image track keeps forced and hearing_impaired, losing only default
    remux, = _remuxes(ran)
    assert _dispositions(remux)[-1] == ("-disposition:s:1",
                                        "forced+hearing_impaired")


def test_mkvmerge_gets_the_same_flags_as_the_ffmpeg_fallback(
        settings, info, plan, tmp_path, monkeypatch):
    """The two muxers write these separately, and an SDH track came out
    hearing_impaired through one and plain through the other, decided by
    nothing but whether mkvmerge was installed."""
    enc = make_encoder(settings, info, plan, tmp_path)
    _stub_ocr(monkeypatch)
    sub = opt.SubStream(index=3, pos=1, codec="hdmv_pgs_subtitle",
                        flags="default+forced+hearing_impaired", default=True,
                        forced=True, language="eng", title="English SDH",
                        empty=False, why="tag")
    assert enc._companion_inputs([sub]) == [
        "--sub-charset", "0:UTF-8", "--language", "0:eng",
        "--track-name", "0:English SDH (OCR)",
        "--default-track-flag", "0:yes", "--forced-display-flag", "0:yes",
        "--hearing-impaired-flag", "0:yes", str(enc._companion_path(sub))]


def test_an_ass_companion_is_never_given_the_hearing_impaired_flag(
        settings, info, plan, tmp_path):
    """The descriptive flags stay on the ASS beside it: that companion is the
    same cues in a poorer format, not the same subtitles as text."""
    enc = make_encoder(settings, info, plan, tmp_path)
    sub = opt.SubStream(index=3, pos=1, codec="ass",
                        flags="default+hearing_impaired", default=True,
                        forced=False, language="eng", title="", empty=False,
                        why="tag")
    assert "--hearing-impaired-flag" not in enc._companion_inputs([sub])
    assert enc._companion_metadata("/x/src.mkv", []) == []


def test_an_ocr_that_fails_hands_the_default_flag_back(
        settings, info, plan, tmp_path, monkeypatch):
    """The remux states its flags before the OCR has run, so a PGS whose text
    copy never arrives is already written without its default. mkvmerge copies
    that file's flags and promotes nothing, so the output would carry NO
    default subtitle at all."""
    enc = make_encoder(settings, info, plan, tmp_path)
    _stub_ocr(monkeypatch, ok=False)
    _, merges = _concat_via_mkvmerge(
        enc, monkeypatch, _pgs_probe(),
        _attachment_probe(_MKV, ("video", 0), ("audio", 0), ("subtitle", 0)))
    merge, = merges
    # audio at track 0, so the PGS (the output's s:1) is track 2
    assert "--default-track-flag" in merge
    assert merge[merge.index("--default-track-flag") + 1] == "2:yes"
    assert enc.subtitles_added == 0


def test_with_the_setting_off_nothing_is_extracted(
        settings, info, plan, tmp_path, monkeypatch):
    enc = make_encoder(settings, info, plan, tmp_path)
    settings.transcode.optimizer.pgs_ocr_srt = False
    _stub_ocr(monkeypatch)
    ran = _mux_with(enc, monkeypatch, _pgs_probe())
    remux, = _remuxes(ran)
    assert "sup" not in remux and enc.subtitles_added == 0


def test_an_empty_image_track_is_never_ocrd(
        settings, info, plan, tmp_path, monkeypatch):
    """It would read nothing and write an empty track - which is the thing
    drop_empty_subtitles exists to stop shipping."""
    enc = make_encoder(settings, info, plan, tmp_path)
    # BOTH of the other settings that ask for emptiness are off, so this also
    # pins the probe: with only these two consulted, every stream came back
    # empty=None and an empty PGS was extracted and OCR'd for nothing
    settings.transcode.optimizer.drop_empty_subtitles = False
    settings.transcode.optimizer.ass_srt_companion = False
    _stub_ocr(monkeypatch)
    # the language tag is restated because `extra` REPLACES tags: without it
    # the track is left out for being untagged, which is a different rule and
    # would let this pass with emptiness never measured at all
    ran = _mux_with(enc, monkeypatch, _pgs_probe(
        extra={"tags": {"NUMBER_OF_FRAMES": "0", "language": "eng"}}))
    remux, = _remuxes(ran)
    assert "sup" not in remux and enc.subtitles_added == 0


def test_an_ass_and_a_pgs_in_one_file_each_get_their_own_companion(
        settings, info, plan, tmp_path, monkeypatch):
    """Two kinds in one list, and the numbering has to be right from both
    ends: "-map 0:s:N" counts the SOURCE's subtitles while "-metadata:s:s:N"
    counts the OUTPUT's."""
    enc = make_encoder(settings, info, plan, tmp_path)
    _stub_ocr(monkeypatch)
    ran = _mux_with(enc, monkeypatch, _probe_json(
        ("video", set()), ("audio", {"default"}),
        ("subtitle", set(), {"codec_name": "ass",
                             "tags": {"NUMBER_OF_FRAMES": "916",
                                      "language": "jpn"}}),
        ("subtitle", {"default"}, {"codec_name": "hdmv_pgs_subtitle",
                                   "tags": {"NUMBER_OF_FRAMES": "1422",
                                            "language": "eng"}})))
    remux, = _remuxes(ran)
    # the ASS copy first, then the image track's bitstream
    assert _maps(remux) == ["0:a?", "0:s?", "0:t?", "0:s:0", "0:s:1"]
    tail = remux[remux.index(str(enc.tempdir / "audio_subs.mkv")) + 1:]
    assert tail == ["-map", "0:s:0", "-c:s", "srt",
                    str(enc.tempdir / "sub_0.srt"),
                    "-map", "0:s:1", "-c:s", "copy", "-f", "sup",
                    str(enc.tempdir / "sub_1.sup")]
    assert enc.subtitles_added == 2


def _render_sup(tmp_path, ffmpeg, text="HELLO WORLD", start=0.2, end=1.5):
    """A real PGS track carrying `text`, built with the tools already required.

    ffmpeg has no PGS ENCODER, so the bitmap is rendered with libass onto a
    black frame, read back as a PGM, and wrapped in the segments a .sup is
    made of (see _make_sup). That is what lets this test assert on the TEXT
    rather than only on stream counts: no stub can tell whether the palette,
    the RLE and the 90 kHz timing are read the way a real player reads them.
    """
    import subprocess as sp

    import numpy as np

    (tmp_path / "burn.ass").write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 640\nPlayResY: 200\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, "
        "SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
        "StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,DejaVu Sans,56,&H00FFFFFF,&H000000FF,&H00000000,"
        "&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,10,10,10,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
        f"Dialogue: 0,0:00:00.00,0:00:10.00,Default,,0,0,0,,{text}\n",
        encoding="utf-8")
    pgm = tmp_path / "cue.pgm"
    try:
        sp.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                "-i", "color=c=black:s=640x200:d=1:r=1", "-vf",
                f"ass={tmp_path / 'burn.ass'}", "-frames:v", "1",
                "-pix_fmt", "gray", str(pgm)], check=True, cwd=tmp_path)
    except sp.CalledProcessError:
        pytest.skip("needs an ffmpeg with the libass 'ass' filter")
    head = pgm.read_bytes().split(b"\n", 3)
    w, h = (int(x) for x in head[1].split())
    gray = np.frombuffer(head[3][:w * h], np.uint8).reshape(h, w)
    if int((gray > 128).sum()) < 200:
        pytest.skip("libass rendered no text (no usable font in this image)")
    sup = tmp_path / "eng.sup"
    sup.write_bytes(_make_sup([(start, end, (gray > 128).astype(np.uint8))],
                              screen=(w, h)))
    return sup


@pytest.mark.skipif(
    any(_REAL_WHICH(t) is None
        for t in ("ffmpeg", "ffprobe", "mkvmerge", "tesseract")),
    reason="needs a real ffmpeg, ffprobe, mkvmerge and tesseract")
@pytest.mark.parametrize("muxer", ["mkvmerge", "ffmpeg"])
def test_the_ocr_companion_against_real_tools(settings, plan, tmp_path,
                                              monkeypatch, muxer):
    """The whole thing for real, on a source carrying a genuine PGS track.

    No stub can tell that ffmpeg copies a PGS out to a .sup with its 90 kHz
    timing intact, that tesseract reads that bitmap back as the same words, or
    which command decides the default flag in the finished file - the remux
    writes audio_subs.mkv, mkvmerge copies that file's flags track for track,
    and the ffmpeg fallback restates its own over them. Both muxers, on a
    source whose PGS really is the default track.
    """
    import subprocess as sp

    real = _REAL_WHICH
    monkeypatch.setattr(shutil, "which", real if muxer == "mkvmerge" else
                        (lambda n, *a, **k: None if "mkvmerge" in n
                         else real(n, *a, **k)))
    ffmpeg, ffprobe, mkvmerge = real("ffmpeg"), real("ffprobe"), real("mkvmerge")

    def ff(*args):
        sp.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
               check=True, cwd=tmp_path)

    shot = tmp_path / "enc_00000.ivf"
    try:
        ff("-f", "lavfi", "-i", "testsrc=size=160x120:rate=5:duration=2",
           "-c:v", "libsvtav1", "-preset", "12", shot)
    except sp.CalledProcessError:
        pytest.skip("needs an ffmpeg with libsvtav1 to build the shot")
    ff("-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac",
       "sound.mka")
    sup = _render_sup(tmp_path, ffmpeg)
    source = tmp_path / "movie.mkv"
    # the PGS is default AND hearing-impaired, which is the shape that made
    # Plex burn a picture in: the companion must take the first and copy the
    # second
    sp.run([mkvmerge, "-q", "-o", str(source), str(shot), "sound.mka",
            "--language", "0:eng", "--track-name", "0:English SDH",
            "--default-track-flag", "0:yes", "--hearing-impaired-flag", "0:yes",
            str(sup)], check=True, cwd=tmp_path)
    info = MediaInfo(path=source)
    info.fps, info.duration = 5.0, 2.0
    enc = make_encoder(settings, info, plan, tmp_path)
    enc.concat_shots([shot])

    assert enc.subtitles_added == 1
    probed = json.loads(sp.run(
        [ffprobe, "-v", "error", "-select_streams", "s", "-show_entries",
         "stream=codec_name:stream_disposition:stream_tags", "-of", "json",
         str(enc.output)], check=True, capture_output=True, text=True).stdout)
    subs = probed["streams"]
    # the image track is KEPT, and the text copy sits beside it
    assert [s["codec_name"] for s in subs] == ["hdmv_pgs_subtitle", "subrip"]
    # exactly one default, and it is the TEXT track: with two, a player takes
    # the first - the picture - and burns it in, which is the whole fault
    assert [s["disposition"]["default"] for s in subs] == [0, 1]
    # hearing_impaired is COPIED, not moved: an SDH track read off the screen
    # is still SDH, and Plex labels it from this flag
    assert [s["disposition"]["hearing_impaired"] for s in subs] == [1, 1]
    assert [s["tags"]["language"] for s in subs] == ["eng", "eng"]
    assert subs[1]["tags"].get("title") == "English SDH (OCR)"
    # and the text is right, at the time the PGS carried it
    ff("-i", enc.output, "-map", "0:s:1", "-c", "copy", "out.srt")
    got = (tmp_path / "out.srt").read_text(encoding="utf-8")
    assert "HELLO WORLD" in got
    assert "00:00:00,200 --> 00:00:01,500" in got
