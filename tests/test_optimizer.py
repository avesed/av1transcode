import json
import shutil
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
        # the SYCL backend announces itself on stderr, which _run folds in;
        # the preflight refuses the GPU without it
        if "sycl_device=" in " ".join(args):
            return "[vmaf-sycl] timing: 30 frames, gpu%=100%"
        return ""

    enc._run = fake_run.__get__(enc)
    enc._probe_shot(0, 0, 90, [28], lp=4)
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


def _source_read(cmd, source):
    """The `-ss X -t D -i <source>` slice of a probe command."""
    i = next(k for k, a in enumerate(cmd) if a == str(source))
    return cmd[max(0, i - 5):i + 1]


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
    assert _source_read(encode_cmd, enc.source) == _source_read(vmaf_cmd, enc.source)
    # and it is a real read, not two empty slices comparing equal
    assert "-ss" in _source_read(encode_cmd, enc.source)


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
            Path(args[-1]).write_bytes(b"YUV4MPEG2 dummy")
            return ""
        # probe/final encode -> write ivf
        if "-f" in args and args[args.index("-f") + 1] == "ivf" and "libsvtav1" in args:
            Path(args[-1]).write_bytes(b"ivf-dummy")
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
            Path(args[-1]).write_bytes(b"shard")
        elif any("libvmaf=" in a for a in args):
            lavfi = args[args.index("-lavfi") + 1]
            log_path = lavfi.split("log_path=")[1].split(":")[0]
            Path(log_path).write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 92.0}}}))
        elif "-f" in args and args[args.index("-f") + 1] == "ivf":
            Path(args[-1]).write_bytes(b"ivf")
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
    """1080p CPU scoring is already ~1.8s; the GPU's per-call overhead is not
    amortised there, and the warm number has never been measured."""
    settings.transcode.optimizer.vmaf_sycl_device = 0
    info.width, info.height = 1920, 1080
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
            Path(args[-1]).write_bytes(b"shard")
            return ""
        if "app.vsmetrics" in args:
            return "93.512\n"
        Path(args[-1]).write_bytes(b"ivf")
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

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        if any("xpsnr" in a for a in args):
            return ("[Parsed_xpsnr_2 @ 0x1] XPSNR  y: 43.6691  u: 49.5918  "
                    "v: 51.1706  (minimum: 43.6691)\n")
        Path(args[-1]).write_bytes(b"ivf")
        return ""

    enc._run = fake_run.__get__(enc)
    score = enc._probe_shot(0, 0, 90, [32], lp=4)[32]
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
        enc._probe_shot(0, 0, 90, [32], lp=4)


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


# ---- fractional CRF must reach SVT-AV1, not ffmpeg's integer -crf option ----
def _encode_cmd(enc, crf):
    cmds = []

    def fake_run(self, args, timeout=None):
        args = [str(a) for a in args]
        cmds.append(args)
        Path(args[-1]).write_bytes(b"ivf")
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
            Path(args[-1]).write_bytes(b"shard")
        elif any("libvmaf=" in a for a in args):
            lavfi = args[args.index("-lavfi") + 1]
            log = lavfi.split("log_path=")[1].split(":")[0]
            Path(log).write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 90.0}}}))
        elif "-f" in args and args[args.index("-f") + 1] == "ivf":
            Path(args[-1]).write_bytes(b"ivf")
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
        Path(args[-1]).write_bytes(b"ivf")
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
        Path(args[-1]).write_bytes(b"ivf")
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
        Path(args[-1]).write_bytes(b"ivf")
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
        Path(args[-1]).write_bytes(b"win")
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
            if "stream=codec_type" in args:
                return "video\n"           # video only: nothing to remux
            return ""
        Path(args[-1]).write_bytes(b"\x1aE\xdf\xa3")
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
            if "stream=codec_type" in args:
                return "video\naudio\n"    # an audio stream is present
            return ""                      # ... but no subtitle streams
        Path(args[-1]).write_bytes(b"\x1aE\xdf\xa3")
        return ""

    enc._run = fake_run.__get__(enc)
    enc.concat_shots([tmp_path / "enc_00000.ivf"])

    muxes = [a for a in ran if "ffprobe" not in a[0]]
    assert any("audio_subs.mkv" in " ".join(a) for a in muxes)
    final = muxes[-1]
    assert "1:a?" in final and "1:s?" in final


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
