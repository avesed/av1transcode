"""Netflix-style shot-based encoding engine.

This is an alternative to the av1an pipeline. Instead of letting av1an probe
chunks for --target-quality SERIALLY (one chunk fully probed before the next),
this engine:

  scenedetect  - splits the source into shots with PySceneDetect
  probing      - probes EVERY shot at a grid of CRFs in PARALLEL (bounded
                 worker pool), computing the quality metric per (shot, crf)
  optimizing   - interpolates a fine-grained per-shot CRF that hits the
                 target_quality (instead of av1an's 4-probe binary search)
  encoding     - encodes all shots in PARALLEL with their chosen CRF
  concat       - concatenates the per-shot encodes and re-muxes audio/subs

The metric is VMAF or SSIMULACRA2 (both higher-is-better, 0-100), run through
ffmpeg's libvmaf filter, so both go through identical plumbing.
"""

from __future__ import annotations

import contextlib
import glob
import itertools
import json
import math
import os
import re
import shutil
import statistics
import sys
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

from loguru import logger

from app import sysres
from app.analyzer import MediaInfo
from app.config import Settings, VideoParams
from app.decisions import TranscodePlan
from app.transcoder import TranscodeError

Shot = Tuple[int, int]  # (start_frame, end_frame), end exclusive
ProbeSamples = Dict[int, Dict[int, float]]  # shot_idx -> {crf -> score}

# ---- pure helpers ---------------------------------------------------------


def parse_target(target: str) -> Tuple[float, Optional[float]]:
    """Parse target_quality into (floor, ceiling). '75' -> (75, None);
    '75-85' -> (75, 85). The engine targets the FLOOR (guarantee at least
    that quality); the ceiling (if given) is informational."""
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(?:[-–]\s*(\d+(?:\.\d+)?))?\s*$", target)
    if not m:
        raise TranscodeError(f"invalid target_quality: {target!r} (expected e.g. '75' or '75-85')")
    lo = float(m.group(1))
    hi = float(m.group(2)) if m.group(2) else None
    if hi is not None and hi < lo:
        raise TranscodeError(f"invalid target_quality range: {target!r}")
    return lo, hi


def pick_crf(samples: List[Tuple[int, float]], target: float) -> float:
    """Pick the CRF whose quality score is closest to `target` by linear
    interpolation on the sampled (crf, score) points.

    scores must be monotone-decreasing in crf (as VMAF/ssimulacra2 are).
    Clamps to the sampled range when the target is out of reach.

    Know what this inversion costs. Measured on 4K at native resolution with
    the 4k model, the curve runs 0.14-0.36 VMAF per CRF, and it is FLATTEST at
    the top - 0.14 around VMAF 96, 0.36 down at 91. So one point of VMAF is
    worth three to seven CRF, and a target in the mid-90s sits on the flattest
    part of it, where the inversion amplifies hardest.

    That is not a defect to fix here, it is the shape of the problem, but it
    explains a whole class of otherwise baffling results and sets the floor on
    what any of this can resolve:

      - the probe window is a SAMPLE of the shot, and measured on real 4K,
        taking 64 frames of a shot instead of 120 moved the score by up to 2.5
        VMAF - which lands as ~12 CRF (observed: +12.2 on one shot)
      - conversely, tiny metric differences are harmless: the SYCL backend
        differs from the CPU one by ~1e-4 VMAF, i.e. ~7e-4 CRF
      - probe_bracket_width=6 is not coarse. Bisecting finer would be chasing
        precision the measurement does not have.
    """
    pts = sorted((crf, score) for crf, score in samples if score is not None)
    if not pts:
        raise TranscodeError("no valid probe scores to pick a CRF")
    if len(pts) == 1:
        return float(pts[0][0])
    # Even the best (lowest crf) cannot reach the target -> use the best CRF.
    if pts[0][1] <= target:
        return float(pts[0][0])
    # Even the worst (highest crf) still beats the target -> stop spending bits.
    if pts[-1][1] >= target:
        return float(pts[-1][0])
    for (crf_lo, s_lo), (crf_hi, s_hi) in zip(pts, pts[1:]):
        if s_lo >= target >= s_hi:
            if s_lo == s_hi:
                return float(crf_lo)
            t = (s_lo - target) / (s_lo - s_hi)
            return crf_lo + t * (crf_hi - crf_lo)
    # Non-monotone noise fallback: nearest sampled score.
    return float(min(pts, key=lambda p: abs(p[1] - target))[0])


def _csv_first(line: str) -> str:
    """The first field of one ffprobe `-of csv=p=0` line.

    ffprobe prints a stream's child sections as extra fields even when only
    one entry was asked for, so a stream with side data comes out as
    "audio," rather than "audio". Every eac3 track in an mp4 carries an
    "Audio Service Type" side data, so on the ATVP and DSNP WEB-DLs this
    library is made of, `stream=codec_type` reads "video," "audio," - and a
    membership test against "audio" said the file had no audio, the mux ran
    video-only, and the output check failed every one of those jobs.
    """
    return line.split(",", 1)[0].strip()


def concat_quote(path: Path) -> str:
    """`path` as one field of an ffmpeg concat demuxer list.

    The demuxer's quoting is the shell's, not Python's: inside single quotes
    everything is literal, so a single quote in the path has to close the
    string, emit an escaped quote and reopen it -> 'it'\\''s'. Writing
    f"'{path}'" instead silently truncates the filename at the first
    apostrophe, and every shot after it lands in the wrong place or vanishes.

    Reachable because these paths are rooted at the configured dirs.work, not
    at a name this code chose.
    """
    return "'" + str(path).replace("'", "'\\''") + "'"


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def bracket_for(samples: List[Tuple[int, float]], target: float) -> Optional[Tuple[int, int]]:
    """The two adjacent probed CRFs the target score falls between.

    None when the target does not fall between any pair - either the best
    (lowest) CRF probed already scores at or below it, so no bracket can
    exist and pick_crf clamps, or the worst already beats it and there is
    nothing to buy by spending more bits. Both cases mean more probing on this
    shot cannot change the answer.
    """
    pts = sorted((c, sc) for c, sc in samples if sc is not None)
    if len(pts) < 2 or pts[0][1] <= target or pts[-1][1] >= target:
        return None
    for (c_lo, s_lo), (c_hi, s_hi) in zip(pts, pts[1:]):
        if s_lo >= target >= s_hi:
            return c_lo, c_hi
    return None


def seed_crfs(grid: List[int], count: int = 3) -> List[int]:
    """The coarse CRFs an adaptive search starts from: both ends plus enough
    interior points to have something to interpolate, taken from `grid` so the
    configured search space still decides the range."""
    if not grid:
        return []
    ordered = sorted(set(grid))
    count = max(2, min(count, len(ordered)))
    last = len(ordered) - 1
    return [ordered[j] for j in sorted({round(i * last / (count - 1))
                                        for i in range(count)})]


def predict_score(samples: List[Tuple[int, float]], crf: float) -> Optional[float]:
    """The score the probes predict at `crf`, interpolated the way pick_crf
    inverts. Returns None when there is nothing to interpolate.

    Used to compare what the probe grid promised against what the finished
    encode actually delivers, which is the only way the probe-side biases
    (a faster probe preset, a 120-frame probe window, linear interpolation on
    a curved rate-distortion relationship) become a number rather than a guess.
    """
    pts = sorted((c, sc) for c, sc in samples if sc is not None)
    if not pts:
        return None
    if crf <= pts[0][0]:
        return float(pts[0][1])
    if crf >= pts[-1][0]:
        return float(pts[-1][1])
    for (c_lo, s_lo), (c_hi, s_hi) in zip(pts, pts[1:]):
        if c_lo <= crf <= c_hi:
            if c_hi == c_lo:
                return float(s_lo)
            t = (crf - c_lo) / (c_hi - c_lo)
            return float(s_lo + t * (s_hi - s_lo))
    return float(pts[-1][1])


def smooth_crfs(crfs: List[float], max_delta: float, iterations: int = 32) -> List[float]:
    """Bound the CRF jump between adjacent shots to <= max_delta.

    Each shot is independently picked to hit the target metric, so two
    neighbouring shots can land far apart (e.g. CRF 22 then CRF 31) and look
    discontinuous even though both meet the target. This runs alternating
    forward/backward projections: every shot is clipped to within +/- max_delta
    of its neighbours, repeated until stable. Shots already within the bound
    are left untouched. Returns a copy; max_delta <= 0 disables smoothing.
    """
    n = len(crfs)
    if n <= 1 or max_delta <= 0:
        return [float(c) for c in crfs]
    s = [float(c) for c in crfs]
    for _ in range(max(1, iterations)):
        moved = False
        for i in range(1, n):
            lo = s[i - 1] - max_delta
            hi = s[i - 1] + max_delta
            new = min(max(s[i], lo), hi)
            if new != s[i]:
                s[i] = new
                moved = True
        for i in range(n - 2, -1, -1):
            lo = s[i + 1] - max_delta
            hi = s[i + 1] + max_delta
            new = min(max(s[i], lo), hi)
            if new != s[i]:
                s[i] = new
                moved = True
        if not moved:
            break
    return s


def merge_short_shots(shots: List[Shot], min_frames: int) -> List[Shot]:
    """Fold shots shorter than `min_frames` into a neighbour.

    Every shot is encoded standalone, so each one costs a keyframe and cannot
    predict across its own boundary, and SVT-AV1's mini-GOP is 32 frames - a
    shot below that cannot even fill one and falls back to a shorter prediction
    structure. Both are paid per shot, so a run of very short shots is
    expensive out of proportion to the frames it holds, while a sub-second shot
    is also the one that gains least from having its own CRF.

    The shortest offender is merged with its shorter neighbour and the scan
    repeats, so a run of tiny shots coalesces instead of all piling onto one
    long neighbour. 0 disables. A merged shot still under the bound is merged
    again, so the result has no shot below `min_frames` unless only one is left.
    """
    if min_frames <= 0:
        return list(shots)
    shots = list(shots)

    def span(i: int) -> int:
        return shots[i][1] - shots[i][0]

    while len(shots) > 1:
        i = min(range(len(shots)), key=span)
        if span(i) >= min_frames:
            break
        if i == 0:
            j = 1
        elif i == len(shots) - 1:
            j = i - 1
        else:
            j = i - 1 if span(i - 1) <= span(i + 1) else i + 1
        lo, hi = min(i, j), max(i, j)
        shots[lo] = (shots[lo][0], shots[hi][1])
        del shots[hi]
    return shots


def merge_to_max(shots: List[Shot], max_shots: int) -> List[Shot]:
    """Merge the shortest adjacent shots until the count is <= max_shots."""
    shots = list(shots)
    while len(shots) > max_shots and len(shots) > 1:
        def span(i: int) -> int:
            return shots[i][1] - shots[i][0]
        i = min(range(len(shots) - 1), key=lambda k: span(k) + span(k + 1))
        shots[i] = (shots[i][0], shots[i + 1][1])
        del shots[i + 1]
    return shots


def _svt_params_dict(video: VideoParams) -> Dict[str, object]:
    """Map VideoParams to SVT-AV1 params (as a dict for -svtav1-params)."""
    svt: Dict[str, object] = {"tune": video.tune}
    if video.film_grain:
        svt["film-grain"] = video.film_grain
        if not video.film_grain_denoise:
            svt["film-grain-denoise"] = 0
    if video.luminance_qp_bias:
        svt["luminance-qp-bias"] = video.luminance_qp_bias
    if video.additional_video_params:
        toks = video.additional_video_params.split()
        i = 0
        while i < len(toks):
            t = toks[i]
            if "=" in t:
                k, v = t.split("=", 1)
                svt[k.lstrip("-")] = v
                i += 1
            elif t.startswith("-") and i + 1 < len(toks) and not toks[i + 1].startswith("-"):
                svt[t.lstrip("-")] = toks[i + 1]
                i += 2
            else:
                i += 1
    return svt


def parse_score(json_path: Path, metric: str) -> float:
    """Extract the pooled-mean metric score from a libvmaf JSON log."""
    data = json.loads(json_path.read_text())
    pooled = data.get("pooled_metrics") or {}
    if metric in pooled:
        mean = pooled[metric].get("mean")
        if mean is not None:
            return float(mean)
    # Fall back to the only metric present, but ONLY if there is exactly one.
    # A default VMAF run pools integer_adm2, integer_motion2, integer_vif_* and
    # vmaf together, so picking "the first one" for an unknown metric name
    # returns integer_adm2 (~0.95) as if it were a 0-100 score: every shot then
    # reads far below any target, every shot falls back to the lowest CRF, and
    # the output balloons. Fail loudly instead.
    if len(pooled) == 1:
        mean = next(iter(pooled.values())).get("mean")
        if mean is not None:
            return float(mean)
    agg = data.get("aggregateVMAF")
    if agg is not None:
        return float(agg)
    if pooled:
        raise TranscodeError(
            f"{json_path} has no '{metric}' metric; libvmaf reported "
            f"{sorted(pooled)}. Check transcode.optimizer."
            f"{'ssimulacra2_model' if metric == 'ssimulacra2' else 'vmaf_model'} "
            f"- the configured model does not produce '{metric}'."
        )
    raise TranscodeError(f"could not parse {metric} score from {json_path}")


class MemCalibration:
    """Multiplicative correction to the static memory model, learned in-job.

    The static model is fitted offline and deliberately biased high, because
    under-estimating is what OOM-kills an encoder while over-estimating only
    leaves budget unused. That bias is not free though - the probe phase in
    particular is over-estimated by 40-60% at 4K, because SVT-AV1 forces
    preset<=M9 there and halves its mini-GOP, which the resolution-independent
    prior cannot know. So the prior is only a starting point: every task
    reports what it actually peaked at, and this scales the model to match.

    Asymmetric on purpose. An observation ABOVE the prediction is acted on at
    once - that direction is the one that OOMs - while a reading below it only
    eases the factor down, so one unusually cheap shot cannot talk the pool
    into over-admitting the next expensive one.
    """

    FLOOR, CAP, EASE = 0.30, 3.0, 0.25

    def __init__(self, name: str) -> None:
        self.name = name
        self._factor = 1.0
        self.samples = 0

    def factor(self) -> float:
        return self._factor

    def observe(self, predicted_gb: float, observed_gb: float) -> None:
        """Fold one measured peak into the correction.

        `predicted_gb` must be the RAW model value, not the corrected one, so
        the factor stays absolute instead of compounding run over run.
        """
        if predicted_gb <= 0 or observed_gb <= 0:
            return
        self.samples += 1
        ratio = observed_gb / predicted_gb
        if ratio > self._factor:
            self._factor = ratio * 1.05      # a little clearance over the peak
        else:
            self._factor += self.EASE * (ratio - self._factor)
        self._factor = min(self.CAP, max(self.FLOOR, self._factor))


def plan_admission(
    pending: List[int],
    mem_free: float,
    cpu_free: float,
    cost: Callable[[int, int], float],
    lp_ladder: List[int],
    idle: bool,
    cpu_charge: float = 1.0,
) -> Optional[Tuple[int, int, float]]:
    """Pick the next (position in `pending`, lp, GB) to start, or None to wait.

    Best fit: `pending` is ordered longest shot first, and this takes the first
    one that still fits both budgets. That ordering is what keeps a mix in
    flight - one long encode plus however many short ones the remaining memory
    holds - which is the whole point of admitting by cost instead of by a fixed
    worker count. Measured on a real 146-shot list, per-instance peak RSS spans
    2.5x (4.7GB for a 24-frame shot against 10.5GB for a 1000-frame one at 4K),
    so no single worker count is right for both ends.

    Taking the LONGEST that fits rather than the first that fits matters:
    draining the long shots up front (a plain longest-first queue) empties the
    pool of anything cheap to pair them with, and simulating that on the same
    shot list came out slower than not sorting at all.

    The CPU budget picks the lp; memory only decides admit-or-wait. Trading lp
    away to fit one more instance in memory looks free - lp does not change the
    bitstream - but it is not, because a lower lp makes that instance slower:
    measured at 4K on a 144-frame shot, 22.5s at lp=4 against 27.6s at lp=3,
    41.0s at lp=2 and 97.3s at lp=1. Taking that trade greedily costs far more
    than the extra concurrency returns. Simulated over a real 146-shot list
    with the measured contention curve, descending the ladder for memory ran
    3338s against 1575s for waiting instead - and the same 2x at every budget
    from 8GB up.

    So the ladder is a fallback, not a routine choice: it is walked only when
    NOTHING is running and the shot does not fit even so, which is the case it
    was added for - a shot too big for the whole budget has to run at some lp
    or the phase deadlocks. On the same list at an 8GB budget that path takes
    10 shots of 146; the other 136 still run at the top of the ladder.

    `idle` also forces an admission when no rung fits at all. The caller warns;
    there is nothing else to do but overshoot.
    """
    ladder = sorted({max(1, lp) for lp in lp_ladder}, reverse=True)
    # cpu_charge lets a phase book fewer tokens than the lp it hands the
    # encoder. A probe is not an encoder for its whole life - it also reads
    # and decodes the window and then scores it - so charging the full lp
    # across all of that reserves cores nothing is using. See _schedule.
    top = next((lp for lp in ladder if lp * cpu_charge <= cpu_free), None)
    if top is not None:
        for pos, idx in enumerate(pending):
            gb = cost(idx, top)
            if gb <= mem_free:
                return pos, top, gb
    if idle and pending:
        for lp in ladder:
            gb = cost(pending[0], lp)
            if gb <= mem_free and lp * cpu_charge <= cpu_free:
                return 0, lp, gb
        lp = ladder[-1]
        return 0, lp, cost(pending[0], lp)
    return None


# ---- the engine -----------------------------------------------------------

def run_shot_transcode(
    settings: Settings,
    info: MediaInfo,
    plan: TranscodePlan,
    source: Path,
    output: Path,
    tempdir: Path,
    log_path: Optional[Path] = None,
    progress_cb: Optional[Callable[[float, Optional[dict]], None]] = None,
    cancel_flag: Optional[Callable[[], bool]] = None,
    stage_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """Run the full shot-based encode. Mirrors run_av1an's callback contract."""
    ShotEncoder(settings, info, plan, source, output, tempdir, log_path,
                progress_cb, cancel_flag, stage_cb).run()


def _render_nodes() -> List[str]:
    """The DRM render nodes this process can see, in name order."""
    return sorted(glob.glob("/dev/dri/renderD*"))


def _render_pdev(node: str) -> Optional[str]:
    """The PCI address behind a render node, e.g. "0000:c6:00.0".

    fdinfo tags every DRM client with the device it belongs to, and this box
    has four render nodes (two 3090s, the B580, and the iGPU); without the
    address a memory reading would sum clients of cards this job never
    touches.
    """
    name = os.path.basename(node)
    try:
        return os.path.basename(
            os.path.realpath(f"/sys/class/drm/{name}/device"))
    except OSError:
        return None


def drm_vram_used_mb(pdev: Optional[str],
                     root: str = "/proc") -> Optional[float]:
    """VRAM this process tree holds on `pdev`, in MB, or None if unreadable.

    The interface nvtop reads: every open DRM file exposes drm-total-vram0 in
    /proc/<pid>/fdinfo/<fd>, and a client that dup()s or forks its fd shows
    the same allocation under several fds, so the sum has to be deduplicated
    by drm-client-id or it counts the same bytes many times over.

    It sees only what this PID namespace can see. Inside the container that
    is our own ffmpegs - Plex's share of the card is invisible here, which is
    why the budget has to leave room for it rather than measure it.
    """
    total, seen = 0.0, set()
    try:
        pids = [d for d in os.listdir(root) if d.isdigit()]
    except OSError:
        return None
    found = False
    for pid in pids:
        d = f"{root}/{pid}/fdinfo"
        try:
            fds = os.listdir(d)
        except OSError:
            continue                      # the process exited, or is not ours
        for fd in fds:
            try:
                with open(f"{d}/{fd}") as fh:
                    text = fh.read(4096)
            except OSError:
                continue
            if "drm-client-id" not in text:
                continue
            found = True
            cid = dev = None
            vram = 0.0
            for line in text.splitlines():
                k, _, v = line.partition(":")
                v = v.strip()
                if k == "drm-client-id":
                    cid = v
                elif k == "drm-pdev":
                    dev = v
                elif k.startswith("drm-total-vram"):
                    vram += _kib_to_mb(v)
            if cid is None or cid in seen:
                continue
            if pdev and dev and dev != pdev:
                continue
            seen.add(cid)
            total += vram
    return total if found or total == 0.0 else None


def _kib_to_mb(v: str) -> float:
    """One fdinfo size field ("1234 KiB", "12 MiB", bare bytes) in MB."""
    parts = v.split()
    if not parts:
        return 0.0
    try:
        n = float(parts[0])
    except ValueError:
        return 0.0
    unit = (parts[1] if len(parts) > 1 else "KiB").lower()
    scale = {"b": 1 / 1048576.0, "kib": 1 / 1024.0, "mib": 1.0,
             "gib": 1024.0}.get(unit, 1 / 1024.0)
    return n * scale


class VramBudget:
    """Admission control for the card's memory, in MB.

    What the counting semaphore this replaces got wrong is the unit. Six
    slots is six OPERATIONS, and operations are not the same size: a SYCL
    score of a 120-frame 4K window holds ~440MB, the same score with
    reference_hwaccel on holds a VA-API surface pool beside it for ~880MB,
    and a QSV probe encode of a 480-frame window is a different number again.
    Six of the cheap kind is a third of the card; six of the expensive kind is
    most of it. Sizing one constant for the worst case wastes the card the
    rest of the time, and sizing it for the average is how E07 died:
    OUT_OF_DEVICE_MEMORY, then DEVICE_LOST, then a reboot to get the card
    back (see _gpu_workers).

    So admission is on bytes, against two numbers that disagree in useful
    ways:

      - RESERVED, the sum of the estimates of everything in flight. Covers
        the ramp: a VA-API session allocates its pool within a second or two
        of launch and a SYCL context grows over the score, so for the first
        moments of an operation the measurement has not caught up yet.
      - MEASURED, what fdinfo says the card actually holds right now. Covers
        the model being wrong, in either direction.

    Occupancy is the larger of the two, which is conservative exactly where
    being wrong is expensive and lets the budget breathe everywhere else.
    Reservations are never trusted after the fact: the model is recalibrated
    from the measurement (see `calibrate`), so a systematic error costs a
    little throughput for a few operations rather than for the whole job.
    """

    def __init__(self, budget_mb: float, max_ops: int,
                 pdev: Optional[str] = None,
                 measure: Optional[Callable[[], Optional[float]]] = None):
        self.budget = max(256.0, float(budget_mb))
        self.max_ops = max(1, int(max_ops))
        self._pdev = pdev
        self._measure = measure or (lambda: drm_vram_used_mb(self._pdev))
        self._cv = threading.Condition()
        self._reserved = 0.0
        self._ops = 0
        self._measured = 0.0
        self._measured_at = 0.0
        self._peak = 0.0
        # worst megabytes-actually-held per megabyte-estimated seen so far.
        # Keyed on the reservation rather than on the peak reading, because
        # the largest reading can land at a moment with nothing booked.
        self._worst = 0.0
        self._booked = 0.0
        # the same reservations before the model's correction is applied:
        # the denominator calibration divides by, so that the sample is not
        # a function of the very ratio it is meant to correct
        self._reserved_raw = 0.0
        self._readable: Optional[bool] = None
        # model correction, in the shape MemCalibration uses for RSS: the
        # ratio of what operations really cost to what they were estimated at
        self._ratio = 1.0
        self._samples = 0
        self._calibrated_at = 0.0

    # -- measurement ---------------------------------------------------
    _CACHE_S = 1.0
    # A score takes seconds and there are thousands of them in an episode;
    # recalibrating on every one would walk /proc thousands of times for a
    # number that moves slowly.
    _CALIBRATE_EVERY_S = 5.0
    # Most any single sample may claim an operation costs against its
    # booking. Without it one mistimed reading pinned the model at its
    # ceiling for the whole job and the pool collapsed to one operation.
    _SAMPLE_CEILING = 4.0
    # The worst case decays, so it is the worst RECENT case. A permanent
    # high-water mark cannot recover from a bad sample by construction.
    _WORST_DECAY = 0.98

    def measured_mb(self, force: bool = False) -> float:
        """Current usage, cached for a second - admission runs per operation
        and walking /proc is not free."""
        now = time.monotonic()
        if not force and now - self._measured_at < self._CACHE_S:
            return self._measured
        mb = self._measure()
        self._measured_at = now
        if mb is None:
            if self._readable is None:
                self._readable = False
            return self._measured
        self._readable = True
        self._measured = mb
        self._peak = max(self._peak, mb)
        return mb

    @property
    def readable(self) -> bool:
        """Whether fdinfo answered at least once. When it never does the
        budget degrades to the reservation model alone, which is the old
        counting behaviour with better units."""
        return self._readable is not False

    def occupancy(self) -> float:
        return max(self._reserved, self.measured_mb())

    # -- admission -----------------------------------------------------
    def booked(self, mb: float) -> float:
        """What `reserve` would actually book for an estimate of `mb`.

        Callers hold on to this and hand it back to `release`: recomputing
        it there reads a _ratio that calibration has moved in between, so
        releases over- or under-subtracted and _reserved drifted to zero (or
        upward forever). Zero reserved silently deletes the half of the model
        that covers the ramp - the seconds before a VA-API pool or a SYCL
        context shows up in fdinfo - which is the path to
        OUT_OF_DEVICE_MEMORY this class exists to prevent.
        """
        with self._cv:
            return max(0.0, float(mb)) * self._ratio

    def reserve(self, mb: float, timeout: float) -> bool:
        """Book `mb` for one operation. False when the wait ran out.

        One operation always fits: a budget smaller than a single score would
        otherwise deadlock the phase rather than slow it down, and the CPU
        fallback the caller takes on failure is meant for contention, not for
        an impossible sum.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        self.measured_mb()          # cached; the reservation alone is half the story
        with self._cv:
            want = max(0.0, float(mb)) * self._ratio
            while True:
                free = self.budget - max(self._reserved, self._measured)
                if self._ops == 0 or (want <= free and self._ops < self.max_ops):
                    self._reserved += want
                    self._reserved_raw += max(0.0, float(mb))
                    self._ops += 1
                    self._booked = want
                    return True
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(min(left, self._CACHE_S))
                self.measured_mb()

    def release(self, booked: float, mb: float = 0.0) -> None:
        """Give back exactly what `reserve` took - see `booked`."""
        with self._cv:
            self._reserved = max(0.0, self._reserved - max(0.0, float(booked)))
            self._reserved_raw = max(0.0, self._reserved_raw - max(0.0, float(mb)))
            self._ops = max(0, self._ops - 1)
            self._cv.notify_all()

    @contextlib.contextmanager
    def hold(self, mb: float, timeout: float, what: str) -> Iterator[None]:
        want = self.booked(mb)
        if not self.reserve(mb, timeout):
            raise TranscodeError(
                f"no room on the GPU for {what} within {timeout:.0f}s "
                f"({self.occupancy():.0f}MB of {self.budget:.0f}MB in use)")
        try:
            yield
        finally:
            self.release(want, mb)

    # -- calibration ---------------------------------------------------
    def calibrate(self) -> None:
        """Pull the model towards what the card actually holds AT ITS PEAK.

        The instantaneous reading is the wrong number to divide by the
        reservation, and measuring said so: a SYCL context does not allocate
        until libvmaf has initialised the device a second or two into the
        run, and a 120-frame 4K score is over in a few seconds, so most
        samples catch operations mid-ramp. On a 163-shot episode that pulled
        the model to its floor, x0.25 - a quarter of an estimate that the
        same run showed was already 21% LOW (peak 3967MB over six concurrent
        scores is 661MB each against an estimate of 524MB). Only the
        operation-count cap stopped that from over-admitting, which is not a
        safety margin to rely on.

        So the target is the WORST megabytes-held per megabyte-booked seen so
        far, not the latest one. That is the case admission has to be sized
        for, and no quiet moment can drag it down. Keyed on the reservation
        rather than on the largest reading, because the largest reading can
        land at a moment with nothing booked at all. The approach stays slow
        - a twentieth of the gap per sample - and the floor is no longer low
        enough to matter.
        """
        now = time.monotonic()
        with self._cv:
            if self._ops <= 0 or self._reserved <= 0:
                return
            if now - self._calibrated_at < self._CALIBRATE_EVERY_S:
                return
            self._calibrated_at = now
        mb = self.measured_mb(force=True)
        with self._cv:
            if self._ops <= 0 or self._reserved <= 0 or mb <= 0:
                return
            # Under the lock, so _reserved cannot be zeroed by a release
            # between the two reads - which raised ZeroDivisionError, and a
            # ZeroDivisionError is not a TranscodeError, so it failed the job
            # rather than falling back to the CPU.
            raw = self._reserved_raw
            if raw <= 0:
                return
            # One sample, clamped: a reading taken while a finished ffmpeg is
            # still tearing down counts memory whose booking has already gone,
            # and an unbounded high-water mark turned one such sample into a
            # permanent x4.0 for the rest of the job.
            sample = min(self._SAMPLE_CEILING, mb / raw)
            # decay, so the worst case is the worst RECENT case
            self._worst = max(sample, self._worst * self._WORST_DECAY)
            self._ratio += (self._worst - self._ratio) / 20.0
            self._ratio = min(4.0, max(0.5, self._ratio))
            self._samples += 1

    def summary(self) -> str:
        return (f"peak {self._peak:.0f}MB of a {self.budget:.0f}MB budget, "
                f"model x{self._ratio:.2f} after {self._samples} sample(s)"
                if self.readable else
                f"{self.budget:.0f}MB budget, unmeasured (no readable fdinfo)")


class CommandTimeout(TranscodeError):
    """A subprocess hit its _run timeout and was killed."""


class ShotEncoder:
    """Parallel shot-based encoder with per-shot interpolated CRF selection."""

    def __init__(
        self,
        settings: Settings,
        info: MediaInfo,
        plan: TranscodePlan,
        source: Path,
        output: Path,
        tempdir: Path,
        log_path: Optional[Path] = None,
        progress_cb: Optional[Callable[[float, Optional[dict]], None]] = None,
        cancel_flag: Optional[Callable[[], bool]] = None,
        stage_cb: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.settings = settings
        self.info = info
        self.plan = plan
        self.source = Path(source)
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.tempdir = Path(tempdir)
        self.log_path = log_path
        self.progress_cb = progress_cb
        self.cancel_flag = cancel_flag
        self.stage_cb = stage_cb

        self.video: VideoParams = plan.params or settings.transcode.video
        self.opt = settings.transcode.optimizer
        self.ffmpeg = settings.tool_path("ffmpeg")

        # Every shot boundary is converted from a frame number to a -ss
        # timestamp with this, so a made-up frame rate silently encodes the
        # wrong parts of the source. Refuse rather than guess 25.
        if not info.fps or info.fps <= 0:
            raise TranscodeError(
                f"engine=optimizer needs a frame rate for {self.source.name} but "
                "ffprobe reported none (avg_frame_rate and r_frame_rate are both "
                "unset). Use engine=av1an for this source."
            )
        self.fps = float(info.fps)
        self.total_frames = int(self.fps * max(info.duration, 0.0)) or 1

        self.metric = self.video.target_metric
        if not (self.video.target_quality or "").strip():
            raise TranscodeError(
                "engine=optimizer requires a target_quality (e.g. '75' or '75-85'). "
                "Set target_quality in the preset."
            )
        self.target, _ = parse_target(self.video.target_quality or "")
        self.probe_dir = self.tempdir / "probes"
        self.probe_dir.mkdir(parents=True, exist_ok=True)

        self._procs: set[subprocess.Popen] = set()
        self._proc_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._log_handle = open(log_path, "a", buffering=1) if log_path else None
        # one-time warning for av1an-style probing_vmaf_features
        self._feature_warned = False
        # SYCL preflight result: None = not yet checked (see _sycl_device).
        # Probe workers call it concurrently, hence the lock.
        self._sycl_ok: Optional[bool] = None
        self._sycl_lock = threading.Lock()
        self._dataset_lock = threading.Lock()
        self._dataset_warned = False
        nodes = _render_nodes()
        self._gpu_vram = VramBudget(self._vram_budget_mb(), self._gpu_workers(),
                                    _render_pdev(nodes[0]) if nodes else None)
        # reference_hwaccel: None until the first scoring read runs the
        # preflight; then whether the source decodes on QSV for this job
        self._hwdec_ok: Optional[bool] = None
        self._hwdec_streak = 0        # hardware reads failed in a row, reset on success
        self._hwdec_bad: Set[Tuple[int, int]] = set()   # windows that failed once
        self._hwdec_lock = threading.Lock()
        # vmaf_zero_copy: None until the first probe score runs the preflight
        # (see _zero_copy); the rest mirrors the reference_hwaccel bookkeeping
        self._zc_ok: Optional[bool] = None
        self._zc_lock = threading.Lock()
        self._zc_bad: Set[Tuple[int, int]] = set()
        self._zc_streak = 0
        self._zc_scored = 0
        self._zc_fallbacks = 0
        # scorings the SYCL backend failed to finish in time (see _score_vmaf)
        self._sycl_timeouts = 0        # consecutive; a good score clears it
        self._sycl_retired_at: Optional[float] = None   # when the device was given up
        # every frame's pts_time from the scdet pass (see _run_scdet), and the
        # timeline slot each frame sits in (see _slots_from_pts / _slot)
        self._frame_pts: List[float] = []
        self._slots: List[int] = []
        # probe pool size, used to auto-size libvmaf threads (see _vmaf_threads)
        self._probe_worker_count = 1
        # affinity slices currently taken (only used when encode_threads is set)
        self._slots_used: set[int] = set()
        self._slot_lock = threading.Lock()
        # per-phase correction to the static memory model, learned as the job
        # runs (see MemCalibration)
        self._cal = {"encoding": MemCalibration("encoding"),
                     "probing": MemCalibration("probing")}
        # peak RSS of the subprocesses one scheduled task ran, accumulated on
        # that task's own thread so concurrent tasks cannot mix their readings
        self._task_peak = threading.local()
        # Dolby Vision Profile 5: the base layer is ICtCp, so it has to have its
        # RPU applied before it means anything. Done per shot rather than once
        # over the whole file - see _acquire_shard.
        self._p5 = bool(plan.p5 and settings.transcode.dovi.enabled)
        self._shards: Dict[int, Path] = {}
        self._shard_lock = threading.Lock()
        # seconds each file's video starts after its container, by path; see
        # _lead_of. Probed once per file, so the source and the output.
        self._leads: Dict[str, float] = {}
        # (peak_mb, command) of the heaviest child this phase, for diagnostics
        self._heaviest_cmd: Tuple[float, str] = (0.0, "")

    # ---------- callbacks / logging ----------
    def _log(self, line: str) -> None:
        logger.debug("optimizer: {}", line)
        if self._log_handle is None:
            return
        with self._log_lock:
            self._log_handle.write(line.rstrip() + "\n")

    def _dataset_path(self) -> Optional[Path]:
        if not getattr(self.opt, "probe_dataset", False):
            return None
        try:
            d = Path(self.settings.dirs.logs)
            d.mkdir(parents=True, exist_ok=True)
            return d / "probe_dataset.jsonl"
        except OSError:
            return None

    def _dataset_write(self, row: Dict[str, object]) -> None:
        """Append one record. Never lets a diagnostics file fail a transcode.

        Written as each shot finishes rather than in one batch at the end,
        because a probe phase is hours long and the jobs that most need
        explaining are the ones that get killed part way through.
        """
        path = self._dataset_path()
        if path is None:
            return
        row = dict(row)
        row.setdefault("job", self.log_path.stem if self.log_path else "")
        row.setdefault("t", round(time.time(), 3))
        try:
            with self._dataset_lock:
                with open(path, "a") as fh:
                    fh.write(json.dumps(row, separators=(",", ":"),
                                        default=float) + "\n")
        except Exception as e:
            # Not just OSError. This runs inside the scheduler's on_done,
            # which holds the condition variable: anything raised there skips
            # the budget release and the notify, and the phase waits forever
            # for a worker slot that never comes back. A diagnostics file must
            # not be able to do that, whatever json.dumps decides to raise.
            if not self._dataset_warned:
                self._dataset_warned = True
                logger.warning("optimizer: cannot write the probe dataset "
                               "({}); continuing without it", e)

    def _dataset_header(self, shots: List[Shot], grid: List[int]) -> None:
        self._dataset_write({
            "type": "job",
            "source": self.source.name,
            "width": self.info.width, "height": self.info.height,
            "fps": self.info.fps, "duration": self.info.duration,
            "src_bitrate": self.info.bitrate,
            "codec": getattr(self.info, "video_codec", ""),
            "shots": len(shots), "frames": sum(b - a for a, b in shots),
            "metric": self.metric, "target": self.target,
            "preset": self.video.preset, "probe_preset": self._probe_preset(),
            "probing_rate": self._probing_rate(),
            "probe_max_frames": self.opt.probe_max_frames,
            "probe_grid": list(grid), "probe_encoder": self._probe_mode(),
            "probe_scale": self._probe_scale() or "",
            "pix_fmt": self._pix_fmt(),
        })

    def _dataset_shot(self, idx: int, s0: int, s1: int, w0: int, w1: int,
                      svt: Dict[int, float],
                      qsv: Optional[Dict[int, float]] = None,
                      qsv_bpf: Optional[Dict[int, float]] = None,
                      seed: Optional[float] = None) -> None:
        """One shot's probes, with the features a later model might want that
        cost nothing to record here: where the shot sits, how long it is, and
        whether its probe window covered the whole of it."""
        total = max(1, self._total_frames())
        row: Dict[str, object] = {
            "type": "shot", "idx": idx, "s0": s0, "s1": s1,
            "frames": s1 - s0, "pos": round(s0 / total, 5),
            "w0": w0, "w1": w1, "window": w1 - w0,
            "whole_shot": (w1 - w0) >= (s1 - s0),
            "svt": {str(k): v for k, v in sorted(svt.items())},
            "crf_star": self._crossing(svt),
        }
        if qsv is not None:
            row["qsv"] = {str(k): v for k, v in sorted(qsv.items())}
            row["q_star"] = self._crossing(qsv)
        if qsv_bpf:
            row["qsv_bpf"] = {str(k): v for k, v in sorted(qsv_bpf.items())}
        if seed is not None:
            row["seed"] = seed
        self._dataset_write(row)

    def _total_frames(self) -> int:
        return int(round((self.info.duration or 0) * (self.info.fps or 0))) or 1

    def _mem_log(self, line: str, level: str = "info") -> None:
        """Memory diagnostics: always written to the job log file; console
        shows summaries at INFO and per-command detail at DEBUG (set
        AV1TC_LOGGING_LEVEL=DEBUG to see it in the console)."""
        if self._log_handle is not None:
            with self._log_lock:
                self._log_handle.write(line.rstrip() + "\n")
        if level == "info":
            logger.info("optimizer: {}", line)
        else:
            logger.debug("optimizer: {}", line)

    def _stage(self, stage: str, pct: float) -> None:
        if self.stage_cb:
            self.stage_cb(stage)
        self._report(pct, 0, 0)

    def _report(self, pct: float, done: int = 0, total: int = 0, fps: float = 0.0) -> None:
        if self.progress_cb:
            pct = min(max(pct, 0.0), 100.0)
            self.progress_cb(pct, {"pct": pct, "done": done, "total": total, "fps": fps})

    @staticmethod
    def _cores() -> int:
        """Cores we may actually use - the cgroup quota, not the machine.

        os.cpu_count() reports the box: a container run with `--cpus=4` still
        reads 32 here, so every derived number (worker count, lp, affinity
        slices) was sized for hardware this process cannot have.
        """
        return max(1, int(sysres.cpu_budget()))

    def _mem_budget_gb(self) -> float:
        """Memory this phase may hold, in GB.

        Under a cgroup limit this is the headroom below it; otherwise the
        kernel's MemAvailable, which already discounts what everything else on
        the box is holding. The old rule was 75% of MemTotal, which on a shared
        machine is not memory we have - measured here, MemTotal*0.75 came to
        23.5GB against 22GB actually available, and encoding at that budget
        pushed 3.8GB into swap.

        The remaining margin covers what the budget does not model: page cache
        for a multi-GB source read, the Python process, and the tmpfs shards
        the DV/SSIMULACRA2 paths stage.
        """
        return max(1.0, sysres.memory_available_gb() * 0.85)

    def _megapixels(self) -> float:
        px = (self.info.width or 0) * (self.info.height or 0)
        return max(0.5, px / 1e6) if px else 2.0

    # Per-instance memory relative to lp=4, measured at 4K on 144-frame shots
    # (3.42 / 4.19 / 5.23 / 6.71 / - / 8.80 GB at lp 1..6).
    #
    # NB these are NOT the ratios of SVT-AV1's frame pool. The pool is
    # min_input + (1 + mg_size) * n_extra_mg pictures with n_extra_mg = 0 for
    # lp<=3, 1 at lp=4, 2 at lp=5 and 7 at lp=6 (enc_handle.c), which predicts
    # equal memory for lp 1-3 and 2.85x for lp=6. Neither holds in practice: the
    # pool is reserved up front but only TOUCHED as frames flow, so a shot short
    # enough not to fill a 305-picture pool never pays for it, and lp 1-3 differ
    # because they fill their shared pool to different depths.
    _LP_MEM_RATIO = {1: 0.55, 2: 0.65, 3: 0.80, 4: 1.00, 5: 1.15, 6: 1.35}

    # Under-estimating here is what OOM-kills an encoder, over-estimating only
    # leaves budget unused, so the fit is deliberately biased high.
    _MEM_SAFETY = 1.10

    def _est_encode_gb(self, frames: int, lp: int = 4) -> float:
        """Peak RSS of ONE final-encode instance, in GB.

        Two things drive it, and the flat estimate this replaces modelled
        neither.

        SHOT LENGTH is the larger one. SVT-AV1 reserves its whole pool up front
        - virtual size is a flat ~14GB at 4K whatever the shot - but only
        touches what the frames in flight need, so RSS tracks the shot. Measured
        at 4K (3840x1920, preset 4, lp=4), peak RSS per instance:

            24f 3.41GB   96f 6.19GB   288f 7.70GB   1152f 10.46GB
            48f 4.69GB  144f 6.71GB   576f 9.12GB

        i.e. 2.5x across the shot lengths of one real film. A flat estimate is
        wrong at both ends: the previous 0.8 + 0.9/Mpx read 7.43GB at 4K, 58%
        too high for a 48-frame shot and 28% too low for a 1000-frame one -
        and it is the low end that OOM-kills the encoder.

        The fit is logarithmic in frames and linear in megapixels, with a fixed
        ~0.4GB of process overhead. Before the safety factor it tracks both the
        4K series above and the 1080p one (1.08GB at 24f to 2.72GB at 960f)
        within 6%; with it, every measured point is over-estimated by 1-23%.
        """
        demand = max(0.30, 0.2455 * math.log(max(frames, 8)) - 0.368)
        ratio = self._LP_MEM_RATIO.get(lp, 1.0)
        return (0.40 + self._megapixels() * demand * ratio) * self._MEM_SAFETY

    def _est_probe_gb(self, frames: int, lp: int = 4) -> float:
        """Peak RSS of ONE probe task, in GB. Same shape as _est_encode_gb.

        A probe is cheaper than a final encode at the same length because
        probe_preset is far faster, but how much cheaper depends on the
        resolution in a way this model cannot see: at 4K, SVT-AV1 forces the
        preset down to M9 and halves its mini-GOP (32 -> 16 frames), which the
        1080p probes do not get. Measured per instance at lp=4:

            4K     24f 2.29GB   48f 2.95GB   96f 3.61GB   120f 3.81GB
            1080p  24f 1.23GB                             120f 1.66GB

        i.e. 4K costs only ~2.9x of 1080p where the frames are 4x the size.
        The fit is therefore taken from the 1080p (expensive per pixel) series,
        which leaves it 10% over at 1080p and 57-79% over at 4K. The 4K slack
        is real but it is the safe direction, and MemCalibration measures it
        away within the first few probes of a job.
        """
        demand = max(0.30, 0.145 * math.log(max(frames, 8)) - 0.0105)
        ratio = self._LP_MEM_RATIO.get(lp, 1.0)
        return (0.40 + self._megapixels() * demand * ratio) * self._MEM_SAFETY

    def _mem_bounded_workers(self, per_instance_gb: float) -> int:
        """How many encoder instances fit in the memory budget.

        Also capped at cores//4, because throughput saturates long before that
        anyway - measured at 4K on 32 cores, 3 workers already reach 23.8fps and
        4 or 6 add nothing.
        """
        cores = self._cores()
        return max(1, min(int(self._mem_budget_gb() / max(0.5, per_instance_gb)),
                          max(1, cores // 4)))

    # lp sizes the frame buffer pool, and the top of its range does not pay for
    # itself: at 4K the pool jumps from 107 pictures (lp=4) to 305 (lp=6) while
    # one instance alone goes 6.7GB -> 8.8GB and gets no faster. So the ladder
    # starts at 4. Descending, because admission walks it looking for the
    # cheapest way to fit a shot that does not fit at the top.
    _ENCODE_LP_LADDER = (4, 3, 2, 1)

    def _lp_ladder(self) -> List[int]:
        return [lp for lp in self._ENCODE_LP_LADDER if lp <= self._cores()] or [1]

    def _max_concurrency(self, num_shots: int) -> int:
        """Ceiling on instances in flight, before the budgets are consulted.

        Only an explicit encode_workers pins this now; otherwise the memory and
        CPU budgets decide, per shot, in plan_admission.
        """
        w = self.opt.encode_workers
        if not w or w <= 0:
            w = max(1, self._cores())
        return max(1, min(w, num_shots))

    def _affinity_prefix(self, slot: int, threads: int) -> List[str]:
        """taskset prefix pinning an instance to a disjoint slice of `threads`
        cores. Empty when the slice covers every core, letting SVT-AV1 use all
        cores without extra subprocess overhead.

        Only used when encode_threads is set explicitly. Static core slices and
        variable concurrency do not mix: sized for N instances they strand
        cores whenever fewer than N are running, and measured that way the
        admission scheduler came out 5% SLOWER than the fixed pool it replaced
        purely from the stranding. With lp bounded by the CPU budget instead,
        the instances in flight cannot oversubscribe the cores by construction.

        That holds for the SVT-AV1 encode, which is what this pins. It does
        NOT hold for the decode side of a probe: nothing here passes -threads,
        so ffmpeg sizes its own frame threads from the host core count, not
        from the lp this task was admitted for. At 10 probes in flight on 40
        cores that is a real oversubscription, and it is why the admission
        accounting reads lower than the machine actually behaves. Measuring
        before capping it, rather than capping it and hoping.
        """
        cores = self._cores()
        if threads >= cores:
            return []
        start = (slot * threads) % cores
        end = start + threads - 1
        if end >= cores:
            start, end = 0, threads - 1
        return ["taskset", "-c", f"{start}-{end}"]

    # Real memory kept unreserved on top of the budget's own accounting. The
    # cost model covers encoder processes and nothing else, while a running job
    # also holds page cache for a multi-GB source read, the tmpfs shards the DV
    # and SSIMULACRA2 paths stage, and the metric process.
    _HEADROOM_GB = 1.0

    def _schedule(self, tasks: List[int], *, phase: str,
                  cost: Callable[[int, int], float],
                  run_one: Callable[[int, int, int, int], object],
                  on_done: Callable[[int, object], None],
                  progress: Callable[[int], None],
                  max_conc: int, ladder: List[int],
                  cpu_charge: float = 1.0) -> None:
        """Run `tasks` concurrently, admitting each against a memory/CPU budget.

        Shared by probing and encoding because both have the same shape: many
        independent per-shot tasks whose cost varies by an order of magnitude
        with the shot's length. A fixed worker pool has to be sized for the
        worst task, which then under-uses the budget on every other one.

        Two things bound admission. The BUDGET is bookkeeping - what the cost
        model says the tasks in flight have reserved. Real headroom is checked
        as well, because the model deliberately covers only the encoder
        processes; when the two disagree the tighter one wins, so page cache,
        tmpfs shards and anything else sharing the cgroup push back on their
        own without needing to be modelled.
        """
        cal = self._cal[phase]
        budget = self._mem_budget_gb()
        threads = self.opt.encode_threads if (self.opt.encode_threads or 0) > 0 else 0
        pending: List[int] = list(tasks)
        errors: List[BaseException] = []
        started: List[threading.Thread] = []
        state = {"mem": budget, "cpu": float(self._cores()), "live": 0,
                 "done": 0, "peak_conc": 0}
        cv = threading.Condition()
        squeezed = [False]

        def corrected(key: int, lp: int) -> float:
            return cost(key, lp) * cal.factor()

        def _worker(key: int, lp: int, gb: float, raw: float) -> None:
            slot = self._take_slot() if threads else -1
            result: object = None
            err: Optional[BaseException] = None
            self._begin_task_peak()
            try:
                result = run_one(key, lp, slot, threads)
            except BaseException as e:  # noqa: BLE001 - re-raised on the caller
                err = e
            finally:
                observed = self._end_task_peak()
                if slot >= 0:
                    self._free_slot(slot)
            with cv:
                if err is not None:
                    errors.append(err)
                else:
                    on_done(key, result)
                    state["done"] += 1
                    cal.observe(raw, observed)
                state["mem"] += gb
                state["cpu"] += lp * cpu_charge
                state["live"] -= 1
                cv.notify_all()

        last_reported = -1
        try:
            while True:
                with cv:
                    if errors or (not pending and state["live"] == 0):
                        break
                    pick = None
                    if pending and state["live"] < max_conc:
                        real = sysres.memory_available_gb() - self._HEADROOM_GB
                        if real < state["mem"] and not squeezed[0]:
                            squeezed[0] = True
                            self._log(
                                f"{phase}: real headroom {real:.1f}GB is below "
                                f"the {state['mem']:.1f}GB the budget still "
                                f"shows free; admitting against the smaller")
                        pick = plan_admission(pending, min(state["mem"], real),
                                              state["cpu"], corrected, ladder,
                                              idle=state["live"] == 0,
                                              cpu_charge=cpu_charge)
                    if pick is None:
                        cv.wait(timeout=0.5)
                    else:
                        pos, lp, gb = pick
                        key = pending.pop(pos)
                        raw = cost(key, lp)
                        if gb > state["mem"]:
                            logger.warning(
                                "optimizer: {} task needs ~{:.1f}GB but only "
                                "{:.1f}GB of the {:.1f}GB budget is free; "
                                "starting it anyway because nothing else is "
                                "running", phase, gb, state["mem"], budget)
                        state["mem"] -= gb
                        state["cpu"] -= lp * cpu_charge
                        state["live"] += 1
                        state["peak_conc"] = max(state["peak_conc"], state["live"])
                        th = threading.Thread(target=_worker,
                                              args=(key, lp, gb, raw),
                                              daemon=True)
                        started.append(th)
                        th.start()
                    done_now = state["done"]
                self._check_cancel()
                if done_now != last_reported:
                    last_reported = done_now
                    progress(done_now)
        finally:
            # a failure or a cancel leaves subprocesses running; stop them
            # before joining, or this blocks for as long as the longest task.
            if errors or (self.cancel_flag and self.cancel_flag()):
                self._kill_all()
            for th in started:
                th.join()
        self._sched_peak_conc = state["peak_conc"]
        self._sched_budget = budget
        if errors:
            raise errors[0]
        # The loop breaks on re-entry, so a task that finished while the main
        # thread was between iterations is counted but never reported - which
        # for the LAST task means the phase ends showing less than 100%.
        if state["done"] != last_reported:
            progress(state["done"])

    def _take_slot(self) -> int:
        """Lowest free affinity slot (only meaningful with encode_threads set)."""
        with self._slot_lock:
            slot = 0
            while slot in self._slots_used:
                slot += 1
            self._slots_used.add(slot)
            return slot

    def _free_slot(self, slot: int) -> None:
        with self._slot_lock:
            self._slots_used.discard(slot)

    def _check_cancel(self) -> None:
        if self.cancel_flag and self.cancel_flag():
            self._kill_all()
            raise TranscodeError("Job cancelled by user")

    # ---------- subprocess ----------
    @staticmethod
    def _rss_kb(pid: int) -> int:
        """Current RSS of a process in kB, 0 if it is gone."""
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except (OSError, ValueError):
            return 0
        return 0

    def _py_rss_mb(self) -> float:
        return self._rss_kb(os.getpid()) / 1024.0

    # ---------- per-task memory accounting (feeds MemCalibration) ----------
    def _note_task_peak(self, gb: float) -> None:
        """Record a subprocess peak against the scheduled task that ran it.

        A task may run several subprocesses in sequence - a probe encode then
        a metric pass, or a shard extraction first - and what it costs the
        budget is the heaviest of them, not their sum, since they do not
        overlap. Kept on a thread-local because tasks run concurrently.
        """
        cur = getattr(self._task_peak, "gb", None)
        if cur is not None and gb > cur:
            self._task_peak.gb = gb

    def _begin_task_peak(self) -> None:
        self._task_peak.gb = 0.0

    def _end_task_peak(self) -> float:
        gb = getattr(self._task_peak, "gb", 0.0)
        self._task_peak.gb = None
        return gb

    def _children_rss_mb(self) -> float:
        total = 0.0
        with self._proc_lock:
            procs = list(self._procs)
        for p in procs:
            total += self._rss_kb(p.pid) / 1024.0
        return total

    def _start_mem_sampler(self) -> Tuple["threading.Event", List[float]]:
        """Background sampler tracking peak total child RSS while a parallel
        phase runs. Caller must stop.set() when done, then read peak[0]."""
        stop = threading.Event()
        peak = [0.0]

        def _sample() -> None:
            while not stop.is_set():
                total = self._children_rss_mb()
                if total > peak[0]:
                    peak[0] = total
                stop.wait(2.0)

        threading.Thread(target=_sample, daemon=True).start()
        return stop, peak

    def _monitor_peak_rss(self, pid: int) -> Tuple["threading.Event", List[float], "threading.Thread"]:
        """Sample a child's live VmRSS until stopped; returns (stop, peak, thread)."""
        peak = [0.0]
        stop = threading.Event()

        def _sample() -> None:
            while not stop.is_set():
                kb = self._rss_kb(pid)
                if kb > peak[0]:
                    peak[0] = kb
                stop.wait(0.4)

        t = threading.Thread(target=_sample, daemon=True)
        t.start()
        return stop, peak, t

    @staticmethod
    def _cmd_desc(args: List[str]) -> str:
        """Short descriptor for a subprocess: the output filename, or 'vmaf'."""
        out = str(args[-1]) if args else ""
        if out in ("-", "null", ""):
            return "vmaf"
        return Path(out).name

    def _run(self, args: List[str], timeout: Optional[int] = None) -> str:
        self._check_cancel()
        self._log("$ " + " ".join(map(str, args)))
        try:
            proc = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace", start_new_session=True,
            )
        except FileNotFoundError:
            raise TranscodeError(f"command not found: {args[0]}")
        with self._proc_lock:
            self._procs.add(proc)
        stop, peak, mon = self._monitor_peak_rss(proc.pid)
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._terminate(proc)
            # release the pipes so communicate()'s reader threads and fds
            # don't linger after we abandon the process
            for stream in (proc.stdout, proc.stderr):
                if stream:
                    try:
                        stream.close()
                    except OSError:
                        pass
            raise CommandTimeout(f"command timed out after {timeout}s: {args[0]}")
        finally:
            with self._proc_lock:
                self._procs.discard(proc)
            stop.set()
            mon.join(timeout=2)
        peak_mb = peak[0] / 1024.0
        desc = self._cmd_desc(args)
        if peak_mb > self._heaviest_cmd[0]:
            self._heaviest_cmd = (peak_mb, desc)
        self._note_task_peak(peak_mb / 1024.0)
        self._mem_log(f"[mem] {desc} peak={peak_mb:.0f}MB rc={proc.returncode}",
                      level="debug")
        # keep the job log small: ffmpeg's full stdout (SVT config dumps,
        # progress bars) is dropped on success; the failure path below still
        # surfaces the tail of the output.
        if proc.returncode != 0:
            msg = f"{args[0]} failed (rc={proc.returncode}):\n{out[-2000:]}"
            if proc.returncode == -9:
                msg += (
                    "\nHint: the encoder was killed (likely out of memory). "
                    "Lower transcode.optimizer.encode_workers (parallel final "
                    "encodes; each 4K instance holds a multi-GB frame pool) or "
                    "probe_workers."
                )
            raise TranscodeError(msg)
        return out

    def _terminate(self, proc: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), 15)  # SIGTERM
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    def _kill_all(self) -> None:
        with self._proc_lock:
            procs = list(self._procs)
        for proc in procs:
            self._terminate(proc)

    def close(self) -> None:
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

    # ---------- phase 1: scene detection ----------
    # A plain "W:H" scale spec, with -1/-2 meaning "derive from the other side
    # and round to that multiple". Anything more elaborate is left alone.
    _SCALE_WH = re.compile(r"^\s*(-?\d+)\s*:\s*(-?\d+)\s*$")

    def _scaled_size(self, spec: str) -> Optional[Tuple[int, int]]:
        """Explicit (w, h) that a `W:H` scale spec produces, or None.

        The hardware path has to be told the size outright: scale_qsv's own
        `w=-1` rounds to its surface alignment rather than to the value the
        software `scale` filter would pick, and a detection copy of a different
        size is a different copy - which for a detector reading it frame by
        frame means different cuts.
        """
        m = self._SCALE_WH.match(spec or "")
        sw, sh = (self.info.width or 0), (self.info.height or 0)
        if not m or sw <= 0 or sh <= 0:
            return None
        w, h = int(m.group(1)), int(m.group(2))
        if w > 0 and h > 0:
            return w, h
        if w > 0:
            mult = abs(h) if h < 0 else 1
            return w, max(mult, round(sh * w / sw / mult) * mult)
        if h > 0:
            mult = abs(w) if w < 0 else 1
            return max(mult, round(sw * h / sh / mult) * mult), h
        return None

    def _detection_copy_cmds(self, out: Path, scale: str) -> List[Tuple[str, List[str]]]:
        """(label, ffmpeg args) to try in order for the detection copy.

        The hardware variant decodes and scales on the GPU but still encodes
        with x264. Encoding on the GPU too is barely faster and its artefacts
        move more cuts, so the encoder stays put.

        NB this does NOT reproduce the software copy exactly. Only the scaler
        can be kept, not both: on a discrete card the frames have to come back
        over PCIe, and reading them at full 4K to scale with swscale costs more
        than decoding them on the CPU did - measured 53.6s against software's
        23.0s. Scaling on the GPU keeps the readback small and is the only
        variant that actually wins, and its scaler is not swscale, so borderline
        cuts can land differently. Measured end to end on 90-second clips:

            hevc 3840x2160 SDR   46.4s -> 17.2s   22 shots, identical
            hevc 3840x2160 HDR   29.5s -> 20.1s   33 -> 34 shots
            hevc 3840x1606 HDR   19.3s -> 25.2s   13 shots, identical

        One shot in 33 is the same order as the 540p downscale this pass already
        does (2 cuts of 59 against detecting at native resolution), so it is
        within what the detection copy already costs - but it is a change, and
        scenedetect_hwaccel=off turns it off.

        The last row is the other half of the trade: below 2160p the decode is
        cheap enough that the x264 pass dominates and the wall clock gets worse,
        though the CPU cost still drops about fivefold.
        """
        # -g 9999: x264 defaults to a keyframe every 250 frames, and the
        # quality jump at each IDR reads as a content change to the detector -
        # measured, it invents a shot boundary landing exactly on a keyframe
        # (frame 250 on one clip, 250 and 750 on another; 1 of 104 shots over
        # five minutes). The copy is only ever read forward, so nothing needs
        # the keyframes and dropping them removes the artefact.
        keyint = ["-g", "9999"]
        sw = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
              "-i", str(self.source), "-vf", f"scale={scale}",
              "-c:v", "libx264", "-preset", "ultrafast", *keyint, "-an", "-sn",
              "-f", "matroska", str(out)]
        cmds: List[Tuple[str, List[str]]] = []
        size = self._scaled_size(scale)
        if (self.opt.scenedetect_hwaccel or "auto").lower() != "off" and size:
            w, h = size
            # No -qsv_device: with one render node passed into the container
            # ffmpeg picks it, and naming a fixed /dev/dri/renderDNN here would
            # be wrong on any other host.
            cmds.append(("qsv", [
                self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-hwaccel", "qsv", "-hwaccel_output_format", "qsv",
                "-i", str(self.source),
                "-vf", f"scale_qsv=w={w}:h={h}:format=p010le,"
                       f"hwdownload,format=p010le",
                "-c:v", "libx264", "-preset", "ultrafast", *keyint,
                "-pix_fmt", "yuv420p10le", "-an", "-sn",
                "-f", "matroska", str(out)]))
        cmds.append(("software", sw))
        return cmds

    def _make_detection_copy(self) -> Optional[Path]:
        """Downscale the source so PySceneDetect/OpenCV isn't decoding 4K
        frame-by-frame (that is unusably slow on a 4K HEVC source). The copy
        keeps the same fps and frame count, so cut frame numbers map 1:1 back
        to the source. Reports progress while ffmpeg runs."""
        scale = (self.opt.scenedetect_scale or "").strip()
        if not scale:
            return None
        out = self.probe_dir / "detect_copy.mkv"
        total_sec = max(self.info.duration, 1.0)
        cmds = self._detection_copy_cmds(out, scale)
        for i, (label, args) in enumerate(cmds):
            last = i == len(cmds) - 1
            try:
                self._run_with_progress(args, timeout=7200,
                                        total_seconds=total_sec,
                                        tag=f"downscale for detection ({label})")
                if out.exists() and out.stat().st_size > 0:
                    return out
                # ffmpeg can exit 0 having written nothing - a QSV decode the
                # card cannot do (AV1 on this one) ends exactly that way.
                reason = "produced no output"
            except TranscodeError as e:
                reason = str(e).splitlines()[0]
            if last:
                raise TranscodeError(
                    f"scene detection downscale failed ({label}): {reason}")
            logger.info("optimizer: {} detection downscale unavailable ({}); "
                        "falling back", label, reason)
        return None

    def _run_with_progress(self, args: List[str], timeout: int,
                           total_seconds: float, tag: str) -> None:
        """Run ffmpeg with -progress and report out_time progress to the UI."""
        args = args + ["-progress", "pipe:1", "-nostats"]
        self._log("$ " + " ".join(map(str, args)))
        self._check_cancel()
        try:
            proc = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace", start_new_session=True,
            )
        except FileNotFoundError:
            raise TranscodeError(f"command not found: {args[0]}")
        with self._proc_lock:
            self._procs.add(proc)
        stop, peak, mon = self._monitor_peak_rss(proc.pid)
        last_pct = -1.0
        _noise = ("frame=", "fps=", "stream_", "bitrate=", "total_size=",
                  "out_time_ms=", "out_time=", "dup_frames=", "drop_frames=",
                  "speed=", "progress=", "out_time_us=")
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.rstrip()
                if line.startswith("out_time_us="):
                    try:
                        sec = int(line.split("=", 1)[1]) / 1e6
                        pct = min(sec / max(total_seconds, 1.0) * 100, 100.0)
                        if pct > last_pct + 0.5:
                            last_pct = pct
                            self._report(pct, int(sec), int(total_seconds))
                    except ValueError:
                        pass
                elif line and not line.startswith(_noise):
                    self._log(line)
        finally:
            stop.set()
            mon.join(timeout=2)
            with self._proc_lock:
                self._procs.discard(proc)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._terminate(proc)
                for stream in (proc.stdout, proc.stderr):
                    if stream:
                        try:
                            stream.close()
                        except OSError:
                            pass
                raise TranscodeError(f"command timed out: {args[0]}")
        self._mem_log(f"[mem] {tag} ({self._cmd_desc(args)}) peak={peak[0] / 1024:.0f}MB rc={rc}",
                      level="debug")
        if rc != 0:
            raise TranscodeError(f"{args[0]} failed (rc={rc}): {tag}")

    def _detect_frame(self, video) -> int:
        """Current frame number decoded by a PySceneDetect VideoStream."""
        try:
            return int(video.frame_number)
        except Exception:
            try:
                return int(video.position.frame_num)
            except Exception:
                return 0

    def detect_shots(self) -> List[Shot]:
        """Shot boundaries, then the merging every engine shares.

        scdet reads the source through ffmpeg in one pass and stages nothing;
        pyscenedetect writes a downscaled copy and reads it back with OpenCV.
        See _detect_shots_scdet for what the two measured against each other.
        """
        engine = (self.opt.scenedetect_engine or "scdet").lower()
        if engine == "scdet":
            shots = self._detect_shots_scdet()
        else:
            shots = self._detect_shots_pyscenedetect()
        if not shots:
            shots = [(0, self.total_frames)]
        detected = len(shots)
        shots = merge_short_shots(shots, self.opt.min_shot_frames)
        shots = merge_to_max(shots, max(1, self.opt.max_shots))
        self._validate_shots(shots)
        if len(shots) != detected:
            logger.info(
                "optimizer: {} shot(s) detected -> {} after merging "
                "(min_shot_frames={}, max_shots={})",
                detected, len(shots), self.opt.min_shot_frames,
                self.opt.max_shots)
        else:
            logger.info("optimizer: {} shot(s) detected ({})", len(shots), engine)
        return shots

    # "frame:123 pts:... " and "lavfi.scd.score=1.234" from metadata=print
    _SCD_FRAME = re.compile(r"^frame:(\d+)(?:\s+pts:\S+\s+pts_time:(-?[\d.]+))?")
    _SCD_SCORE = re.compile(r"^lavfi\.scd\.score=([\d.]+)")

    def _scdet_cmds(self) -> List[Tuple[str, List[str]]]:
        """(label, ffmpeg args) to try for a scdet pass over the source.

        One pass, no staged copy: scdet is a filter, so the frames never leave
        ffmpeg and nothing is written to disk. Downscaling first is purely for
        speed - measured, detection at 540p and at full resolution pick the same
        cuts at the same threshold.
        """
        scale = (self.opt.scenedetect_scale or "").strip()
        size = self._scaled_size(scale) if scale else None
        chain_sw = ([f"scale={scale}"] if scale else []) + [
            "scdet=threshold=100", "metadata=print:file=-"]
        sw = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
              "-i", str(self.source), "-map", "0:v:0",
              "-vf", ",".join(chain_sw), "-f", "null", "-"]
        cmds: List[Tuple[str, List[str]]] = []
        if (self.opt.scenedetect_hwaccel or "auto").lower() != "off" and size:
            w, h = size
            cmds.append(("qsv", [
                self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
                "-hwaccel", "qsv", "-hwaccel_output_format", "qsv",
                "-i", str(self.source), "-map", "0:v:0",
                "-vf", f"scale_qsv=w={w}:h={h}:format=nv12,hwdownload,"
                       f"format=nv12,scdet=threshold=100,metadata=print:file=-",
                "-f", "null", "-"]))
        cmds.append(("software", sw))
        return cmds

    def _run_scdet(self, args: List[str]) -> List[float]:
        """Per-frame scene-change scores from one ffmpeg pass.

        scdet is run wide open (threshold=100, which never fires) and the score
        is thresholded here instead, so the knob can be changed without another
        pass over the source.
        """
        self._log("$ " + " ".join(args))
        self._check_cancel()
        try:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, text=True,
                                    errors="replace", start_new_session=True)
        except FileNotFoundError:
            raise TranscodeError(f"command not found: {args[0]}")
        with self._proc_lock:
            self._procs.add(proc)
        scores: List[float] = []
        # every frame's presentation time, from the same lines: the only pass
        # that sees the whole timeline, and what _assert_constant_frame_rate
        # judges it by
        self._frame_pts = []
        total = max(self.total_frames, 1)
        last_pct = -1.0
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                m = self._SCD_SCORE.match(line)
                if m:
                    scores.append(float(m.group(1)))
                    continue
                m = self._SCD_FRAME.match(line)
                if m:
                    n = int(m.group(1))
                    if m.group(2) is not None:
                        self._frame_pts.append(float(m.group(2)))
                    pct = min(n / total * 100, 100.0)
                    if pct > last_pct + 0.5:
                        last_pct = pct
                        self._report(pct, n, total)
                    if n % 500 == 0:
                        self._check_cancel()
        finally:
            with self._proc_lock:
                self._procs.discard(proc)
            rc = proc.wait(timeout=60)
        if rc != 0:
            raise TranscodeError(f"scdet pass failed (rc={rc})")
        return scores

    # How far a frame interval may stray from 1/fps before it counts as a
    # missing or a duplicated frame. Matroska rounds timestamps to whole
    # milliseconds, so at 23.976fps the intervals alternate 41 and 42ms
    # (0.98-1.01 periods); a dropped frame is 2.0. Half to one-and-a-half.
    _CFR_INTERVAL = (0.5, 1.5)
    # how far the timestamps' average rate may sit from the container's
    _CFR_RATE_TOLERANCE = 0.005
    # Missing frames are tolerated up to this share of the intervals: the
    # timeline is kept through them (see _slot), so what is lost is only the
    # frame itself. Beyond it the source is variable frame rate in earnest -
    # a clip with every 7th frame dropped is 16% - and refused.
    _CFR_MAX_GAP_SHARE = 0.005

    def _assert_constant_frame_rate(self, pts: List[float]) -> None:
        """Refuse a source whose timestamps this engine cannot follow.

        Everything here maps frame numbers to time: shot boundaries come from
        a frame count, -ss is a frame's slot times 1/fps, and the encoder
        reads emit one frame per decoded frame. The scdet pass prints every
        frame's pts_time on its way past, so the timeline is judged from that
        (the container cannot tell: an mkv with frames missing still reports
        avg_frame_rate == r_frame_rate), and it is also what _slot uses to
        keep seeks exact.

        Three faults, two of them fatal. An interval shorter than half a
        frame is a duplicated or out-of-order timestamp: nothing downstream
        can place that frame, refuse. An average rate away from the
        container's is a wrong period on every frame number, refuse. A gap -
        a frame missing from an otherwise regular timeline - is what real
        files have: a 4K Blu-ray remux had 2 in 68686 intervals. Those are
        kept: the missing frame's slot stays empty in the output so the
        picture does not creep against the audio, and seeks are taken from
        the real timestamps. Only a source with more than
        _CFR_MAX_GAP_SHARE of its intervals missing is refused as variable
        frame rate. Measured on a 30s clip with every 7th frame dropped,
        before any of this: the job reported success, delivered VMAF median
        54, and the video track ended 4.2s before its audio.

        Only the scdet engine has the timestamps; the PySceneDetect path sees
        none and gets no check.
        """
        n = len(pts)
        if n < 3:
            if n == 0:
                logger.warning("optimizer: the scene-detection pass reported no "
                               "timestamps, so the source's frame rate could not "
                               "be verified as constant")
            return
        period = 1.0 / self.fps
        lo, hi = self._CFR_INTERVAL
        short, gaps = [], []
        for i, (a, b) in enumerate(zip(pts, pts[1:])):
            d = b - a
            if d < lo * period:
                short.append((i + 1, d))
            elif d > hi * period:
                gaps.append((i + 1, d))
        span = pts[-1] - pts[0]
        measured = (n - 1) / span if span > 0 else 0.0
        rate_off = abs(measured - self.fps) / self.fps
        what = None
        if short:
            i, d = short[0]
            what = (f"{len(short)} of {n - 1} frame intervals are shorter than half a "
                    f"frame (the first is frame {i} at {pts[i]:.3f}s, {d * 1000:.1f}ms "
                    f"after the previous one), which is a duplicated or out-of-order "
                    f"timestamp")
        elif len(gaps) > self._CFR_MAX_GAP_SHARE * (n - 1):
            # judged before the rate: this many holes is what makes the rate
            # wrong, and naming them is the useful message
            i, d = gaps[0]
            what = (f"{len(gaps)} of {n - 1} frame intervals are longer than one frame "
                    f"(the first is frame {i} at {pts[i]:.3f}s, {d * 1000:.0f}ms where "
                    f"{period * 1000:.1f}ms is one frame); the timestamps average "
                    f"{measured:.3f} fps against the container's {self.fps:g} and the "
                    f"picture would run {measured / self.fps * 100 - 100:+.1f}% against "
                    f"the audio")
        elif rate_off > self._CFR_RATE_TOLERANCE:
            what = (f"its timestamps average {measured:.3f} fps against the container's "
                    f"{self.fps:g}, so every frame number would be converted with the "
                    f"wrong period")
        if what:
            raise TranscodeError(
                f"engine=optimizer needs a constant frame rate and {self.source.name} "
                f"does not have one: {what}. Use engine=av1an for it, or convert it to "
                f"a constant frame rate first.")
        if gaps:
            where = ", ".join(f"frame {i} ({pts[i]:.1f}s, {d * 1000:.0f}ms)" for i, d in gaps[:5])
            logger.warning(
                "optimizer: {} has {} frame(s) missing from its timeline ({}{}); their "
                "slots are kept empty in the output so the picture stays in step with "
                "the audio, and seeks follow the real timestamps", self.source.name,
                len(gaps), where, ", ..." if len(gaps) > 5 else "")
            self._log(f"timeline: {n} frames at {measured:.4f} fps (container says "
                      f"{self.fps:g}), {len(gaps)} gap(s): {where}")
        else:
            self._log(f"timeline: {n} frames at a constant {measured:.4f} fps "
                      f"(container says {self.fps:g})")

    def _slots_from_pts(self, pts: List[float]) -> List[int]:
        """Each frame's position on the 1/fps grid, counted from the first."""
        if len(pts) < 2:
            return []
        p0 = pts[0]
        return [int(round((t - p0) * self.fps)) for t in pts]

    def _span(self, w0: int, w1: int) -> float:
        """Seconds from frame w0's slot to frame w1's: the -t of a window."""
        return (self._slot(w1) - self._slot(w0)) / self.fps

    def _slot(self, frame: int) -> int:
        """The timeline slot of `frame`: the frame number itself on a regular
        timeline, one more for every missing frame before it (see
        _assert_constant_frame_rate). What every -ss and -t here is made of,
        for the source and the output alike - the output keeps the same
        holes, by construction (see _exact_frames and concat_shots)."""
        slots = self._slots
        if not slots:
            return frame
        if frame < len(slots):
            return slots[frame]
        return slots[-1] + (frame - (len(slots) - 1))

    def _cuts_from_scores(self, scores: List[float]) -> List[int]:
        """Frames whose score clears the threshold, min_scene_len apart."""
        th = float(self.opt.scdet_threshold)
        gap = max(1, int(self.opt.min_scene_len))
        cuts: List[int] = []
        last = -gap
        for i, v in enumerate(scores):
            if i > 0 and v >= th and i - last >= gap:
                cuts.append(i)
                last = i
        return cuts

    def _detect_shots_scdet(self) -> List[Shot]:
        """Shot boundaries from ffmpeg's scdet, without staging a copy.

        Measured against the PySceneDetect path on three 4K sources at
        threshold 2.0: 29/29 boundaries identical on one, 28 of 29 with one
        extra on another, 13 of 13 with one extra on the third. Where the two
        disagreed at the frame level, scdet was reading ~0.0 - no pixels had
        changed - which is what led to the keyframe artefact fixed separately.

        The whole detection phase drops from 44.7s and 255s of CPU to 12.2s and
        8s on a 2160p source, and writes nothing to disk.

        Threshold: 2.0 tracks the old detector closely. 0.8-1.5 picks up softer
        transitions (measured peak 1.5-4.5 over a single frame). Below ~0.5 the
        extra hits are broad and shallow - peak ~0.5 spread over 4-5 frames,
        which is camera motion or a lighting change rather than a cut - and
        they also start displacing correct boundaries, because min_scene_len
        merging takes the first candidate it sees rather than the strongest.
        """
        scores: List[float] = []
        cmds = self._scdet_cmds()
        for i, (label, args) in enumerate(cmds):
            last = i == len(cmds) - 1
            try:
                scores = self._run_scdet(args)
                if scores:
                    break
                reason = "produced no frames"
            except TranscodeError as e:
                reason = str(e).splitlines()[0]
            if last:
                raise TranscodeError(f"scene detection failed ({label}): {reason}")
            logger.info("optimizer: {} scene detection unavailable ({}); "
                        "falling back", label, reason)
        if len(scores) != self.total_frames:
            self._log(f"scdet saw {len(scores)} frames, "
                      f"fps x duration estimated {self.total_frames}")
        self._assert_constant_frame_rate(self._frame_pts)
        self._slots = self._slots_from_pts(self._frame_pts)
        cuts = self._cuts_from_scores(scores)
        bounds = [0] + cuts + [len(scores)]
        return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]

    def _detect_shots_pyscenedetect(self) -> List[Shot]:
        try:
            from scenedetect import ContentDetector, open_video, SceneManager
        except ImportError as e:  # pragma: no cover
            raise TranscodeError(
                "engine=optimizer requires PySceneDetect. "
                "pip install scenedetect (and opencv-python-headless)."
            ) from e
        total = max(self.total_frames, 1)
        det_path = self._make_detection_copy()
        try:
            video = open_video(str(det_path) if det_path else str(self.source))
            sm = SceneManager()
            sm.add_detector(ContentDetector(
                threshold=self.opt.scenedetect_threshold,
                min_scene_len=self.opt.min_scene_len,
            ))

            def on_cut(_frame, _position) -> None:
                nonlocal cuts_found
                cuts_found += 1

            cuts_found = 0

            def _detect() -> None:
                sm.detect_scenes(video, show_progress=False, callback=on_cut)

            thread = threading.Thread(target=_detect, daemon=True)
            thread.start()
            # report real frame progress while detection runs in the background
            while thread.is_alive():
                cur = self._detect_frame(video)
                self._report(min(cur / total * 100, 100.0), cur, total)
                thread.join(timeout=0.5)
            thread.join()
            scenes = sm.get_scene_list()
            return [(int(a.frame_num), int(b.frame_num)) for a, b in scenes]
        finally:
            if det_path is not None:
                try:
                    det_path.unlink()
                except OSError:
                    pass

    def _validate_shots(self, shots: List[Shot]) -> None:
        """Check the shot list covers the source exactly once, in order.

        The list decides which frames get encoded at all, and nothing
        downstream can tell a missing scene from a short one: a gap silently
        drops those frames from the output, an overlap encodes them twice, and
        a list that stops early truncates the tail. All three are structural -
        no source produces them legitimately - so they abort here rather than
        surfacing as a duration mismatch hours later.

        The tail is a different case. total_frames is int(fps * duration), an
        ESTIMATE, while the shot list ends at the real frame count of whatever
        scene detection read. So a disagreement there means the estimate was
        off, not that the shots are wrong: warn, and take the shot list as
        authoritative for progress reporting.
        """
        if not shots:
            raise TranscodeError("scene detection produced an empty shot list")
        if shots[0][0] != 0:
            raise TranscodeError(
                f"the shot list starts at frame {shots[0][0]}, not 0: the "
                f"first {shots[0][0]} frame(s) would never be encoded")
        for i, (a, b) in enumerate(shots):
            if b <= a:
                raise TranscodeError(
                    f"shot {i} spans frames [{a}, {b}), which is empty")
        for i, ((_, prev_end), (next_start, _)) in enumerate(zip(shots, shots[1:])):
            if prev_end != next_start:
                kind = "gap" if next_start > prev_end else "overlap"
                raise TranscodeError(
                    f"{kind} between shot {i} (ends at frame {prev_end}) and "
                    f"shot {i + 1} (starts at frame {next_start}): the shot "
                    f"list does not cover the source exactly once")
        covered = shots[-1][1]
        if covered != self.total_frames:
            logger.warning(
                "optimizer: shots cover {} frames but fps x duration estimated "
                "{}; trusting the shot list", covered, self.total_frames)
            self._log(f"total_frames {self.total_frames} -> {covered} (from shots)")
            self.total_frames = covered

    # ---------- probe configuration ----------
    def _probe_grid(self) -> List[int]:
        grid = sorted(set(self.opt.probe_crfs or []))
        if not grid:
            raise TranscodeError("transcode.optimizer.probe_crfs is empty")
        probes = self.video.probes
        if probes and 0 < probes < len(grid):
            # Spread the allowed probes evenly and KEEP BOTH ENDPOINTS. Simply
            # truncating (grid[:probes]) drops the high-CRF end, so any target
            # cheaper than the surviving max is clamped to it and the "fewer
            # probes" knob silently biases every shot towards a bigger file.
            if probes == 1:
                grid = [grid[len(grid) // 2]]
            else:
                last = len(grid) - 1
                picks = {round(i * last / (probes - 1)) for i in range(probes)}
                grid = [grid[j] for j in sorted(picks)]
        return grid

    def _probe_preset(self) -> int:
        """The preset probe encodes run at.

        Note this saturates: SVT-AV1 forces the preset down to M9 at 4K (see
        the note on _est_probe_gb), so on 4K sources anything above 9 here is
        the same encode. Raising it to chase a faster probe phase will do
        nothing; probe_max_frames is the knob that actually scales.
        """
        raw = (self.video.probe_video_params or "").strip()
        if not raw:
            return self.opt.probe_preset
        if raw.lower() == "copy":
            return self.video.preset
        m = re.search(r"(?:--)?preset[=\s]+(\d+)", raw)
        if m:
            return int(m.group(1))
        return self.opt.probe_preset

    def _probe_scale(self) -> str:
        """The probe-side scale spec, or "" for none. "0", "none" and "off"
        read as none too: the settings page's placeholder shows 960x540, and
        someone clearing it typed 0, which ffmpeg's scale filter rejects as
        an invalid size - every probe of the job would have failed."""
        for v in (self.video.probe_res, self.opt.probe_scale):
            v = str(v or "").strip()
            if v and v.lower() not in ("0", "none", "off"):
                return v
        return ""

    def _probing_rate(self) -> int:
        rate = self.video.probing_rate or self.opt.probing_rate
        return max(1, rate)

    def _probe_window(self, s0: int, s1: int) -> Tuple[int, int]:
        """The frame window of a shot that is actually probed.

        Probes now encode at the source resolution (see _probe_input), so a
        minutes-long take would cost minutes of 4K encoding per CRF. Long shots
        are therefore probed over a bounded CONTIGUOUS window taken from the
        middle of the shot rather than by subsampling the whole of it:
        subsampling widens the gap between consecutive frames, which makes
        inter prediction artificially hard and biases the measured quality
        downwards (i.e. towards a lower CRF and a larger file).
        """
        rate = self._probing_rate()
        want = max(1, self.opt.probe_max_frames) * rate
        n = max(1, s1 - s0)
        if n <= want:
            return s0, s1
        start = s0 + (n - want) // 2
        return start, start + want

    def _pix_fmt(self) -> str:
        """ffmpeg pixel format for probe + final encodes.

        VideoParams spells 8-bit as "yuv420p8le", which is not an ffmpeg
        format name; everything else passes through.
        """
        pf = self.video.pixel_format
        return "yuv420p" if pf == "yuv420p8le" else pf

    # ---------- Dolby Vision Profile 5 shards ----------
    def _shard_bytes(self, frames: int) -> int:
        """Rough size of a lossless FFV1 shard, for the tmpfs fit check.

        Measured 1.4-1.8MB per 4K 10-bit frame on real content; 0.35MB per
        megapixel per frame stays comfortably above that.
        """
        return int(frames * self._megapixels() * 0.35 * 1024 * 1024)

    def _shard_dir(self, frames: int) -> Path:
        """Where to put a shard of `frames` frames.

        Preference is tmpfs (p5_cache_dir), which keeps the DV path off disk
        completely: the alternative is a whole-file intermediate that runs to
        ~100GB at 4K for a 45-minute episode, and gets re-read 11 times per
        shot. Falls back to the work dir when the shard would not comfortably
        fit - free space is checked live, so concurrent shards are accounted
        for without a separate budget knob.
        """
        cache = self.settings.transcode.dovi.p5_cache_dir
        if cache:
            try:
                st = os.statvfs(cache)
                free = st.f_bavail * st.f_frsize
                if self._shard_bytes(frames) * 2 < free:
                    return Path(cache)
            except OSError:
                pass
        return self.tempdir

    def _lead_of(self, path: Path) -> float:
        """Seconds `path`'s video stream starts after its container does.

        ffmpeg adds the CONTAINER's start_time (its earliest stream) to an
        input -ss, and the frame numbers this class works in count from the
        first VIDEO frame. Those coincide only while the picture is the first
        thing in the file. When audio or subtitles begin earlier, every seek
        lands `lead * fps` frames early - proven by frame hashes: a clip with
        video at 0.066 and audio at 0 returned frame 17 for _seek(19), and a
        Better Call Saul remux (video 1.955, audio 0.008, PGS 0) returned frame
        953 for _seek(1000). Shot 0 still started right (the seek clamps at
        0), so what came out was every later shot shifted, the shifted-over
        frames encoded twice at the first cut, the tail never encoded, probe
        windows describing the previous shot, and verification comparing
        frames that were never meant to match. -frames:v kept every count
        exact, so none of the checks fired. Of 100 library files sampled, 11
        have such a lead - every Better Call Saul episode, 0.96-1.96s.

        Probed with ffprobe rather than taken from MediaInfo because the encode
        input is not always the analysed file: the Dolby Vision paths read a
        video-only intermediate whose lead is 0 whatever the original's was,
        and verification reads the finished output, which has its own.

        Falls back to 0 - today's behaviour - if ffprobe cannot answer, and
        says so.
        """
        key = str(path)
        if key in self._leads:
            return self._leads[key]
        lead = 0.0
        try:
            proc = subprocess.run(
                [self.settings.tool_path("ffprobe"), "-v", "error",
                 "-select_streams", "v:0",
                 "-show_entries", "format=start_time:stream=start_time",
                 "-of", "json", key],
                capture_output=True, text=True, timeout=120)
            data = json.loads(proc.stdout or "")
            fmt = float(data.get("format", {}).get("start_time") or 0.0)
            streams = data.get("streams") or [{}]
            vid = float(streams[0].get("start_time") or 0.0)
            lead = max(0.0, vid - fmt)
        except (OSError, ValueError, subprocess.SubprocessError, AttributeError) as e:
            logger.warning("optimizer: could not read the video start time of "
                           "{} ({}); assuming the video starts with the "
                           "container", Path(key).name, e)
        if lead > 0:
            logger.info("optimizer: {} video starts {:.3f}s after the "
                        "container; seeks and the final mux account for it",
                        Path(key).name, lead)
        self._leads[key] = lead
        return lead

    def _seek(self, frame: int, path: Optional[Path] = None) -> str:
        """-ss value that reliably lands ON `frame` of `path` (default: the
        encode input), never past it.

        -ss discards frames whose timestamp is below the one asked for, and
        containers store those rounded - Matroska keeps whole milliseconds
        while a 23.976fps frame time is an infinite decimal - so asking for a
        frame's exact time lands just above its stored timestamp often enough
        to skip it. Measured against per-frame hashes of a real 16-shot split,
        6 of 16 shots began one frame late. Half a frame of lead cannot
        overshoot: the preceding frame is a whole period further back, against
        at most 0.5ms of rounding (8x the margin even at 119.88fps, 42x at
        23.976). Measured after the change: 0 of 16 shots slip.

        Plus the file's lead (see _lead_of): frame 0 of a video that begins
        1.955s into its container sits at -ss 1.955, not 0. And on the
        frame's SLOT rather than its number (see _slot), so a frame missing
        earlier in the file does not pull every later seek a frame early.
        """
        lead = self._lead_of(path or self.source)
        return f"{max(0.0, lead + (self._slot(frame) - 0.5) / self.fps):.6f}"

    def _exact_frames(self) -> Tuple[List[str], List[str]]:
        """(video filters, output options) that make an encoder read emit
        exactly one frame per decoded frame, each on its own timeline slot.

        ffmpeg's default constant-frame-rate sync duplicates frames on these
        reads. The half-frame seek lead leaves every decoded frame at +0.5
        frame of the output grid, and Matroska's millisecond timestamps land
        that on either side of the rounding threshold - per shot. Measured on
        a 120-frame probe window: cfr emitted 121 frames ("*** 1 dup!" in the
        debug log, then "Clipping frame in rate conversion by 0.508"), every
        frame after the duplicate paired one off, and the probe pooled 82.41
        where the same read without sync scores 93.97. Timestamps regenerated
        exactly on the grid still got one duplicate. In the final encode
        -frames:v hides it completely: the count is right, one real frame is
        gone, and everything after it is a frame late.

        So frame sync is switched off and the timestamps are the source's
        own, rebased to the first frame in a fine timebase (AVTB is
        microseconds; the millisecond one would round again). Rescaled to the
        encoder's 1/fps that is the frame's slot: the millisecond jitter
        rounds away, and a frame missing from the source leaves its slot
        empty - which is exactly the hole the audio expects. Regenerating
        from the frame INDEX would close that hole and let the picture creep.
        Appended AFTER the read's other filters, which is where the frames
        come out.
        """
        return (["settb=AVTB", "setpts=PTS-STARTPTS"], ["-fps_mode", "passthrough"])

    def _extract_window(self, w0: int, w1: int, dest: Path, *,
                        source: Optional[Path] = None,
                        vf: Optional[List[str]] = None,
                        apply_dv: bool = False) -> None:
        """Copy frames [w0, w1) of `source` (default: the encode input) into a
        lossless file.

        -ss + -frames:v, the same frame-exact pairing the final encode uses; a
        `-t` duration here is what let windows come out a frame short.

        But -frames:v counts the frames that LEAVE the chain, and a probe-side
        `vf` does not emit one per frame it reads: at probing_rate 2 its fps=
        keeps every other frame, so a shard capped by -frames:v alone read
        twice the window and ran on into the next shot - which the probe
        encode and its VMAF reference, both reading the shard, then measured.
        Every DV P5 and SSIMULACRA2 probe at probing_rate > 1 did this.
        Measured on a 23.976fps mkv: [0, 90) came out as source frames
        0..178; with a timeline hole, [120, 240) reached 357. So the window is
        bounded where the frames go IN, by trim ahead of every other filter
        (after -ss has dropped the frames before w0): all 20 windows then held
        exactly the live probe read's frames, at probing_rate 1 as before, and
        the read stops at the window rather than decoding on to the end.
        """
        pre: List[str] = []
        chain: List[str] = [f"trim=end_frame={w1 - w0}", *(vf or [])]
        if apply_dv:
            from app import dovi  # local import avoids a cycle

            pre, dv = dovi.dv_apply_chain(self.settings)
            chain.append(dv)
        exact_vf, exact_out = self._exact_frames()
        chain += exact_vf
        args = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *pre,
                "-ss", self._seek(w0, source), "-i", str(source or self.source),
                "-frames:v", str(w1 - w0), "-map", "0:v:0",
                "-vf", ",".join(chain), *exact_out]
        args += ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p10le",
                 "-an", "-sn", "-f", "matroska", str(dest)]
        self._run(args, timeout=3600)
        if not dest.exists() or dest.stat().st_size == 0:
            raise TranscodeError(
                f"extracting frames [{w0}, {w1}) produced no output")

    def _make_shard(self, idx: int, w0: int, w1: int, vf: List[str], dest: Path) -> None:
        """Apply the DV RPU to frames [w0, w1) and store them losslessly.

        Every probe of a shot reads its frames 11 times over (5 probe encodes,
        5 VMAF reference reads, 1 final encode), so converting once per shot
        and re-reading a small shard costs one libplacebo pass and leaves the
        repeat reads in page cache.
        """
        try:
            self._extract_window(w0, w1, dest, vf=vf, apply_dv=self._p5)
        except TranscodeError as e:
            raise TranscodeError(f"DV shard for shot {idx}: {e}") from e

    def _needs_shard(self) -> bool:
        """Whether this job stages a per-shot reference file.

        Two reasons, both requiring the same machinery: a Dolby Vision P5 base
        layer has to have its RPU applied before it means anything, and
        SSIMULACRA2 reads through bestsource, which indexes a whole file before
        serving frames - pointing that at a multi-GB source once per probe
        would be far more expensive than staging the window.
        """
        return self._p5 or self.metric == "ssimulacra2"

    def _acquire_shard(self, idx: int, w0: int, w1: int, vf: List[str]) -> Optional[Path]:
        """Staged reference shard for a shot, or None when the job reads its
        reference straight from the source.

        One pool task owns a shot for the whole of its probing, so the shard is
        built, used by every probe of that shot and dropped without ever being
        shared: no refcount, no per-shard lock. The dict is only kept so a
        failed phase can find what is still on disk (see _cleanup_shards).
        """
        if not self._needs_shard():
            return None
        dest = self._shard_dir(w1 - w0) / f"dv_{id(self):x}_{idx:05d}.mkv"
        with self._shard_lock:
            self._shards[idx] = dest
        self._make_shard(idx, w0, w1, vf, dest)
        return dest

    def _cleanup_shards(self) -> None:
        """Drop any shard a failed phase left staged.

        Shards live in transcode.dovi.p5_cache_dir (tmpfs by default), NOT
        under tempdir, so the caller's rmtree of the temp directory never
        reaches them. A job that dies mid-probe would otherwise leave multi-GB
        files sitting in /dev/shm - i.e. holding RAM - until the container is
        restarted.
        """
        with self._shard_lock:
            shards = list(self._shards.values())
            self._shards.clear()
        for dest in shards:
            _unlink(dest)

    def _release_shard(self, idx: int) -> None:
        """Drop the shot's shard once its probing is done."""
        if not self._needs_shard():
            return
        with self._shard_lock:
            dest = self._shards.pop(idx, None)
        if dest is not None:
            _unlink(dest)

    def _vmaf_threads(self) -> int:
        """Threads for the libvmaf calculation.

        ffmpeg's libvmaf defaults n_threads to 0, which is single-threaded,
        and the measurement then ends up slower than the probe encode it is
        measuring (7.0s vs 2.1s at n_threads=8 for the same clip, identical
        score). Auto-size it to the cores each probe worker has to itself.

        Note this is the CPU path only - _score_vmaf drops n_threads entirely
        when the SYCL backend is on, where more threads are strictly worse.
        And the frames are no longer 1080p: a source at or above
        vmaf_4k_min_width is scored against the 4k model at native resolution
        (see _vmaf_scale_filter), so this sizes threads for 4K work.
        """
        explicit = self.video.vmaf_threads or self.opt.vmaf_threads
        if explicit:
            return explicit
        cores = self._cores()
        return max(1, cores // max(1, self._probe_worker_count))

    def _verify_threads(self) -> int:
        """libvmaf threads for the verification pass.

        _vmaf_threads() divides the cores by the PROBE pool, which is right
        while that pool is running and wrong two phases later: verification
        runs on its own, after the encode, and would otherwise ask for the
        cores/probe_workers handful it needed when ten probes were competing.

        Half the cores is the knee, and all of them is never better. Measured
        on 32 cores, one 120-frame 3840x1920 window against the 4k model,
        reading both sides in place:

            n_threads      4      8     16     32
            wall       11.2s   6.7s   5.5s   5.8s
            cpu        60.3s  62.1s  66.9s  69.5s

        and the same shape on a second window (12.6 / 7.9 / 6.6 / 7.2s). The
        score is identical at every count, to six decimals.

        An explicit vmaf_threads still wins, as it does for probing.
        """
        explicit = self.video.vmaf_threads or self.opt.vmaf_threads
        return explicit if explicit else max(1, self._cores() // 2)

    def _sycl_device(self) -> int:
        """SYCL device index to hand the libvmaf filter, or -1 for the CPU.

        Gated on three things, in order: the setting, the source being big
        enough to be worth it (see vmaf_sycl_min_width), and the device
        actually being there.
        """
        dev = int(self.opt.vmaf_sycl_device)
        if dev < 0 or self.metric != "vmaf":
            return -1
        min_w = int(self.opt.vmaf_sycl_min_width or 0)
        if min_w and (self.info.width or 0) < min_w:
            return -1
        with self._sycl_lock:
            if self._sycl_ok is None:
                self._sycl_ok = self._sycl_preflight(dev)
            elif (not self._sycl_ok and self._sycl_retired_at is not None
                  and time.time() - self._sycl_retired_at >= self._SYCL_REARM_AFTER):
                # One attempt per cool-off, whatever the outcome: clearing the
                # timestamp first means a device that is really gone is asked
                # once and then left alone for the rest of the job.
                self._sycl_retired_at = None
                if self._sycl_preflight(dev):
                    self._sycl_ok = True
                    self._sycl_timeouts = 0
                    logger.info("optimizer: libvmaf SYCL device {} works again "
                                "after {:.0f}s; scoring returns to the GPU",
                                dev, self._SYCL_REARM_AFTER)
        return dev if self._sycl_ok else -1

    # How far the GPU and CPU backends may disagree on the same pair. Measured
    # across 5 sources x 3 CRFs the worst was 1e-4, and 5e-5 on the preflight's
    # own synthetic pair; 1e-3 leaves room for a sane build without letting a
    # broken one through.
    _SYCL_MAX_DELTA = 1e-3

    def _sycl_preflight(self, device: int) -> bool:
        """Prove the SYCL device works before any probe depends on it.

        The libvmaf filter aborts the ffmpeg run outright when
        vmaf_sycl_state_init fails, which is the behaviour we want - libvmaf's
        own CLI instead falls back to the CPU and returns a perfectly valid
        score, so a broken driver would show up only as everything being three
        times slower. But "abort" once per probe would mean a job that fails
        hundreds of times. So spend one tiny comparison up front: if the device
        is not usable, say so loudly and run the whole job on the CPU.
        """
        log = self.probe_dir / "sycl_preflight.json"
        cpu_log = self.probe_dir / "sycl_preflight_cpu.json"
        # testsrc2, not a flat colour: a black frame has zero variance
        # everywhere, and VIF's log ratios on that are a good way to fail the
        # preflight for a reason that has nothing to do with the device.
        #
        # And the distorted side is blurred, not identical: a pair scored
        # against itself returns 100.000000 from any backend, working or not,
        # which would make the comparison below pure theatre. boxblur puts it
        # near 49 VMAF, where adm/vif/motion all actually run.
        src = "testsrc2=s=256x256:d=1:r=30"
        def _lavfi(dev: Optional[int], path: Path) -> str:
            sycl = f"sycl_device={dev}:" if dev is not None else ""
            return ("[0:v]boxblur=2,format=yuv420p10le[d];"
                    "[1:v]format=yuv420p10le[r];"
                    f"[d][r]libvmaf=model={self._model_cfg()}:{sycl}"
                    f"log_fmt=json:log_path={path}")
        def _cmd(dev: Optional[int], path: Path) -> List[str]:
            return [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", src, "-f", "lavfi", "-i", src,
                    "-lavfi", _lavfi(dev, path), "-f", "null", "-"]
        try:
            out = self._run(_cmd(device, log), timeout=300)
            # Score the same pair on the CPU and compare. libvmaf now comes
            # from a fork pinned to a commit, and a device that runs but
            # computes differently is invisible otherwise: every score shifts
            # together, pick_crf picks differently, and the delivered-vs-
            # predicted report still looks self-consistent because both sides
            # use the same scorer. Costs one more 256x256 pass.
            #
            # Note what this does NOT catch: a VMAFX_REF bump that moves the
            # shared feature code shifts CPU and GPU together and passes. That
            # would need a pinned expected value, which in turn needs a fixed
            # model rather than the job's - worth doing, not done here.
            self._run(_cmd(None, cpu_log), timeout=300)
            gpu, cpu = parse_score(log, "vmaf"), parse_score(cpu_log, "vmaf")
        # OSError/ValueError as well as TranscodeError: ffmpeg can exit 0 and
        # still leave no usable log, and parse_score then raises FileNotFound
        # or JSONDecodeError. Letting those out would kill the job, which is
        # the exact opposite of this function's contract - and "exited fine but
        # wrote nothing" is what a bad build looks like, i.e. the case this
        # exists for.
        except (TranscodeError, OSError, ValueError) as e:
            # A cancel arrives as a TranscodeError too (see _check_cancel), and
            # reporting that as a dead GPU would be a lie in the logs of every
            # cancelled job.
            self._check_cancel()
            detail = "; ".join(
                [ln for ln in str(e).strip().splitlines() if ln.strip()][-3:])
            logger.warning(
                "optimizer: libvmaf SYCL device {} unusable, scoring this job "
                "on the CPU instead ({}). Expect the probe phase to cost "
                "roughly 3x the wall time and 30x the cores it would on the "
                "GPU; fix the device or set vmaf_sycl_device=-1 to silence "
                "this.", device, detail)
            return False
        finally:
            _unlink(log)
            _unlink(cpu_log)
        # Agreeing scores prove the numbers are right; they do not prove the
        # GPU produced them. An ignored sycl_device would give agreement too -
        # and that is not hypothetical, the fork's own ffmpeg patch gated its
        # code behind a CONFIG_ symbol it never defined, so "compiles, runs,
        # silently on the CPU" is a shape this build has already taken once.
        # The backend announces itself on stderr, which _run folds into stdout.
        if "vmaf-sycl" not in out:
            logger.warning(
                "optimizer: libvmaf accepted sycl_device={} but never announced "
                "the SYCL backend, so it is probably scoring on the CPU while "
                "claiming otherwise. Scoring on the CPU explicitly instead.",
                device)
            return False
        if abs(gpu - cpu) > self._SYCL_MAX_DELTA:
            logger.warning(
                "optimizer: libvmaf SYCL device {} scores {:.6f} where the CPU "
                "backend scores {:.6f} on the same pair - a {:.2e} gap against "
                "an expected ~1e-4. The libvmaf build is suspect, so this job "
                "scores on the CPU.", device, gpu, cpu, abs(gpu - cpu))
            return False
        logger.info("optimizer: scoring on libvmaf SYCL device {} (agrees with "
                    "the CPU backend to {:.2e})", device, abs(gpu - cpu))
        return True

    def _use_4k_model(self) -> bool:
        """Whether this source should be scored with the 4K VMAF model."""
        if self.metric != "vmaf" or not self.opt.vmaf_model_4k:
            return False
        return (self.info.width or 0) >= int(self.opt.vmaf_4k_min_width or 0) > 0

    def _vmaf_scale_filter(self) -> str:
        """scale filter applied to BOTH libvmaf inputs before comparison.

        vmaf_v0.6.1 is trained on 1080p viewed at 3H, so scoring a 4K (or a
        540p) pair with it is outside the model's domain and compresses the
        whole CRF range into a couple of points. Downscale-only and
        aspect-preserving: upscaling a smaller source would invent detail, and
        forcing an exact WxH would distort non-16:9 sources (e.g. 3840x1920).

        Not applied when the 4K model is in use - that model IS trained for
        this resolution, and downscaling to reach the 1080p model reads about
        a point optimistic, which the encoder spends as lost sharpness.
        """
        if self._use_4k_model():
            return ""
        w = int(self.opt.vmaf_width or 0)
        if w <= 0:
            return ""
        return f"scale=w='min(iw,{w})':h=-2:flags=bicubic"

    def _vmaf_features(self) -> str:
        return (self.video.probing_vmaf_features or "").strip()

    def _model_cfg(self) -> str:
        """libvmaf model config string for ffmpeg's libvmaf filter.

        Accepts either a plain file path (wrapped as path=...), or an explicit
        libvmaf model config such as "version=ssimulacra2" or "path=/x.json".
        """
        if self.metric == "ssimulacra2":
            raw = self.opt.ssimulacra2_model
        elif self._use_4k_model():
            raw = self.opt.vmaf_model_4k
        else:
            raw = self.opt.vmaf_model
        raw = (raw or "").strip()
        if not raw:
            raw = "version=vmaf_v0.6.1"
        if "=" in raw:
            return raw
        return f"path={raw}"

    # ---------- phase 2: parallel probing ----------
    def _probe_input(self, w0: int, w1: int,
                     shard: Optional[Path] = None,
                     lp: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """(input args, video filters) reading frames [w0, w1) of the source.

        Used identically by the probe encode and by the VMAF reference read, so
        the two are frame-aligned by construction and there is no multi-GB y4m
        intermediate to cache: at 4K 10-bit a y4m frame is 22MB, and the frames
        would have to be re-read once per CRF anyway.

        A DV shard already holds exactly these frames with the probe-side
        filters baked in, so it is read whole and needs no filters of its own.
        """
        # -threads before -i is a DECODER thread count, and it wants to be the
        # lp this task was admitted for. Left to itself ffmpeg sizes it from
        # the host core count, so ten probes admitted 4 cores each open enough
        # frame threads to oversubscribe 40 - which is not merely untidy
        # accounting, it is slower AND more expensive. Measured on 10
        # concurrent 120-frame 4K reads: unbounded 7.18s wall and 261
        # CPU-seconds, against 4.61s and 130 at -threads 4. Both axes.
        threads = ["-threads", str(lp if lp and lp > 0 else self._lp_ladder()[0])]
        if shard is not None:
            return threads + ["-i", str(shard)], []
        # _seek, not w0/fps: the same half-frame lead the final encode and the
        # shard staging already use. Without it the probe reads a window that
        # starts one frame later than the one that actually gets encoded -
        # measured on a 4K mp4, 1 of 12 sampled windows; the note on _seek
        # records 6 of 16 on a Matroska split, whose millisecond timebase is
        # coarser. With it, all 12 come out frame-identical to the encode's
        # read. That matters more than it sounds: the VMAF/CRF curve here runs
        # 0.14-0.36 VMAF per CRF (see pick_crf), so measuring a window the
        # encoder never sees is worth whole CRF steps.
        args = threads + ["-ss", self._seek(w0),
                          "-t", f"{self._span(w0, w1):.6f}",
                          "-i", str(self.source)]
        vf: List[str] = []
        rate = self._probing_rate()
        if rate > 1:
            # sample every nth frame by re-timing to fps/rate. Note: the
            # select='not(mod(n,N))' filter does NOT reliably drop frames on
            # ffmpeg master, while fps= cleanly subsamples.
            vf.append(f"fps={self.fps / rate:g}")
        scale = self._probe_scale()
        if scale:
            vf.append(f"scale={scale}")
        return args, vf

    def _probe_workers(self, n_tasks: int) -> int:
        """Roughly how many probes will be in flight at once.

        Admission decides the real number per shot, but libvmaf still has to be
        given a thread count up front (see _vmaf_threads), so this estimates
        the typical concurrency from the budget and a mid-sized probe window.
        """
        w = self.opt.probe_workers
        if not w or w <= 0:
            typical = self._est_probe_gb(
                max(1, min(self.opt.probe_max_frames or 120, 120)),
                self._lp_ladder()[0])
            w = self._mem_bounded_workers(typical)
        return max(1, min(w, n_tasks))

    def probe_all(self, shots: List[Shot], grid: List[int]) -> ProbeSamples:
        """Probe every shot, one scheduled task per shot.

        Per SHOT rather than per (shot, CRF): the CRFs of a shot are now chosen
        adaptively, so each one depends on the scores before it and they have
        to run in order. Parallelism comes from the shots, of which there are
        hundreds, and the staged reference shard a shot may need is built, used
        and dropped inside one task instead of being shared across a pool.

        Admitted against the same budget as the encode phase. Probe windows run
        from a whole short shot up to probe_max_frames, a 5x span in length and
        roughly 2x in memory, so a fixed pool has the same problem here that it
        had there - and this phase was the one over-committing: sized from a
        flat 4.19GB estimate it ran 5 workers that measured 5.34GB each, 27%
        past the budget it had been given.
        """
        ladder = self._lp_ladder()
        self._probe_worker_count = self._probe_workers(len(shots) or 1)
        scale = self._probe_scale()
        if scale:
            logger.warning(
                "optimizer: probe_scale={!r} makes the probes encode at a "
                "different resolution than the final encode, so the CRF picked "
                "from them does not transfer. Leave it empty unless you are "
                "trading accuracy for speed on purpose.", scale)
        results: ProbeSamples = {}
        self._heaviest_cmd = (0.0, "")
        width = int(self.opt.probe_bracket_width or 0)
        plan = (f"adaptive from {seed_crfs(grid)} down to a {width}-wide bracket"
                if width > 0 else f"the full {len(grid)}-point grid {grid}")
        with self._slot_lock:
            self._slots_used.clear()

        def probe_frames(idx: int) -> int:
            w0, w1 = self._probe_window(*shots[idx])
            return w1 - w0

        def cost(idx: int, lp: int) -> float:
            return self._est_probe_gb(probe_frames(idx), lp)

        def run_one(idx: int, lp: int, _slot: int, _threads: int) -> object:
            s0, s1 = shots[idx]
            return self._probe_shot(idx, s0, s1, grid, lp)

        def on_done(idx: int, result: object) -> None:
            results[idx] = result           # type: ignore[assignment]
            s0, s1 = shots[idx]
            self._dataset_shot(idx, s0, s1, *self._probe_window(s0, s1),
                               svt=result)  # type: ignore[arg-type]

        def progress(done: int) -> None:
            self._report(done / max(len(shots), 1) * 100, done, len(shots))

        self._log(f"probing {len(shots)} shots (budget "
                  f"{self._mem_budget_gb():.1f}GB, lp ladder {ladder}), {plan}")
        stop, peak = self._start_mem_sampler()
        try:
            self._schedule(list(range(len(shots))), phase="probing", cost=cost,
                           run_one=run_one, on_done=on_done, progress=progress,
                           max_conc=self._max_probe_concurrency(len(shots)),
                           ladder=ladder,
                           cpu_charge=self.opt.probe_cpu_charge)
        finally:
            stop.set()
        self._log_phase_memory("probing", peak[0])
        spent = sum(len(v) for v in results.values())
        if results:
            self._mem_log(
                f"probes: {spent} for {len(results)} shot(s), "
                f"{spent / len(results):.2f} per shot "
                f"(a full grid would have been {len(grid)}){self._zc_summary()}")
        return results

    def _max_probe_concurrency(self, n_tasks: int) -> int:
        """Hard cap on probes in flight; the budget decides the real number."""
        w = self.opt.probe_workers
        if not w or w <= 0:
            w = max(1, self._cores())
        return max(1, min(w, n_tasks))

    def _probe_shot(self, idx: int, s0: int, s1: int, grid: List[int],
                    lp: int) -> Dict[int, float]:
        """Probe one shot and return {crf: score}.

        With probe_bracket_width set, this is a bisection rather than a sweep:
        probe a coarse seed, find the two CRFs the target falls between, and
        halve that interval until it is narrow enough or the budget runs out.
        A uniform grid spends the same probes everywhere regardless of where
        the target lands; bisection spends them where the answer is, so the
        interval that decides the CRF ends up at least as tight as the grid's
        for fewer probes - and shots whose target is out of reach altogether
        stop as soon as that is known instead of sweeping to the end.
        """
        self._check_cancel()
        w0, w1 = self._probe_window(s0, s1)
        _, probe_vf = self._probe_input(w0, w1)
        shard = self._acquire_shard(idx, w0, w1, probe_vf)
        scores: Dict[int, float] = {}
        try:
            width = int(self.opt.probe_bracket_width or 0)
            budget = self._probe_budget(grid)
            order = seed_crfs(grid) if width > 0 else sorted(set(grid))
            for crf in order[:budget]:
                _, _, scores[crf] = self._probe_encode_and_score(
                    idx, w0, w1, crf, lp, shard)
            if width <= 0:
                return scores
            while len(scores) < budget:
                self._check_cancel()
                span = bracket_for(list(scores.items()), self.target)
                if span is None or span[1] - span[0] <= width:
                    break
                mid = (span[0] + span[1]) // 2
                if mid in scores:
                    break
                _, _, scores[mid] = self._probe_encode_and_score(
                    idx, w0, w1, mid, lp, shard)
            return scores
        finally:
            self._release_shard(idx)

    def _probe_budget(self, grid: List[int]) -> int:
        """Most probes one shot may spend. The grid already carries the
        video.probes cap (see _probe_grid), so its size is the budget - which
        also means adaptive probing can never cost more than the sweep it
        replaces, only less."""
        return max(2, len(set(grid)))

    def _probe_encode_and_score(self, idx: int, w0: int, w1: int, crf: int,
                                lp: int, shard: Optional[Path]) -> Tuple[int, int, float]:
        ivf = self.probe_dir / f"probe_{idx:05d}_{crf}.ivf"
        in_args, vf = self._probe_input(w0, w1, shard, lp=lp)
        exact_vf, exact_out = self._exact_frames()
        args = ([self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
                + in_args + ["-vf", ",".join(vf + exact_vf)] + exact_out)
        # the probe must use the same encoder configuration as the final encode
        # (tune, film grain, keyint, extra params) or it measures a different
        # rate-distortion curve than the one that gets delivered. Only the
        # preset differs, for speed.
        svt = _svt_params_dict(self.video)
        svt["lp"] = lp
        args += ["-map", "0:v:0", "-c:v", "libsvtav1",
                 "-preset", str(self._probe_preset()), "-crf", str(crf)]
        if self.video.keyint:
            args += ["-g", str(self.video.keyint), "-keyint_min", str(self.video.keyint)]
        args += ["-svtav1-params", ":".join(f"{k}={v}" for k, v in svt.items()),
                 "-pix_fmt", self._pix_fmt(), "-f", "ivf", str(ivf)]
        self._run(args, timeout=3600)
        score = self._score_probe(w0, w1, ivf, idx, crf, shard)
        if not self.opt.keep_probes:
            try:
                ivf.unlink()
            except OSError:
                pass
        return idx, crf, score

    # ------------------------------------------------------------------
    # The GPU probe path. Opt-in through probe_encoder="qsv"; nothing above
    # this block changes behaviour when it is off, and every failure here
    # falls back to the SVT probe rather than degrading a shot silently.
    # ------------------------------------------------------------------

    def _probe_mode(self) -> str:
        """"svt", "qsv" or "qsv+svt" - what the probe phase actually runs.

        Both card modes need a render node; without one they degrade to the
        SVT path rather than failing the job.
        """
        mode = (self.opt.probe_encoder or "svt").lower()
        if mode == "svt":
            return "svt"
        if not self._hwdec_args():
            logger.warning("optimizer: probe_encoder={} needs a /dev/dri render "
                           "node; probing with SVT instead", mode)
            return "svt"
        return mode

    def _gpu_probe_on(self) -> bool:
        """Whether the card is doing probe ENCODES this job - either card
        mode. Sizes the SYCL timeout, which cares that the card is busy and
        not why."""
        return self._probe_mode() in ("qsv", "qsv+svt")

    def _qsv_grid(self) -> List[int]:
        g = sorted(set(self.opt.gpu_probe_qs or []))
        if len(g) < 2:
            raise TranscodeError(
                "transcode.optimizer.gpu_probe_qs needs at least two points")
        return g

    @contextlib.contextmanager
    def _gpu_slot(self, what: str, mb: Optional[float] = None) -> Iterator[None]:
        """Hold room on the card for one operation. The same budget the scores
        book against: encode engine and compute engine are separate, but they
        share the one pool of memory, and it was exhausting that which cost a
        reboot."""
        with self._gpu_vram.hold(mb if mb is not None else self._est_score_mb(None),
                                 self._GPU_SLOT_WAIT, what):
            yield

    def _qsv_probe_encode(self, w0: int, w1: int, q: int, out: Path) -> None:
        """One probe encode that really never leaves the card.

        It used to say that and not do it: VA-API decoded into a hwdownload,
        the 4K frames came back to system memory, and av1_qsv uploaded them
        again. Measured on a 120-frame 4K window that round trip costs 5.39s
        of wall against 4.29s for the SVT probe it was supposed to undercut -
        SLOWER, on 1.2 cores against 8.5 - because the pool is ten wide and a
        probe that idles on a copy holds its slot just as long as one that
        works. That is where the 57% the GPU probe path was losing actually
        went; it was never the CPU, and it was never the admission cap
        (raising it from six to ten moved nothing and the card never went
        above 3.97GB either way).

        Encoding through VA-API instead keeps the frames where they were
        decoded: 0.78s wall and 0.53 CPU-seconds for the same window, 6.9x
        the throughput on a twelfth of the CPU.

        rc_mode=ICQ, not the CQP the option names suggest: with CQP the
        driver ignores -qp outright and returns the same 35690 KiB at 20, 32
        and 44.

        The two encoders' quality indices are NOT the same scale, and an
        earlier note here claiming they agreed "within 2%" was one window at
        one q extrapolated across the grid. Measured shot for shot on the
        same windows at the same q, av1_vaapi spends 1.32x the bytes of
        av1_qsv (331 points, both ends of the grid) and scores +1.44 VMAF
        (279 points); the crossing moves +2.2 q. Nothing breaks, because the
        map refits online from whatever the card produces - but every offline
        number that justified the byte feature was computed from av1_qsv
        output and does not describe what this path now writes.
        """
        vf: List[str] = []
        rate = self._probing_rate()
        if rate > 1:
            vf.append(f"fps={self.fps / rate:.6f}")
        scale = self._probe_scale()
        if scale:
            # on the card the scaler is the card's too, or the frames would
            # have to come back for it
            vf.append(f"scale_vaapi={scale.replace('x', ':')}")
        exact_vf, exact_out = self._exact_frames()
        args = ([self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-threads", "4", *self._hwdec_args(),
                 "-ss", self._seek(w0), "-t", f"{self._span(w0, w1):.6f}",
                 "-i", str(self.source), "-map", "0:v:0"]
                + (["-vf", ",".join(vf + exact_vf)] if (vf or exact_vf) else [])
                + exact_out
                + ["-c:v", "av1_vaapi", "-rc_mode", "ICQ",
                   "-global_quality", str(q), "-f", "ivf", str(out)])
        with self._gpu_slot(f"probe encode q={q}",
                            self._est_qsv_probe_mb(w1 - w0)):
            self._run(args, timeout=3600)

    def _probe_shot_qsv(self, idx: int, s0: int, s1: int,
                        qgrid: List[int]) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Both ends of the q grid. Two probes, and the chord between them.

        NOT the bisection the SVT path runs, and deliberately so. There the
        probed CRF is the answer, so the bracket has to be tight enough to
        interpolate inside. Here q* only feeds a fitted plane, and a chord
        drawn across the whole grid is a perfectly good input to a fit.

        A third, interpolated point used to be probed, on the reasoning that
        the chord must miss a curve that bends - and it does, by -2.46 q on
        average. That measurement was the wrong test. The shift is not noise
        around a mean, it is a LINEAR FUNCTION of q* itself (slope -0.37,
        R^2 0.96 over 156 shots), so refitting simply re-estimates the slope
        and absorbs it. The test that decides is the refit, and it says the
        third probe is worth nothing:

            run    n     3-point LOO   ends-only LOO
            feat  100       2.58           2.53
            wide   99       2.58           2.54
            card  115       2.63           2.63

        So the pass costs two probes a shot rather than three - a third of
        all the card's probe work, and a third of the VMAF scores that go
        with it, which are the part that is actually scarce.

        Ends that do not bracket the target mean the shot has no crossing on
        this curve, and no fit can use it.
        """
        self._check_cancel()
        w0, w1 = self._probe_window(s0, s1)
        lo, hi = min(qgrid), max(qgrid)
        scores: Dict[int, float] = {}
        bpf: Dict[int, float] = {}
        for q in (lo, hi):
            self._check_cancel()
            scores[q], bpf[q] = self._qsv_score(idx, w0, w1, q)
        return scores, bpf

    def _qsv_score(self, idx: int, w0: int, w1: int, q: int) -> Tuple[float, float]:
        """Score one QSV probe, and report what it cost in bytes per frame.

        The bytes are free - the probe writes the file either way - and they
        are the single most valuable thing the card produces after the score
        itself. Measured on a 163-shot 4K episode, predicting the SVT CRF
        from the crossing alone leaves a leave-one-out residual of 4.53 CRF
        (35% of the variance); adding the bytes per frame at the high-q end
        takes it to 2.72 (77%). The two carry different information: the
        crossing says where the curve meets the target, the bytes say what
        the encoder had to spend to get there.
        """
        ivf = self.probe_dir / f"gpuprobe_{idx:05d}_{q}.ivf"
        self._qsv_probe_encode(w0, w1, q, ivf)
        try:
            size = ivf.stat().st_size
        except OSError:
            size = 0
        frames = max(1, (w1 - w0) // self._probing_rate())
        try:
            return self._score_probe(w0, w1, ivf, idx, q), size / frames
        finally:
            if not self.opt.keep_probes:
                try:
                    ivf.unlink()
                except OSError:
                    pass

    def _crossing(self, scores: Dict[int, float]) -> Optional[float]:
        """Where a probed curve crosses the target, or None when it never
        does - clamping would put the answer at an endpoint the probes never
        justified, which is precisely the case this path must abstain on."""
        pts = [(k, v) for k, v in scores.items() if v is not None]
        if bracket_for(pts, self.target) is None:
            return None
        return pick_crf(pts, self.target)

    @staticmethod
    def _theil_sen(pts: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Median-of-slopes fit. Least squares would let one easy shot - the
        kind that reaches the target 15 CRF above the bulk - tilt the line for
        every other shot in the job."""
        slopes = [(y1 - y0) / (x1 - x0)
                  for (x0, y0), (x1, y1) in itertools.combinations(sorted(pts), 2)
                  if x1 != x0]
        if not slopes:
            return 0.0, (pts[0][1] if pts else 0.0)
        a = statistics.median(slopes)
        return a, statistics.median(y - a * x for x, y in pts)

    def _gpu_anchor_indices(self, shots: List[Shot]) -> List[int]:
        """Which shots get probed both ways to fit the mapping.

        Stratified by shot LENGTH rather than spread over the timeline: the
        one case the line was measured to miss is an easy shot whose target
        CRF sits far above the bulk, and shot length is the cheapest handle on
        that kind of variation that costs nothing to compute.
        """
        n = len(shots)
        k = max(4, min(int(self.opt.gpu_probe_anchors or 16), n))
        order = sorted(range(n), key=lambda i: shots[i][1] - shots[i][0])
        if k <= 1:
            return [order[0]]
        return sorted({order[round(j * (n - 1) / (k - 1))] for j in range(k)})

    def _map_crf(self, q: float, fit: Dict[str, float], grid: List[int]) -> float:
        crf = fit["a"] * q + fit["b"] + float(self.opt.probe_crf_offset or 0.0)
        return max(self._crf_floor(min(grid)), min(crf, float(max(grid))))

    def probe_all_gpu(self, shots: List[Shot],
                      grid: List[int]) -> Tuple[ProbeSamples, Dict[int, float]]:
        """Probe on the card and map back to SVT CRF, falling back to the SVT
        path whenever the mapping cannot be trusted - for the whole job if the
        calibration is loose, for a single shot if its quality index lands
        outside the range the anchors actually cover."""
        qgrid = self._qsv_grid()
        ladder = self._lp_ladder()
        anchors = self._gpu_anchor_indices(shots)
        samples: ProbeSamples = {}
        chosen: Dict[int, float] = {}
        pairs: List[Tuple[float, float]] = []
        lock = threading.Lock()
        self._log(f"gpu probing: {len(anchors)} calibration anchor(s) of "
                  f"{len(shots)} shot(s), q grid {qgrid}")

        def cost(idx: int, lp: int) -> float:
            w0, w1 = self._probe_window(*shots[idx])
            return self._est_probe_gb(w1 - w0, lp)

        def anchor_one(idx: int, lp: int, _slot: int, _threads: int) -> object:
            s0, s1 = shots[idx]
            svt = self._probe_shot(idx, s0, s1, grid, lp)
            crf_star = self._crossing(svt)
            qs, _ = self._probe_shot_qsv(idx, s0, s1, qgrid)
            q_star = self._crossing(qs)
            with lock:
                samples[idx] = svt
                if crf_star is not None and q_star is not None:
                    pairs.append((q_star, crf_star))
            return None

        self._schedule(anchors, phase="probing", cost=cost, run_one=anchor_one,
                       on_done=lambda *_: None,
                       progress=lambda d: self._report(d / max(len(shots), 1) * 100,
                                                       d, len(shots)),
                       max_conc=self._max_probe_concurrency(len(anchors)),
                       ladder=ladder, cpu_charge=self.opt.probe_cpu_charge)

        fit = self._fit_gpu_map(pairs)
        rest = [i for i in range(len(shots)) if i not in samples]
        if fit is None:
            self._log("gpu probing: calibration rejected; the rest of this job "
                      "probes with SVT")
            svt_samples = self.probe_all([shots[i] for i in rest], grid)
            for k, idx in enumerate(rest):
                samples[idx] = svt_samples[k]
            return samples, self.pick_all_crfs(samples, grid)

        mapped: Dict[int, float] = {}

        def bulk_one(idx: int, lp: int, _slot: int, _threads: int) -> object:
            s0, s1 = shots[idx]
            q_star = self._crossing(self._probe_shot_qsv(idx, s0, s1, qgrid)[0])
            if q_star is not None and fit["q_lo"] <= q_star <= fit["q_hi"]:
                with lock:
                    mapped[idx] = self._map_crf(q_star, fit, grid)
                return None
            # outside what the anchors cover, or never crossed the target:
            # an extrapolated line is exactly what missed by 10 CRF once
            svt = self._probe_shot(idx, s0, s1, grid, lp)
            with lock:
                samples[idx] = svt
            return None

        done0 = len(anchors)
        self._schedule(rest, phase="probing", cost=cost, run_one=bulk_one,
                       on_done=lambda *_: None,
                       progress=lambda d: self._report((done0 + d) / max(len(shots), 1) * 100,
                                                       done0 + d, len(shots)),
                       max_conc=self._max_probe_concurrency(max(1, len(rest))),
                       ladder=ladder, cpu_charge=self.opt.probe_cpu_charge)

        chosen = self.pick_all_crfs(samples, grid)
        chosen.update(mapped)
        self._mem_log(f"gpu probing: {len(mapped)} shot(s) mapped from QSV, "
                      f"{len(samples)} probed with SVT "
                      f"({len(anchors)} of them calibration anchors)")
        return samples, chosen

    # ------------------------------------------------------------------
    # Predict on the card, confirm with SVT (probe_encoder="qsv+svt").
    #
    # The difference from the block above is where the answer comes from.
    # There the mapped CRF IS the answer for a bulk shot, so the mapping has
    # to be trusted outright, which is why it carries a refusal gate and why
    # a job whose line does not hold throws its whole QSV pass away. Here the
    # mapped CRF only chooses where to put the first SVT probe: the CRF that
    # ships still comes from SVT probes scored the usual way and interpolated
    # by pick_crf, exactly as the plain path produces it.
    #
    # That changes what a bad prediction costs. It cannot ship a CRF nobody
    # measured any more - the worst it can do is start the bisection in the
    # wrong place and spend an extra probe getting back, which is bounded by
    # the same probe budget. So the gate can go, and with it the case that
    # kept this path out of production: a 150s clip refused calibration at
    # target 94 (leave-one-out 3.00 CRF against a 2.00 limit) because 12 of
    # its 22 shots could not reach 94 at any CRF, and both curves had to be
    # crossed at their ends where inverting them is worst conditioned.
    #
    # It also lets the line keep learning. Every verified shot yields another
    # (q*, crf*) pair, so the fit runs on hundreds of them by mid-episode
    # instead of on the anchors alone - and on the most recent hundreds, so
    # it tracks content that drifts across an episode rather than averaging
    # over it.
    # ------------------------------------------------------------------

    # Pairs the FIRST fit needs. Four is enough to draw a line and far too
    # few to trust one: measured on a 163-shot episode the first fit landed
    # on four pairs at a residual of 6.06 CRF, which is above the threshold,
    # so nothing was seeded until the next refit 25 pairs later - and the
    # same episode's fit over 104 pairs came out at 4.14. Only 48 of 163
    # shots ended up seeded, and the QSV pass was paid for all 163.
    _VERIFY_MIN_PAIRS = 12
    # Refit this often. Theil-Sen is quadratic in the pairs, so refitting on
    # every shot would grow into real time by the end of an episode. While
    # there is no fit worth following yet, refit sooner: every 25 pairs is a
    # long time to keep probing from the grid because of one noisy line.
    _VERIFY_REFIT_EVERY = 25
    _VERIFY_REFIT_EVERY_UNUSABLE = 10
    # Pairs the fit looks at, most recent first. Bounds the refit cost and
    # gives the line locality: an episode's content is not one population.
    _VERIFY_FIT_WINDOW = 200
    def _verify_max_sd(self, grid: List[int]) -> float:
        """Residual spread above which the prediction is not worth following.

        Not a correctness gate - it only decides whether to seed from the
        line or from the middle of the grid, the way the plain path always
        does. A sixth of the grid is where the arithmetic stops working: at
        that spread the second probe goes out at _verify_step's ceiling and
        still misses more often than not, so the walk costs the probes the
        prediction was supposed to save.
        """
        span = max(grid) - min(grid) if len(grid) > 1 else 0
        return max(3.0, span / 6.0)

    def _verify_step(self, sd: float) -> int:
        """How far from the prediction to place the SECOND probe.

        Far enough that the two straddle the target most of the time - the
        prediction lands on one side of it and the pair has to enclose it
        before pick_crf can interpolate - and no further, because a probe
        spent out in the tail measures a part of the curve nothing will use.
        1.5 sd covers ~87% of a normal residual; the floor keeps it useful
        when the fit looks better than the measurement underneath it really
        is, and the ceiling keeps one bad episode from throwing probes at the
        end of the grid.
        """
        return int(max(2, min(8, round(1.5 * max(0.0, sd)))))

    def _probe_shot_seeded(self, idx: int, s0: int, s1: int, grid: List[int],
                           lp: int, seed: float, step: int) -> Dict[int, float]:
        """Two probes placed at once around a prediction, not a search.

        The plain path opens on seed_crfs and bisects. That spends probes
        narrowing a bracket - and narrowing it turns out not to matter:
        measured on 449 shot curves at the production grid, false position
        instead of bisection gives a bracket 17% tighter and an answer that
        differs by 0.076 CRF on average, never by more than 0.67, because
        pick_crf's interpolation across a wide bracket already lands where
        the extra probes would have put it.

        So there is nothing to gain from converging, only from STRADDLING.
        Two probes at seed +- step, placed together with no direction-finding
        probe between them. Measured on the same data with leave-one-out
        predictions, +-5 straddles 92% of the time for 2.08 probes a shot
        against the plain path's 3.89, and the answer lands 0.27 CRF from the
        measured crossing - inside the 0.3 the plain path's own interpolation
        carries.

        When they do not straddle, one more probe at the bound in the
        direction they both point finishes it: below the floor and it is a
        floor shot, above the ceiling and it is a ceiling shot, and either
        way the clamp pick_crf applies is against the grid's own end, which
        is the only thing that makes clamping correct. Three probes, worst
        case, against the plain path's four.

        This is also how a shot with no crossing at all stops being
        expensive. Those shots - 27 of 163 on the measured episode - burn the
        whole budget in the plain path to discover the curve never crosses.
        Here the prediction points at a bound, one probe confirms the target
        is out of reach there, and that is the answer. The prediction does
        NOT have to be right: outside the fitted range it is poor (7.70 CRF
        mean error against 2.4 inside it, because the fit only ever sees
        shots that do cross), so it is used to choose which bound to try
        first and never as a value. A wrong guess costs one probe.
        """
        self._check_cancel()
        w0, w1 = self._probe_window(s0, s1)
        _, probe_vf = self._probe_input(w0, w1)
        shard = self._acquire_shard(idx, w0, w1, probe_vf)
        scores: Dict[int, float] = {}
        try:
            lo, hi = min(grid), max(grid)
            budget = self._probe_budget(grid)
            floor, ceil = self._crf_floor(lo), self._crf_ceiling(hi)

            def probe(c: int) -> float:
                c = min(hi, max(lo, int(c)))
                if c not in scores:
                    self._check_cancel()
                    _, _, scores[c] = self._probe_encode_and_score(
                        idx, w0, w1, c, lp, shard)
                return scores[c]

            # A prediction at or past a bound: try that bound first. One probe
            # settles it when the guess is right.
            if seed <= floor:
                if probe(int(floor)) <= self.target:
                    return scores                  # confirmed: a floor shot
            elif seed >= ceil:
                if probe(int(ceil)) >= self.target:
                    return scores                  # confirmed: a ceiling shot

            a = min(hi, max(lo, int(round(seed)) - step))
            b = min(hi, max(lo, int(round(seed)) + step))
            if a == b:
                a, b = lo, hi
            probe(a)
            probe(b)
            while len(scores) < budget:
                if bracket_for(list(scores.items()), self.target) is not None:
                    break                          # straddled: pick_crf does the rest
                pts = sorted(scores.items())
                if pts[-1][1] >= self.target:
                    nxt = hi                       # everything beats it: go cheaper
                elif pts[0][1] <= self.target:
                    nxt = lo                       # nothing reaches it: go dearer
                else:
                    break
                if nxt in scores:
                    break
                probe(nxt)
            return scores
        finally:
            self._release_shard(idx)

    @staticmethod
    def _lstsq(X: List[List[float]], y: List[float]) -> List[float]:
        """Normal equations with partial pivoting. Three columns at most, so
        the numerics are not worth a dependency."""
        n, m = len(X), len(X[0])
        S = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(m)]
             + [sum(X[i][a] * y[i] for i in range(n))] for a in range(m)]
        for c in range(m):
            piv = max(range(c, m), key=lambda r: abs(S[r][c]))
            S[c], S[piv] = S[piv], S[c]
            if abs(S[c][c]) < 1e-12:
                continue
            for r in range(m):
                if r != c:
                    f = S[r][c] / S[c][c]
                    for k in range(m + 1):
                        S[r][k] -= f * S[c][k]
        return [S[i][m] / S[i][i] if abs(S[i][i]) > 1e-12 else 0.0
                for i in range(m)]

    @staticmethod
    def _design(pairs: List[Tuple[float, float, float]]) -> List[List[float]]:
        return [[1.0, q, bpf] for q, bpf, _ in pairs]

    def _refit_verified(self, pairs: List[Tuple[float, float, float]]
                        ) -> Optional[Dict[str, object]]:
        """Refit CRF from the QSV crossing AND what the probe cost in bytes.

        The crossing alone is a poor predictor and no amount of data fixes
        it: on a 163-shot 4K episode a line through the crossings left a
        leave-one-out residual of 4.53 CRF against a target whose own spread
        is 5.61, i.e. 35% of the variance. That residual is not measurement
        noise - 94% of those shots were probed WHOLE, and refitting the
        crossings with a curve model instead of linear interpolation did not
        move it (4.58) - it is a real, content-dependent difference between
        how the two encoders respond, which is exactly the kind of thing
        another feature can carry.

        Adding log bytes per frame at the high-q end of the probe takes it to
        2.72 CRF and 77% of the variance, and it costs nothing: the probe
        writes the file anyway, and false position always probes both ends,
        so that q is the same for every shot. Forward selection over shot
        length, timeline position, the QSV curve's level and slope, and
        whether the window covered the whole shot rejected all of them; the
        bytes were the only feature that paid.

        Least squares rather than the Theil-Sen the one-feature fit used,
        with one robustness pass instead: drop what sits beyond three
        residual deviations and refit, so a single freak shot cannot tilt the
        plane for the rest of the job.
        """
        if len(pairs) < self._VERIFY_MIN_PAIRS:
            return None
        recent = pairs[-self._VERIFY_FIT_WINDOW:]
        X, y = self._design(recent), [c for _, _, c in recent]
        beta = self._lstsq(X, y)
        res = [y[i] - sum(beta[j] * X[i][j] for j in range(len(beta)))
               for i in range(len(X))]
        sd = statistics.stdev(res) if len(res) > 1 else 0.0
        if sd > 0:
            keep = [k for k in range(len(recent)) if abs(res[k]) <= 3 * sd]
            if self._VERIFY_MIN_PAIRS <= len(keep) < len(recent):
                X, y = [X[k] for k in keep], [y[k] for k in keep]
                beta = self._lstsq(X, y)
        # Leave-one-out, not the fit's own residual. _fit_gpu_map says it in
        # the same file - "a line always fits its own points" - and this path
        # had quietly abandoned the principle for the two decisions that use
        # the number: the seeding gate and the step size. The trim makes it
        # worse, because it drops exactly the tail the step has to cover.
        loo = []
        for k in range(len(X)):
            b = self._lstsq(X[:k] + X[k + 1:], y[:k] + y[k + 1:])
            loo.append(y[k] - sum(b[j] * X[k][j] for j in range(len(b))))
        sd = statistics.stdev(loo) if len(loo) > 1 else sd
        return {"beta": beta, "sd": sd, "n": len(X)}

    def _predict_crf(self, fit: Dict[str, object], q: float, bpf: float,
                     grid: List[int]) -> float:
        beta = fit["beta"]                      # type: ignore[index]
        crf = beta[0] + beta[1] * q + beta[2] * bpf
        return min(self._crf_ceiling(max(grid)),
                   max(self._crf_floor(min(grid)), crf))

    @staticmethod
    def _log_bpf(bpf: Optional[Dict[int, float]]) -> Optional[float]:
        """log bytes per frame at the highest q probed - the bit-starved end.

        The margin over the low-q end is much larger than first reported: as
        single extra features on top of q*, bytes at q38 give 2.72 CRF and
        q34 2.95, while q18 and q14 give 4.34 and 4.33 against 4.53 for q*
        alone - i.e. the low end is worth essentially nothing. The 2.87 once
        quoted here was a different, two-column model.
        """
        if not bpf:
            return None
        v = bpf[max(bpf)]
        return math.log(v) if v > 0 else None

    def probe_all_verified(self, shots: List[Shot],
                           grid: List[int]) -> ProbeSamples:
        """Probe every shot with SVT, using the card to choose where to start.

        Returns ordinary ProbeSamples - {crf: score} per shot from real SVT
        probes - so everything downstream (pick_all_crfs, smoothing, the
        verification report) sees exactly what the plain path produces and
        needs no knowledge of this mode at all.
        """
        qgrid = self._qsv_grid()
        ladder = self._lp_ladder()
        # The shots that start the line off, stratified by length the same way
        # the "qsv" path picks its anchors. They pay a QSV pass on top of a
        # full SVT bisection; at 16 of a 1400-shot episode that is ~1% of the
        # phase.
        seeds = set(self._gpu_anchor_indices(shots))
        samples: ProbeSamples = {}
        pairs: List[Tuple[float, float, float]] = []   # q*, log bytes/frame, crf*
        state: Dict[str, object] = {"fit": None, "since": 0, "warned": False,
                                    "seeded": 0, "walked": 0}
        qsv_spent = [0]
        lock = threading.Lock()
        self._log(f"verified gpu probing: {len(seeds)} seed shot(s) of "
                  f"{len(shots)}, q grid {qgrid}")

        def cost(idx: int, lp: int) -> float:
            w0, w1 = self._probe_window(*shots[idx])
            return self._est_probe_gb(w1 - w0, lp)

        def run_one(idx: int, lp: int, _slot: int, _threads: int) -> object:
            s0, s1 = shots[idx]
            q_star, qs, bpf = None, None, None
            try:
                qs, bpf = self._probe_shot_qsv(idx, s0, s1, qgrid)
                with lock:
                    qsv_spent[0] += len(qs)
                q_star = self._crossing(qs)
            except TranscodeError as e:
                # the card is busy, wedged, or this shot simply never crosses
                # on the QSV curve. None of that is a reason not to probe it.
                self._log(f"qsv prediction failed for shot {idx:05d}: {e}")
            with lock:
                fit = state["fit"]
            lb = self._log_bpf(bpf)
            seed = None
            if q_star is not None and lb is not None and fit and idx not in seeds \
                    and fit["sd"] <= self._verify_max_sd(grid):
                # no probe_crf_offset here: that corrects the DELIVERED crf
                # and pick_all_crfs applies it at the end, so adding it would
                # shift where we probe by a correction made again downstream
                seed = self._predict_crf(fit, q_star, lb, grid)
            if seed is None:
                scores = self._probe_shot(idx, s0, s1, grid, lp)
            else:
                scores = self._probe_shot_seeded(
                    idx, s0, s1, grid, lp, seed,
                    self._verify_step(float(fit["sd"])))
            crf_star = self._crossing(scores)
            self._dataset_shot(idx, s0, s1, *self._probe_window(s0, s1),
                               svt=scores, qsv=qs if q_star is not None else None,
                               qsv_bpf=bpf, seed=seed)
            with lock:
                samples[idx] = scores
                if seed is not None:
                    state["seeded"] = int(state["seeded"]) + 1
                    if crf_star is not None:
                        state["walked"] = int(state["walked"]) + \
                            (1 if abs(crf_star - seed) > self._verify_step(
                                float(fit["sd"])) else 0)
                snapshot = None
                if q_star is not None and lb is not None and crf_star is not None:
                    pairs.append((q_star, lb, crf_star))
                    state["since"] = int(state["since"]) + 1
                    cur = state["fit"]
                    usable = bool(cur) and cur["sd"] <= self._verify_max_sd(grid)
                    every = (self._VERIFY_REFIT_EVERY if usable
                             else self._VERIFY_REFIT_EVERY_UNUSABLE)
                    if (len(pairs) >= self._VERIFY_MIN_PAIRS
                            and int(state["since"]) >= every):
                        state["since"] = 0
                        snapshot = list(pairs)
            if snapshot is not None:
                # outside the lock: Theil-Sen is quadratic in the pairs and
                # every other probe worker wants this lock
                new = self._refit_verified(snapshot)
                if new is not None:
                    with lock:
                        state["fit"] = new
                        warn = (new["sd"] > self._verify_max_sd(grid)
                                and not state["warned"])
                        if warn:
                            state["warned"] = True
                    if warn:
                        logger.warning(
                            "optimizer: the QSV prediction is not worth "
                            "following on this source (residual {:.2f} CRF "
                            "over {}); probing from the grid instead",
                            new["sd"], new["n"])
            return None

        stop, peak = self._start_mem_sampler()
        try:
            self._schedule(list(range(len(shots))), phase="probing", cost=cost,
                           run_one=run_one, on_done=lambda *_: None,
                           progress=lambda d: self._report(d / max(len(shots), 1) * 100,
                                                           d, len(shots)),
                           max_conc=self._max_probe_concurrency(len(shots)),
                           ladder=ladder, cpu_charge=self.opt.probe_cpu_charge)
        finally:
            stop.set()
        self._log_phase_memory("probing", peak[0])
        spent = sum(len(v) for v in samples.values())
        if samples:
            self._mem_log(f"probes: {spent} for {len(samples)} shot(s), "
                          f"{spent / len(samples):.2f} per shot (SVT), plus "
                          f"{qsv_spent[0]} on the card{self._zc_summary()}")

        fit = state["fit"]
        if fit:
            b = fit["beta"]
            self._mem_log(
                "verified gpu probing: {} of {} shot(s) seeded from "
                "CRF = {:.2f}q {:+.2f}log(bytes/frame) {:+.2f} (residual "
                "{:.2f} CRF over {} pairs), {} needed more than one step to "
                "bracket".format(state["seeded"], len(shots), b[1], b[2], b[0],
                                 fit["sd"], fit["n"], state["walked"]))
        else:
            self._mem_log("verified gpu probing: the line never fitted; every "
                          "shot probed from the grid")
        return samples

    def _fit_gpu_map(self, pairs: List[Tuple[float, float]]) -> Optional[Dict[str, float]]:
        """Fit CRF = a*q + b and refuse the mapping when it does not hold.

        The leave-one-out residual is the number that decides it, not the fit
        residual: a line always fits its own points. Measured on ten 4K
        windows, LOO came out 0.52 CRF at target 91 and 1.21 at target 94 once
        the one shot outside the anchors' range was excluded - against a
        default ceiling of 2.0.
        """
        need = 4
        if len(pairs) < need:
            logger.warning("optimizer: gpu probing calibrated on only {} shot(s), "
                           "fewer than the {} it needs; probing with SVT",
                           len(pairs), need)
            return None
        res = []
        for k in range(len(pairs)):
            a, b = self._theil_sen(pairs[:k] + pairs[k + 1:])
            res.append(pairs[k][1] - (a * pairs[k][0] + b))
        sd = statistics.stdev(res) if len(res) > 1 else 0.0
        limit = float(self.opt.gpu_probe_max_residual or 2.0)
        a, b = self._theil_sen(pairs)
        qs = [q for q, _ in pairs]
        margin = float(self.opt.gpu_probe_max_q_margin or 0.0)
        if sd > limit:
            logger.warning(
                "optimizer: the QSV->SVT mapping does not hold on this source "
                "(leave-one-out {:.2f} CRF over {}, limit {:.2f}); probing the "
                "rest with SVT", sd, len(pairs), limit)
            return None
        logger.info("optimizer: QSV->SVT mapping CRF = {:.2f}q {:+.2f} over {} "
                    "anchor(s), leave-one-out {:.2f} CRF (~{:.2f} {}), trusting "
                    "it for q in [{:.1f}, {:.1f}]",
                    a, b, len(pairs), sd, sd * 0.24, self.metric,
                    min(qs) - margin, max(qs) + margin)
        return {"a": a, "b": b, "sd": sd,
                "q_lo": min(qs) - margin, "q_hi": max(qs) + margin}

    def _score_probe(self, w0: int, w1: int, dist: Path, idx: int, crf: int,
                     shard: Optional[Path] = None,
                     threads: Optional[int] = None) -> float:
        """Score a distorted window that already exists as a file.

        Zero-copy first when the job has it (see _zero_copy); a window that
        fails there is scored again the usual way.
        """
        if self.metric == "ssimulacra2":
            return self._score_ssimulacra2(w0, w1, dist, idx, crf, shard)
        frames = (w1 - w0) // self._probing_rate()
        if shard is None and (w0, w1) not in self._zc_bad and self._zero_copy():
            try:
                score = self._score_zero_copy(w0, w1, dist, idx, crf, frames)
            except TranscodeError as e:
                if not self._zc_fallback(w0, w1, e):
                    raise
                score = None
            if score is not None:
                return score
        ref_args, ref_vf = self._probe_input(w0, w1, shard)
        hw = False
        if shard is None and (w0, w1) not in self._hwdec_bad:
            # a DV shard is a small file already carrying the probe-side
            # filters; only the 4K source read is worth the GPU
            ref_args, ref_vf, hw = self._reference_read(ref_args, ref_vf)
        dist_args = ["-i", str(dist)]
        try:
            score = self._score_pair(dist_args, ref_args, ref_vf, idx, crf, threads, frames)
        except TranscodeError as e:
            if not hw or not self._hwdec_fallback(w0, w1, e):
                raise
            ref_args, ref_vf = self._probe_input(w0, w1, shard)
            return self._score_pair(dist_args, ref_args, ref_vf, idx, crf, threads, frames)
        if hw:
            self._hwdec_read_ok()
        return score

    def _score_pair(self, dist_args: List[str], ref_args: List[str],
                    ref_vf: List[str], idx: int, crf: int,
                    threads: Optional[int], frames: int) -> float:
        if self.metric == "xpsnr":
            return self._score_xpsnr(dist_args, ref_args, ref_vf, idx, crf)
        return self._score_vmaf(dist_args, ref_args, ref_vf, idx, crf,
                                threads=threads, frames=frames)

    def _window_input(self, w0: int, w1: int, source: Path,
                      threads: int) -> List[str]:
        """ffmpeg input args reading frames [w0, w1) of `source` unfiltered.

        Not _probe_input: that one also applies the probe-side filters
        (probing_rate subsampling, probe_scale). Verification compares what was
        delivered against the source at native everything, and applying a probe
        filter to one side of that would be measuring the wrong thing.
        """
        # each file's own lead: the output's video begins where the mux put
        # it, the source's where its container did
        return ["-threads", str(max(1, threads)), "-ss", self._seek(w0, source),
                "-t", f"{self._span(w0, w1):.6f}", "-i", str(source)]

    _HWDEC_PREFLIGHT_FRAMES = 8

    def _hwdec_args(self) -> List[str]:
        """The hardware-decode options for one input, or [] without a GPU.

        VA-API, not QSV, and that is not a preference. ffmpeg's QSV path is a
        separate decoder (h264_qsv, hevc_qsv) with its own timestamp handling,
        and after an input-side -ss it kept different frames than the software
        decoder on an mkv with timeline gaps, on a DV P5 mp4, and on an 8-bit
        H.264 WEB-DL at two of three seek points (55 VMAF where the software
        read scored 92) - while reading clean Blu-ray remuxes exactly. VA-API
        is a hwaccel OF the native decoder: the frame selection is the software
        path's by construction and only the pixels come from the GPU. Nine
        cases, nine identical scores.
        """
        nodes = _render_nodes()
        if not nodes:
            return []
        return ["-hwaccel", "vaapi", "-hwaccel_device", nodes[0],
                "-hwaccel_output_format", "vaapi"]

    def _surface_format(self) -> str:
        """The pixel format a hardware decode of the source lands in."""
        return "p010le" if (self.info.color.bit_depth or 8) > 8 else "nv12"

    def _native_format(self) -> str:
        """What the software decoder would hand the chain."""
        return self.info.color.pix_fmt or (
            "yuv420p10le" if self._surface_format() == "p010le" else "yuv420p")

    def _download_vf(self) -> str:
        """hwdownload plus the conversion that hands the rest of the chain the
        very pixel format the software decoder would have produced, so
        everything downstream (fps=, scale=, setpts, the scorer's format=)
        runs on identical input either way. p010le -> yuv420p10le is a shift
        and nv12 -> yuv420p a plane split; both exact."""
        return f"hwdownload,format={self._surface_format()},format={self._native_format()}"

    def _reference_read(self, args: List[str],
                        vf: List[str]) -> Tuple[List[str], List[str], bool]:
        """Move a source read onto the GPU decoder, when reference_hwaccel
        allows it and the preflight passed; the flag says whether it moved.

        The hwaccel options precede the input they apply to, and the download
        has to be the first filter in the chain: everything after it wants
        system-memory frames.
        """
        if not self._hwdec():
            return args, vf, False
        return [*self._hwdec_args(), *args], [self._download_vf(), *vf], True

    _HWDEC_MAX_STREAK = 8

    def _hwdec_fallback(self, w0: int, w1: int, err: TranscodeError) -> bool:
        """Whether a failed hardware read of [w0, w1) should be scored again
        on the CPU. True for a decode failure; False for a timeout or a
        cancelled job, which are not the GPU's doing.

        The preflight cannot prove every window, and two different things
        fail. E06 of Stranger Things (a Blu-ray remux, DV P7 with its
        enhancement layer) begins with a broken group of pictures: its first
        frames reference pictures not in the file ("Could not find ref with
        POC 2..12"). Software conceals and carries on; VA-API answers "Failed
        to sync surface: internal decoding error" and exits 251. That window
        fails every time, so it is remembered and its other CRF probes skip
        the doomed attempt. E07 is the other kind: five mid-file windows that
        decode cleanly on their own each failed once under ten concurrent
        VA-API reads plus SYCL scoring on the one B580 - contention, not the
        stream, and the next read succeeds.

        So the whole-job give-up counts CONSECUTIVE failures, reset by any
        successful hardware read (see _hwdec_read_ok). A device that has
        genuinely died fails read after read and trips it; a scattered burst
        under load never does, because successes keep resetting it. Either
        way the window at hand is scored on the CPU and the job goes on.
        """
        if isinstance(err, CommandTimeout) or (self.cancel_flag and self.cancel_flag()):
            return False
        with self._hwdec_lock:
            self._hwdec_bad.add((w0, w1))
            self._hwdec_streak += 1
            n = self._hwdec_streak
            flip = n >= self._HWDEC_MAX_STREAK and self._hwdec_ok
            if flip:
                self._hwdec_ok = False
        tail = str(err).strip().splitlines()[-1][:160] if str(err).strip() else err.__class__.__name__
        logger.warning("reference_hwaccel: the VA-API read of frames [{}, {}) failed "
                       "({}); scoring that window on the CPU{}", w0, w1, tail,
                       f"; {n} in a row, the rest of the job scores on the CPU" if flip else "")
        return True

    def _hwdec_read_ok(self) -> None:
        """A hardware read succeeded: clear the consecutive-failure streak so
        a scattered failure under load never adds up to giving up the GPU."""
        if self._hwdec_streak:
            with self._hwdec_lock:
                self._hwdec_streak = 0

    def _hwdec(self) -> bool:
        if (self.opt.reference_hwaccel or "auto").lower() == "off":
            return False
        with self._hwdec_lock:
            if self._hwdec_ok is None:
                self._hwdec_ok = self._hwdec_preflight()
            return self._hwdec_ok

    def _hwdec_preflight(self) -> bool:
        """Prove a seeked hardware read of this source is frame-identical to
        the software read before any score depends on it.

        Two things go wrong, and both did on a B580. A stream the card cannot
        decode still exits 0 with no frames (the scdet path learnt that on
        AV1). And a decoder can keep different frames after an input-side
        seek: QSV began five frames BEFORE the seek target on an 8-bit H.264
        WEB-DL - every frame bit-exact, every one the wrong frame - which is
        why the reads go through VA-API now (see _hwdec_args). A slip of one
        frame in the reference window measured 66 VMAF against 92, so the two
        reads are still compared frame by frame after a seek, in the pixel
        format the scorer sees. A mismatch keeps the job on the CPU and says
        which kind it was.
        """
        n = self._HWDEC_PREFLIGHT_FRAMES
        if not self._hwdec_args():
            logger.warning("reference_hwaccel: no /dev/dri render node in this "
                           "container; scoring reads stay on the CPU")
            return False
        # two seconds in, so the seek path is the one that is exercised, but
        # never past the end of a short source
        seek = self._seek(min(int(round(2 * self.fps)), max(self.total_frames - n, 0)))
        common = ["-hide_banner", "-loglevel", "error", "-nostats"]
        tail = ["-i", str(self.source), "-map", "0:v:0", "-frames:v", str(n)]
        sw = [self.ffmpeg, *common, "-threads", "2", "-ss", seek, *tail,
              "-vf", f"format={self._native_format()}", "-f", "framemd5", "-"]
        hw = [self.ffmpeg, *common, *self._hwdec_args(), "-ss", seek, *tail,
              "-vf", self._download_vf(), "-f", "framemd5", "-"]

        def sums(out: str) -> List[str]:
            return [line.rsplit(",", 1)[-1].strip()
                    for line in out.splitlines() if line.startswith("0,")]

        try:
            want = sums(self._run(sw, timeout=300))
            got = sums(self._run(hw, timeout=300))
        except (TranscodeError, OSError) as e:
            logger.warning("reference_hwaccel: VA-API cannot decode {} ({}); "
                           "scoring reads stay on the CPU for this job",
                           self.source.name, e)
            return False
        if not want or not got:
            logger.warning("reference_hwaccel: {} decoded no frames of {}; "
                           "scoring reads stay on the CPU for this job",
                           "software" if not want else "VA-API", self.source.name)
            return False
        if want != got:
            shift = next((k for k in range(1, min(len(want), len(got)))
                          if want[:-k] == got[k:] or got[:-k] == want[k:]), None)
            logger.warning("reference_hwaccel: the VA-API read of {} is not the "
                           "software read after a seek ({}); scoring reads "
                           "stay on the CPU for this job", self.source.name,
                           f"offset by {shift} frame(s)" if shift else "different frames")
            return False
        logger.info("reference_hwaccel: scoring reads of {} decode on VA-API "
                    "({}); {} frames after a seek verified frame-identical",
                    self.source.name, self._surface_format(), len(got))
        return True

    # ---------- zero-copy scoring: VA-API decode straight into libvmaf_sycl ----------
    _ZC_DEVICE = "zc"
    # framesync holds frames of whichever input runs ahead while the other
    # catches up, and the decoder's surface pool must not run dry under it
    _ZC_EXTRA_HW_FRAMES = 8
    _ZC_MAX_STREAK = 8
    # scored frames in the preflight window: enough for motion and the
    # de-tile to see real content, cheap enough to run once per job
    _ZC_PREFLIGHT_FRAMES = 24

    def _zero_copy(self) -> bool:
        """Whether probe scores keep their frames on the card (vmaf_zero_copy).

        Both inputs - the source window and the probe's ivf - decode on
        VA-API and reach libvmaf_sycl as surfaces, which libvmaf imports as
        DMA-BUFs and de-tiles on the GPU. Measured on the B580 at 4K: 1.18s
        and 1.36 CPU-seconds per score against 5.4s and 23, scores identical
        to 0.0 on every frame over 247 runs. Decided once per job, like the
        SYCL and reference_hwaccel preflights, and under a lock for the same
        reason: probe workers ask concurrently.
        """
        if (self.opt.vmaf_zero_copy or "off").lower() == "off":
            return False
        sycl = self._sycl_device()
        if sycl < 0:
            return False
        with self._zc_lock:
            if self._zc_ok is None:
                why = self._zc_unsupported()
                if why:
                    logger.info("optimizer: vmaf_zero_copy stays off for this job: {}", why)
                    self._zc_ok = False
                else:
                    self._zc_ok = self._zc_preflight(sycl)
            return bool(self._zc_ok)

    def _zc_unsupported(self) -> Optional[str]:
        """Why this job cannot score zero-copy, or None when it can."""
        if not self._hwdec_args():
            return "no /dev/dri render node in this container"
        if self._vmaf_scale_filter():
            return ("scores are scaled for the 1080p model, and libvmaf_sycl "
                    "compares the frames as decoded")
        if self._probe_scale():
            return "probe_scale is set"
        if self._vmaf_features():
            return "probing_vmaf_features is set, and the import carries luma only"
        src = self.info.color.bit_depth or 8
        probe = 10 if "10" in self._pix_fmt() else 8
        if src != probe:
            return (f"the source is {src}-bit and the probes {probe}-bit, and both "
                    "sides have to decode to the same surface format")
        return None

    def _zc_input_args(self) -> List[str]:
        """Input options that decode one side of a zero-copy score on the card."""
        return ["-hwaccel", "vaapi", "-hwaccel_device", self._ZC_DEVICE,
                "-hwaccel_output_format", "vaapi",
                "-extra_hw_frames", str(self._ZC_EXTRA_HW_FRAMES)]

    def _zc_preflight(self, sycl: int) -> bool:
        """Prove a zero-copy score is the score before any probe relies on one.

        Three things have to hold. The seeked VA-API read must keep the frames
        the software read does - the reference_hwaccel preflight, reused. The
        import must really stay on the card: libvmaf falls back to vaGetImage
        plus an upload without failing and says so only at INFO. And the
        score must equal the usual read's on the same pair, which a wrong
        de-tile or a missed P010 shift would not.
        """
        if not self._hwdec_preflight():
            logger.info("optimizer: vmaf_zero_copy stays off for this job (the "
                        "seeked VA-API read above did not match software)")
            return False
        n = self._ZC_PREFLIGHT_FRAMES * self._probing_rate()
        w0 = min(int(round(2 * self.fps)), max(self.total_frames - n, 0))
        w1 = min(w0 + n, max(self.total_frames, w0 + 1))
        ivf = self.probe_dir / "zc_preflight.ivf"
        out_json = self.probe_dir / "zc_preflight.json"
        ref_args, ref_vf = self._probe_input(w0, w1)
        exact_vf, exact_out = self._exact_frames()
        encode = ([self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *ref_args,
                   "-vf", ",".join(ref_vf + exact_vf), *exact_out, "-map", "0:v:0",
                   "-c:v", "libsvtav1", "-preset", "12", "-crf", "40",
                   "-pix_fmt", self._pix_fmt(), "-f", "ivf", str(ivf)])
        try:
            self._run(encode, timeout=600)
            want = self._score_vmaf_on(sycl, ["-i", str(ivf)], ref_args, ref_vf,
                                       -1, 0, None, timeout=300)
            out = self._run(self._vmaf_cmd(sycl, ["-i", str(ivf)], ref_args, ref_vf,
                                           out_json, None, zero_copy=True,
                                           loglevel="info"), timeout=300)
            got = parse_score(out_json, self.metric)
        except (TranscodeError, OSError, ValueError) as e:
            self._check_cancel()
            detail = "; ".join(
                [ln for ln in str(e).strip().splitlines() if ln.strip()][-3:])
            logger.warning("optimizer: vmaf_zero_copy stays off for this job: the "
                           "preflight could not score ({})", detail)
            return False
        finally:
            _unlink(ivf)
            _unlink(out_json)
        if "readback path" in out or "de-tile" not in out:
            logger.warning(
                "optimizer: vmaf_zero_copy stays off for this job: libvmaf {} "
                "instead of importing the surfaces in place",
                "read the surfaces back" if "readback path" in out
                else "never reported a zero-copy import")
            return False
        if abs(got - want) > self._SYCL_MAX_DELTA:
            logger.warning(
                "optimizer: vmaf_zero_copy stays off for this job: it scores "
                "{:.6f} where the usual read scores {:.6f} on the same pair",
                got, want)
            return False
        logger.info("optimizer: probe scores decode on VA-API and stay on the card "
                    "(zero-copy agrees with the usual read to {:.2e})", abs(got - want))
        return True

    def _score_zero_copy(self, w0: int, w1: int, dist: Path, idx: int, crf: int,
                         frames: int) -> Optional[float]:
        """One probe score with both sides decoded on the card, or None when
        there is no room on the card for it now and the usual read should run."""
        sycl = self._sycl_device()
        if sycl < 0:
            return None
        need = self._est_zc_score_mb()
        booked = self._gpu_vram.booked(need)
        if not self._gpu_vram.reserve(need, self._GPU_SLOT_WAIT):
            return None
        ref_args, ref_vf = self._probe_input(w0, w1)
        try:
            score = self._score_vmaf_on(sycl, ["-i", str(dist)], ref_args, ref_vf,
                                        idx, crf, None,
                                        timeout=self._sycl_timeout(frames),
                                        zero_copy=True)
        finally:
            self._gpu_vram.calibrate()
            self._gpu_vram.release(booked, need)
        with self._zc_lock:
            self._zc_streak = 0
            self._zc_scored += 1
        self._sycl_scored_ok()
        return score

    def _zc_fallback(self, w0: int, w1: int, err: TranscodeError) -> bool:
        """Whether a failed zero-copy score of [w0, w1) should be scored again
        the usual way: always, unless the job was cancelled.

        The same shape as _hwdec_fallback: the window is remembered so its
        other probes skip the attempt, and consecutive failures - reset by any
        success - retire zero-copy for the rest of the job.
        """
        if self.cancel_flag and self.cancel_flag():
            return False
        with self._zc_lock:
            self._zc_bad.add((w0, w1))
            self._zc_streak += 1
            self._zc_fallbacks += 1
            n = self._zc_streak
            flip = n >= self._ZC_MAX_STREAK and bool(self._zc_ok)
            if flip:
                self._zc_ok = False
        text = str(err).strip()
        tail = text.splitlines()[-1][:160] if text else err.__class__.__name__
        logger.warning("optimizer: zero-copy scoring of frames [{}, {}) failed ({}); "
                       "scoring that window the usual way{}", w0, w1, tail,
                       f"; {n} in a row, the rest of the job scores the usual way"
                       if flip else "")
        return True

    def _zc_summary(self) -> str:
        """The zero-copy share of the probe phase, for its summary line."""
        if not (self._zc_scored or self._zc_fallbacks):
            return ""
        fell = f", {self._zc_fallbacks} fell back" if self._zc_fallbacks else ""
        return f"; {self._zc_scored} score(s) zero-copy{fell}"

    def _score_windows(self, w0: int, w1: int, idx: int, crf: int,
                       threads: int) -> float:
        """Score [w0, w1) of the finished output against the same frames of the
        source, reading both in place."""
        # the AV1 side stays on the CPU: dav1d beats a GPU decode plus
        # download there
        dist_args = self._window_input(w0, w1, self.output, threads)
        ref_args, ref_vf, hw = self._window_input(w0, w1, self.source, threads), [], False
        if (w0, w1) not in self._hwdec_bad:
            ref_args, ref_vf, hw = self._reference_read(ref_args, ref_vf)
        try:
            score = self._score_pair(dist_args, ref_args, ref_vf, idx, crf, threads, w1 - w0)
        except TranscodeError as e:
            if not hw or not self._hwdec_fallback(w0, w1, e):
                raise
            return self._score_pair(dist_args, self._window_input(w0, w1, self.source, threads),
                                    [], idx, crf, threads, w1 - w0)
        if hw:
            self._hwdec_read_ok()
        return score

    # ---- colour description of the reference, for metrics that need it ----
    # ffmpeg's colour names are not zimg's, and a wrong one is not a rounding
    # error: "2020nc" is simply rejected, and mislabelling PQ as gamma would
    # linearise with the wrong curve and measure something else entirely.
    _ZIMG_MATRIX = {"bt709": "709", "bt2020nc": "2020ncl", "bt2020c": "2020cl",
                    "smpte170m": "170m", "bt470bg": "470bg"}
    _ZIMG_TRANSFER = {"bt709": "709", "smpte2084": "st2084",
                      "arib-std-b67": "std-b67", "bt470bg": "470bg"}
    _ZIMG_PRIMARIES = {"bt709": "709", "bt2020": "2020", "smpte170m": "170m"}

    def _colour_args(self) -> List[str]:
        """Matrix/transfer/primaries of the signal being compared, in zimg's
        spelling. SSIMULACRA2 is defined on linear-light RGB, so the coded
        transfer has to be undone before scoring."""
        return [
            "--matrix", self._ZIMG_MATRIX.get(self.plan.colorspace or "", "709"),
            "--transfer", self._ZIMG_TRANSFER.get(self.plan.color_trc or "", "709"),
            "--primaries", self._ZIMG_PRIMARIES.get(self.plan.color_primaries or "", "709"),
            "--range", "full" if self.plan.color_range == "pc" else "limited",
        ]

    def _score_ssimulacra2(self, w0: int, w1: int, dist: Path, idx: int,
                           crf: int, shard: Optional[Path]) -> float:
        if shard is None:
            raise TranscodeError(
                "target_metric=ssimulacra2 needs a staged reference shard; "
                "this is a bug in the probe path")
        args = [sys.executable, "-m", "app.vsmetrics", str(shard), str(dist),
                "--step", str(max(1, self.opt.ssimulacra2_frame_step)),
                "--vszip", str(self.opt.vszip_plugin),
                "--bestsource", str(self.opt.bestsource_plugin),
                "--threads", str(self._vmaf_threads()),
                *self._colour_args()]
        try:
            out = self._run(args, timeout=3600)
        except TranscodeError as e:
            raise TranscodeError(
                f"{e}\nHint: target_metric=ssimulacra2 needs the VapourSynth "
                f"vszip and bestsource plugins ({self.opt.vszip_plugin}, "
                f"{self.opt.bestsource_plugin}). libvmaf cannot compute "
                "SSIMULACRA2 itself.") from e
        try:
            score = float(out.strip().splitlines()[-1])
        except (ValueError, IndexError) as e:
            raise TranscodeError(f"could not parse SSIMULACRA2 output: {out[-300:]}") from e
        self._log(f"shot {idx:05d} crf {crf} ssimulacra2={score:.3f}")
        return score

    # A metric run must output the metric's video and nothing else. Without
    # these, ffmpeg also maps the source's audio into the null output - decodes
    # it, encodes it to pcm, and interleaves it with the scored frames in the
    # muxer. That interleaving is a deadlock waiting for a slow video path:
    # when the scorer lags (the SYCL backend under contention, mostly) the
    # audio fills the mux queue, the demuxer blocks on its packet queue, the
    # video decoder starves and framesync waits forever. Measured: a scoring
    # that stalled at frame 56 of 120 had every thread - dav1d workers, filter
    # threads, dec1:1:eac3, enc0:1:pcm_s16le, mux0:null - parked in
    # futex_do_wait, none in a GPU ioctl; with -an the same pair never stalls.
    # The one-in-a-few-hundred version of this ate the full 3600s command
    # timeout and failed a real job. The DV paths score a video-only remux,
    # which is why they never saw it.
    _VIDEO_ONLY_OUTPUT = ("-an", "-sn", "-dn")

    def _score_xpsnr(self, dist_args: List[str], ref_args: List[str],
                     ref_vf: List[str], idx: int, crf: int) -> float:
        """ffmpeg's xpsnr filter: a dB scale, not 0-100. Weighted luma is what
        the ITU work reports, so that is what is returned."""
        fmt = f"format={self._pix_fmt()}"
        # same index pairing as _score_vmaf, for the same reason
        rebase = "setpts=PTS-STARTPTS"
        dist_chain = f"{rebase},{fmt}"
        ref_chain = ",".join(f for f in (*ref_vf, rebase, fmt) if f)
        lavfi = (f"[0:v]{dist_chain}[dist];[1:v]{ref_chain}[ref];"
                 f"[dist][ref]xpsnr=shortest=1")
        args = ([self.ffmpeg, "-hide_banner", "-y", "-loglevel", "info"]
                + dist_args + ref_args
                + ["-lavfi", lavfi, *self._VIDEO_ONLY_OUTPUT, "-f", "null", "-"])
        out = self._run(args, timeout=3600)
        m = re.findall(r"XPSNR\s+y:\s*([0-9.]+)", out)
        if not m:
            raise TranscodeError(f"could not parse XPSNR output: {out[-300:]}")
        score = float(m[-1])
        self._log(f"shot {idx:05d} crf {crf} xpsnr={score:.3f}dB")
        return score

    # After this many SYCL scorings have had to be killed in one job, the
    # rest of the job scores on the CPU: a GPU that keeps stalling is not
    # going to get better, and every stall already cost a full timeout.
    _SYCL_MAX_TIMEOUTS = 3
    # How long a retired SYCL device is left alone before the preflight is
    # given one more go. Measured the hard way: an episode reported
    # OUT_OF_DEVICE_MEMORY at 8% of its probe phase, the device was retired
    # for the whole job, and a selfcheck minutes later passed - so the card
    # had recovered while the job spent its remaining 90% on the CPU, about
    # half an hour of it. A card that is genuinely gone fails the preflight
    # again and costs only that.
    _SYCL_REARM_AFTER = 300.0
    # How long a probe waits for a GPU slot before scoring on the CPU instead.
    # Generous: a slot frees every few seconds, and the CPU score costs real
    # cores that the encodes want.
    _GPU_SLOT_WAIT = 120.0
    _GPU_WORKERS_AUTO = 6

    def _gpu_workers(self) -> int:
        """How many scores may use the GPU at once.

        Not the probe pool's width. The card is one device with one pool of
        memory, and every GPU score holds a SYCL context plus - when
        reference_hwaccel is on - a VA-API decode session with its own 4K
        surface pool. Ten of each is what killed E07 of Stranger Things:
        "SYCL memcpy H2D: OUT_OF_DEVICE_MEMORY", then "DEVICE_LOST", ffmpeg
        exit 234, job failed. Measured afterwards on an idle B580 with
        240-frame 4K windows, the whole point of a wider queue is missing
        anyway - total throughput plateaus at four:

            n=2 0.11/s   n=4 0.21/s   n=6 0.22/s   n=8 0.22/s   n=10 0.23/s
            n=12 every one of the twelve failed, and the device stayed
            broken afterwards: the next run of TWO failed as well, and only
            rebooting the host brought the card back.

        The memory is measurable, and worth measuring before changing this.
        DRM fdinfo carries it - the interface nvtop reads: sum
        drm-total-vram0 over /proc/*/fdinfo, deduplicated by drm-client-id,
        inside the container holding the device. On a 4K probe run six
        concurrent scores peak at 5.27GB across 12 clients (~440MB each,
        two clients per score), 44% of a 12GB B580; ten works out to ~8.8GB
        and the twelve that broke it to ~10.5GB, before Plex and the
        framebuffer take their share.

        So the default sits above the throughput plateau and at half of what
        broke: margin where it costs little. Measured cost of the cap on a
        150s 4K clip at 40 cores: probe phase 447.7s -> 482.5s, scores
        identical. 0 = auto.
        """
        w = int(self.opt.vmaf_sycl_workers or 0)
        return max(1, w if w > 0 else self._GPU_WORKERS_AUTO)

    # The card's memory to book when nothing else says otherwise. Six 4K
    # scores with reference_hwaccel on measured 5.27GB, so this lands the
    # default where vmaf_sycl_workers already put it - and leaves half of a
    # 12GB B580 for Plex, which shares the card and cannot be measured from
    # inside this container.
    _VRAM_BUDGET_AUTO_MB = 6000.0

    def _vram_budget_mb(self) -> float:
        v = float(self.opt.gpu_vram_budget_mb or 0)
        return v if v > 0 else self._VRAM_BUDGET_AUTO_MB

    # Per megapixel. The figure behind _gpu_workers - one 4K score, one DRM
    # client, ~440MB - was read off a run whose windows were shorter; a
    # 163-shot episode of 3840x2160 peaked at 3967MB over six concurrent
    # scores, which is 661MB each, or 80MB per megapixel. With
    # reference_hwaccel on a VA-API decode session with its own surface pool
    # sits beside it for about as much again.
    #
    # Flat in window length on purpose. libvmaf streams frames rather than
    # holding the window, and across the 120- and 240-frame runs measured the
    # per-client figure did not move with it; if that turns out to be wrong
    # on some other content, `calibrate` corrects the model from what the
    # card actually holds rather than waiting for someone to remeasure.
    _VRAM_MB_PER_MPX = 80.0
    # Over-booking leaves the card idle; under-booking is what turns into
    # OUT_OF_DEVICE_MEMORY and then a device that needs a reboot, so the
    # estimate is biased the safe way.
    _VRAM_SAFETY = 1.15

    def _est_score_mb(self, frames: Optional[int] = None) -> float:
        """Card memory one GPU score holds, in MB."""
        clients = 2.0 if self._hwdec_ok else 1.0
        return (self._megapixels() * self._VRAM_MB_PER_MPX * clients
                * self._VRAM_SAFETY)

    def _est_qsv_probe_mb(self, frames: Optional[int] = None) -> float:
        """Card memory one QSV probe encode holds, in MB.

        Two clients like a score with the reference read on - a VA-API decode
        session feeding a hardware encoder - but the encoder's reference
        frames are its own, so this starts a little above a score and lets
        the calibration carry it the rest of the way. Unlike the score figure
        this one is a projection, not a measurement: nothing has run the GPU
        probe path long enough on this card to read it off fdinfo.
        """
        return (self._megapixels() * self._VRAM_MB_PER_MPX * 2.5
                * self._VRAM_SAFETY)

    # A zero-copy score holds two VA-API decode sessions beside its SYCL
    # context: measured 1.49-1.61GB per 4K probe window on the B580, against
    # 661MB for a score fed by the CPU - about 200MB per megapixel.
    _VRAM_MB_PER_MPX_ZC = 200.0

    def _est_zc_score_mb(self) -> float:
        """Card memory one zero-copy score holds, in MB (see _zero_copy)."""
        return self._megapixels() * self._VRAM_MB_PER_MPX_ZC * self._VRAM_SAFETY

    def _sycl_timeout(self, frames: Optional[int]) -> int:
        """Seconds a SYCL scoring may take before it is killed and retried.

        A 120-frame 4K window scores in 1-6s on the GPU, ~30s with ten
        scorers contending. A stall never finishes at all (see
        _VIDEO_ONLY_OUTPUT for the one that was found), so what matters is
        that the budget is a small multiple of the honest case, not the
        3600s the CPU path keeps: 60s plus a second per frame.

        With the GPU probe path on, the card is ALSO running hardware AV1
        encodes, and a score waits behind them. Measured on an episode: the
        budget sized for a scoring-only card tripped three times around shot
        180 and flipped the whole job to CPU scoring for its remaining 400
        shots - a self-inflicted stall, not a sick device. So the budget
        widens with the work the card has been given.
        """
        base = 60 + max(0, int(frames or 0))
        return base * (2 if self._gpu_probe_on() else 1)

    def _score_vmaf(self, dist_args: List[str], ref_args: List[str],
                    ref_vf: List[str], idx: int, crf: int,
                    threads: Optional[int] = None,
                    frames: Optional[int] = None) -> float:
        """Score one distorted window against one reference window.

        Both sides arrive as ffmpeg INPUT ARGUMENTS rather than paths, because
        the two callers hand over different shapes: a probe has already written
        its distorted window to an ivf, while verification seeks into the
        finished output and the source in place. See _score_windows.

        `frames` sizes the SYCL timeout; a SYCL scoring that overruns it is
        killed and the window is scored again on the CPU, so a stalled GPU
        costs one short wait rather than the job.
        """
        sycl = self._sycl_device()
        need = self._est_score_mb(frames)
        booked = self._gpu_vram.booked(need)
        if sycl >= 0 and not self._gpu_vram.reserve(need, self._GPU_SLOT_WAIT):
            # The card is full and staying full. Scoring on the CPU is slower
            # per shot but it is not queued behind anything, and the point of
            # the budget is that a deeper GPU queue buys nothing.
            self._log(f"gpu full ({self._gpu_vram.occupancy():.0f}MB of "
                      f"{self._gpu_vram.budget:.0f}MB, wanted {need:.0f}MB); "
                      f"scoring shot {idx:05d} crf {crf} on the CPU")
            sycl = -1
        if sycl >= 0:
            try:
                score = self._score_vmaf_on(sycl, dist_args, ref_args, ref_vf,
                                            idx, crf, threads,
                                            timeout=self._sycl_timeout(frames))
                self._sycl_scored_ok()
                return score
            except TranscodeError as e:
                # Not just timeouts. E07 of Stranger Things died on
                # "SYCL memcpy H2D: OUT_OF_DEVICE_MEMORY" followed by
                # "DEVICE_LOST": ten concurrent VA-API reads and ten SYCL
                # contexts exhausted the B580's memory, ffmpeg exited 234 and
                # the whole three-hour job failed. A score is a score - if the
                # device cannot produce it, the CPU can, and only the job
                # dying is unrecoverable. A non-GPU error (a bad model path,
                # say) fails on the CPU too and surfaces from there.
                if self.cancel_flag and self.cancel_flag():
                    raise
                stalled = isinstance(e, CommandTimeout)
                # DEVICE_LOST is terminal for the CONTEXT, which is not the
                # same as the device being gone, and treating it as the latter
                # cost an episode its GPU scoring twice. Caught in the act on
                # S03E01: one scorer aborted with "SYCL graph wait: ...
                # DEVICE_LOST" followed by OUT_OF_DEVICE_MEMORY, while the
                # scorers running beside it finished normally in the next
                # breath and a selfcheck minutes later passed. So this counts
                # like any other failure: the streak below retires the device
                # only when nothing at all is getting through, which is what a
                # device that has really gone looks like.
                text = str(e)
                # Under the lock: probe workers fail concurrently, and the
                # flip to the CPU should be announced exactly once.
                with self._sycl_lock:
                    self._sycl_timeouts += 1
                    n = self._sycl_timeouts
                    # consecutive, not cumulative: see _sycl_scored_ok
                    flip = n >= self._SYCL_MAX_TIMEOUTS and self._sycl_ok
                    if flip:
                        self._sycl_ok = False
                        self._sycl_retired_at = time.time()
                if stalled:
                    logger.warning(
                        "optimizer: SYCL scoring of shot {} crf {} did not finish "
                        "within {}s (stall {} this job); scoring it on the CPU "
                        "instead", idx, crf, self._sycl_timeout(frames), n)
                else:
                    tail = text.strip().splitlines()[-1][:160] if text.strip() else e.__class__.__name__
                    logger.warning(
                        "optimizer: SYCL scoring of shot {} crf {} failed ({}); "
                        "scoring it on the CPU instead", idx, crf, tail)
                self._log(f"sycl failure shot {idx:05d} crf {crf}: {e}")
                if flip:
                    logger.warning(
                        "optimizer: libvmaf SYCL device {} failed {} times in a "
                        "row; scoring moves to the CPU until it passes a "
                        "preflight again", sycl, n)
            finally:
                self._gpu_vram.calibrate()
                self._gpu_vram.release(booked, need)
        return self._score_vmaf_on(-1, dist_args, ref_args, ref_vf, idx, crf,
                                   threads, timeout=3600)

    def _sycl_scored_ok(self) -> None:
        """A SYCL score came back: clear the stall streak.

        Cumulative counting cost an episode 400 shots of CPU scoring once -
        three stalls scattered among hundreds of good scores were enough to
        retire the device for the whole job. What should retire it is a run
        of failures with nothing working in between, which is what a sick
        device looks like; occasional slowness under a busy card is not.
        """
        if self._sycl_timeouts:
            with self._sycl_lock:
                self._sycl_timeouts = 0

    def _score_vmaf_on(self, sycl: int, dist_args: List[str], ref_args: List[str],
                       ref_vf: List[str], idx: int, crf: int,
                       threads: Optional[int], timeout: int,
                       zero_copy: bool = False) -> float:
        """One libvmaf run on the given backend (`sycl` < 0 = CPU); with
        `zero_copy` both inputs decode on the card instead (see _zero_copy)."""
        out_json = self.probe_dir / f"score_{idx:05d}_{crf}.json"
        args = self._vmaf_cmd(sycl, dist_args, ref_args, ref_vf, out_json, threads,
                              zero_copy=zero_copy)
        try:
            self._run(args, timeout=timeout)
        except CommandTimeout:
            raise
        except TranscodeError as e:
            raise TranscodeError(
                f"{e}\n"
                f"Hint: the quality probe could not run (model={self._model_cfg()}). "
                "If target_metric=ssimulacra2, note stock libvmaf has no ssimulacra2 "
                "built in - set transcode.optimizer.ssimulacra2_model to a compatible "
                "model (e.g. path=/path/to/ssimulacra2.json) or use target_metric=vmaf."
            ) from e
        score = parse_score(out_json, self.metric)
        self._log(f"shot {idx:05d} crf {crf} {self.metric}={score:.3f}")
        if not self.opt.keep_probes:
            try:
                out_json.unlink()
            except OSError:
                pass
        return score

    def _vmaf_cmd(self, sycl: int, dist_args: List[str], ref_args: List[str],
                  ref_vf: List[str], out_json: Path, threads: Optional[int],
                  zero_copy: bool = False, loglevel: str = "error") -> List[str]:
        """The ffmpeg command behind one libvmaf run (see _score_vmaf_on)."""
        # ts_sync_mode=nearest is NOT optional. The reference is read straight
        # from the source container while the distorted side is an ivf carrying
        # an exact frame-rate timebase, and matroska stores timestamps in whole
        # milliseconds: at 23.976fps the rounded mkv PTS drift up to ~1ms either
        # side of the ivf's, so framesync's default "nearest lower or equal"
        # picks the PREVIOUS frame about half the time. Measured on a 1080p mkv,
        # that one-frame slip scores 66.3 where the aligned pair scores 92.4 -
        # a 26 point error, all of it in the direction of a lower CRF and a
        # bigger file. Frame periods are ~42ms, so "nearest" cannot mis-pick.
        opts = [f"model={self._model_cfg()}", "log_fmt=json",
                f"log_path={out_json}", "shortest=1", "ts_sync_mode=nearest"]
        feats = self._vmaf_features()
        if feats:
            # av1an's --probing-vmaf-features uses its own CLI syntax
            # (e.g. "default motionless", "weighted neg") which ffmpeg's
            # libvmaf filter cannot parse. Only forward valid ffmpeg feature
            # configs (contain '=' / '|'); ignore the rest, warning once.
            if "=" in feats or "|" in feats:
                opts.append(f"feature={feats}")
            elif not self._feature_warned:
                self._feature_warned = True
                logger.warning(
                    "optimizer: ignoring probing_vmaf_features {!r} (av1an-style, "
                    "not understood by ffmpeg libvmaf; use feature=name=... "
                    "if needed, or leave empty for default VMAF features)",
                    feats,
                )
        if sycl >= 0:
            # Deliberately WITHOUT n_threads. On the GPU the CPU threads only
            # add frame pools and synchronisation, and the cost is monotone in
            # both directions - measured on a B580, same clip and same score:
            # unset 0.98s / 0.16GB, 4 threads 1.07s / 0.30GB, 8 threads 1.25s /
            # 0.43GB, 40 threads 2.04s / 1.54GB. Passing the CPU path's thread
            # count here would give back most of the memory the GPU just saved.
            opts.append(f"sycl_device={sycl}")
        else:
            n_threads = threads if threads is not None else self._vmaf_threads()
            if n_threads:
                opts.append(f"n_threads={n_threads}")
        # Input 0 is the DISTORTED encode and input 1 the REFERENCE source:
        # ffmpeg's libvmaf takes #0 as main (distorted) and #1 as reference.
        # Passing them the other way round makes libvmaf treat the encode as
        # the reference - the motion feature is then measured on the smoothed
        # encode and VIF sees detail being *added* rather than lost, which
        # inflates and flattens the whole CRF curve (measured +5 VMAF at CRF 20
        # and +18 at CRF 44 on 4K HDR10).
        scale = self._vmaf_scale_filter()
        fmt = f"format={self._pix_fmt()}"
        # Both sides are rebased to t=0 before they meet, so libvmaf pairs
        # frame k with frame k. Pairing by timestamp cannot work here because
        # the two sides never share one: a probe's ivf starts at 0 with exact
        # 1/fps periods, while the reference comes out of -ss still carrying
        # _seek's half-frame lead, i.e. at +0.5 frame, and Matroska rounds each
        # of those to a millisecond. ts_sync_mode=nearest is then choosing
        # between two frames exactly equidistant, and the rounding decides -
        # per frame. Measured on the dovi_split=bl mkv of a DV-P8 4K source,
        # one 120-frame window at CRF 20: per-frame scores alternate 48 / 93
        # and pool to 76.86; rebased, the same pair scores 93.97, which is what
        # the un-remuxed source gives. Every shot of that job read ~17 low and
        # fell back to CRF 20. Index pairing is right by construction: the ivf
        # was encoded from the very read the reference is (_probe_input), and
        # verification reads both files with the same seek, which the
        # half-frame lead makes land on the same index either side.
        rebase = "setpts=PTS-STARTPTS"
        hw: List[str] = []
        scorer = "libvmaf"
        if zero_copy:
            # Surfaces all the way in: libvmaf_sycl takes the frames as the
            # card decoded them, so there is no format= to convert and no
            # scale (see _zc_unsupported). The fps= subsampling and the
            # rebases only re-time frames and run on surfaces unchanged.
            scorer = "libvmaf_sycl"
            hw = ["-init_hw_device", f"vaapi={self._ZC_DEVICE}:{_render_nodes()[0]}"]
            dist_args = [*self._zc_input_args(), *dist_args]
            ref_args = [*self._zc_input_args(), *ref_args]
            dist_chain = rebase
            ref_chain = ",".join((*ref_vf, rebase))
        else:
            dist_chain = ",".join(f for f in (rebase, scale, fmt) if f)
            # the reference goes through the same probe-side filters the
            # distorted copy was encoded with, then both land on the same
            # comparison raster. The rebase comes AFTER those filters: fps=
            # subsampling re-times its output, and it is that output the ivf
            # holds.
            ref_chain = ",".join(f for f in (*ref_vf, rebase, scale, fmt) if f)
        lavfi = (f"[0:v]{dist_chain}[dist];[1:v]{ref_chain}[ref];"
                 f"[dist][ref]{scorer}={':'.join(opts)}")
        return ([self.ffmpeg, "-hide_banner", "-loglevel", loglevel, "-y", *hw]
                + dist_args + ref_args
                + ["-lavfi", lavfi, *self._VIDEO_ONLY_OUTPUT, "-f", "null", "-"])

    # ---------- phase 3: per-shot CRF selection ----------
    def _crf_floor(self, grid_lo: int) -> float:
        """Lowest CRF any shot may be assigned. Without min_crf this is just the
        bottom of the probe grid, which pick_crf falls back to whenever the
        target is unreachable - and at 4K that is the most expensive setting
        there is."""
        return float(max(grid_lo, self.opt.min_crf or 0))

    def _crf_ceiling(self, grid_hi: int) -> float:
        """Highest CRF any shot may be assigned - the mirror of _crf_floor.

        Separating this from the probe grid is what lets the grid be widened
        for free: probe_crfs says where to look, min_crf/max_crf say what may
        be shipped.
        """
        cap = int(self.opt.max_crf or 0)
        return float(min(grid_hi, cap) if cap > 0 else grid_hi)

    def pick_all_crfs(self, samples: ProbeSamples, grid: List[int]) -> Dict[int, float]:
        lo, hi = min(grid), max(grid)
        floor, ceil = self._crf_floor(lo), self._crf_ceiling(hi)
        offset = float(self.opt.probe_crf_offset or 0.0)
        chosen: Dict[int, float] = {}
        unreachable: List[float] = []
        for idx, by_crf in samples.items():
            pts = [(crf, score) for crf, score in by_crf.items() if score is not None]
            crf = pick_crf(pts, self.target) + offset
            chosen[idx] = max(floor, min(crf, ceil))
            best = max((s for _, s in pts), default=None)
            if best is not None and best < self.target:
                unreachable.append(best)
        if unreachable:
            logger.warning(
                "optimizer: {}/{} shot(s) cannot reach {} {:g} even at CRF {} "
                "(best probed score {:.1f}-{:.1f}); they fall back to CRF {:g}, "
                "which is what inflates the output size. Lower target_quality, "
                "extend probe_crfs downwards, or set optimizer.min_crf.",
                len(unreachable), len(samples), self.metric, self.target, lo,
                min(unreachable), max(unreachable), floor,
            )
            self._log(f"{len(unreachable)}/{len(samples)} shots below target "
                      f"{self.metric} {self.target:g} at CRF {lo}")
        return chosen

    def smooth_chosen(self, chosen: Dict[int, float]) -> Dict[int, float]:
        """Bound adjacent-shot CRF jumps (see smooth_crfs) to keep the picture
        visually continuous. max_crf_delta <= 0 disables smoothing."""
        max_delta = float(self.opt.max_crf_delta or 0)
        if max_delta <= 0 or len(chosen) <= 1:
            return chosen
        ordered_idx = sorted(chosen)
        smoothed = smooth_crfs([chosen[i] for i in ordered_idx], max_delta)
        grid = self._probe_grid()
        lo, hi = self._crf_floor(min(grid)), self._crf_ceiling(max(grid))
        out = {}
        for i, crf in zip(ordered_idx, smoothed):
            out[i] = max(lo, min(crf, hi))
        return out

    # ---------- phase 4: parallel final encode ----------
    def encode_all(self, shots: List[Shot], chosen: Dict[int, float]) -> List[Path]:
        """Encode every shot, admitting as many at once as the budgets allow.

        Not a fixed worker pool. Per-instance memory follows the shot length
        (2.5x across one real film's shots), so a constant worker count is
        simultaneously too many for the long shots - which is what OOM-kills the
        encoder - and too few for the short ones. Admission takes the longest
        shot that still fits the remaining memory and CPU, which keeps a long
        encode and several short ones in flight together. Measured against the
        fixed pool on a real 24-shot list at a 3.7GB budget: 23.3fps against
        21.4fps, and the budget actually filled (3.47GB of 3.7GB) where the
        fixed pool left 30% of it unused.
        """
        ladder = self._lp_ladder()
        ivf_paths: Dict[int, Path] = {}
        done_frames = [0]
        t0 = time.monotonic()
        self._heaviest_cmd = (0.0, "")
        with self._slot_lock:
            self._slots_used.clear()
        longest = max((b - a) for a, b in shots) if shots else 0

        def cost(idx: int, lp: int) -> float:
            s0, s1 = shots[idx]
            return self._est_encode_gb(s1 - s0, lp)

        def run_one(idx: int, lp: int, slot: int, threads: int) -> object:
            s0, s1 = shots[idx]
            return self._encode_shot(idx, s0, s1,
                                     chosen.get(idx, self.video.crf), lp,
                                     slot, threads)

        def on_done(idx: int, result: object) -> None:
            ivf_paths[idx] = result         # type: ignore[assignment]
            done_frames[0] += shots[idx][1] - shots[idx][0]

        def progress(_done: int) -> None:
            elapsed = max(time.monotonic() - t0, 1e-6)
            frames = done_frames[0]
            self._report(frames / max(self.total_frames, 1) * 100, frames,
                         self.total_frames, fps=frames / elapsed)

        self._log(
            f"encoding {len(shots)} shots (budget {self._mem_budget_gb():.1f}GB / "
            f"{self._cores()} cpu units, lp ladder {ladder}, longest shot "
            f"{longest}f -> {self._est_encode_gb(longest, ladder[0]):.1f}GB)")
        stop, peak = self._start_mem_sampler()
        try:
            self._schedule(list(range(len(shots))), phase="encoding", cost=cost,
                           run_one=run_one, on_done=on_done, progress=progress,
                           max_conc=self._max_concurrency(len(shots)),
                           ladder=ladder)
        finally:
            stop.set()
        self._log_phase_memory("encoding", peak[0])
        missing = [i for i in range(len(shots)) if i not in ivf_paths]
        if missing:
            raise TranscodeError(
                f"{len(missing)} shot(s) produced no encode: {missing[:5]}")
        return [ivf_paths[i] for i in range(len(shots))]

    def _log_phase_memory(self, phase: str, peak_children_mb: float) -> None:
        heavy = f"; heaviest: {self._heaviest_cmd[1]} {self._heaviest_cmd[0]:.0f}MB" \
            if self._heaviest_cmd[1] else ""
        cal = self._cal[phase]
        drift = (f", model x{cal.factor():.2f} after {cal.samples} samples"
                 if cal.samples else "")
        self._mem_log(
            f"[mem] {phase} peak children RSS={peak_children_mb:.0f}MB "
            f"(python={self._py_rss_mb():.0f}MB), peak concurrency "
            f"{getattr(self, '_sched_peak_conc', 0)} of a "
            f"{getattr(self, '_sched_budget', 0.0):.1f}GB budget{drift}{heavy}")
        if phase == "probing" and self._sycl_device() >= 0:
            self._mem_log(f"[vram] {phase}: {self._gpu_vram.summary()}")

    def _encode_shot(self, idx: int, s0: int, s1: int, crf: float, lp: int,
                     slot: int = -1, threads: int = 0) -> Path:
        dst = self.probe_dir / f"enc_{idx:05d}.ivf"
        # encode exactly (s1 - s0) frames: `-t` on input-seeked shots is not
        # frame-exact (off by a frame per shot), and 762 shots x 1 frame drift
        # = seconds of A/V desync once the audio is muxed whole. `-frames:v`
        # guarantees the exact frame count so the concat sums to the source.
        args = self._affinity_prefix(slot, threads) if threads and slot >= 0 else []
        args += [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
        vf: List[str] = []
        if self._p5:
            # The final encode reads each shot exactly once, so there is nothing
            # for a shard to amortise here: apply the RPU straight into the
            # encoder instead of staging a file.
            from app import dovi  # local import avoids a cycle

            pre, chain = dovi.dv_apply_chain(self.settings)
            args += pre
            vf.append(chain)
        exact_vf, exact_out = self._exact_frames()
        vf += exact_vf
        args += ["-ss", self._seek(s0), "-i", str(self.source),
                 "-frames:v", str(s1 - s0), "-map", "0:v:0",
                 "-vf", ",".join(vf), *exact_out]
        svt = _svt_params_dict(self.video)
        svt["lp"] = lp
        args += ["-c:v", "libsvtav1", "-preset", str(self.video.preset)]
        args += self._crf_args(crf, svt)
        if self.video.keyint:
            args += ["-g", str(self.video.keyint), "-keyint_min", str(self.video.keyint)]
        args += ["-svtav1-params", ":".join(f"{k}={v}" for k, v in svt.items())]
        args += ["-pix_fmt", self._pix_fmt(), "-f", "ivf", str(dst)]
        self._run(args, timeout=7200)
        if not dst.exists() or dst.stat().st_size == 0:
            raise TranscodeError(f"shot {idx} produced no output")
        self._assert_shot_length(idx, dst, s1 - s0)
        return dst

    def _assert_shot_length(self, idx: int, dst: Path, expected: int) -> None:
        """Fail the shot if the encoder did not write exactly `expected` frames.

        The point is to make this class of fault LOUD. A shot that starts on
        the wrong frame still emits the right count (-frames:v guarantees it),
        which is how a seek that mis-landed on 6 of 16 shots stayed invisible
        for months: frame count, duration and A/V sync all still checked out,
        and only the final shot - with no frame left to borrow - came up short.
        Counting here catches that at the shot that caused it rather than as a
        42ms discrepancy at the end of a multi-hour job.

        IVF carries one packet per displayed frame, so this is an index walk on
        a few-MB file: measured 58ms for a 96-frame 4K shot, i.e. ~29s spread
        across the encode pool for a 500-shot feature.

        NB: when scene detection finds nothing at all the shot list falls back
        to a single shot of int(fps * duration), which is an estimate and can
        overshoot the real frame count by a frame. That path would report a
        shortfall here; the message says exactly what was seen either way.
        
        The source read here stays on the CPU, and that is measured rather
        than inherited. On 300 4K frames with the production SVT parameters
        at lp=4, moving the decode to VA-API took CPU from 484s to 470s
        (-2.9%) and wall from 70.0s to 68.1s - but peak RSS from 7.30GiB to
        7.94GiB (+8.7%), because hwdownload allocates host frames on top of
        the surface pool rather than instead of them. Memory is what caps
        this phase's concurrency (see the encode budget log: peak RSS sits
        on the budget and concurrency drops below 10 whenever a long shot is
        in flight), so paying 8.7% of it to save 2.9% of CPU loses.
        What decides this per phase is how expensive the ENCODE is: scene
        detection is pure decode and gains hugely on the GPU, a probe encode
        at preset 9 spends ~30% of its cost decoding and gains, and a
        delivery encode at preset 4 spends 3% and does not.
        """
        written = self._count_frames(dst)
        if written is None or written == expected:
            return
        raise TranscodeError(
            f"shot {idx} encoded {written} frames but the shot spans "
            f"{expected}: the source range for this shot did not yield the "
            f"frames it should have (a seek that landed on the wrong frame, or "
            f"a shot list running past the end of the source)")

    def _count_frames(self, path: Path) -> Optional[int]:
        """Frames in a per-shot ivf, or None when ffprobe cannot say."""
        try:
            out = self._run(
                [self.settings.tool_path("ffprobe"), "-v", "error",
                 "-select_streams", "v:0", "-count_packets",
                 "-show_entries", "stream=nb_read_packets",
                 "-of", "csv=p=0", str(path)], timeout=300)
        except (TranscodeError, FileNotFoundError) as e:
            logger.warning("could not count frames in {} ({}); "
                           "skipping the shot length check", path.name, e)
            return None
        text = out.strip().splitlines()[-1].strip().rstrip(",") if out.strip() else ""
        try:
            return int(text)
        except ValueError:
            logger.warning("unexpected ffprobe frame count for {}: {!r}",
                           path.name, text)
            return None

    def _crf_args(self, crf: float, svt: Dict[str, object]) -> List[str]:
        """The chosen CRF as ffmpeg args, written into `svt` when fractional.

        ffmpeg's -crf is an INTEGER AVOption for libsvtav1, so a decimal is
        truncated on the way in: measured on SVT-AV1 v4.2, "-crf 28.5" produces
        a byte-identical file to "-crf 28" while "-crf 29" differs. That made
        fractional_crf a silent no-op and threw away the fine-grained CRF this
        engine exists to interpolate. SVT-AV1 itself accepts a fractional CRF,
        and -svtav1-params is handed to the library verbatim, so that is the
        only route which reaches it (measured: crf=28.5 lands between 28 and 29).
        """
        text = self._fmt_crf(crf)
        if self.opt.fractional_crf:
            svt["crf"] = text
            return []
        return ["-crf", text]

    def _fmt_crf(self, crf: float) -> str:
        crf = max(0.0, min(63.0, crf))
        if self.opt.fractional_crf:
            return f"{crf:g}"
        return str(round(crf))

    # ---------- phase 4b: verify what was actually delivered ----------
    def _verify_sample(self, n_shots: int, want: int) -> List[int]:
        """Shot indices to re-score: evenly spread across the timeline.

        Spread rather than random, because the faults worth catching are
        positional. A seek that starts mis-landing partway through, or a shot
        list that runs out early, reads low on every shot after that point -
        which a spread sample shows as a cliff and a clustered sample misses.
        """
        want = min(max(0, want), n_shots)
        if want <= 0:
            return []
        if want == 1:
            return [n_shots // 2]
        return sorted({round(i * (n_shots - 1) / (want - 1)) for i in range(want)})

    def _score_delivered(self, idx: int, w0: int, w1: int, crf: float) -> float:
        """Score frames [w0, w1) of the FINISHED file against the source.

        Both windows are read in place and piped straight into the metric.
        Staging them as lossless FFV1 first - which is what this used to do -
        buys nothing at all. Measured on a 4K DV-P8 pair, 120 frames of
        3840x1920 against the 4k model, three runs each:

            staged   17.0s wall   166s cpu   377MB written and re-read
            direct   11.2s wall    61s cpu     0MB

        and the two score IDENTICALLY, 99.081524 to six decimals, on every run
        and on a second window. So the staging was buying a 2.7x CPU bill and
        a third of a gigabyte per shot for a number that does not move.

        Two cases still stage, and they are the same two the probe path stages
        for, so _needs_shard() decides here as well: SSIMULACRA2 reads through
        bestsource, which indexes a file rather than a stream, and a Dolby
        Vision P5 reference has to have its RPU applied before it means
        anything.
        """
        threads = self._verify_threads()
        if not self._needs_shard():
            return self._score_windows(w0, w1, idx, int(round(crf)), threads)
        ref = self.probe_dir / f"verify_ref_{idx:05d}.mkv"
        dist = self.probe_dir / f"verify_out_{idx:05d}.mkv"
        try:
            self._extract_window(w0, w1, ref, apply_dv=self._p5)
            self._extract_window(w0, w1, dist, source=self.output)
            return self._score_probe(w0, w1, dist, idx, int(round(crf)),
                                     shard=ref, threads=threads)
        finally:
            for path in (ref, dist):
                try:
                    path.unlink()
                except OSError:
                    pass

    def verify_delivered(self, shots: List[Shot], chosen: Dict[int, float],
                         samples: ProbeSamples) -> None:
        """Re-score a sample of shots from the finished file and report.

        Everything before this trusts the probes: a CRF is chosen from a fast
        probe encode of a 120-frame window and then applied to the delivery,
        and nothing ever checks that the delivery landed where the probe said
        it would. That gap is where target_quality stops meaning anything - the
        probe preset under-reports, the probe window over-reports, and linear
        interpolation on a curved rate-distortion relationship under-reports
        again, all uncontrolled and all in different directions.

        This is diagnostic only: a failure to score never fails the job, since
        an encode that is otherwise fine should not be thrown away because
        ffprobe or libvmaf had a bad day.

        What it cannot see: a misalignment that both sides reproduce. Reference
        and delivered windows are both located with _seek, so if _seek were
        wrong again the source-side window would slip by exactly the amount the
        encode slipped and the two would still agree. Verified by running this
        against an output built with the pre-fix seek - every sampled shot came
        back clean. The seek class is covered by _assert_shot_length and the
        source-versus-output duration check instead, which is how it surfaced.
        Scoring each window at offsets -1/0/+1 and reporting which one wins
        would close the gap at three times the metric cost.
        """
        idxs = self._verify_sample(len(shots), int(self.opt.verify_shots or 0))
        if not idxs:
            return
        self._stage("verifying", 0.0)
        logger.info("optimizer: verifying delivered {} on {} sampled shot(s)",
                    self.metric, len(idxs))
        rows: List[Tuple[int, float, Optional[float], float]] = []
        for done, idx in enumerate(idxs, start=1):
            s0, s1 = shots[idx]
            w0, w1 = self._probe_window(s0, s1)
            crf = float(chosen.get(idx, self.video.crf))
            try:
                score = self._score_delivered(idx, w0, w1, crf)
            except (TranscodeError, OSError) as e:
                logger.warning("optimizer: could not verify shot {} ({})", idx, e)
                continue
            predicted = predict_score(list((samples.get(idx) or {}).items()), crf)
            rows.append((idx, crf, predicted, score))
            self._log(f"verify shot {idx:05d} crf {crf:g} "
                      f"predicted={predicted if predicted is None else round(predicted, 2)} "
                      f"delivered={score:.2f}")
            self._report(done / len(idxs) * 100, done, len(idxs))
        if not rows:
            logger.warning("optimizer: no shot could be verified")
            return
        self._report_verification(rows)

    def _report_verification(
            self, rows: List[Tuple[int, float, Optional[float], float]]) -> None:
        scores = sorted(r[3] for r in rows)
        n = len(scores)
        median = scores[n // 2] if n % 2 else (scores[n // 2 - 1] + scores[n // 2]) / 2
        below = [s for s in scores if s < self.target]
        summary = (f"delivered {self.metric} over {n} sampled shot(s): "
                   f"min {scores[0]:.2f}, median {median:.2f}, "
                   f"max {scores[-1]:.2f} (target {self.target:g})")
        self._mem_log(summary)
        if below:
            logger.warning(
                "optimizer: {}/{} sampled shot(s) delivered below target "
                "{} {:g} (lowest {:.2f})", len(below), n, self.metric,
                self.target, scores[0])
        deltas = [score - pred for _, _, pred, score in rows if pred is not None]
        if not deltas:
            return
        bias = sum(deltas) / len(deltas)
        self._mem_log(
            f"probe prediction vs delivered: mean {bias:+.2f} {self.metric} "
            f"over {len(deltas)} shot(s)")
        if bias < -5.0:
            # An encoder-vs-probe difference is a point or two. A gap this size
            # is the signature of the two sides not holding the same frames.
            logger.warning(
                "optimizer: delivered {} is {:.1f} below what the probes "
                "predicted. A difference that large is usually misalignment - "
                "the compared windows not holding the same frames - rather "
                "than an encoder-versus-probe difference.", self.metric, -bias)
        elif abs(bias) >= 0.5:
            direction = ("under-reports" if bias > 0 else "over-reports")
            logger.info(
                "optimizer: the probes {} the delivered {} by {:.2f} on "
                "average; transcode.optimizer.probe_crf_offset trades that "
                "bias back for size.", direction, self.metric, abs(bias))

    # ---------- phase 5: concat + mux ----------
    # tx3g only exists in MP4; Matroska cannot carry it, so those have to be
    # converted. Everything else must be copied - notably the PGS and VobSub
    # streams a Blu-ray remux carries, which are BITMAP subtitles: asking
    # ffmpeg to make srt out of them fails the whole job with "Subtitle
    # encoding currently only possible from text to text or bitmap to bitmap".
    _SUBS_NEEDING_CONVERSION = {"mov_text"}

    def _subtitle_codec_args(self, source: str) -> List[str]:
        """Per-stream -c:s arguments for remuxing `source`'s subtitles to mkv."""
        try:
            out = self._run([self.settings.tool_path("ffprobe"), "-v", "error",
                             "-select_streams", "s",
                             "-show_entries", "stream=codec_name",
                             "-of", "csv=p=0", source], timeout=120)
        except TranscodeError as e:
            logger.warning("could not probe subtitle codecs ({}); copying", e)
            return ["-c:s", "copy"]
        codecs = [_csv_first(c) for c in out.splitlines() if _csv_first(c)]
        if not codecs:
            return ["-c:s", "copy"]
        args: List[str] = []
        for i, codec in enumerate(codecs):
            convert = codec in self._SUBS_NEEDING_CONVERSION
            args += [f"-c:s:{i}", "srt" if convert else "copy"]
        return args

    def concat_shots(self, ivf_paths: List[Path],
                     shots: Optional[List[Shot]] = None) -> None:
        if not ivf_paths:
            raise TranscodeError("no shot encodes to concatenate")
        list_file = self.tempdir / "concat.txt"
        # Each entry carries its shot's duration in slots. The concat demuxer
        # otherwise starts the next file where the previous one's last frame
        # ended, and a frame missing at the very end of a shot (see _slot)
        # would then pull every later shot a frame early.
        lines = []
        for i, p in enumerate(ivf_paths):
            lines.append(f"file {concat_quote(p)}\n")
            if shots is not None and i < len(shots):
                lines.append(f"duration {self._span(*shots[i]):.6f}\n")
        list_file.write_text("".join(lines), encoding="utf-8")
        video_only = self.tempdir / "video_only.mkv"
        args = [self.ffmpeg, "-hide_banner", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file), "-c", "copy", "-fflags", "+genpts",
                "-f", "matroska", str(video_only)]
        self._run(args, timeout=1800)
        # audio + subs from the ORIGINAL source (info.path, not self.source:
        # for Dolby Vision the encode input is a video-only stripped
        # intermediate, so muxing from it would drop audio).
        audio_src = str(self.info.path if self.info.path else self.source)
        audio_subs: Optional[Path] = self.tempdir / "audio_subs.mkv"
        if not self._has_audio_or_subs(audio_src):
            # Nothing to carry over, and asking anyway is actively dangerous:
            # "-map 0:a? -map 0:s?" against a video-only source maps NOTHING,
            # and ffmpeg with zero output streams grows to ~8GB before it
            # exits. Measured on a 15MB, 12-second 4K file: 8.19GB peak, and
            # 8.20GB for the 24-second version - a flat allocation, unrelated
            # to the input size. Under a container memory limit that is an
            # OOM kill at the very last step, after the whole encode is done.
            logger.info("optimizer: source has no audio or subtitle streams; "
                        "muxing video only")
            audio_subs = None
        else:
            # NB: no -map_metadata -1 here: it strips per-stream LANGUAGE tags
            # from the subtitle/audio streams (Plex then shows every subtitle
            # as English). The source's global "DV.HDR10.PLUS" title is
            # harmless in this intermediate - mkvmerge does not copy it into
            # the final file.
            base_args = [self.ffmpeg, "-hide_banner", "-y", "-loglevel", "error",
                         "-i", audio_src, "-map", "0:a?", "-map", "0:s?",
                         "-c:a", "copy"]
            try:
                self._run(base_args + self._subtitle_codec_args(audio_src)
                          + [str(audio_subs)], timeout=1800)
            except TranscodeError:
                logger.warning("subtitle remux failed, retrying with a plain copy")
                self._run(base_args + ["-c:s", "copy", str(audio_subs)],
                          timeout=1800)
        # Final mux via mkvmerge: ffmpeg's -c copy remux of the concat leaves
        # the shot-boundary structure that Plex's 4K AV1 transcode hangs on
        # (runs for 10-20 min then stops producing HLS segments; verified on
        # 4K DV-P8 output while a single-continuous encode is fine). mkvmerge
        # rebuilds the container so the same bitstream plays and seeks fine.
        # NB: mkvmerge does NOT copy the source's global metadata (which would
        # carry a misleading "...DV.HDR10.PLUS..." title onto an HDR10 stream).
        mkvmerge = self.settings.tools.mkvmerge
        if shutil.which(mkvmerge) and self._mkvmerge_mux(mkvmerge, video_only,
                                                        audio_subs):
            return
        logger.warning("mkvmerge unavailable or failed; falling back to ffmpeg mux")
        # fallback: global metadata comes from the first input (video_only,
        # which has no title), stream language tags ride along with the
        # mapped streams, so no -map_metadata -1 needed here either.
        base = [self.ffmpeg, "-hide_banner", "-y"]
        lead = self._mux_lead(audio_subs)
        if lead > 0:
            base += ["-itsoffset", f"{lead:.6f}"]
        base += ["-i", str(video_only)]
        if audio_subs is not None:
            base += ["-i", str(audio_subs)]
        base += ["-map", "0:v:0"]
        if audio_subs is not None:
            base += ["-map", "1:a?", "-map", "1:s?"]
        base += ["-c", "copy"]
        try:
            self._run(base + [str(self.output)], timeout=1800)
        except TranscodeError:
            if self.output.exists():
                try:
                    self.output.unlink()
                except OSError:
                    pass
            logger.warning("subtitle stream copy failed, converting subtitles to srt")
            self._run(base + ["-c:s", "srt", str(self.output)], timeout=1800)
        if not self.output.exists() or self.output.stat().st_size == 0:
            raise TranscodeError("optimizer produced no output file")

    def _has_audio_or_subs(self, source: str) -> bool:
        """Whether `source` carries anything the final mux needs to copy over.

        Assumed present when ffprobe cannot say: running the remux for nothing
        wastes a pass, but skipping it wrongly silently drops the audio.
        """
        # One probe over every stream's type. NB not "-select_streams a,s":
        # ffprobe takes a single stream specifier, not a list, and rejects that
        # with "Invalid stream specifier" - which this method would then treat
        # as "cannot tell", quietly restoring the behaviour it exists to avoid.
        try:
            out = self._run([self.settings.tool_path("ffprobe"), "-v", "error",
                             "-show_entries", "stream=codec_type",
                             "-of", "csv=p=0", source], timeout=120)
        except (TranscodeError, FileNotFoundError) as e:
            logger.warning("could not probe {} for audio/subtitle streams ({}); "
                           "assuming there are some", Path(source).name, e)
            return True
        kinds = {_csv_first(line) for line in out.splitlines()}
        return bool(kinds & {"audio", "subtitle"})

    def _mux_lead(self, audio_subs: Optional[Path]) -> float:
        """Seconds to delay the video track by in the final mux.

        The encoded video starts at 0 - every shot ivf does - while the audio
        and subtitles are copied from the ORIGINAL file with their timestamps
        kept. So a source whose picture began after its sound would come out
        with the picture that much early: measured on a clip with video at
        0.066 and audio at 0, the finished file had both at 0. The original's
        lead is the amount, and it applies whichever file was actually encoded
        (the Dolby Vision intermediates start at 0 but the audio does not come
        from them). Nothing to align when there is no audio.
        """
        if audio_subs is None:
            return 0.0
        return self._lead_of(self.info.path if self.info.path else self.source)

    def _mkvmerge_mux(self, mkvmerge: str, video_only: Path,
                      audio_subs: Optional[Path]) -> bool:
        """Final mux with mkvmerge. Returns True on a valid output file."""
        try:
            if self.output.exists():
                self.output.unlink()
            inputs: List[str] = []
            lead_ms = int(round(self._mux_lead(audio_subs) * 1000))
            if lead_ms > 0:
                # --sync applies to the track of the input that FOLLOWS it
                inputs += ["--sync", f"0:{lead_ms}"]
            inputs.append(str(video_only))
            if audio_subs is not None:
                inputs.append(str(audio_subs))
            proc = subprocess.run(
                [mkvmerge, "-o", str(self.output), *inputs],
                capture_output=True, text=True, timeout=1800,
            )
            if proc.returncode != 0:
                logger.error("mkvmerge failed: {}", (proc.stderr or "")[-500:])
                return False
            return self.output.exists() and self.output.stat().st_size > 0
        except Exception as e:  # noqa: BLE001
            logger.error("mkvmerge mux error: {}", e)
            return False

    # ---------- orchestration ----------
    def run(self) -> None:
        try:
            self._stage("scenedetect", 0.0)
            self._mem_log(f"[mem] job start python={self._py_rss_mb():.0f}MB "
                          f"source={self.info.duration:.1f}s @ {self.fps:g}fps "
                          f"~{self.total_frames} frames")
            shots = self.detect_shots()
            self._log(f"{len(shots)} shot(s) from scene detection")
            # Probe the source's video lead once, here, rather than letting
            # the first few probe workers all discover it at the same time.
            self._lead_of(self.source)

            self._stage("probing", 0.0)
            self._report(0.0, 0, len(shots))  # surface the shot count to the UI
            grid = self._probe_grid()
            mode = self._probe_mode()
            self._dataset_header(shots, grid)
            if mode == "qsv":
                samples, chosen = self.probe_all_gpu(shots, grid)
            else:
                samples = (self.probe_all_verified(shots, grid)
                           if mode == "qsv+svt" else self.probe_all(shots, grid))
                chosen = self.pick_all_crfs(samples, grid)
            if float(self.opt.max_crf_delta or 0) > 0:
                ideal = ", ".join(f"{i}:{chosen[i]:g}" for i in sorted(chosen))
                chosen = self.smooth_chosen(chosen)
                self._log(f"ideal per-shot CRFs -> {ideal}")
            else:
                chosen = self.smooth_chosen(chosen)
            crf_line = ", ".join(f"{i}:{chosen[i]:g}" for i in sorted(chosen))
            self._log(f"chosen per-shot CRFs -> {crf_line}")
            # the training target, after smoothing: what each shot is really
            # encoded at, which is not always what its own probes picked
            self._dataset_write({"type": "crfs",
                                 "final": {str(i): chosen[i] for i in sorted(chosen)}})

            self._stage("encoding", 0.0)
            ivf_paths = self.encode_all(shots, chosen)
            self._report(100.0, self.total_frames, self.total_frames)

            self.concat_shots(ivf_paths, shots)
            self._report(100.0, self.total_frames, self.total_frames)

            self.verify_delivered(shots, chosen, samples)
        finally:
            self._cleanup_shards()
            self.close()
