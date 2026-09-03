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

import json
import math
import os
import re
import shutil
import sys
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
        # scorings the SYCL backend failed to finish in time (see _score_vmaf)
        self._sycl_timeouts = 0
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
    _SCD_FRAME = re.compile(r"^frame:(\d+)")
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
        return self.video.probe_res or self.opt.probe_scale or ""

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
        1.955s into its container sits at -ss 1.955, not 0.
        """
        lead = self._lead_of(path or self.source)
        return f"{max(0.0, lead + (frame - 0.5) / self.fps):.6f}"

    def _exact_frames(self, rate: float) -> Tuple[List[str], List[str]]:
        """(video filters, output options) that make an encoder read emit
        exactly one frame per decoded frame, with pts 0, 1, 2, ...

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

        So the timestamps are rebuilt from the frame index in a fine timebase
        (AVTB is microseconds; the millisecond one would round again) and
        frame sync is switched off. `rate` is the frame rate the frames arrive
        at - the source's, or fps/probing_rate after the probe subsampling.
        Appended AFTER the read's other filters, which is where the frames it
        counts come out.
        """
        return (["settb=AVTB", f"setpts=N/{rate:.6f}/TB"],
                ["-fps_mode", "passthrough"])

    def _extract_window(self, w0: int, w1: int, dest: Path, *,
                        source: Optional[Path] = None,
                        vf: Optional[List[str]] = None,
                        apply_dv: bool = False,
                        rate: Optional[float] = None) -> None:
        """Copy frames [w0, w1) of `source` (default: the encode input) into a
        lossless file.

        -ss + -frames:v, the same frame-exact pairing the final encode uses; a
        `-t` duration here is what let windows come out a frame short.
        """
        pre: List[str] = []
        chain: List[str] = list(vf or [])
        if apply_dv:
            from app import dovi  # local import avoids a cycle

            pre, dv = dovi.dv_apply_chain(self.settings)
            chain.append(dv)
        exact_vf, exact_out = self._exact_frames(rate or self.fps)
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
            self._extract_window(w0, w1, dest, vf=vf, apply_dv=self._p5,
                                 rate=self.fps / self._probing_rate())
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
                          "-t", f"{(w1 - w0) / self.fps:.6f}",
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
                f"(a full grid would have been {len(grid)})")
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
        # a shard already holds the subsampled frames, so the rate is the
        # same either way: what the frames arrive at after probing_rate
        exact_vf, exact_out = self._exact_frames(self.fps / self._probing_rate())
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

    def _score_probe(self, w0: int, w1: int, dist: Path, idx: int, crf: int,
                     shard: Optional[Path] = None,
                     threads: Optional[int] = None) -> float:
        """Score a distorted window that already exists as a file."""
        if self.metric == "ssimulacra2":
            return self._score_ssimulacra2(w0, w1, dist, idx, crf, shard)
        ref_args, ref_vf = self._probe_input(w0, w1, shard)
        dist_args = ["-i", str(dist)]
        if self.metric == "xpsnr":
            return self._score_xpsnr(dist_args, ref_args, ref_vf, idx, crf)
        return self._score_vmaf(dist_args, ref_args, ref_vf, idx, crf,
                                threads=threads,
                                frames=(w1 - w0) // self._probing_rate())

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
                "-t", f"{(w1 - w0) / self.fps:.6f}", "-i", str(source)]

    def _score_windows(self, w0: int, w1: int, idx: int, crf: int,
                       threads: int) -> float:
        """Score [w0, w1) of the finished output against the same frames of the
        source, reading both in place."""
        dist_args = self._window_input(w0, w1, self.output, threads)
        ref_args = self._window_input(w0, w1, self.source, threads)
        if self.metric == "xpsnr":
            return self._score_xpsnr(dist_args, ref_args, [], idx, crf)
        return self._score_vmaf(dist_args, ref_args, [], idx, crf,
                                threads=threads, frames=w1 - w0)

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

    def _sycl_timeout(self, frames: Optional[int]) -> int:
        """Seconds a SYCL scoring may take before it is killed and retried.

        A 120-frame 4K window scores in 1-6s on the GPU, ~30s with ten
        scorers contending. A stall never finishes at all (see
        _VIDEO_ONLY_OUTPUT for the one that was found), so what matters is
        that the budget is a small multiple of the honest case, not the
        3600s the CPU path keeps: 60s plus a second per frame.
        """
        return 60 + max(0, int(frames or 0))

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
        if sycl >= 0:
            try:
                return self._score_vmaf_on(sycl, dist_args, ref_args, ref_vf,
                                           idx, crf, threads,
                                           timeout=self._sycl_timeout(frames))
            except CommandTimeout as e:
                # Under the lock: probe workers time out concurrently, and
                # the flip to the CPU should be announced exactly once.
                with self._sycl_lock:
                    self._sycl_timeouts += 1
                    n = self._sycl_timeouts
                    flip = n >= self._SYCL_MAX_TIMEOUTS and self._sycl_ok
                    if flip:
                        self._sycl_ok = False
                logger.warning(
                    "optimizer: SYCL scoring of shot {} crf {} did not finish "
                    "within {}s (stall {} this job); scoring it on the CPU "
                    "instead", idx, crf, self._sycl_timeout(frames), n)
                self._log(f"sycl timeout shot {idx:05d} crf {crf}: {e}")
                if flip:
                    logger.warning(
                        "optimizer: libvmaf SYCL device {} stalled {} times; "
                        "the rest of this job scores on the CPU", sycl, n)
        return self._score_vmaf_on(-1, dist_args, ref_args, ref_vf, idx, crf,
                                   threads, timeout=3600)

    def _score_vmaf_on(self, sycl: int, dist_args: List[str], ref_args: List[str],
                       ref_vf: List[str], idx: int, crf: int,
                       threads: Optional[int], timeout: int) -> float:
        """One libvmaf run on the given backend (`sycl` < 0 = CPU)."""
        out_json = self.probe_dir / f"score_{idx:05d}_{crf}.json"
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
        dist_chain = ",".join(f for f in (rebase, scale, fmt) if f)
        # the reference goes through the same probe-side filters the distorted
        # copy was encoded with, then both land on the same comparison raster.
        # The rebase comes AFTER those filters: fps= subsampling re-times its
        # output, and it is that output the ivf holds.
        ref_chain = ",".join(f for f in (*ref_vf, rebase, scale, fmt) if f)
        lavfi = (f"[0:v]{dist_chain}[dist];[1:v]{ref_chain}[ref];"
                 f"[dist][ref]libvmaf={':'.join(opts)}")
        args = ([self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
                + dist_args + ref_args
                + ["-lavfi", lavfi, *self._VIDEO_ONLY_OUTPUT, "-f", "null", "-"])
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

    # ---------- phase 3: per-shot CRF selection ----------
    def _crf_floor(self, grid_lo: int) -> float:
        """Lowest CRF any shot may be assigned. Without min_crf this is just the
        bottom of the probe grid, which pick_crf falls back to whenever the
        target is unreachable - and at 4K that is the most expensive setting
        there is."""
        return float(max(grid_lo, self.opt.min_crf or 0))

    def pick_all_crfs(self, samples: ProbeSamples, grid: List[int]) -> Dict[int, float]:
        lo, hi = min(grid), max(grid)
        floor = self._crf_floor(lo)
        offset = float(self.opt.probe_crf_offset or 0.0)
        chosen: Dict[int, float] = {}
        unreachable: List[float] = []
        for idx, by_crf in samples.items():
            pts = [(crf, score) for crf, score in by_crf.items() if score is not None]
            crf = pick_crf(pts, self.target) + offset
            chosen[idx] = max(floor, min(crf, float(hi)))
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
        lo, hi = self._crf_floor(min(grid)), max(grid)
        out = {}
        for i, crf in zip(ordered_idx, smoothed):
            out[i] = max(lo, min(crf, float(hi)))
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
        exact_vf, exact_out = self._exact_frames(self.fps)
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
        codecs = [c.strip() for c in out.splitlines() if c.strip()]
        if not codecs:
            return ["-c:s", "copy"]
        args: List[str] = []
        for i, codec in enumerate(codecs):
            convert = codec in self._SUBS_NEEDING_CONVERSION
            args += [f"-c:s:{i}", "srt" if convert else "copy"]
        return args

    def concat_shots(self, ivf_paths: List[Path]) -> None:
        if not ivf_paths:
            raise TranscodeError("no shot encodes to concatenate")
        list_file = self.tempdir / "concat.txt"
        list_file.write_text(
            "".join(f"file {concat_quote(p)}\n" for p in ivf_paths),
            encoding="utf-8")
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
        kinds = {line.strip() for line in out.splitlines()}
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
            samples = self.probe_all(shots, grid)
            chosen = self.pick_all_crfs(samples, grid)
            if float(self.opt.max_crf_delta or 0) > 0:
                ideal = ", ".join(f"{i}:{chosen[i]:g}" for i in sorted(chosen))
                chosen = self.smooth_chosen(chosen)
                self._log(f"ideal per-shot CRFs -> {ideal}")
            else:
                chosen = self.smooth_chosen(chosen)
            crf_line = ", ".join(f"{i}:{chosen[i]:g}" for i in sorted(chosen))
            self._log(f"chosen per-shot CRFs -> {crf_line}")

            self._stage("encoding", 0.0)
            ivf_paths = self.encode_all(shots, chosen)
            self._report(100.0, self.total_frames, self.total_frames)

            self.concat_shots(ivf_paths)
            self._report(100.0, self.total_frames, self.total_frames)

            self.verify_delivered(shots, chosen, samples)
        finally:
            self._cleanup_shards()
            self.close()
