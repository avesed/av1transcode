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
import os
import re
import shutil
import sys
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger

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
        # probe pool size, used to auto-size libvmaf threads (see _vmaf_threads)
        self._probe_worker_count = 1
        # pool-thread -> core slice index, for taskset affinity (see _worker_slot)
        self._slots: Dict[int, int] = {}
        self._slot_lock = threading.Lock()
        # Dolby Vision Profile 5: the base layer is ICtCp, so it has to have its
        # RPU applied before it means anything. Done per shot rather than once
        # over the whole file - see _acquire_shard.
        self._p5 = bool(plan.p5 and settings.transcode.dovi.enabled)
        self._shards: Dict[int, Path] = {}
        self._shard_lock = threading.Lock()
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

    # SVT-AV1 only accepts a level of parallelism in [0, 6]; anything higher is
    # clamped with a warning on every single probe. It is also the main driver
    # of per-instance memory at 4K, since it sizes the picture buffer pool.
    MAX_LP = 6

    def _svt_lp(self, workers: int) -> int:
        """Bound SVT-AV1 parallelism per parallel probe instance.

        N encoders run at once; giving each the full core count spawns
        N x cores threads and can exhaust RAM on large machines. Scale
        per-instance parallelism so the total stays near the core count.
        """
        cores = os.cpu_count() or 1
        return max(1, min(cores, self.MAX_LP, round(cores / max(1, workers))))

    @staticmethod
    def _ram_gb() -> int:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // (1024 * 1024)
        except (OSError, ValueError):
            return 8
        return 8

    def _megapixels(self) -> float:
        px = (self.info.width or 0) * (self.info.height or 0)
        return max(0.5, px / 1e6) if px else 2.0

    def _est_encode_gb(self) -> float:
        """Peak RSS of ONE final-encode instance, in GB.

        SVT-AV1's picture buffer pool dominates, and it scales with frame size,
        which the old flat "ram_gb // 8" rule ignored entirely. Measured on this
        codebase's defaults (preset 4, 10-bit, 144-frame shots, uncontended):

            3840x1920  9.2GB @ lp=6   7.25GB @ lp=4
            1920x960   2.6GB @ lp=6   2.50GB @ lp=4
            1280x640   1.8GB @ lp=6   1.64GB @ lp=4

        0.8 + 0.9/Mpx tracks the lp<=4 column and stays slightly above every
        measured point, which is the right side to err on for an OOM guard.
        """
        return 0.8 + 0.9 * self._megapixels()

    def _est_probe_gb(self) -> float:
        """Same, for a probe instance: probe_preset is far faster and lp is
        smaller, measured 3.5GB at 4K against 9.2GB for the final encode."""
        return 0.5 + 0.5 * self._megapixels()

    def _mem_bounded_workers(self, per_instance_gb: float) -> int:
        """How many encoder instances fit in RAM, with headroom.

        75% of total memory: the rest is the OS, page cache for a multi-GB
        source read, and whatever else shares the box. Also capped at cores//4,
        because throughput saturates long before that anyway - measured at 4K on
        32 cores, 3 workers already reach 23.8fps and 4 or 6 add nothing.
        """
        cores = os.cpu_count() or 1
        budget = self._ram_gb() * 0.75
        return max(1, min(int(budget / max(0.5, per_instance_gb)),
                          max(1, cores // 4)))

    def _encode_workers(self, num_shots: int) -> int:
        """Final-encode concurrency, bounded by RAM rather than cores."""
        w = self.opt.encode_workers
        if not w or w <= 0:
            w = self._mem_bounded_workers(self._est_encode_gb())
        return max(1, min(w, num_shots))

    def _encode_threads(self, workers: int) -> int:
        """CPU cores allotted per final-encode instance. SVT-AV1 does not honour
        -threads and spawns ~80+ threads at 4K regardless of cores, so this is
        enforced with taskset affinity (each worker gets a disjoint core range).
        It does not reduce per-instance memory; it only stops parallel instances
        from oversubscribing cores and thrashing each other."""
        cores = os.cpu_count() or 1
        t = self.opt.encode_threads
        if not t or t <= 0:
            t = max(1, cores // max(1, workers))
        return max(1, min(t, cores))

    def _affinity_prefix(self, worker_idx: int, threads: int) -> List[str]:
        """taskset prefix pinning worker `worker_idx` to a disjoint slice of
        `threads` cores. Empty when the slice covers every core (single worker
        on a free machine), letting SVT-AV1 use all cores without extra
        subprocess overhead."""
        cores = os.cpu_count() or 1
        if threads >= cores:
            return []
        start = (worker_idx * threads) % cores
        end = start + threads - 1
        if end >= cores:
            start, end = 0, threads - 1
        return ["taskset", "-c", f"{start}-{end}"]

    def _worker_slot(self, workers: int) -> int:
        """Stable 0..workers-1 slot for the calling pool thread.

        Core ranges must be keyed on the WORKER, not the shot index: shots
        finish out of order, so `shot_idx % workers` puts two concurrent
        encoders on the same core slice while another slice sits idle.
        ThreadPoolExecutor reuses its threads, so one slot per thread ident is
        stable for the whole phase.
        """
        tid = threading.get_ident()
        with self._slot_lock:
            slot = self._slots.get(tid)
            if slot is None:
                slot = len(self._slots) % max(1, workers)
                self._slots[tid] = slot
            return slot

    def _encode_lp(self, workers: int) -> int:
        """SVT-AV1 level of parallelism for the FINAL encode.

        lp sizes the frame buffer pool that dominates 4K memory, and the top of
        its range does not pay for itself: at 4K the picture buffer count jumps
        from 107 (lp=4) to 305 (lp=6) for no throughput gain. Measured with 3
        parallel workers on 32 cores, lp=4 was both lighter and marginally
        faster than lp=6 (23.8fps / 21.2GB vs 23.3fps / 22.4GB), and one
        instance alone drops from 9.2GB to 7.25GB. So cap at 4, not 6.
        """
        cores = os.cpu_count() or 1
        return max(1, min(cores, round(cores / max(1, workers)), 4))

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
            raise TranscodeError(f"command timed out after {timeout}s: {args[0]}")
        finally:
            with self._proc_lock:
                self._procs.discard(proc)
            stop.set()
            mon.join(timeout=2)
        peak_mb = peak[0] / 1024.0
        desc = self._cmd_desc(args)
        if peak_mb > self._heaviest_cmd[0]:
            self._heaviest_cmd = (peak_mb, desc)
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
        args = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(self.source), "-vf", f"scale={scale}",
                "-c:v", "libx264", "-preset", "ultrafast", "-an", "-sn",
                "-f", "matroska", str(out)]
        self._run_with_progress(args, timeout=7200, total_seconds=total_sec,
                                tag="downscale for detection")
        if not out.exists() or out.stat().st_size == 0:
            raise TranscodeError("scene detection downscale produced no output")
        return out

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
            shots = [(int(a.frame_num), int(b.frame_num)) for a, b in scenes]
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
                logger.info("optimizer: {} shot(s) detected", len(shots))
            return shots
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

    def _seek(self, frame: int) -> str:
        """-ss value that reliably lands ON `frame`, never past it.

        -ss discards frames whose timestamp is below the one asked for, and
        containers store those rounded - Matroska keeps whole milliseconds
        while a 23.976fps frame time is an infinite decimal - so asking for a
        frame's exact time lands just above its stored timestamp often enough
        to skip it. Measured against per-frame hashes of a real 16-shot split,
        6 of 16 shots began one frame late. Half a frame of lead cannot
        overshoot: the preceding frame is a whole period further back, against
        at most 0.5ms of rounding (8x the margin even at 119.88fps, 42x at
        23.976). Measured after the change: 0 of 16 shots slip.
        """
        return f"{max(0.0, (frame - 0.5) / self.fps):.6f}"

    def _extract_window(self, w0: int, w1: int, dest: Path, *,
                        source: Optional[Path] = None,
                        vf: Optional[List[str]] = None,
                        apply_dv: bool = False) -> None:
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
        args = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *pre,
                "-ss", self._seek(w0), "-i", str(source or self.source),
                "-frames:v", str(w1 - w0), "-map", "0:v:0"]
        if chain:
            args += ["-vf", ",".join(chain)]
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

        ffmpeg's libvmaf defaults n_threads to 0, which is single-threaded, and
        the score is now computed on 1080p frames decoded from 4K sources - the
        measurement ends up slower than the probe encode it is measuring (7.0s
        vs 2.1s at n_threads=8 for the same clip, identical score). Auto-size it
        to the cores each probe worker has to itself.
        """
        explicit = self.video.vmaf_threads or self.opt.vmaf_threads
        if explicit:
            return explicit
        cores = os.cpu_count() or 1
        return max(1, cores // max(1, self._probe_worker_count))

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
                     shard: Optional[Path] = None) -> Tuple[List[str], List[str]]:
        """(input args, video filters) reading frames [w0, w1) of the source.

        Used identically by the probe encode and by the VMAF reference read, so
        the two are frame-aligned by construction and there is no multi-GB y4m
        intermediate to cache: at 4K 10-bit a y4m frame is 22MB, and the frames
        would have to be re-read once per CRF anyway.

        A DV shard already holds exactly these frames with the probe-side
        filters baked in, so it is read whole and needs no filters of its own.
        """
        if shard is not None:
            return ["-i", str(shard)], []
        args = ["-ss", f"{w0 / self.fps:.6f}", "-t", f"{(w1 - w0) / self.fps:.6f}",
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
        """Probe concurrency.

        Probes encode at the source resolution now, so each SVT-AV1 instance
        holds a frame buffer pool of the same order as the final encode and the
        limit is RAM, not cores: a 4K probe measured 3.5GB peak RSS, against
        ~166MB for the old 540p probes.
        """
        w = self.opt.probe_workers
        if not w or w <= 0:
            w = self._mem_bounded_workers(self._est_probe_gb())
        return max(1, min(w, n_tasks))

    def probe_all(self, shots: List[Shot], grid: List[int]) -> ProbeSamples:
        """Probe every shot, one pool task per shot.

        Per SHOT rather than per (shot, CRF): the CRFs of a shot are now chosen
        adaptively, so each one depends on the scores before it and they have
        to run in order. Parallelism comes from the shots, of which there are
        hundreds - far more than the worker count - so nothing is lost, and
        the staged reference shard a shot may need is now built, used and
        dropped inside one task instead of being shared across the pool.
        """
        workers = self._probe_workers(len(shots) or 1)
        self._probe_worker_count = workers
        lp = self._svt_lp(workers)
        scale = self._probe_scale()
        if scale:
            logger.warning(
                "optimizer: probe_scale={!r} makes the probes encode at a "
                "different resolution than the final encode, so the CRF picked "
                "from them does not transfer. Leave it empty unless you are "
                "trading accuracy for speed on purpose.", scale)
        results: ProbeSamples = {}
        done_shots = 0
        self._heaviest_cmd = (0.0, "")
        width = int(self.opt.probe_bracket_width or 0)
        plan = (f"adaptive from {seed_crfs(grid)} down to a {width}-wide bracket"
                if width > 0 else f"the full {len(grid)}-point grid {grid}")
        self._log(f"probing {len(shots)} shots with {workers} workers "
                  f"(svt lp={lp}), {plan}")
        stop, peak = self._start_mem_sampler()
        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(self._probe_shot, i, s0, s1, grid, lp): i
                        for i, (s0, s1) in enumerate(shots)}
                for fut in as_completed(futs):
                    self._check_cancel()
                    results[futs[fut]] = fut.result()
                    done_shots += 1
                    # stage-local progress: the bar matches done/total shots
                    pct = done_shots / max(len(shots), 1) * 100
                    self._report(pct, done_shots, len(shots))
        finally:
            stop.set()
        heavy = f"; heaviest: {self._heaviest_cmd[1]} {self._heaviest_cmd[0]:.0f}MB" \
            if self._heaviest_cmd[1] else ""
        self._mem_log(f"[mem] probing peak children RSS={peak[0]:.0f}MB "
                      f"(python={self._py_rss_mb():.0f}MB){heavy}")
        spent = sum(len(v) for v in results.values())
        if results:
            self._mem_log(
                f"probes: {spent} for {len(results)} shot(s), "
                f"{spent / len(results):.2f} per shot "
                f"(a full grid would have been {len(grid)})")
        return results

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
        in_args, vf = self._probe_input(w0, w1, shard)
        args = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y"] + in_args
        if vf:
            args += ["-vf", ",".join(vf)]
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
                     shard: Optional[Path] = None) -> float:
        if self.metric == "ssimulacra2":
            return self._score_ssimulacra2(w0, w1, dist, idx, crf, shard)
        if self.metric == "xpsnr":
            return self._score_xpsnr(w0, w1, dist, idx, crf, shard)
        return self._score_vmaf(w0, w1, dist, idx, crf, shard)

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

    def _score_xpsnr(self, w0: int, w1: int, dist: Path, idx: int,
                     crf: int, shard: Optional[Path]) -> float:
        """ffmpeg's xpsnr filter: a dB scale, not 0-100. Weighted luma is what
        the ITU work reports, so that is what is returned."""
        if shard is not None:
            ref_args, ref_vf = ["-i", str(shard)], []
        else:
            ref_args, ref_vf = self._probe_input(w0, w1)
        fmt = f"format={self._pix_fmt()}"
        dist_chain = fmt
        ref_chain = ",".join(f for f in (*ref_vf, fmt) if f)
        lavfi = (f"[0:v]{dist_chain}[dist];[1:v]{ref_chain}[ref];"
                 f"[dist][ref]xpsnr=shortest=1")
        args = ([self.ffmpeg, "-hide_banner", "-y", "-loglevel", "info",
                 "-i", str(dist)] + ref_args
                + ["-lavfi", lavfi, "-f", "null", "-"])
        out = self._run(args, timeout=3600)
        m = re.findall(r"XPSNR\s+y:\s*([0-9.]+)", out)
        if not m:
            raise TranscodeError(f"could not parse XPSNR output: {out[-300:]}")
        score = float(m[-1])
        self._log(f"shot {idx:05d} crf {crf} xpsnr={score:.3f}dB")
        return score

    def _score_vmaf(self, w0: int, w1: int, dist: Path, idx: int, crf: int,
                    shard: Optional[Path]) -> float:
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
        n_threads = self._vmaf_threads()
        if n_threads:
            opts.append(f"n_threads={n_threads}")
        # Input 0 is the DISTORTED encode and input 1 the REFERENCE source:
        # ffmpeg's libvmaf takes #0 as main (distorted) and #1 as reference.
        # Passing them the other way round makes libvmaf treat the encode as
        # the reference - the motion feature is then measured on the smoothed
        # encode and VIF sees detail being *added* rather than lost, which
        # inflates and flattens the whole CRF curve (measured +5 VMAF at CRF 20
        # and +18 at CRF 44 on 4K HDR10).
        ref_args, ref_vf = self._probe_input(w0, w1, shard)
        scale = self._vmaf_scale_filter()
        fmt = f"format={self._pix_fmt()}"
        dist_chain = ",".join(f for f in (scale, fmt) if f)
        # the reference goes through the same probe-side filters the distorted
        # copy was encoded with, then both land on the same comparison raster.
        ref_chain = ",".join(f for f in (*ref_vf, scale, fmt) if f)
        lavfi = (f"[0:v]{dist_chain}[dist];[1:v]{ref_chain}[ref];"
                 f"[dist][ref]libvmaf={':'.join(opts)}")
        args = ([self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-i", str(dist)] + ref_args
                + ["-lavfi", lavfi, "-f", "null", "-"])
        try:
            self._run(args, timeout=3600)
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
        workers = self._encode_workers(len(shots))
        threads = self._encode_threads(workers)
        lp = self._encode_lp(workers)
        ivf_paths: Dict[int, Path] = {}
        with self._slot_lock:
            self._slots.clear()
        done_frames = 0
        t0 = time.monotonic()
        self._heaviest_cmd = (0.0, "")
        aff = f" taskset {threads}c" if threads < (os.cpu_count() or 1) else ""
        self._log(f"encoding {len(shots)} shots in parallel"
                  f" (svt lp={lp}, workers={workers}, threads={threads}/instance{aff})")
        stop, peak = self._start_mem_sampler()
        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {
                    ex.submit(self._encode_shot, i, s0, s1,
                              chosen.get(i, self.video.crf), lp, threads,
                              workers):
                    (i, s1 - s0) for i, (s0, s1) in enumerate(shots)}
                for fut in as_completed(futs):
                    self._check_cancel()
                    i, span = futs[fut]
                    ivf = fut.result()
                    ivf_paths[i] = ivf
                    done_frames += span
                    # stage-local progress so the bar matches done/total frames
                    elapsed = max(time.monotonic() - t0, 1e-6)
                    pct = done_frames / max(self.total_frames, 1) * 100
                    self._report(pct, done_frames, self.total_frames,
                                 fps=done_frames / elapsed)
        finally:
            stop.set()
        heavy = f"; heaviest: {self._heaviest_cmd[1]} {self._heaviest_cmd[0]:.0f}MB" \
            if self._heaviest_cmd[1] else ""
        self._mem_log(f"[mem] encoding peak children RSS={peak[0]:.0f}MB "
                      f"(python={self._py_rss_mb():.0f}MB){heavy}")
        return [ivf_paths[i] for i in range(len(shots))]

    def _encode_shot(self, idx: int, s0: int, s1: int, crf: float, lp: int,
                     threads: int, workers: int = 1) -> Path:
        dst = self.probe_dir / f"enc_{idx:05d}.ivf"
        # encode exactly (s1 - s0) frames: `-t` on input-seeked shots is not
        # frame-exact (off by a frame per shot), and 762 shots x 1 frame drift
        # = seconds of A/V desync once the audio is muxed whole. `-frames:v`
        # guarantees the exact frame count so the concat sums to the source.
        args = self._affinity_prefix(self._worker_slot(workers), threads)
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
        args += ["-ss", self._seek(s0), "-i", str(self.source),
                 "-frames:v", str(s1 - s0), "-map", "0:v:0"]
        if vf:
            args += ["-vf", ",".join(vf)]
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

        Both sides are staged as lossless windows so the comparison is between
        two files holding exactly these frames - the same footing the probes
        score on, so a delivered score is directly comparable to the probe's
        prediction for the same window.
        """
        ref = self.probe_dir / f"verify_ref_{idx:05d}.mkv"
        dist = self.probe_dir / f"verify_out_{idx:05d}.mkv"
        try:
            self._extract_window(w0, w1, ref, apply_dv=self._p5)
            self._extract_window(w0, w1, dist, source=self.output)
            return self._score_probe(w0, w1, dist, idx, int(round(crf)), shard=ref)
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
            "".join(f"file '{p}'\n" for p in ivf_paths), encoding="utf-8")
        video_only = self.tempdir / "video_only.mkv"
        args = [self.ffmpeg, "-hide_banner", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file), "-c", "copy", "-fflags", "+genpts",
                "-f", "matroska", str(video_only)]
        self._run(args, timeout=1800)
        # audio + subs from the ORIGINAL source (info.path, not self.source:
        # for Dolby Vision the encode input is a video-only stripped
        # intermediate, so muxing from it would drop audio).
        audio_src = str(self.info.path if self.info.path else self.source)
        audio_subs = self.tempdir / "audio_subs.mkv"
        # NB: no -map_metadata -1 here: it strips per-stream LANGUAGE tags
        # from the subtitle/audio streams (Plex then shows every subtitle as
        # English). The source's global "DV.HDR10.PLUS" title is harmless in
        # this intermediate - mkvmerge does not copy it into the final file.
        base_args = [self.ffmpeg, "-hide_banner", "-y", "-loglevel", "error",
                     "-i", audio_src, "-map", "0:a?", "-map", "0:s?",
                     "-c:a", "copy"]
        try:
            self._run(base_args + self._subtitle_codec_args(audio_src)
                      + [str(audio_subs)], timeout=1800)
        except TranscodeError:
            logger.warning("subtitle remux failed, retrying with a plain copy")
            self._run(base_args + ["-c:s", "copy", str(audio_subs)], timeout=1800)
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
        base = [self.ffmpeg, "-hide_banner", "-y", "-i", str(video_only),
                "-i", audio_subs, "-map", "0:v:0", "-map", "1:a?",
                "-map", "1:s?", "-c", "copy"]
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

    def _mkvmerge_mux(self, mkvmerge: str, video_only: Path,
                      audio_subs: Path) -> bool:
        """Final mux with mkvmerge. Returns True on a valid output file."""
        try:
            if self.output.exists():
                self.output.unlink()
            proc = subprocess.run(
                [mkvmerge, "-o", str(self.output), str(video_only),
                 str(audio_subs)],
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
