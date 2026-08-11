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
import subprocess
import threading
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
    if pooled:
        first = next(iter(pooled.values()))
        mean = first.get("mean")
        if mean is not None:
            return float(mean)
    agg = data.get("aggregateVMAF")
    if agg is not None:
        return float(agg)
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
        self.tempdir = Path(tempdir)
        self.log_path = log_path
        self.progress_cb = progress_cb
        self.cancel_flag = cancel_flag
        self.stage_cb = stage_cb

        self.video: VideoParams = plan.params or settings.transcode.video
        self.opt = settings.transcode.optimizer
        self.ffmpeg = settings.tool_path("ffmpeg")

        self.fps = float(info.fps or 25.0)
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
        # per-shot y4m cache: path + refcount + lock so each shot is decoded once
        self._cache: Dict[int, _ShotCacheEntry] = {}
        self._cache_lock = threading.Lock()

    # ---------- callbacks / logging ----------
    def _log(self, line: str) -> None:
        logger.debug("optimizer: {}", line)
        if self._log_handle is None:
            return
        with self._log_lock:
            self._log_handle.write(line.rstrip() + "\n")

    def _stage(self, stage: str, pct: float) -> None:
        if self.stage_cb:
            self.stage_cb(stage)
        self._report(pct, 0, 0)

    def _report(self, pct: float, done: int = 0, total: int = 0) -> None:
        if self.progress_cb:
            self.progress_cb(min(max(pct, 0.0), 100.0),
                             {"pct": min(max(pct, 0.0), 100.0), "done": done,
                              "total": total, "fps": 0.0})

    def _check_cancel(self) -> None:
        if self.cancel_flag and self.cancel_flag():
            self._kill_all()
            raise TranscodeError("Job cancelled by user")

    # ---------- subprocess ----------
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
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._terminate(proc)
            raise TranscodeError(f"command timed out after {timeout}s: {args[0]}")
        finally:
            with self._proc_lock:
                self._procs.discard(proc)
        if out.strip():
            self._log(out[-4000:])
        if proc.returncode != 0:
            raise TranscodeError(
                f"{args[0]} failed (rc={proc.returncode}):\n{out[-2000:]}"
            )
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
    def detect_shots(self) -> List[Shot]:
        try:
            from scenedetect import ContentDetector, open_video, SceneManager
        except ImportError as e:  # pragma: no cover
            raise TranscodeError(
                "engine=optimizer requires PySceneDetect. "
                "pip install scenedetect (and opencv-python-headless)."
            ) from e
        video = open_video(str(self.source))
        sm = SceneManager()
        sm.add_detector(ContentDetector(
            threshold=self.opt.scenedetect_threshold,
            min_scene_len=self.opt.min_scene_len,
        ))
        sm.detect_scenes(video, show_progress=False)
        scenes = sm.get_scene_list()
        shots = [(int(a.frame_num), int(b.frame_num)) for a, b in scenes]
        if not shots:
            shots = [(0, self.total_frames)]
        shots = merge_to_max(shots, max(1, self.opt.max_shots))
        logger.info("optimizer: {} shot(s) detected", len(shots))
        return shots

    # ---------- probe configuration ----------
    def _probe_grid(self) -> List[int]:
        grid = list(self.opt.probe_crfs or [])
        if not grid:
            raise TranscodeError("transcode.optimizer.probe_crfs is empty")
        probes = self.video.probes
        if probes and probes > 0:
            grid = grid[:probes]
        return sorted(grid)

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

    def _vmaf_threads(self) -> int:
        return self.video.vmaf_threads or self.opt.vmaf_threads

    def _vmaf_features(self) -> str:
        return (self.video.probing_vmaf_features or "").strip()

    def _model_cfg(self) -> str:
        """libvmaf model config string for ffmpeg's libvmaf filter.

        Accepts either a plain file path (wrapped as path=...), or an explicit
        libvmaf model config such as "version=ssimulacra2" or "path=/x.json".
        """
        raw = (self.opt.ssimulacra2_model if self.metric == "ssimulacra2"
               else self.opt.vmaf_model)
        raw = (raw or "").strip()
        if not raw:
            raw = "version=vmaf_v0.6.1"
        if "=" in raw:
            return raw
        return f"path={raw}"

    # ---------- phase 2: parallel probing ----------
    def _shot_cache(self, idx: int) -> "_ShotCacheEntry":
        with self._cache_lock:
            entry = self._cache.get(idx)
            if entry is None:
                entry = _ShotCacheEntry(self.probe_dir / f"shot_{idx:05d}.y4m")
                self._cache[idx] = entry
            entry.refs += 1
            return entry

    def _release_cache(self, idx: int) -> None:
        with self._cache_lock:
            entry = self._cache.get(idx)
            if entry is None:
                return
            entry.refs -= 1
            if entry.refs <= 0:
                if not self.opt.keep_probes:
                    entry.unlink()
                self._cache.pop(idx, None)

    def _extract_shot(self, idx: int, s0: int, s1: int, path: Path) -> None:
        start = s0 / self.fps
        dur = (s1 - s0) / self.fps
        args = [self.ffmpeg, "-hide_banner", "-y", "-ss", f"{start:.6f}",
                "-i", str(self.source), "-t", f"{dur:.6f}"]
        vf = []
        scale = self._probe_scale()
        if scale:
            vf.append(f"scale={scale}")
        rate = self._probing_rate()
        if rate > 1:
            # sample every nth frame; no -r re-timing so the sample stays
            # down-sampled (both ref and probe then use the same sampled frames)
            vf.append(f"select='not(mod(n\\,{rate}))'")
        if vf:
            args += ["-vf", ",".join(vf)]
        args += ["-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe", str(path)]
        self._run(args, timeout=1800)

    def probe_all(self, shots: List[Shot], grid: List[int]) -> ProbeSamples:
        workers = self.opt.probe_workers or os.cpu_count() or 1
        workers = max(1, min(workers, len(shots) * len(grid) or 1))
        tasks = [(i, s0, s1, crf) for i, (s0, s1) in enumerate(shots) for crf in grid]
        total = len(tasks)
        results: ProbeSamples = {}
        done = 0
        self._log(f"probing {len(shots)} shots x {len(grid)} crfs with {workers} workers")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(self._probe_one, i, s0, s1, crf)
                    for i, s0, s1, crf in tasks]
            for fut in as_completed(futs):
                self._check_cancel()
                i, crf, score = fut.result()
                results.setdefault(i, {})[crf] = score
                done += 1
                # probing spans 2%..47% of overall progress
                pct = 2 + done / max(total, 1) * 45
                self._report(pct, done, total)
        return results

    def _probe_one(self, idx: int, s0: int, s1: int, crf: int) -> Tuple[int, int, float]:
        entry = self._shot_cache(idx)
        try:
            self._check_cancel()
            with entry.lock:
                if not entry.path.exists():
                    self._extract_shot(idx, s0, s1, entry.path)
            if not entry.path.exists():
                raise TranscodeError(f"shot {idx} extraction produced no frames")
            ivf = self.probe_dir / f"probe_{idx:05d}_{crf}.ivf"
            args = [self.ffmpeg, "-hide_banner", "-y", "-i", str(entry.path),
                    "-c:v", "libsvtav1", "-preset", str(self._probe_preset()),
                    "-crf", str(crf), "-pix_fmt", "yuv420p", "-f", "ivf", str(ivf)]
            self._run(args, timeout=3600)
            score = self._score_probe(entry.path, ivf, idx, crf)
            if not self.opt.keep_probes:
                try:
                    ivf.unlink()
                except OSError:
                    pass
            return idx, crf, score
        finally:
            self._release_cache(idx)

    def _score_probe(self, ref: Path, dist: Path, idx: int, crf: int) -> float:
        out_json = self.probe_dir / f"score_{idx:05d}_{crf}.json"
        opts = [f"model={self._model_cfg()}", "log_fmt=json",
                f"log_path={out_json}"]
        feats = self._vmaf_features()
        if feats:
            opts.append(f"feature={feats}")
        n_threads = self._vmaf_threads()
        if n_threads:
            opts.append(f"n_threads={n_threads}")
        args = [self.ffmpeg, "-hide_banner", "-y", "-i", str(ref), "-i", str(dist),
                "-lavfi", f"[0:v][1:v]libvmaf={':'.join(opts)}",
                "-f", "null", "-"]
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
        if not self.opt.keep_probes:
            try:
                out_json.unlink()
            except OSError:
                pass
        return score

    # ---------- phase 3: per-shot CRF selection ----------
    def pick_all_crfs(self, samples: ProbeSamples, grid: List[int]) -> Dict[int, float]:
        chosen: Dict[int, float] = {}
        for idx, by_crf in samples.items():
            pts = [(crf, score) for crf, score in by_crf.items()]
            crf = pick_crf(pts, self.target)
            crf = max(min(grid), min(crf, max(grid)))
            chosen[idx] = crf
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
        lo, hi = min(grid), max(grid)
        out = {}
        for i, crf in zip(ordered_idx, smoothed):
            out[i] = max(lo, min(crf, hi))
        return out

    # ---------- phase 4: parallel final encode ----------
    def encode_all(self, shots: List[Shot], chosen: Dict[int, float]) -> List[Path]:
        workers = self.opt.probe_workers or os.cpu_count() or 1
        workers = max(1, min(workers, len(shots) or 1))
        ivf_paths: Dict[int, Path] = {}
        done_frames = 0
        self._log(f"encoding {len(shots)} shots in parallel")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self._encode_shot, i, s0, s1, chosen.get(i, self.video.crf)):
                    (i, s1 - s0) for i, (s0, s1) in enumerate(shots)}
            for fut in as_completed(futs):
                self._check_cancel()
                i, span = futs[fut]
                ivf = fut.result()
                ivf_paths[i] = ivf
                done_frames += span
                # encoding spans 47%..100% of overall progress
                pct = 47 + done_frames / max(self.total_frames, 1) * 53
                self._report(pct, done_frames, self.total_frames)
        return [ivf_paths[i] for i in range(len(shots))]

    def _encode_shot(self, idx: int, s0: int, s1: int, crf: float) -> Path:
        dst = self.probe_dir / f"enc_{idx:05d}.ivf"
        start = s0 / self.fps
        dur = (s1 - s0) / self.fps
        args = [self.ffmpeg, "-hide_banner", "-y", "-ss", f"{start:.6f}",
                "-i", str(self.source), "-t", f"{dur:.6f}", "-map", "0:v:0",
                "-c:v", "libsvtav1", "-preset", str(self.video.preset),
                "-crf", self._fmt_crf(crf)]
        if self.video.keyint:
            args += ["-g", str(self.video.keyint), "-keyint_min", str(self.video.keyint)]
        svt = _svt_params_dict(self.video)
        if svt:
            args += ["-svtav1-params", ":".join(f"{k}={v}" for k, v in svt.items())]
        args += ["-pix_fmt", self.video.pixel_format, "-f", "ivf", str(dst)]
        self._run(args, timeout=7200)
        if not dst.exists() or dst.stat().st_size == 0:
            raise TranscodeError(f"shot {idx} produced no output")
        return dst

    def _fmt_crf(self, crf: float) -> str:
        if self.opt.fractional_crf:
            return f"{crf:g}"
        return str(max(0, min(63, round(crf))))

    # ---------- phase 5: concat + mux ----------
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
        # re-mux: video from the concat, audio + subs copied from the source
        args = [self.ffmpeg, "-hide_banner", "-y", "-i", str(video_only),
                "-i", str(self.source), "-map", "0:v:0", "-map", "1:a?",
                "-map", "1:s?", "-c", "copy", "-map_metadata", "1",
                str(self.output)]
        self._run(args, timeout=1800)
        if not self.output.exists() or self.output.stat().st_size == 0:
            raise TranscodeError("optimizer produced no output file")

    # ---------- orchestration ----------
    def run(self) -> None:
        try:
            self._stage("scenedetect", 0.0)
            shots = self.detect_shots()
            self._log(f"{len(shots)} shot(s) from scene detection")

            self._stage("probing", 2.0)
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

            self._stage("encoding", 47.0)
            ivf_paths = self.encode_all(shots, chosen)
            self._report(99.0, self.total_frames, self.total_frames)

            self.concat_shots(ivf_paths)
            self._report(100.0, self.total_frames, self.total_frames)
        finally:
            self.close()


class _ShotCacheEntry:
    """Ref-counted per-shot extracted frame file (decoded once per shot)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.refs = 0
        self.lock = threading.Lock()

    def unlink(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass
