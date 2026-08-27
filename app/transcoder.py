from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from loguru import logger

from app.analyzer import MediaInfo, analyze
from app.config import Settings, VideoParams
from app.decisions import TranscodePlan


class TranscodeError(Exception):
    pass


def _to_svt_flags(params: str) -> str:
    """Normalize "key=value key=value" input to "--key value --key value".
    Also accepts "--key=value" and converts to "--key value".

    av1an forwards --probe-video-params verbatim to the encoder binary, and
    SvtAv1EncApp (SVT-AV1 v4) only accepts the "--key value" form.
    """
    parts = []
    for tok in params.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            # If key already starts with '-', don't add another '--'
            if k.startswith("-"):
                parts.append(f"{k} {v}")
            else:
                parts.append(f"--{k} {v}")
        else:
            parts.append(tok)
    return " ".join(parts)


def av1_video_params(video: VideoParams) -> List[str]:
    """svt-av1 flags passed via av1an --video-params."""
    parts = [
        f"--preset {video.preset}",
        f"--crf {video.crf}",
        f"--keyint {video.keyint}",
        f"--tune {video.tune}",
    ]
    if video.film_grain:
        parts.append(f"--film-grain {video.film_grain}")
        if not video.film_grain_denoise:
            parts.append("--film-grain-denoise 0")
    if video.additional_video_params:
        parts.append(_to_svt_flags(video.additional_video_params))
    return parts


def build_av1an_cmd(
    settings: Settings,
    video: VideoParams,
    src_input: Path,
    output: Path,
    tempdir: Path,
    scenes: Optional[str] = None,
    workers: Optional[int] = None,
) -> List[str]:
    av1an = settings.tool_path("av1an")
    cmd = [
        av1an, "-i", str(src_input), "-o", str(output),
        "--encoder", "svt-av1",
        "--video-params", " ".join(av1_video_params(video)),
        "--pix-format", video.pixel_format,
        "--min-scene-len", str(video.min_scene_len),
        "-y",
    ]
    if video.target_quality:
        # Per-scene target quality: av1an picks a per-chunk quantizer to hit
        # the target metric score (--crf in video-params is the search start).
        cmd += ["--target-quality", video.target_quality]
        cmd += ["--vmaf-path", "/usr/share/model/vmaf_v0.6.1.json"]
        if video.probes:
            cmd += ["--probes", str(video.probes)]
        if video.probing_rate:
            cmd += ["--probing-rate", str(video.probing_rate)]
        if video.probe_res:
            cmd += ["--probe-res", video.probe_res]
        if video.probing_vmaf_features:
            cmd += ["--probing-vmaf-features", *video.probing_vmaf_features.split()]
        if video.vmaf_threads:
            cmd += ["--vmaf-threads", str(video.vmaf_threads)]
        if video.probe_video_params:
            # SvtAv1EncApp only accepts "--key value" style args; av1an passes
            # probe params through verbatim, so normalize "key=value" input.
            cmd += ["--probe-video-params", _to_svt_flags(video.probe_video_params)]
    if workers:
        cmd += ["--workers", str(workers)]
    if settings.transcode.video.extra_split_sec:
        cmd += ["--extra-split-sec", str(settings.transcode.video.extra_split_sec)]
    if scenes:
        cmd += ["--scenes", scenes]
    if video.passes > 1:
        cmd += ["--passes", str(video.passes)]
    cmd += ["--temp", str(tempdir)]
    return cmd


def run_av1an(
    cmd: List[str],
    log_path: Optional[Path] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
    cancel_flag: Optional[Callable[[], bool]] = None,
    stage_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """Run av1an under a pty, streaming stdout to log, reporting progress %.

    av1an only renders its progress bar when stdout is a terminal, so it is
    started on a pty (master/slave pair): the slave is the child's stdout, the
    parent drains the stream. Progress lines like "NN% frames/total" are parsed
    and forwarded to progress_cb. Cancel is honored by polling both the pty
    and the cancel flag, killing the whole process group on cancel.

    For "--target-quality" jobs the VMAF probe phase runs silently on the pty:
    progress is then derived from the per-chunk probe files that av1an writes
    into its temp dir (split/v_AAAAA_<q>.ivf): each new chunk index means one
    more chunk has been probed. The chunk count is known from av1an's scenecut
    line ("found N scene(s) [with extra_splits (..): M scene(s)]" where M is
    the final chunk count, or N when no extra splits), so the probe progress is
    reported as (chunks probed / total chunks). Once the real encode progress
    bar appears the probe phase ends and normal frame-based progress resumes.
    """
    import fcntl  # noqa: PLC0415
    import pty  # noqa: PLC0415

    env = dict(os.environ)
    env.setdefault("AV1AN_LOG_LEVEL", "info")
    # enable full backtrace for panics
    env["RUST_BACKTRACE"] = "full"
    logger.debug("av1an env: {}", env)
    log_handle = open(log_path, "w", buffering=1) if log_path else None
    proc: Optional[subprocess.Popen] = None
    master_fd: Optional[int] = None

    def _read_available() -> str:
        """Non-blocking read of everything currently buffered on the pty."""
        out = []
        while True:
            try:
                chunk = os.read(master_fd, 65536)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            if not chunk:
                break
            out.append(chunk)
        return b"".join(out).decode("utf-8", errors="replace")

    try:
        master_fd, slave_fd = pty.openpty()
        os.set_blocking(master_fd, False)
        proc = subprocess.Popen(
            cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            start_new_session=True,  # own process group so we can kill children
        )
        os.close(slave_fd)

        buf = ""
        last_report: float = -1.0
        last_done: Optional[int] = None
        last_stage: Optional[str] = None
        last_output = time.monotonic()
        # target-quality probe tracking: av1an silently runs VMAF probes per
        # chunk, writing split/v_AAAAA_<q>.ivf files. The last probed chunk
        # index is our probe progress denominator-free numerator.
        probing: bool = "--target-quality" in cmd
        probe_total_chunks: Optional[int] = None
        probe_done_chunks: int = 0
        probe_last_pct: float = -1.0
        temp_dir: Optional[Path] = None
        for i, arg in enumerate(cmd):
            if arg == "--temp" and i + 1 < len(cmd):
                temp_dir = Path(cmd[i + 1])
                break
        while True:
            if cancel_flag and cancel_flag():
                logger.info("Cancel requested - terminating av1an pid {}", proc.pid)
                _terminate_proc(proc)
                raise TranscodeError("Job cancelled by user")
            # drain whatever the pty has buffered
            raw = _read_available()
            if raw:
                last_output = time.monotonic()
                buf += raw
            # split on newlines; keep a trailing partial line as buf
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.rstrip("\r")
                # progress bar redraws combine many \r-separated states into one
                # line: keep only the latest redraw for the log
                if "\r" in line:
                    line = line.rsplit("\r", 1)[-1]
                clean = _strip_ansi(line)
                if log_handle and clean:
                    log_handle.write(clean + "\n")
                last_output = time.monotonic()
                if progress_cb:
                    stats = parse_progress_stats(clean)
                    if stats and stats["pct"] > last_report:
                        progress_cb(stats["pct"], stats)
                        last_report = stats["pct"]
                logger.debug("av1an: {}", clean)
                # surface av1an milestones at INFO so stage transitions are visible
                if any(k in clean for k in ("Scene detection", "scenecut", "Encoding", "Queue", "Params", "Worker")):
                    logger.info("av1an: {}", clean)
                # report coarse stage to the UI: scenedetect -> probing -> encoding
                if stage_cb:
                    new_stage = None
                    if "Scene detection" in clean:
                        new_stage = "scenedetect"
                    elif "scenecut" in clean:
                        # scene detection finished: VMAF probing begins now
                        new_stage = "probing"
                    elif "Chunking" in clean or "Queue" in clean:
                        # chunk queue built: probing done, real encoding starts
                        new_stage = "encoding"
                        probing = False
                    if new_stage and new_stage != last_stage:
                        stage_cb(new_stage)
                        last_stage = new_stage
                        # the progress bar restarts from 0 at each stage:
                        # allow the fresh lower % to be reported
                        last_report = -1.0
                        last_done = None
                # av1an's chunk count is the denominator for probe progress.
                # Two sources, whichever appears first:
                #   scenecut: found 4 scene(s) [with extra_splits (300 frames): 5 scene(s)]
                #   Queue 5 Workers ...   (workers line has total chunk count)
                if probing and probe_total_chunks is None:
                    m = re.search(
                        r"scenecut: found \d+ scene\(s\)"
                        r"(?: \[with extra_splits \(\d+ frames\): (\d+) scene\(s\)\])?",
                        clean,
                    )
                    if m:
                        probe_total_chunks = int(m.group(1) or re.search(r"found (\d+)", clean).group(1))
                    else:
                        m = re.search(r"Queue (\d+) Worker", clean)
                        if m:
                            probe_total_chunks = int(m.group(1))
            # av1an's progress bar redraws in-place with \r, never \n, and can
            # accumulate in buf for the whole encode: keep only the latest
            # redraw, then scan it so the UI percentage stays current.
            if buf:
                if "\r" in buf:
                    buf = buf.rsplit("\r", 1)[-1]
                if progress_cb:
                    stats = parse_progress_stats(_strip_ansi(buf))
                    if stats:
                        # probing normally ends at the Queue line; the scene
                        # detection bar also carries fps>0, so only treat an
                        # fps>0 bar as encode progress once probing has started
                        # (stage=probing) and the pty suddenly emits frames
                        if probing and last_stage == "probing" and stats.get("fps", 0) > 0:
                            probing = False
                            last_report = -1.0
                            last_done = None
                            if stage_cb and last_stage != "encoding":
                                stage_cb("encoding")
                                last_stage = "encoding"
                        # bar restarted (new chunk): old values are stale
                        if last_done is not None and stats["done"] < last_done:
                            last_report = -1.0
                        if stats["pct"] > last_report:
                            progress_cb(stats["pct"], stats)
                            last_report = stats["pct"]
                        last_done = stats["done"]
            # VMAF probe phase: av1an emits nothing on the pty while probing, but it
            # writes a probe-encode file split/v_AAAAA_<q>.ivf as each chunk's
            # probing starts (one file per CRF sample). The set of chunk
            # indices seen so far is a reliable, monotonic probe progress.
            if probing and progress_cb and temp_dir is not None:
                probed = 0
                split_dir = temp_dir / "split"
                if split_dir.is_dir():
                    try:
                        chunks = set()
                        for p in split_dir.iterdir():
                            m = re.match(r"v_(\d{5})_", p.name)
                            if m:
                                chunks.add(int(m.group(1)))
                        probed = len(chunks)
                    except OSError:
                        pass
                if probed > probe_done_chunks:
                    probe_done_chunks = probed
                    last_output = time.monotonic()
                    # total chunk count may still be unknown while probing is
                    # in flight: report probed chunks as done/total whenever
                    # possible; progress_cb sees done>0 either way
                    if probe_total_chunks:
                        pct = min(probe_done_chunks / probe_total_chunks * 100, 99.0)
                    else:
                        pct = 0.0
                    if pct > probe_last_pct or (not probe_total_chunks and probe_done_chunks):
                        probe_last_pct = pct
                        progress_cb(
                            pct,
                            {
                                "pct": pct,
                                "done": probe_done_chunks,
                                "total": probe_total_chunks or 0,
                                "fps": 0.0,
                            },
                        )
                    if stage_cb and last_stage != "probing":
                        stage_cb("probing")
                        last_stage = "probing"
            rc = proc.poll()
            if rc is not None:
                if rc != 0:
                    # try to extract more detail from the log file
                    tail = []
                    lines = []
                    if log_path and log_path.exists():
                        try:
                            with open(log_path) as f:
                                lines = f.read().strip().splitlines()
                                tail = lines[-20:]  # last 20 lines
                        except Exception:
                            pass
                        # if log ends with "Scene detection" with no further output,
                        # likely scene detection failed
                        if len(lines) >= 1 and "Scene detection" in lines[-1]:
                            raise TranscodeError(
                                f"av1an scene detection failed (no scenes found?); exit code {rc}; "
                                f"log: {log_path}"
                            )
                    error_msg = f"av1an exited with code {rc}; log: {log_path}"
                    if tail:
                        error_msg += f"\nLast log lines:\n" + "\n".join(tail[-10:])
                    raise TranscodeError(error_msg)
                break
            # watchdog: no output for 90s => log a warning (scene detection
            # on long files is single-threaded and can look stalled)
            if time.monotonic() - last_output > 90:
                pstate = "alive" if proc.poll() is None else "dead"
                logger.warning(
                    "av1an produced no output for {}s (pstate={}); cmd: {}",
                    90, pstate, " ".join(cmd),
                )
                last_output = time.monotonic()
            time.sleep(1.5)
    finally:
        if log_handle:
            log_handle.close()
        if master_fd is not None:
            try:
                os.close(master_fd)
            except OSError:
                pass
        if proc is not None and proc.poll() is None:
            logger.info("Cleaning up leftover av1an process (pid {})", proc.pid)
            _terminate_proc(proc)


def _terminate_proc(proc: "subprocess.Popen") -> None:
    """Kill av1an and its whole process group (av1an spawns ffmpeg/svt children)."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("av1an (pid {}) did not die after SIGKILL", proc.pid)


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (colours, cursor movement) from a line."""
    return re.sub(r"\x1b(?:\[[0-9;?]*[A-Za-z]|\][^\x07\x1b]*(?:\x07|\x1b\\)|\(B|\)[0-9A-B])", "", text)


def parse_progress(line: str) -> Optional[float]:
    """Extract the most recent percentage from av1an output.

    Matches both plain log lines ("40% ...") and av1an's terminal progress bar
    ("▐██▌ 64% 700/1000 (80 fps, eta)") once ANSI codes are stripped. Returns
    the LAST percentage found in the text (in-place bar redraws accumulate).
    """
    hits = re.findall(r"([\d.]+)\s*%", line)
    for raw in reversed(hits):
        val = float(raw)
        if 0 <= val <= 100:
            return val
    return None


def parse_progress_stats(line: str) -> Optional[dict]:
    """Parse av1an's progress bar line into structured stats.

    Bar format (ANSI stripped): "00:00:07 [1/5 Chunks] ▐▌  60% 724/1200
    (97.15 fps, eta 5s, ...)". Returns {pct, done, total, fps} from the LAST
    redraw found (in-place redraws accumulate in one text buffer), or None.
    """
    hits = re.findall(
        r"([\d.]+)\s*%\s+(\d+)/(\d+)\s+\(([\d.]+)\s*fps",
        line,
    )
    for raw_pct, raw_done, raw_total, raw_fps in reversed(hits):
        pct = float(raw_pct)
        if 0 <= pct <= 100:
            return {
                "pct": pct,
                "done": int(raw_done),
                "total": int(raw_total),
                "fps": float(raw_fps),
            }
    return None


def run_full_transcode(
    settings: Settings,
    info: MediaInfo,
    plan: TranscodePlan,
    source: Path,
    output: Path,
    log_path: Optional[Path] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
    cancel_flag: Optional[Callable[[], bool]] = None,
    stage_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """Execute the full encode of a plan: DV preprocessing, av1an, mux, metadata."""
    from app import dovi  # local import avoids cycles

    work_dir = settings.dirs.work
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_files: List[Path] = []
    encode_input = source
    video = plan.params or settings.transcode.video

    # --- Dolby Vision pre-processing ---
    if info.dovi.present:
        if plan.rpu_path is not None:
            ok = dovi.extract_rpu(settings, source, plan.rpu_path, info.dovi.profile)
            if not ok:
                logger.warning("RPU extraction failed for {}, continuing without it", source)

        dovi_cfg = settings.transcode.dovi
        if dovi_cfg.enabled:
            if plan.p5 and video.engine == "optimizer":
                # The shot-based engine applies the RPU per shot, into tmpfs.
                # Converting the whole file up front instead would write a
                # lossless intermediate of ~100GB for a 45-minute 4K episode
                # and then re-read it 11 times per shot.
                logger.info("P5: RPU applied per shot by the optimizer engine "
                            "(no whole-file intermediate)")
            elif plan.p5:
                method = plan.p5_method
                intermediate = work_dir / f"{source.stem}.dv_p5.mkv"
                ok_file = dovi.convert_p5_to_hdr10(settings, str(source), str(intermediate), method)
                if not ok_file:
                    raise TranscodeError(f"P5->HDR10 conversion failed ({method})")
                tmp_files.append(intermediate)
                encode_input = intermediate
            elif info.dovi.profile in (7, 8):
                stripped = work_dir / f"{source.stem}.dv_bl.mkv"
                outfile = dovi.strip_dv_from_hevc(settings, str(source), str(stripped))
                if outfile is None:
                    raise TranscodeError("failed to strip DV layers for encode")
                tmp_files.append(stripped)
                encode_input = stripped
            else:
                logger.warning("Unsupported DV profile {} - encoding BL directly", info.dovi.profile)

    # --- av1an encode ---
    tempdir = work_dir / f"av1an_{time.time_ns()}"
    tempdir.mkdir(parents=True, exist_ok=True)
    tmp_files.append(tempdir)

    def on_progress(pct: float, stats: Optional[dict] = None) -> None:
        if progress_cb:
            progress_cb(pct, stats)

    try:
        if video.engine == "optimizer":
            from app.optimizer import run_shot_transcode

            logger.info("Starting optimizer engine (shot-based): {} -> {}", encode_input, output)
            t0 = time.monotonic()
            run_shot_transcode(
                settings, info, plan, encode_input, output, tempdir,
                log_path=log_path, progress_cb=on_progress, cancel_flag=cancel_flag,
                stage_cb=stage_cb,
            )
            logger.info("optimizer finished in {:.1f}s", time.monotonic() - t0)
        else:
            cmd = build_av1an_cmd(
                settings, video, encode_input, output, tempdir,
                workers=settings.workers.av1an_workers,
            )
            logger.info("Starting av1an: {} -> {}", encode_input, output)
            logger.debug("av1an full command: {}", " ".join(cmd))
            t0 = time.monotonic()
            run_av1an(cmd, log_path=log_path, progress_cb=on_progress, cancel_flag=cancel_flag,
                      stage_cb=stage_cb)
            logger.info("av1an finished in {:.1f}s", time.monotonic() - t0)

        try:
            _verify_output(settings, info, output)
        except TranscodeError:
            # A file that failed verification must not stay next to the source:
            # it is the thing that later gets mistaken for a finished archive.
            # The encode is reproducible and the job log keeps the reason.
            if output.exists():
                try:
                    output.unlink()
                except OSError:
                    pass
            raise

        # --- optional HDR metadata tag on the mkv ---
        if settings.transcode.hdr.preserve and (info.is_hdr or info.is_hlg or info.dovi.present):
            _colorpropedit_hdr(settings, output, plan)
    finally:
        # NB: a finally, not the tail of the happy path. Every intermediate
        # here is source-sized or larger (the per-shot encodes under tempdir,
        # a stripped DV base layer, a lossless P5 convert), and a job that
        # raises is exactly the job that gets retried twice more - so leaking
        # on failure meant three copies per file that never encodes.
        _cleanup_temp(tmp_files, keep=settings.transcode.keep_temp)


def _cleanup_temp(tmp_files: List[Path], keep: bool) -> None:
    """Remove the working files of a finished (or failed) job.

    tmp_files mixes directories (av1an/optimizer temp trees) with plain files
    (the stripped DV base layer, the lossless P5 convert). shutil.rmtree only
    handles the former - on a file it raises NotADirectoryError, which
    ignore_errors=True then swallowed - so every Dolby Vision job used to leave
    a source-sized intermediate behind in dirs.work.
    """
    if keep:
        return
    for path in tmp_files:
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        except OSError as e:  # noqa: PERF203
            logger.warning("could not remove temp {}: {}", path, e)


def _verify_output(settings: Settings, info: MediaInfo, output: Path) -> None:
    """Check the finished file against the source before the job is called done.

    Existence and a non-zero size were the only checks, so a truncated encode -
    or one that silently lost its audio and subtitle streams - was reported as
    a success, and with transcode.delete_source that is the point at which the
    source gets deleted. Everything read here comes from the container header,
    so this costs one ffprobe rather than a pass over the file.
    """
    if not output.exists() or output.stat().st_size == 0:
        raise TranscodeError("the encoder produced no output file")
    out = analyze(settings, str(output))
    if out is None:
        raise TranscodeError(f"could not probe the encoded file {output.name}")
    if not out.video_codec:
        raise TranscodeError(f"{output.name} has no video stream")

    problems: List[str] = []
    # Duration catches the dropped tail: a shot list that did not reach the end
    # of the source, or a chunk missing from the concat. The tolerance absorbs
    # container timestamp rounding (measured ~1ms on a 46-minute encode), not a
    # missing scene.
    if info.duration > 0 and out.duration > 0:
        tolerance = max(1.0, info.duration * 0.005)
        drift = abs(out.duration - info.duration)
        if drift > tolerance:
            problems.append(
                f"duration {out.duration:.2f}s against the source's "
                f"{info.duration:.2f}s (off by {drift:.2f}s, tolerated "
                f"{tolerance:.2f}s)")
    elif info.duration > 0:
        problems.append("the output reports no duration")
    # Stream counts catch the mux going wrong - an encode fed a video-only
    # intermediate, or a subtitle stream the container silently refused.
    if out.audio_count != info.audio_count:
        problems.append(f"{out.audio_count} audio stream(s) against the "
                        f"source's {info.audio_count}")
    if out.subtitle_count != info.subtitle_count:
        problems.append(f"{out.subtitle_count} subtitle stream(s) against the "
                        f"source's {info.subtitle_count}")
    if problems:
        raise TranscodeError(
            f"output verification failed for {output.name}: " + "; ".join(problems))
    logger.info(
        "Verified {}: {:.2f}s, {} audio, {} subtitle stream(s)",
        output.name, out.duration, out.audio_count, out.subtitle_count)


def _md_number(token: str, scale: float, real_limit: float) -> float:
    """One mastering-display number, normalised to real units.

    Three spellings reach us:
      "11408507/16777216"  ffprobe side data, an arbitrary rational that IS
                           already the real value (this is what an MKV HDR
                           source reports at stream level)
      "34000"              x265/mkvmerge integer units of 1/scale
      "0.68"               already real
    `real_limit` separates the last two: a chromaticity is always <= 1, a real
    max luminance never reaches 10000 * ... , while the integer-unit spelling
    of either is orders of magnitude larger.
    """
    if "/" in token:
        num, den = token.split("/", 1)
        den_f = float(den)
        if den_f == 0:
            raise ValueError(f"zero denominator in mastering display: {token}")
        return float(num) / den_f
    value = float(token)
    return value / scale if value > real_limit else value


def _md_fmt(value: float) -> str:
    """Format a mastering-display number the way mkvpropedit will accept it.

    mkvpropedit only parses plain decimal notation, while Python's str() flips
    to scientific below 1e-4 - a min-luminance of 0.00005 becomes "5e-05" and
    mkvpropedit then rejects the ENTIRE --set command ("The file has not been
    modified"), so the file silently loses every colour tag, not just the
    mastering display. Six decimals covers real luminance floors (1e-6 nits)
    and rounds float noise out of the chromaticity coordinates.
    """
    text = f"{value:.6f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def _parse_master_display(md: str) -> Optional[dict]:
    """Parse a mastering display string into mkvpropedit coordinate fields.

    Format: G(gx,gy)B(bx,by)R(rx,ry)WP(wx,wy)L(max,min), where each number is
    either an ffprobe rational, integer units of 0.00002 (chromaticity) /
    0.0001 cd/m^2 (luminance), or a plain real. See _md_number.
    """
    m = re.match(
        r"G\(([\d./]+),([\d./]+)\)B\(([\d./]+),([\d./]+)\)R\(([\d./]+),([\d./]+)\)"
        r"WP\(([\d./]+),([\d./]+)\)L\(([\d./]+),([\d./]+)\)",
        md.strip(),
    )
    if not m:
        return None
    gx, gy, bx, by, rx, ry, wx, wy, lmax, lmin = m.groups()
    try:
        coords = [_md_number(t, 50000, 1.0)
                  for t in (gx, gy, bx, by, rx, ry, wx, wy)]
        # a real display peak is 100-10000 nits; the same figure in 1/10000
        # units is >= 1e6. A real display floor is always well under 1 nit.
        max_l = _md_number(lmax, 10000, 10000.0)
        # a real display floor is 0.0001-0.05 nits; the integer-unit spelling of
        # the same figure is >= 1, so anything above half a nit is unit-scaled.
        min_l = _md_number(lmin, 10000, 0.5)
    except (ValueError, ZeroDivisionError):
        return None
    names = ("chromaticity-coordinates-green-x", "chromaticity-coordinates-green-y",
             "chromaticity-coordinates-blue-x", "chromaticity-coordinates-blue-y",
             "chromaticity-coordinates-red-x", "chromaticity-coordinates-red-y",
             "white-coordinates-x", "white-coordinates-y")
    fields: Dict[str, float] = dict(zip(names, coords))
    fields["max-luminance"] = max_l
    fields["min-luminance"] = min_l
    return fields


def _colorpropedit_hdr(settings: Settings, output: Path, plan: TranscodePlan) -> None:
    """Write HDR10 colour + display metadata onto the MKV container.

    AV1 sequence headers default to BT.709, so for a BT.2020/PQ encode we must
    tag primaries/transfer/matrix at container level or players will show wrong
    colors. Uses the ISO colour numbers mkvmerge/mkvpropedit expect. Note that
    mkvtoolnix >= 74 dropped the legacy 'master-display' property in favour of
    individual chromaticity/luminance fields.
    """
    md = plan.master_display
    cll = plan.max_cll
    cmd = [settings.tool_path("mkvpropedit"), str(output), "--edit", "track:v1"]

    # ISO 23091-2 values for HDR10 (BT.2020, PQ, non-constant luminance)
    if "smpte2084" in plan.color_trc or "arib-std-b67" in plan.color_trc:
        cmd += [
            "--set", "colour-primaries=9",      # BT.2020
            "--set", "colour-transfer-characteristics="
            + ("18" if "arib-std-b67" in plan.color_trc else "16"),  # HLG=18, PQ=16
            "--set", "colour-matrix-coefficients=9",  # BT.2020 non-constant
            "--set", "colour-range=1",          # limited range
        ]
    if md:
        fields = _parse_master_display(md)
        if not fields:
            # Don't ship an HDR10 file with no mastering display at all just
            # because the source spelled it in a form we could not read: the
            # configured default is a far better answer than nothing.
            fallback = settings.transcode.hdr.default_master_display
            logger.warning("Could not parse master-display string {!r}; "
                           "falling back to the configured default", md)
            fields = _parse_master_display(fallback) if fallback else None
        if fields:
            for name, value in fields.items():
                cmd += ["--set", f"{name}={_md_fmt(value)}"]
    if cll:
        try:
            max_cll, max_fall = (float(v) for v in str(cll).split(","))
            cmd += ["--set", f"max-content-light={int(max_cll)}"]
            cmd += ["--set", f"max-frame-light={int(max_fall)}"]
        except ValueError:
            logger.warning("Could not parse MaxCLL string: {}", cll)
    try:
        # NB: mkvpropedit is all-or-nothing - one rejected --set value aborts
        # the whole edit and the file keeps NO colour tags at all, so the
        # result has to be checked rather than discarded.
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True,
                              timeout=120)
        if proc.returncode != 0:
            logger.warning(
                "mkvpropedit did not tag {} (rc={}): {} - the output keeps no "
                "HDR colour metadata", output.name, proc.returncode,
                ((proc.stdout or "") + (proc.stderr or "")).strip()[-300:])
    except Exception as e:  # noqa: BLE001
        logger.warning("mkvpropedit failed (ignored): {}", e)