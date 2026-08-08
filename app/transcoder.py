from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from loguru import logger

from app.analyzer import MediaInfo
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
    """Run av1an, streaming stdout to log, reporting progress %, honoring cancel.

    av1an's stdout is drained by a background thread so a stalled process
    (no output) can still be cancelled. The main loop polls both the process
    and the cancel flag, and kills the whole process group on cancel.
    """
    import fcntl  # noqa: PLC0415

    env = dict(os.environ)
    env.setdefault("AV1AN_LOG_LEVEL", "info")
    # enable full backtrace for panics
    env["RUST_BACKTRACE"] = "full"
    logger.debug("av1an env: {}", env)
    log_handle = open(log_path, "w", buffering=1) if log_path else None
    proc: Optional[subprocess.Popen] = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,  # own process group so we can kill children
        )
        assert proc.stdout is not None
        # make stdout reads non-blocking so a stuck av1an can't wedge us
        fd = proc.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        last_output = time.monotonic()
        last_stage = ""
        while True:
            if cancel_flag and cancel_flag():
                logger.info("Cancel requested - terminating av1an pid {}", proc.pid)
                _terminate_proc(proc)
                raise TranscodeError("Job cancelled by user")
            # drain available stdout
            while True:
                try:
                    raw = proc.stdout.readline()
                except Exception:  # noqa: BLE001
                    raw = ""
                if raw == "":
                    break
                line = raw.rstrip("\n")
                last_output = time.monotonic()
                if log_handle:
                    log_handle.write(line + "\n")
                if progress_cb:
                    pct = parse_progress(line)
                    if pct is not None:
                        progress_cb(pct)
                logger.debug("av1an: {}", line)
                # surface av1an milestones at INFO so stage transitions are visible
                if any(k in line for k in ("Scene detection", "scenecut", "Encoding", "Queue", "Params", "Worker")):
                    logger.info("av1an: {}", line)
                # report coarse stage to the UI: scenedetect -> encoding
                if stage_cb:
                    if "Scene detection" in line:
                        stage_cb("scenedetect")
                    elif "scenecut" in line or "Chunking" in line:
                        stage_cb("encoding")
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


def parse_progress(line: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
    if m:
        val = float(m.group(1))
        if 0 <= val <= 100:
            return val
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

    # --- Dolby Vision pre-processing ---
    if info.dovi.present:
        if plan.rpu_path is not None:
            ok = dovi.extract_rpu(settings, source, plan.rpu_path, info.dovi.profile)
            if not ok:
                logger.warning("RPU extraction failed for {}, continuing without it", source)

        dovi_cfg = settings.transcode.dovi
        if dovi_cfg.enabled:
            if plan.p5:
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

    def on_progress(pct: float) -> None:
        if progress_cb:
            progress_cb(pct)

    video = plan.params or settings.transcode.video
    cmd = build_av1an_cmd(
        settings, video, encode_input, output, tempdir, workers=settings.workers.av1an_workers,
    )
    logger.info("Starting av1an: {} -> {}", encode_input, output)
    logger.debug("av1an full command: {}", " ".join(cmd))
    t0 = time.monotonic()
    run_av1an(cmd, log_path=log_path, progress_cb=on_progress, cancel_flag=cancel_flag,
              stage_cb=stage_cb)
    logger.info("av1an finished in {:.1f}s", time.monotonic() - t0)

    if not output.exists() or output.stat().st_size == 0:
        raise TranscodeError("av1an produced no output file")

    # --- optional HDR metadata tag on the mkv ---
    if settings.transcode.hdr.preserve and (info.is_hdr or info.is_hlg or info.dovi.present):
        _colorpropedit_hdr(settings, output, plan)

    # --- cleanup ---
    if not settings.transcode.keep_temp:
        for t in tmp_files:
            shutil.rmtree(t, ignore_errors=True)


def _parse_master_display(md: str) -> Optional[dict]:
    """Parse mkvmerge-style mastering display string into coordinate fields.

    Format: G(gx,gy)B(bx,by)R(rx,ry)WP(wx,wy)L(max,min)
    Coordinates are in units of 0.00002; luminance in units of 0.0001 cd/m^2.
    """
    m = re.match(
        r"G\(([\d.]+),([\d.]+)\)B\(([\d.]+),([\d.]+)\)R\(([\d.]+),([\d.]+)\)"
        r"WP\(([\d.]+),([\d.]+)\)L\(([\d.]+),([\d.]+)\)",
        md,
    )
    if not m:
        return None
    gx, gy, bx, by, rx, ry, wx, wy, lmax, lmin = (float(v) for v in m.groups())
    return {
        "chromaticity-coordinates-green-x": gx / 50000,
        "chromaticity-coordinates-green-y": gy / 50000,
        "chromaticity-coordinates-blue-x": bx / 50000,
        "chromaticity-coordinates-blue-y": by / 50000,
        "chromaticity-coordinates-red-x": rx / 50000,
        "chromaticity-coordinates-red-y": ry / 50000,
        "white-coordinates-x": wx / 50000,
        "white-coordinates-y": wy / 50000,
        "max-luminance": lmax / 10000,
        "min-luminance": lmin / 10000,
    }


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
        if fields:
            for name, value in fields.items():
                cmd += ["--set", f"{name}={value}"]
        else:
            logger.warning("Could not parse master-display string: {}", md)
    if cll:
        try:
            max_cll, max_fall = (float(v) for v in str(cll).split(","))
            cmd += ["--set", f"max-content-light={int(max_cll)}"]
            cmd += ["--set", f"max-frame-light={int(max_fall)}"]
        except ValueError:
            logger.warning("Could not parse MaxCLL string: {}", cll)
    try:
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=120)
    except Exception as e:  # noqa: BLE001
        logger.warning("mkvpropedit failed (ignored): {}", e)