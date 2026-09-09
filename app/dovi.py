from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from app.analyzer import run_command
from app.config import Settings


# Floor for the plausibility check in extract_rpu, in bytes of RPU per second
# of video. Real Profile 7 remuxes measure ~4800; a truncated extraction that
# still exited 0 measured 2.
_RPU_MIN_BYTES_PER_SECOND = 50


def _duration_seconds(settings: Settings, source: Path) -> float:
    """Runtime of `source` in seconds, or 0.0 when it cannot be read."""
    try:
        rc, out = run_command(
            [settings.tool_path("ffprobe"), "-v", "error", "-show_entries",
             "format=duration", "-of", "default=nw=1:nk=1", str(source)],
            timeout=120)
        return float(out.strip().splitlines()[-1]) if rc == 0 and out.strip() else 0.0
    except Exception:  # noqa: BLE001 - a missing duration must not fail the job
        return 0.0


def extract_rpu(settings: Settings, source: Path, dest: Path, profile: int) -> bool:
    """Extract the Dolby Vision RPU into a standalone .bin for archival."""
    dovi = settings.tool_path("dovi_tool")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        # Always through ffmpeg, never dovi_tool's own Matroska reader. Handing
        # it an .mkv directly looks like the obvious thing and silently
        # extracts almost nothing on some remuxes: on a 46-minute Blu-ray
        # remux written by DVDFab it stopped after about 29 frames, wrote
        # 5819 bytes, exited 0 and printed no warning, so `rc == 0 and size >
        # 0` called it a success. The same file through this pipe gives
        # 13,389,697 bytes in 54s, and a 120-second cut of it - remuxed by
        # ffmpeg, which normalises whatever the parser trips on - extracts
        # correctly either way, which is what hid this for a whole season.
        #
        # NB dovi_tool's "-o -" does NOT mean stdout: it creates a file named
        # "-" while its actual stdout only carries log lines. So point -o at
        # the real destination file.
        ffmpeg = settings.tool_path("ffmpeg")
        proc_in = subprocess.Popen(
            [ffmpeg, "-loglevel", "error", "-i", str(source),
             "-c:v", "copy", "-bsf:v", "hevc_mp4toannexb", "-f", "hevc", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        try:
            proc = subprocess.run(
                [dovi, "extract-rpu", "-", "-o", str(dest)],
                stdin=proc_in.stdout,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=1800,
            )
            rc, out = proc.returncode, (proc.stderr or b"").decode(errors="replace")
        finally:
            # Drop the parent's copy of the pipe so ffmpeg gets EPIPE when
            # dovi_tool exits, then make sure it is gone: without this an
            # early dovi_tool exit leaves ffmpeg blocked on a full pipe,
            # decoding a whole 4K movie into nothing.
            if proc_in.stdout:
                proc_in.stdout.close()
            try:
                proc_in.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc_in.kill()
                proc_in.wait(timeout=10)
        if rc != 0 or not dest.exists() or dest.stat().st_size == 0:
            logger.error("RPU extraction failed for {}: {}", source, out[-500:])
            if dest.exists():
                dest.unlink()
            return False
        size = dest.stat().st_size
        # An RPU carries per-frame metadata, so its size tracks the runtime.
        # Measured on P7 FEL remuxes it runs about 4.8KB per second of video;
        # the silent truncation above came to 2 bytes per second. Anything
        # under this floor did not extract, whatever the exit code said - and
        # saying so is the whole point, because the failure it exists for
        # produced a valid, parseable, useless file.
        secs = _duration_seconds(settings, source)
        if secs and size < _RPU_MIN_BYTES_PER_SECOND * secs:
            logger.error(
                "RPU extraction produced {} bytes for {:.0f}s of {} - about {:.1f} "
                "bytes/second against the {} floor. The file is a valid RPU but "
                "covers only the opening frames; treating it as a failure.",
                size, secs, source.name, size / secs, _RPU_MIN_BYTES_PER_SECOND)
            dest.unlink(missing_ok=True)
            return False
        logger.info("Extracted RPU from {} -> {} ({} bytes)", source, dest, size)
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("RPU extraction error for {}: {}", source, e)
        return False


def dv_apply_chain(settings: Settings) -> Tuple[List[str], str]:
    """(ffmpeg pre-input args, filter chain) that applies a Dolby Vision RPU.

    libplacebo is the only thing in ffmpeg that can APPLY an RPU - dovi_rpu can
    only strip or compress the metadata, and the hevc decoder has no switch for
    it - so a Profile 5 base layer (ICtCp) has no other route to HDR10.

    `-init_hw_device vulkan=vk` + `-filter_hw_device` gives libplacebo a device:
    the host GPU when one is passed into the container, Mesa's lavapipe software
    renderer when not. `format=` on both sides of the hwupload/hwdownload pair
    keeps the hwframe sw-formats consistent, and it must stay 10-bit - nv12 is
    8 bits per component and truncating there quantises the base layer BEFORE
    the mapping is applied, baking banding into a PQ signal.
    """
    device = (settings.transcode.dovi.vulkan_device or "").strip()
    pre = [
        "-init_hw_device", f"vulkan=vk:{device}" if device else "vulkan=vk",
        "-filter_hw_device", "vk",
    ]
    chain = (
        "format=yuv420p10le,hwupload,"
        "libplacebo=apply_dolbyvision=1:format=yuv420p10le"
        ":colorspace=bt2020nc:color_primaries=bt2020:color_trc=smpte2084,"
        "hwdownload,format=yuv420p10le"
    )
    return pre, chain


def convert_p5_to_hdr10(settings: Settings, source: str, out: str,
                        method: str = "libplacebo") -> Optional[str]:
    """Convert a whole Dolby Vision Profile 5 BL (ICtCp) into an HDR10 file.

    Writes a lossless FFV1 10-bit Matroska, which at 4K runs to ~100GB for a
    45-minute episode. The shot-based engine avoids it entirely by converting
    one shot at a time (see optimizer); this whole-file path is what the av1an
    engine needs, since av1an owns its own chunking.
    """
    pre, filt = dv_apply_chain(settings)
    cmd = [
        settings.tool_path("ffmpeg"), "-y", "-loglevel", "error",
        *pre,
        "-i", source,
        "-vf", filt,
        "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p10le",
        "-f", "matroska",
        "-an", "-sn",
        str(out),
    ]
    rc, log = run_command(cmd, timeout=3600 * 6)
    if rc != 0 or not Path(out).exists():
        if rc != 0:
            logger.error("P5->HDR10 conversion failed: {}", log[-800:])
        else:
            logger.error("P5->HDR10 conversion produced no output")
        return None
    return str(out)


