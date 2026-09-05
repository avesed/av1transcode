from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from app.analyzer import run_command
from app.config import Settings


def extract_rpu(settings: Settings, source: Path, dest: Path, profile: int) -> bool:
    """Extract the Dolby Vision RPU into a standalone .bin for archival."""
    dovi = settings.tool_path("dovi_tool")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        if source.suffix.lower() == ".mkv":
            # dovi_tool can read RPU directly from Matroska
            cmd = [dovi, "extract-rpu", str(source), "-o", str(dest)]
            rc, out = run_command(cmd, timeout=1800)
        else:
            # Non-mkv: pipe Annex-B HEVC through ffmpeg so dovi_tool sees a stream.
            # NB: dovi_tool's "-o -" does NOT mean stdout - it creates a file named
            # "-" while its actual stdout only carries log lines. So point -o at the
            # real destination file.
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
        logger.info("Extracted RPU from {} -> {} ({} bytes)", source, dest, dest.stat().st_size)
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


