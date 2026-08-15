from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

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


def convert_p5_to_hdr10(settings: Settings, source: str, out: str,
                        method: str = "libplacebo") -> Optional[str]:
    """Convert a Dolby Vision Profile 5 BL (ICtCp) into an HDR10 equivalent before AV1 encode.

    Two strategies:
      libplacebo : GPU (Vulkan) filter that applies the DV RPU in-place and
                   remaps to BT.2020/PQ. Most accurate; falls back to CPU
                   (llvmpipe) rendering when no GPU is available.
      zscale     : software approximation, no Vulkan required but approximate.
    Returns path to an intermediate lossless Matroska (FFV1 10-bit), or the
    source path when no conversion is needed.
    """
    if method == "libplacebo":
        # Run the DV application on the llvmpipe software Vulkan device (works
        # without a GPU). `-init_hw_device vulkan:llvmpipe` + `-filter_hw_device`
        # makes libplacebo accept the software renderer, and `format=` inside the
        # filter keeps the hwframe input/output sw-formats consistent so
        # hwdownload knows how to map them back to system memory.
        # The upload format must stay 10-bit: nv12 is 8 bits per component, so
        # feeding it truncated the P5 base layer BEFORE the RPU mapping was
        # applied and baked banding into a PQ signal. Measured against the
        # 10-bit path that cost 25.4dB PSNR, ran 27% slower (3.56 vs 4.51fps at
        # 4K) and inflated the lossless intermediate by 60% with dither noise.
        filt = (
            "format=yuv420p10le,hwupload,"
            "libplacebo=apply_dolbyvision=1:format=yuv420p10le"
            ":colorspace=bt2020nc:color_primaries=bt2020:color_trc=smpte2084,"
            "hwdownload,format=yuv420p10le"
        )
        # Device selection is deliberately NOT pinned to llvmpipe: ffmpeg's
        # "vulkan=vk:<sel>" picks the device whose name matches <sel>, so
        # hardcoding the software renderer meant a host GPU was never used even
        # when it was passed into the container. Bare "vulkan=vk" takes device
        # 0 - the GPU when present, lavapipe when not - and measured no slower
        # than the pinned form on a GPU-less host.
        device = (settings.transcode.dovi.vulkan_device or "").strip()
        pre = [
            "-init_hw_device", f"vulkan=vk:{device}" if device else "vulkan=vk",
            "-filter_hw_device", "vk",
        ]
    else:
        # zscale: software primaries conversion
        filt = (
            "zscale=matrixin=bt2020:primariesin=bt2020:transferin=smpte2084"
            ":matrix=bt2020nc:primaries=bt2020:transfer=smpte2084"
            ":rangein=limited:range=limited,format=yuv420p10le"
        )
        pre = []
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


def strip_dv_from_hevc(settings: Settings, source: str, out: str,
                       keep_el: bool = False) -> Optional[str]:
    """For P7/P8: copy the BL (or BL+RPU) to a plain HEVC file without DV layers.

    - P7: uses dovi_split bitstream filter (base/bl_rpu).
    - P8: uses dovi_rpu strip filter to drop RPU while keeping HDR10 video.
    Returns the interim HEVC path or None.
    """
    ffmpeg = settings.tool_path("ffmpeg")
    bsf = "dovi_split=bl_rpu" if keep_el else "dovi_split=bl"
    if Path(out).exists():
        Path(out).unlink()
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
        "-map", "0:v:0", "-c:v", "copy", "-bsf:v", bsf, "-f", "matroska", str(out),
    ]
    rc, log = run_command(cmd, timeout=3600)
    if rc != 0:
        if Path(out).exists():
            Path(out).unlink()
        logger.error("strip_dv failed: {}", log[-500:])
        return None
    return out