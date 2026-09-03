from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from loguru import logger

from app.config import Settings


@dataclass
class ColorInfo:
    primaries: Optional[str] = None
    transfer: Optional[str] = None
    space: Optional[str] = None
    range: Optional[str] = None
    pix_fmt: Optional[str] = None
    bit_depth: int = 8
    mastering_display: Optional[str] = None
    max_cll: Optional[str] = None


@dataclass
class DolbyVisionInfo:
    present: bool = False
    profile: int = 0
    rpu_present: bool = False
    el_present: bool = False
    bl_present: bool = False
    compatible_id: int = 0


@dataclass
class MediaInfo:
    path: Path
    container: str = ""
    duration: float = 0.0
    size: int = 0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    bitrate: int = 0
    video_codec: str = ""
    audio_count: int = 0
    subtitle_count: int = 0
    color: ColorInfo = field(default_factory=ColorInfo)
    dovi: DolbyVisionInfo = field(default_factory=DolbyVisionInfo)
    is_hdr: bool = False
    is_hlg: bool = False
    is_av1: bool = False

    @property
    def display(self) -> str:
        parts = [self.video_codec or "?"]
        if self.width and self.height:
            parts.append(f"{self.width}x{self.height}")
        if self.fps:
            parts.append(f"{self.fps:g}fps")
        if self.color.transfer:
            parts.append(f"trc={self.color.transfer}")
        if self.dovi.present:
            parts.append(f"DV-P{self.dovi.profile}")
        elif self.is_hdr:
            parts.append("HDR")
        return " ".join(parts)


def run_command(cmd: List[str], timeout: int = 600) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, text=True, errors="replace",
        )
        return proc.returncode, proc.stdout
    except FileNotFoundError:
        return 127, f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"command timed out after {timeout}s: {cmd[0]}"
    except Exception as e:  # noqa: BLE001
        return 1, str(e)


def _ffprobe(settings: Settings, path: str) -> Optional[dict]:
    cmd = [
        settings.tool_path("ffprobe"), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams",
    ]
    if str(path).endswith((".m2ts", ".ts")):
        cmd.append("-probesize")
        cmd.append("100M")
    cmd.append(path)
    rc, out = run_command(cmd, timeout=1200)
    if rc != 0 or not out:
        logger.warning("ffprobe failed ({}) for {}: {}", rc, path, out[-500:])
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        logger.warning("ffprobe returned non-JSON for {}", path)
        return None


def _side_data_to_dovi(stream: dict) -> DolbyVisionInfo:
    info = DolbyVisionInfo()
    for sd in stream.get("side_data_list", []) or []:
        sd_type = (sd.get("side_data_type") or "").lower()
        if "dolby vision configuration" not in sd_type and "dovi configuration" not in sd_type:
            continue
        info.present = True
        info.profile = int(sd.get("dv_profile", 0))
        info.rpu_present = bool(sd.get("rpu_present_flag", sd.get("rpu_present", 0)))
        info.el_present = bool(sd.get("el_present_flag", sd.get("el_present", 0)))
        info.bl_present = bool(sd.get("bl_present_flag", sd.get("bl_present", 0)))
        info.compatible_id = int(
            sd.get("dv_bl_signal_compatibility_id", sd.get("compatibility_id", 0))
        )
    return info


def _mastering_display(stream: dict) -> Optional[str]:
    for sd in stream.get("side_data_list", []) or []:
        if sd.get("side_data_type") == "Mastering display metadata":
            try:
                r = sd["min_luminance"]
                m = sd["max_luminance"]
                return (
                    f"G({sd['green_x']},{sd['green_y']})"
                    f"B({sd['blue_x']},{sd['blue_y']})"
                    f"R({sd['red_x']},{sd['red_y']})"
                    f"WP({sd['white_point_x']},{sd['white_point_y']})"
                    f"L({m},{r})"
                )
            except KeyError:
                return None
    return None


def _max_cll(stream: dict) -> Optional[str]:
    for sd in stream.get("side_data_list", []) or []:
        if sd.get("side_data_type") == "Content light level":
            return f"{sd.get('max_content', 0)},{sd.get('max_average', 0)}"
    return None


def analyze(settings: Settings, path: str) -> Optional[MediaInfo]:
    """Gather all technical details about a media file via ffprobe."""
    p = Path(path)
    if not p.exists():
        logger.error("File does not exist: {}", path)
        return None

    data = _ffprobe(settings, path)
    if data is None:
        return None

    fmt = data.get("format", {})
    info = MediaInfo(path=p)
    info.container = fmt.get("format_name") or ""
    info.size = p.stat().st_size
    info.duration = float(fmt.get("duration") or 0.0)
    info.bitrate = int(float(fmt.get("bit_rate") or fmt.get("bit_rate", 0) or 0))

    video_picked = False
    for st in data.get("streams", []):
        ct = st.get("codec_type")
        if ct == "video" and not video_picked:
            video_picked = True
            info.width = int(st.get("width") or 0)
            info.height = int(st.get("height") or 0)
            info.video_codec = st.get("codec_name") or ""
            info.is_av1 = info.video_codec == "av1"
            info.color.pix_fmt = st.get("pix_fmt")
            info.color.primaries = st.get("color_primaries")
            info.color.transfer = st.get("color_transfer")
            info.color.space = st.get("color_space")
            info.color.range = st.get("color_range")
            bits = st.get("bits_per_raw_sample")
            if bits:
                info.color.bit_depth = int(bits)
            elif info.color.pix_fmt:
                if "p10" in info.color.pix_fmt:
                    info.color.bit_depth = 10
                elif "p12" in info.color.pix_fmt:
                    info.color.bit_depth = 12
            info.dovi = _side_data_to_dovi(st)
            md = _mastering_display(st)
            cll = _max_cll(st)
            info.color.mastering_display = info.color.mastering_display or md
            info.color.max_cll = info.color.max_cll or cll
            transfer = (info.color.transfer or "").lower()
            if "smpte2084" in transfer or "smpte-st-2084" in transfer or "pq" in transfer:
                info.is_hdr = True
            elif "arib-std-b67" in transfer or "hlg" in transfer:
                info.is_hlg = True
            # avg_frame_rate first (it is the true average on VFR sources), but
            # it is "0/0" on some streams; r_frame_rate then still gives a
            # usable rate. The shot-based engine converts every frame number to
            # a timestamp with this, so a wrong value mis-cuts the whole encode.
            for key in ("avg_frame_rate", "r_frame_rate"):
                try:
                    num, den = (st.get(key) or "0/1").split("/")
                    fps = float(num) / (float(den) or 1.0)
                except (ValueError, ZeroDivisionError, AttributeError):
                    continue
                if fps > 0:
                    info.fps = round(fps, 6)
                    break
            else:
                info.fps = 0.0
        elif ct == "audio":
            info.audio_count += 1
        elif ct == "subtitle":
            info.subtitle_count += 1

    return info


def fingerprint(path: str) -> Optional[tuple[int, float]]:
    """(size, mtime) of `path`, or None when it cannot be stat'ed.

    The watcher decides a file has finished being written by taking this twice
    and comparing. It used to do that inside a single call, with a hard
    time.sleep(2) between the two stats - and since the scan submits files one
    at a time, that cost two seconds PER FILE with the whole scan loop blocked
    behind it: fifty new files meant a hundred seconds during which nothing
    else in the watcher thread ran.

    Nothing needs to be slept through. The scan already repeats every few
    seconds, so the NEXT scan is the second observation, and comparing across
    cycles is a longer settling window than the two seconds it replaces.
    """
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return st.st_size, st.st_mtime


def settled_for(mtime: float, window_sec: float) -> bool:
    """Whether `mtime` is far enough in the past to count as settled."""
    return time.time() - mtime >= window_sec


def safe_stem(path: str) -> str:
    return Path(path).stem
