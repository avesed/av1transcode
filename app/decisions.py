from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from loguru import logger

from app import naming
from app.analyzer import MediaInfo
from app.config import Settings, VideoParams


@dataclass
class TranscodePlan:
    skip: bool = False
    skip_reason: str = ""
    action_name: str = ""
    preset_name: str = "balanced"
    params: Optional[VideoParams] = None
    dv_profile: Optional[int] = None
    dovi_present: bool = False
    p5: bool = False
    p5_method: str = "libplacebo"
    rpu_path: Optional[Path] = None
    color_primaries: str = "bt709"
    color_trc: str = "bt709"
    colorspace: str = "bt709"
    color_range: str = "tv"
    master_display: Optional[str] = None
    max_cll: Optional[str] = None
    output_path: Optional[Path] = None
    notes: List[str] = field(default_factory=list)


# dv_bl_signal_compatibility_id -> what the base layer is
_DV_BL_SIGNAL = {1: "pq", 6: "pq", 4: "hlg", 2: "sdr"}


def dv_base_signal(info: MediaInfo) -> str:
    """What a Dolby Vision stream's base layer carries: "pq", "hlg" or "sdr".

    Outside P5 the base layer IS what gets encoded - the decoder returns it
    and ignores EL and RPU - so it is what the output has to be tagged as.
    Profile 8 comes in three: 8.1 is HDR10 (PQ), 8.4 is HLG (phones,
    broadcast) and 8.2 is SDR BT.709; 7 is HDR10. Everything used to be tagged
    BT.2020+PQ, which makes an HLG or SDR picture play back at the wrong
    brightness and gamut. The DV configuration's compatibility id is the
    authority; the stream's own transfer is the fallback, and a stream that
    says nothing keeps the old PQ answer.
    """
    known = _DV_BL_SIGNAL.get(info.dovi.compatible_id)
    if known:
        return known
    if info.is_hdr:
        return "pq"
    if info.is_hlg:
        return "hlg"
    trc = (info.color.transfer or "").lower()
    if trc and trc not in ("unknown", "unspecified", "reserved"):
        return "sdr"
    return "pq"


def _color_tags_for(settings: Settings, info: MediaInfo, plan: TranscodePlan) -> None:
    """Determine color metadata tags for the encoded video, preserving HDR/DV target."""
    plan.color_primaries = "bt709"
    plan.color_trc = "bt709"
    plan.colorspace = "bt709"
    plan.color_range = info.color.range if info.color.range else "tv"

    # P5 has no backward-compatible base: libplacebo converts it to HDR10
    # (BT.2020 + PQ) before the encode. Any other DV encodes its base layer.
    base = "pq" if plan.p5 else dv_base_signal(info) if plan.dovi_present else None
    if base == "pq" and plan.dovi_present:
        plan.color_primaries = "bt2020"
        plan.color_trc = "smpte2084"
        plan.colorspace = "bt2020nc"
        plan.color_range = "tv"
        if settings.transcode.hdr.preserve:
            plan.master_display = info.color.mastering_display or (
                settings.transcode.hdr.default_master_display if info.is_hdr else None
            )
            plan.max_cll = info.color.max_cll or (
                settings.transcode.hdr.default_max_cll if info.is_hdr else None
            )
        return
    if base == "hlg":
        plan.color_primaries = "bt2020"
        plan.color_trc = "arib-std-b67"
        plan.colorspace = "bt2020nc"
        plan.color_range = "tv"
        return
    if base == "sdr":
        return

    if info.is_hdr:
        plan.color_primaries = "bt2020"
        plan.color_trc = "smpte2084"
        plan.colorspace = "bt2020nc"
        plan.color_range = "tv"
        if settings.transcode.hdr.preserve:
            plan.master_display = info.color.mastering_display or settings.transcode.hdr.default_master_display
            plan.max_cll = info.color.max_cll or settings.transcode.hdr.default_max_cll
    elif info.is_hlg:
        plan.color_primaries = "bt2020"
        plan.color_trc = "arib-std-b67"
        plan.colorspace = "bt2020nc"
        plan.color_range = "tv"


def decide_action(settings: Settings, info: MediaInfo, preset_name: str = "",
                  overrides: Optional[dict] = None) -> TranscodePlan:
    """Decide whether and how to transcode a file, and prepare the plan.

    overrides: optional dict of VideoParams field overrides applied on top of
    the chosen preset (e.g. {"crf": 24, "preset": 3, "target_quality": "75-85"}).
    """
    plan = TranscodePlan()
    if not info.video_codec:
        plan.skip = True
        plan.skip_reason = "no video stream"
        return plan

    # Already AV1?
    if info.is_av1:
        if settings.transcode.skip_existing_av1:
            if info.height >= settings.transcode.min_height_to_transcode:
                plan.skip = True
                plan.skip_reason = f"already AV1 ({info.width}x{info.height})"
                return plan
            plan.notes.append("AV1 source below min height, still re-encoding")

    # Resolve preset
    presets = settings.transcode.presets
    if preset_name == "custom":
        # Custom mode: start from the default video params, apply overrides on top.
        video = settings.transcode.video.model_copy(deep=True)
    else:
        if not preset_name:
            preset_name = settings.transcode.default_preset
        if preset_name not in presets:
            logger.warning("Unknown preset '{}', falling back to '{}'", preset_name,
                           settings.transcode.default_preset)
            preset_name = settings.transcode.default_preset
        video = presets[preset_name]
    if overrides:
        # Validate keys against VideoParams fields, ignore unknowns
        valid = set(VideoParams.model_fields)
        safe = {k: v for k, v in overrides.items() if k in valid and v not in (None, "")}
        if safe:
            video = video.model_copy(update=safe)
            logger.info("Preset '{}' overridden with {}", preset_name, safe)
    plan.preset_name = preset_name
    plan.params = video
    plan.action_name = f"transcode:{preset_name}"

    # Engine notes / validation
    if video.engine == "optimizer":
        if not (video.target_quality or "").strip():
            plan.skip = True
            plan.skip_reason = "engine=optimizer requires target_quality (e.g. 75 or 75-85)"
            return plan
        plan.notes.append(
            f"shot-based engine (optimizer) with target_metric={video.target_metric}, "
            f"target_quality={video.target_quality}"
        )

    # DV detection/handling
    rpu_dir: Optional[Path] = None
    if info.dovi.present:
        plan.dovi_present = True
        plan.dv_profile = info.dovi.profile
        if settings.transcode.dovi.enabled:
            rpu_note = ""
            if settings.transcode.dovi.save_rpu:
                rpu_dir = _output_dir(settings, info, "rpu")   # named with the output below
                rpu_note = " + RPU saved separately"
            target = {"pq": "HDR10", "hlg": "HLG", "sdr": "SDR"}[
                "pq" if info.dovi.profile == 5 else dv_base_signal(info)]
            plan.notes.append(f"Dolby Vision P{info.dovi.profile}: -> {target}{rpu_note}")
            if info.dovi.profile == 5:
                plan.p5 = True
                plan.p5_method = settings.transcode.dovi.p5_method
                plan.notes.append("P5: convert ICtCp BL to HDR10 (BT.2020+PQ)")
            elif info.dovi.profile in (7, 8):
                plan.notes.append(
                    f"P{info.dovi.profile}: encode {('base layer' if info.dovi.profile==7 else 'BL')} "
                    "after stripping RPU/EL"
                )
        else:
            plan.notes.append("Dolby Vision present but conversion disabled")

    _color_tags_for(settings, info, plan)

    stem, says_av1 = naming.av1_stem(info.path.stem, naming.dynamic_range(
        plan.color_trc, bool(plan.master_display or plan.max_cll)))
    out_dir = _output_dir(settings, info, "video")
    # No codec token to turn into AV1: the old marker says it instead (and
    # keeps the output off the source's own name when dirs.output is its dir).
    plan.output_path = out_dir / (f"{stem}.mkv" if says_av1 else f"{stem}.av1.mkv")
    if rpu_dir is not None:
        # Same name as the output, so the two pair up. Not .bin: Sonarr,
        # Radarr and Plex list .bin as a video extension (VCD images) and
        # imported an `S01E01.rpu.bin` as the episode.
        plan.rpu_path = rpu_dir / f"{stem}.rpu"
    return plan


def _output_dir(settings: Settings, info: MediaInfo, kind: str) -> Path:
    """Resolve the output directory for a job.

    kind = "video" | "rpu". By default (dirs.output/dirs.rpu unset) outputs
    go into `<source_dir>/av1/` (RPU into `<source_dir>/av1/rpu/`). If the
    fixed dirs are configured they win.
    """
    if kind == "rpu":
        if settings.dirs.rpu is not None:
            return settings.dirs.rpu
        base = settings.dirs.output if settings.dirs.output is not None else (info.path.parent / "av1")
        return base / "rpu"
    if settings.dirs.output is not None:
        return settings.dirs.output
    return info.path.parent / "av1"


def build_preset(crf: int, preset: int, film_grain: int = 0, passes: int = 1) -> VideoParams:
    return VideoParams(crf=crf, preset=preset, film_grain=film_grain, passes=passes)