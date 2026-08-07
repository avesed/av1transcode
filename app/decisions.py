from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from loguru import logger

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


def _color_tags_for(settings: Settings, info: MediaInfo, plan: TranscodePlan) -> None:
    """Determine color metadata tags for the encoded video, preserving HDR/DV target."""
    plan.color_primaries = "bt709"
    plan.color_trc = "bt709"
    plan.colorspace = "bt709"
    plan.color_range = info.color.range if info.color.range else "tv"

    if plan.p5 or plan.dovi_present:
        # DV always converts/encodes to HDR10 (BT.2020 + PQ)
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

    # DV detection/handling
    if info.dovi.present:
        plan.dovi_present = True
        plan.dv_profile = info.dovi.profile
        if settings.transcode.dovi.enabled:
            rpu_note = ""
            if settings.transcode.dovi.save_rpu:
                rpu_dir = _output_dir(settings, info, "rpu")
                plan.rpu_path = rpu_dir / f"{info.path.stem}.rpu.bin"
                rpu_note = " + RPU saved separately"
            plan.notes.append(f"Dolby Vision P{info.dovi.profile}: -> HDR10{rpu_note}")
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

    stem = info.path.stem
    out_dir = _output_dir(settings, info, "video")
    plan.output_path = out_dir / f"{stem}.av1.mkv"
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