from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"


class Dirs(BaseModel):
    input: Path = Path("media/input")
    # null => transcode into an `av1/` subdir next to each source file
    # (RPU files go into `<source_dir>/av1/rpu/` unless dirs.rpu is set)
    output: Optional[Path] = None
    rpu: Optional[Path] = None
    work: Path = Path("media/work")
    archive: Optional[Path] = None  # None => do not move source after success
    db: Path = Path("data/av1transcode.db")
    logs: Path = Path("data/logs")
    # User-defined presets (added/edited via web UI). Builtin presets come
    # from transcode.presets in config.yaml; user entries override/extend them.
    presets_file: Path = Path("data/presets.json")
    # User settings persisted from the web UI (workers etc.)
    settings_file: Path = Path("data/settings.json")


class Workers(BaseModel):
    # Number of concurrent transcode jobs (each job may use av1an with its own workers)
    concurrency: int = 1
    # av1an internal chunk workers, 0 = auto (detected by av1an)
    av1an_workers: int = 0
    # Max retries on failure
    max_retries: int = 2


class Watcher(BaseModel):
    enabled: bool = True
    recursive: bool = True
    # Seconds a file must be stable (size unchanged) before it is accepted
    stable_seconds: int = 30
    # Minimum file size in MB to consider (ignore small / sidecar files)
    min_size_mb: int = 20
    extensions: List[str] = Field(default_factory=lambda: ["mkv", "mp4", "mov", "m2ts", "ts", "avi"])


class VideoParams(BaseModel):
    codec: Literal["svt-av1"] = "svt-av1"
    # Constant rate factor / quality. Lower = higher quality.
    crf: int = 28
    # SVT-AV1 preset, 0 (slowest/best) - 13 (fastest). 4-6 is a good range.
    preset: int = 4
    # Film grain synthesis level 0-50, 0 = disabled. 8-10 for grainy live action.
    film_grain: int = 0
    film_grain_denoise: bool = False
    passes: int = 1
    keyint: int = 240
    # extra split in seconds: av1an will subdivide long scenes to keep chunks
    # reasonably sized for parallel workers. 0 = disabled.
    extra_split_sec: int = 60
    min_scene_len: int = 24
    tune: int = 0  # 0=VQ (visual quality), 1=PSNR, 2=SSIM
    pixel_format: Literal["yuv420p10le", "yuv420p8le", "yuv420p12le"] = "yuv420p10le"
    additional_video_params: str = ""
    # Per-scene target quality (av1an). Range like "75-85" (VMAF/SSIMULACRA2)
    # or "1.0-1.5" (butteraugli). Empty = fixed-CRF encode.
    target_quality: str = ""
    # ---- av1an target-quality "probing matrix" tuning (0/"" = av1an default) ----
    # Max number of probes allowed for target quality search
    probes: int = 0
    # Only use every nth frame for the probe VMAF calculation (1 = all frames)
    probing_rate: int = 0
    # Probe resolution, e.g. "960x540" (scaled down for faster probes)
    probe_res: str = ""
    # VMAF features for probing, e.g. "default motionless" or "weighted neg"
    probing_vmaf_features: str = ""
    # Threads used for the probe VMAF calculation
    vmaf_threads: int = 0
    # Encoder params used during probing; "copy" = same as --video-params.
    # Recommend a faster preset here, e.g. "preset=10".
    probe_video_params: str = ""


class DolbyVision(BaseModel):
    enabled: bool = True
    # Where to store extracted RPU files (relative to dirs.rpu)
    save_rpu: bool = True
    # Conversion method for Profile 5 (ICtCp -> HDR10):
    #   libplacebo: accurate, requires Vulkan (recommended)
    #   zscale:     fast software, approximate colors
    p5_method: Literal["libplacebo", "zscale"] = "libplacebo"
    # For Profile 7/8: strip RPU/EL from the stream fed to the encoder.
    strip_rpu: bool = True


class Hdr(BaseModel):
    # Preserve HDR10/HLG mastering display and CLL metadata on output.
    preserve: bool = True
    # Fallback mastering display metadata when source has none.
    default_master_display: str = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
    default_max_cll: str = "1000,400"


class Transcode(BaseModel):
    video: VideoParams = Field(default_factory=VideoParams)
    # Presets named in config. Keys used by CLI/API as `--preset name`.
    presets: Dict[str, VideoParams] = Field(default_factory=dict)
    # Builtin presets snapshot (from config.yaml) - user presets override these.
    # Excluded from model_dump so it is never serialized into config.
    builtin_presets: Dict[str, VideoParams] = Field(default_factory=dict, exclude=True)
    # Which preset is the default (falls back to video defaults if unset)
    default_preset: str = "balanced"
    dovi: DolbyVision = Field(default_factory=DolbyVision)
    hdr: Hdr = Field(default_factory=Hdr)
    # Skip files that are already AV1 at >= this resolution height (0 = never skip)
    skip_existing_av1: bool = True
    min_height_to_transcode: int = 0
    # Keep av1an temp files after success
    keep_temp: bool = False
    # Delete source file after successful transcode (false by default - archive manually)
    delete_source: bool = False


class Web(BaseModel):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    # API key for mutating endpoints (optional). Empty = open.
    api_key: str = ""


class Tools(BaseModel):
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    mediainfo: str = "mediainfo"
    av1an: str = "av1an"
    dovi_tool: str = "dovi_tool"
    mkvmerge: str = "mkvmerge"
    mkvextract: str = "mkvextract"
    mkvpropedit: str = "mkvpropedit"

    def path(self, name: str) -> str:
        return getattr(self, name)


class Logging(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # Console + file. File is relative to dirs.logs
    file: bool = True


class Settings(BaseModel):
    dirs: Dirs = Field(default_factory=Dirs)
    workers: Workers = Field(default_factory=Workers)
    watcher: Watcher = Field(default_factory=Watcher)
    transcode: Transcode = Field(default_factory=Transcode)
    web: Web = Field(default_factory=Web)
    tools: Tools = Field(default_factory=Tools)
    logging: Logging = Field(default_factory=Logging)

    @model_validator(mode="after")
    def resolve_paths(self) -> "Settings":
        """Make relative paths absolute against the config file location (or CWD)."""
        base = ROOT_DIR
        for field_name in ("input", "output", "rpu", "work", "archive", "db", "logs", "presets_file", "settings_file"):
            p = getattr(self.dirs, field_name)
            if p is not None and not p.is_absolute():
                setattr(self.dirs, field_name, (base / p).resolve())
        return self

    def ensure_dirs(self) -> None:
        for name in ("input", "work", "logs"):
            p = getattr(self.dirs, name)
            try:
                p.mkdir(parents=True, exist_ok=True)
            except OSError:
                # container-only path on a dev machine; ignore, worker will
                # surface a clear error if the file actually needs writing.
                pass
        for name in ("output", "rpu"):
            p = getattr(self.dirs, name)
            if p is None:
                continue  # source-relative mode: created per-job by the worker
            try:
                p.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
        if self.dirs.archive:
            try:
                self.dirs.archive.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
        try:
            self.dirs.db.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def tool_path(self, name: str) -> str:
        p = self.tools.path(name)
        found = shutil.which(p)
        if not found:
            raise FileNotFoundError(f"Required tool not found in PATH: {p}")
        return found


def _load_config_dict(path: Path) -> Dict[str, Any]:
    if path.exists():
        return yaml.safe_load(path.read_text()) or {}
    return {}


def _merge_env(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Allow overriding every config value via AV1TC_<SECTION>_<KEY> env vars."""
    for key, val in os.environ.items():
        if not key.startswith("AV1TC_"):
            continue
        parts = key[6:].lower().split("_")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        last = parts[-1]
        if last in node and isinstance(node[last], bool):
            node[last] = val.lower() in ("1", "true", "yes", "on")
        elif last in node and isinstance(node[last], int):
            try:
                node[last] = int(val)
            except ValueError:
                pass
        else:
            node[last] = val
    return cfg


def load_settings(config_path: Optional[Path] = None) -> Settings:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    cfg = _load_config_dict(path)
    cfg = _merge_env(cfg)
    settings = Settings.model_validate(cfg)
    # Fill missing presets from defaults
    default_video = settings.transcode.video
    presets = settings.transcode.presets
    for name in ("balanced", "quality", "compact"):
        if name not in presets:
            presets[name] = default_video.model_copy(deep=True)
    if settings.transcode.default_preset in presets:
        settings.transcode.video = presets[settings.transcode.default_preset].model_copy(deep=True)
    # Snapshot builtin presets, then apply user presets (web UI persisted).
    settings.transcode.builtin_presets = {
        name: v.model_copy(deep=True) for name, v in presets.items()
    }
    user = load_user_presets(settings)
    for name, params in user.items():
        presets[name] = params
    # Apply user workers settings (persisted by the web UI)
    usettings = load_user_settings(settings)
    if "concurrency" in usettings and usettings["concurrency"]:
        try:
            settings.workers.concurrency = max(1, int(usettings["concurrency"]))
        except (TypeError, ValueError):
            pass
    if "av1an_workers" in usettings and usettings["av1an_workers"] is not None:
        try:
            settings.workers.av1an_workers = max(0, int(usettings["av1an_workers"]))
        except (TypeError, ValueError):
            pass
    return settings


def load_user_settings(settings: Settings) -> Dict[str, Any]:
    """Read user-persisted settings (workers etc.) from the settings file."""
    p = settings.dirs.settings_file
    try:
        if not p.exists():
            return {}
        data = json.loads(p.read_text()) or {}
        return data.get("workers") or {}
    except (OSError, ValueError):
        return {}


def save_user_settings(settings: Settings, workers: Dict[str, Any]) -> None:
    """Persist user settings (workers) to disk from the web UI."""
    p = settings.dirs.settings_file
    data = {"workers": workers}
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(p)


def load_user_presets(settings: Settings) -> Dict[str, VideoParams]:
    """Read user-defined presets (persisted by the web UI)."""
    p = settings.dirs.presets_file
    try:
        if not p.exists():
            return {}
        data = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        logger = __import__("loguru").logger
        logger.warning("Could not load user presets from {}: {}", p, e)
        return {}
    out: Dict[str, VideoParams] = {}
    for name, params in (data or {}).items():
        try:
            out[name] = VideoParams.model_validate(params)
        except Exception as e:  # noqa: BLE001
            logger = __import__("loguru").logger
            logger.warning("Ignoring invalid user preset '{}': {}", name, e)
    return out


def save_user_presets(settings: Settings, presets: Dict[str, VideoParams]) -> None:
    """Persist user-defined presets to disk (used by the web UI)."""
    p = settings.dirs.presets_file
    data = {name: v.model_dump(mode="json") for name, v in presets.items()}
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(p)


def dump_config_for_docs(settings: Settings) -> str:
    return json.dumps(json.loads(settings.model_dump_json()), indent=2, default=str)
