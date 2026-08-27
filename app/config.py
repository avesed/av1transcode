from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

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
    # Which engine runs the encode:
    #   av1an     - the current av1an pipeline (scene detection + serial VMAF
    #               probing + parallel chunk encode, all inside av1an).
    #   optimizer - Netflix-style shot-based encoding: shot detection via
    #               PySceneDetect, per-shot quality probing in PARALLEL across
    #               chunks, interpolated fine-grained CRF selection, then a
    #               parallel per-shot encode and concat.
    engine: Literal["av1an", "optimizer"] = "av1an"
    # Quality metric used by the "optimizer" engine's probes.
    #   vmaf        - Netflix VMAF (0-100). Default. The model is picked from
    #                 the source resolution, see OptimizerSettings.
    #   ssimulacra2 - SSIMULACRA2 via VapourSynth + vszip (0-100, ~90+ is
    #                 excellent). The most artifact-sensitive of the three, and
    #                 the slowest: measured 2.51fps at 4K against ~23fps for
    #                 libvmaf, so it scores every ssimulacra2_frame_step'th
    #                 frame.
    #   xpsnr       - ITU-standardised perceptually weighted PSNR, in dB, from
    #                 ffmpeg's xpsnr filter. Designed for UHD/HDR and cheap.
    #                 Roughly 42dB+ reads as visually lossless, but the
    #                 threshold is content-dependent, so target_quality has to
    #                 be recalibrated - it is a dB scale, not 0-100.
    target_metric: Literal["vmaf", "ssimulacra2", "xpsnr"] = "vmaf"
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


class OptimizerSettings(BaseModel):
    """Pipeline tuning for the "optimizer" (shot-based) engine.

    These are job-wide defaults; per-preset VideoParams fields (probing_rate,
    probe_res, vmaf_threads, probing_vmaf_features, probe_video_params)
    override the matching values below when set.
    """
    # CRF grid sampled per shot during probing. Denser = more accurate CRF
    # selection but more probe encodes. Probes run in parallel across shots.
    # Probes encode at source resolution, so each extra grid point costs a full
    # extra pass over every shot; 5 well-spread points interpolate fine.
    probe_crfs: List[int] = Field(default_factory=lambda: [20, 26, 32, 38, 44])
    # Fast SVT-AV1 preset used for the probe encodes. NB: SVT-AV1 clamps to M9
    # for 4K and above in random-access mode, so 10+ behaves as 9 there. The
    # probe preset is always faster than the final one, which makes the probe
    # under-report quality a little - see probe_crf_offset.
    probe_preset: int = 10
    # Probe resolution "WxH". Empty (recommended) = probe at source resolution.
    # Setting this encodes the probes at a DIFFERENT resolution than the final
    # encode, so the CRF that hits the target on the probe does not hit it on
    # the real encode - measured on 4K HDR, a 960x540 probe compressed CRF
    # 20..44 into 5 VMAF points and capped the whole curve below 91.
    probe_scale: str = ""
    # Only probe every nth frame (1 = all frames of each shot). Values > 1 widen
    # the gap between consecutive frames, making inter prediction artificially
    # hard, so the probe under-reports quality: prefer probe_max_frames to bound
    # probing cost.
    probing_rate: int = 1
    # Cap on frames probed per shot. Probes encode at source resolution, so a
    # minutes-long take would cost minutes of 4K encoding per CRF. Shots longer
    # than this are probed over a contiguous window taken from their middle
    # (not subsampled - see probing_rate).
    probe_max_frames: int = 120
    # libvmaf model configs. Accepts "path=/x.json", "version=NAME", or a bare
    # path (wrapped as path=...). Note: stock libvmaf <= 2.3.1 has no
    # ssimulacra2 model; a patched libvmaf or a ssimulacra2.json is required
    # for target_metric=ssimulacra2.
    vmaf_model: str = "/usr/share/model/vmaf_v0.6.1.json"
    # Used instead of vmaf_model when the source is at least vmaf_4k_min_width
    # wide, and then scored at NATIVE resolution rather than downscaled: the
    # 0.6.1 model is trained for 1080p at 3H, the 4k model for 4K at 1.5H.
    # Measured on 4K HDR, scoring downscaled with the 1080p model reads about a
    # point optimistic against the 4k model and the gap widens with CRF (+0.13
    # at CRF 26, +0.80 at 32, +1.63 at 38), which quietly costs sharpness.
    vmaf_model_4k: str = "/usr/share/model/vmaf_4k_v0.6.1.json"
    vmaf_4k_min_width: int = 2560
    # NB there is still no HDR model to select here: upstream vmaf carries nine
    # models as of v3.2.0 and none of them is HDR, and no community model has
    # taken hold either. PQ content is scored on its coded signal, which is
    # self-consistent but not perceptually calibrated for HDR - target_metric
    # xpsnr is the standardised option that was designed with HDR in mind.
    ssimulacra2_model: str = "version=ssimulacra2"
    # SSIMULACRA2 is a per-frame still-image metric, so scoring a subset is
    # statistically sound - unlike subsampling the ENCODE, which distorts the
    # rate-distortion curve. At 4K it runs 2.51fps against ~23fps for libvmaf;
    # every 4th frame keeps the metric affordable (measured ~2.1h per 45-minute
    # episode against 8.3h unsubsampled). 1 = score every frame.
    ssimulacra2_frame_step: int = 4
    # VapourSynth plugins backing target_metric=ssimulacra2.
    vszip_plugin: Path = Path("/usr/local/lib/vapoursynth/libvszip.so")
    bestsource_plugin: Path = Path("/usr/local/lib/vapoursynth/libbestsource.so")
    # Both libvmaf inputs are downscaled to at most this width (aspect
    # preserved, never upscaled) before the comparison. vmaf_v0.6.1 is trained
    # on 1080p at 3H viewing distance; scoring a 4K pair with it is outside the
    # model's domain. 0 = compare at native resolution.
    vmaf_width: int = 1920
    # Threads for the libvmaf calculation. 0 = auto (cores / probe_workers).
    # NB: ffmpeg's own libvmaf default is single-threaded, which now costs more
    # than the probe encode it measures (7.0s vs 2.1s at 8 threads, same score).
    vmaf_threads: int = 0
    # Parallel probe workers across shots. Probes now encode at source
    # resolution, so each instance holds a multi-GB frame pool at 4K just like
    # the final encode (measured 3.5GB at 4K). 0 = auto (total_mem / 8).
    probe_workers: int = 0
    # Parallel FINAL ENCODE instances. Separate from probe_workers because a
    # single SVT-AV1 encode of 4K uses several GB regardless of thread count
    # (measured ~8GB at preset 6 / lp=6); too many parallel instances OOM the
    # machine. 0 = auto (total_mem / 8, so total encode memory stays under
    # ~half of RAM).
    encode_workers: int = 0
    # CPU cores allotted to each final-encode instance (0 = auto:
    # cores / encode_workers). Enforced with taskset affinity per instance.
    # This does NOT reduce per-instance memory (SVT-AV1 spawns 80+ threads at
    # 4K regardless); it only prevents thread oversubscription between
    # parallel instances so they do not thrash each other.
    encode_threads: int = 0
    # PySceneDetect ContentDetector threshold (higher = fewer/split less).
    scenedetect_threshold: float = 27.0
    # Minimum shot length in frames (shorter segments are merged).
    min_scene_len: int = 24
    # ffmpeg scale filter used to make a small detection copy before running
    # PySceneDetect. OpenCV decodes frame-by-frame and 4K HEVC is unusably slow
    # (minutes per scene); a downscaled copy (same fps, frame numbers map 1:1)
    # is 5-10x faster. e.g. "-2:540". Empty = detect on the source directly.
    scenedetect_scale: str = "-2:540"
    # Fold shots shorter than this many frames into a neighbour before probing.
    # OFF by default, because the size win it was added for did not survive
    # measurement. Splitting a CONTINUOUS take into short pieces is expensive
    # (+43.5% at 24-frame pieces, +13.8% at 48, +4.1% at 96, measured at 1080p
    # preset 4 CRF 32) - but this engine only ever splits at real scene cuts,
    # where the keyframe is what the encoder would spend anyway. On real
    # detected boundaries the same clip measured -0.08% at 48, +0.08% at 64 and
    # +0.26% at 96, with VMAF flat at 90.82 +/- 0.02: no size to recover.
    # What it does buy is probe time - merging at 48 took that clip from 16
    # shots to 12, i.e. a quarter fewer probe encodes - at the cost of coarser
    # per-shot CRF adaptation. Set it if probing is your bottleneck; 96 and
    # above measurably costs size. 0 = keep every detected shot.
    min_shot_frames: int = 0
    # Cap on the number of shots; shortest adjacent shots are merged past this.
    max_shots: int = 3000
    # Bound the per-shot CRF jump between neighbouring shots (0 = disable).
    # Each shot independently hits target_quality, which can leave adjacent
    # shots with very different CRFs and a visible quality step; smoothing
    # keeps |CRF[i] - CRF[i+1]| <= max_crf_delta. May lower some shots a
    # little below target to keep the picture continuous.
    max_crf_delta: float = 4.0
    # Hard floor on the CRF any shot may be assigned (0 = the bottom of
    # probe_crfs). When target_quality is out of reach - which is easy to do at
    # 4K against an already-compressed source - every such shot otherwise falls
    # back to the lowest probed CRF, i.e. the most expensive setting available,
    # and the output ends up larger than the source.
    min_crf: int = 0
    # Added to every interpolated CRF before clamping. The probes run at
    # probe_preset while the delivery runs at the (slower, better) preset, so
    # the probe under-reports the quality the final encode will actually
    # deliver; a positive offset trades that bias back for size. 0 = off.
    probe_crf_offset: float = 0.0
    # Pass decimal CRF values to SVT-AV1 (finer than integer CRF granularity).
    # SVT-AV1 must accept fractional --crf for this to work.
    fractional_crf: bool = False
    # Keep per-shot probe files in the temp dir for debugging.
    keep_probes: bool = False


class DolbyVision(BaseModel):
    enabled: bool = True
    # Where to store extracted RPU files (relative to dirs.rpu)
    save_rpu: bool = True
    # Conversion method for Profile 5 (ICtCp -> HDR10). libplacebo is the only
    # option: it is the one filter in ffmpeg that can APPLY a DV RPU. The old
    # "zscale" alternative is accepted and ignored - that build of the filter
    # chain converted bt2020/PQ to bt2020/PQ, i.e. it relabelled the ICtCp
    # signal without converting anything, and zscale is not compiled into the
    # shipped ffmpeg either.
    p5_method: Literal["libplacebo"] = "libplacebo"
    # Where the shot-based engine stores its per-shot converted shards. tmpfs
    # keeps the DV path off disk entirely (a whole-file intermediate is ~100GB
    # at 4K); a shard that would not fit falls back to dirs.work automatically.
    p5_cache_dir: Path = Path("/dev/shm")
    # Vulkan device for the libplacebo P5 conversion, as ffmpeg's
    # -init_hw_device selector (an index, or a substring of the device name).
    # Empty (recommended) = let ffmpeg pick, which takes the GPU when one is
    # passed into the container and falls back to Mesa's llvmpipe software
    # renderer when there is none. Pin it to "llvmpipe" only to force software.
    vulkan_device: str = ""
    # For Profile 7/8: strip RPU/EL from the stream fed to the encoder.
    strip_rpu: bool = True

    @field_validator("p5_method", mode="before")
    @classmethod
    def _retire_zscale(cls, v: object) -> object:
        """Existing configs and saved settings may still say "zscale"; accept
        them rather than refusing to start, but say what is happening."""
        if isinstance(v, str) and v.strip().lower() == "zscale":
            logger = __import__("loguru").logger  # module avoids a hard dep here
            logger.warning(
                "transcode.dovi.p5_method=zscale is no longer supported "
                "(it relabelled the ICtCp signal without converting it, and "
                "zscale is not built into the shipped ffmpeg); using libplacebo")
            return "libplacebo"
        return v


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
    optimizer: OptimizerSettings = Field(default_factory=OptimizerSettings)
    # Skip files that are already AV1 at >= this resolution height (0 = never skip)
    skip_existing_av1: bool = True
    min_height_to_transcode: int = 0
    # Keep av1an temp files after success
    keep_temp: bool = False
    # Delete source file after successful transcode (false by default)
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
    # Delete job_*.log files older than this many days (0 = never clean up).
    retention_days: int = 7


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
        for field_name in ("input", "output", "rpu", "work", "db", "logs", "presets_file", "settings_file"):
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
    """Allow overriding every config value via AV1TC_<SECTION>_<KEY> env vars.

    Keys themselves may contain underscores (e.g. DIRS_SETTINGS_FILE,
    TRANSCODE_VIDEO_PROBE_VIDEO_PARAMS): at each level the longest remaining
    underscore-joined token that exists in the current node wins.
    """
    for key, val in os.environ.items():
        if not key.startswith("AV1TC_"):
            continue
        parts = key[6:].lower().split("_")
        node = cfg
        i = 0
        while i < len(parts):
            match = None
            for j in range(len(parts), i, -1):
                cand = "_".join(parts[i:j])
                if cand in node:
                    match = (cand, j)
                    break
            if match is None:
                break
            cand, j = match
            nxt = node[cand]
            if j == len(parts):
                if isinstance(nxt, bool):
                    node[cand] = val.lower() in ("1", "true", "yes", "on")
                elif isinstance(nxt, int):
                    try:
                        node[cand] = int(val)
                    except ValueError:
                        pass
                elif isinstance(nxt, list):
                    node[cand] = [x.strip() for x in val.split(",")]
                else:
                    node[cand] = val
                break
            if isinstance(nxt, dict):
                node = nxt
                i = j
            else:
                break
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
    user_workers = usettings.get("workers") or {}
    if "concurrency" in user_workers and user_workers["concurrency"]:
        try:
            settings.workers.concurrency = max(1, int(user_workers["concurrency"]))
        except (TypeError, ValueError):
            pass
    if "av1an_workers" in user_workers and user_workers["av1an_workers"] is not None:
        try:
            settings.workers.av1an_workers = max(0, int(user_workers["av1an_workers"]))
        except (TypeError, ValueError):
            pass
    if "delete_source" in usettings and isinstance(usettings["delete_source"], bool):
        settings.transcode.delete_source = usettings["delete_source"]
    user_opt = usettings.get("optimizer") or {}
    if user_opt:
        try:
            settings.transcode.optimizer = OptimizerSettings.model_validate(user_opt)
        except Exception as e:  # noqa: BLE001
            logger = __import__("loguru").logger
            logger.warning("Ignoring invalid optimizer settings: {}", e)
    return settings


def load_user_settings(settings: Settings) -> Dict[str, Any]:
    """Read user-persisted settings (workers, delete_source) from the settings file."""
    p = settings.dirs.settings_file
    try:
        if not p.exists():
            return {}
        data = json.loads(p.read_text()) or {}
        out: Dict[str, Any] = {}
        if data.get("workers"):
            out["workers"] = data["workers"]
        if "delete_source" in data:
            out["delete_source"] = data["delete_source"]
        if data.get("optimizer"):
            out["optimizer"] = data["optimizer"]
        return out
    except (OSError, ValueError):
        return {}


def save_user_settings(settings: Settings, data: Dict[str, Any]) -> None:
    """Persist user settings (workers, delete_source) to disk from the web UI."""
    p = settings.dirs.settings_file
    # merge with existing data so workers / delete_source don't clobber each other
    existing: Dict[str, Any] = {}
    try:
        if p.exists():
            existing = json.loads(p.read_text()) or {}
    except (OSError, ValueError):
        existing = {}
    existing.update(data)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
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
