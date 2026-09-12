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
    # SVT-AV1 --luminance-qp-bias: lowers a frame's QP according to its
    # average luma, i.e. spends more bits on dark frames. 0 = off, 1-100.
    #
    # Measured on two 90s 4K cuts against the same configuration without it:
    # output +9.8% (dark passage) and +8.9% (bright), delivered VMAF median
    # +0.37 and +0.06, and shots that could not reach the target at any
    # probed CRF fell from 6/23 to 2/23. So it does reach the shots it is
    # aimed at, it is NOT confined to dark frames (the bright cut paid too),
    # and the quality it buys is small against what it costs - the CRF
    # search does not convert the extra bits back into a higher CRF.
    # Whether that trade is worth taking is a judgement about the library,
    # which is why it is a per-preset switch and not a global default.
    luminance_qp_bias: int = 0
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
    # for 4K and above in random-access mode, so 10+ behaves as 9 there, five
    # presets away from the delivery default of 4.
    #
    # The direction of the resulting bias is NOT the obvious one and is not
    # even fixed. Measured on ten 4K windows at CRF 24, encoding the same
    # window at preset 4 instead of 9 scored 0.80 VMAF LOWER (sd 0.76): the
    # slower preset spends its extra work on rate, not on the metric. Yet
    # end-to-end (probe prediction against delivered) the sign follows the
    # content - clip_A -0.18, a DV P7 remux -0.29, an 8-bit 1080p WEB-DL
    # +1.52. So this is a bias to measure per job, not a constant to assume;
    # see probe_crf_offset, and note the spread it contributes (0.76) is the
    # SMALLER half of the prediction error - probe_max_frames owns the rest.
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
    # Adaptive probing: refine until the two probed CRFs the target falls
    # between are at most this far apart, then interpolate. A uniform sweep
    # spends the same probes on every shot no matter where its target lands;
    # bisection spends them on the interval that actually decides the CRF, and
    # stops as soon as the answer cannot change - a shot whose target is out of
    # reach ends after the 3 seeds instead of sweeping to the end.
    # Simulated over 400 shots against the default grid, concave curves, and
    # the exact root of each curve as ground truth:
    #   full 5-point sweep   5.00 probes/shot   error 0.053 avg / 0.150 max CRF
    #   width 6 (default)    4.00  (-20%)       error 0.053 / 0.148
    #   width 3              5.00  (+0%)        error 0.013 / 0.041
    #   width 1              5.00  (+0%)        error 0.014 / 0.041
    # So 6 is the sweep's own accuracy for a fifth fewer probes (20-25%
    # depending on where the target lands), and 3 is four times more accurate
    # for what the sweep already cost. Below 3 buys nothing: integer CRF is
    # the floor. 0 = probe the whole probe_crfs grid, i.e. the old behaviour.
    # NB this interacts with the spacing of probe_crfs, and at the defaults
    # the two cancel out: seeds 6 apart give a bracket 12 wide, one bisection
    # halves it to 6, and 6 <= width stops. So the adaptive search buys about
    # one probe per shot over the plain grid (measured: 3.3 probes/shot against
    # a grid of 5). Narrow this, or widen probe_crfs, to get more from it.
    probe_bracket_width: int = 6
    # Cap on frames probed per shot. Probes encode at source resolution, so a
    # minutes-long take would cost minutes of 4K encoding per CRF. Shots longer
    # than this are probed over a contiguous window taken from their middle
    # (not subsampled - see probing_rate).
    # Load-bearing, not a conservative default. Measured at 64: the probe
    # phase does get 1.55x faster and the job 1.30x, but a shorter window
    # over-estimates quality systematically - chosen CRFs came out +2.2 higher
    # on average (worst +12.2, some pinned to the ceiling) and the delivered-
    # vs-predicted gap went from -0.38 to -2.28, i.e. the probes became six
    # times less predictive. Change it only while watching that gap.
    #
    # Raising it is the main lever left on prediction error. Decomposing the
    # per-shot residual on ten 4K windows: the probe preset contributes
    # -0.80 VMAF of BIAS with sd 0.76, while truncating a 300-frame shot to
    # its middle 120 contributes sd 1.22 - nearly pure noise, because whether
    # the middle represents the whole shot is a property of the shot. The two
    # together give sd 1.16 against the 1.19 measured end to end, so
    # interpolation accounts for almost none of it.
    # And the truncation reaches further than the shot COUNT suggests: on two
    # real 4K episodes only 13-22% of shots exceed 120 frames, but those
    # shots hold 47-55% of the picture, of which the probe sees about 45%.
    # The cost of reaching further is modest, because long shots are few -
    # measured on a 1035-shot episode, total probed frames go 1.16x at 240,
    # 1.26x at 480 and 1.35x with no cap at all, while the truncated share of
    # the picture falls 26.2% -> 14.1% -> 7.1% -> 0%. Concurrency does not
    # suffer: _est_probe_gb grows only with log(frames) (+29% from 120 to
    # 480) and the pool is capped by cores/lp long before the memory budget.
    probe_max_frames: int = 120
    # Fraction of its lp a probe books against the CPU budget. 1.0 charges
    # the full lp for the task's whole life, which is what the encoding phase
    # wants - there the process really is SVT-AV1 start to finish. A probe is
    # not: it also reads and decodes its window and then scores it, and only
    # the encode segment uses lp cores. Measured on a 4K job, a probe averaged
    # 1.79 cores while holding 4 tokens, so the phase ran at 45% CPU
    # utilisation with concurrency pinned at cores/lp.
    # Below 1.0 admits more probes at once, and it does work: measured on a
    # 4K job, 0.5 took the probe phase from 10 concurrent to 18.
    #
    # It did not make it faster. Same clip, same 158 probes, identical chosen
    # CRFs: 313.1s at 1.0 against 342.4s at 0.5, i.e. 80% more in flight for 9%
    # LESS throughput. The idle-looking cores are not idle - one probe decodes
    # or scores while another encodes, so the phases already interleave across
    # tasks, and the reservation that looks like over-booking per task is about
    # right in aggregate. Adding more only adds contention.
    #
    # Left in because it is the right lever on a differently shaped machine
    # (many cores, little memory, or a much cheaper metric), but do not reach
    # for it expecting free throughput here. Default 1.0 is today's behaviour.
    #
    # Tried again on 2026-09-05 with the GPU build (SYCL scoring, VA-API
    # reference read), where a probe's scoring step measures about two of
    # the four cores it holds and E06 ran its probe phase at 30-36 of 40
    # cores: relaxing HALF the charge during the scoring step only, capped
    # at 30% of the cores, took peak concurrency from 10 to 13 and the probe
    # phase from 234s to 266s on an idle 40-core box (encode phase identical
    # at 178s). The cores that look idle are pipeline bubbles the extra
    # probes cannot fill without slowing every encode. Reverted.
    probe_cpu_charge: float = 1.0
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
    # Score on the GPU via libvmaf's SYCL backend. -1 = off (the CPU path);
    # >= 0 selects a SYCL device index, passed to the libvmaf filter as
    # sycl_device=N. Needs the image built with the VMAFx libvmaf and a Level
    # Zero device (Intel Arc). The filter fails loudly at init when the device
    # is not there rather than quietly scoring on the CPU, so a preflight runs
    # once per job and falls the whole job back to CPU instead of letting
    # every probe die.
    # Measured on a B580 at the real probe scale (120 frames, native 4K):
    # 1.7s on ONE core and 0.25GB, against 5.6s on 10.7 cores and 7.6GB.
    # Scores are unchanged - see the note on the vmaf-builder stage.
    vmaf_sycl_device: int = -1
    # Below this source width the CPU path is used even when vmaf_sycl_device
    # is set. The GPU win scales with frame size while its per-invocation
    # overhead does not: at 1080p CPU scoring is already only ~1.8s and the
    # warm GPU cost there has not been measured. Conservative on purpose -
    # lower it once 1080p is measured, do not assume.
    vmaf_sycl_min_width: int = 2560
    # Parallel probe workers across shots. Probes now encode at source
    # resolution, so each instance holds a multi-GB frame pool at 4K just like
    # the final encode (measured 3.5GB at 4K). 0 = auto, from the memory
    # available to THIS cgroup rather than the machine's total.
    probe_workers: int = 0
    # Hard cap on parallel FINAL ENCODE instances. 0 (recommended) = no cap:
    # the encode phase admits shots against a memory and CPU budget instead,
    # so concurrency floats with what each shot actually costs. That matters
    # because per-instance peak RSS follows the SHOT LENGTH - measured at 4K,
    # 3.4GB for a 24-frame shot against 10.5GB for a 1150-frame one - and any
    # single worker count is therefore too many for the long shots (which is
    # what OOM-kills the encoder) and too few for the short ones.
    encode_workers: int = 0
    # CPU cores pinned to each final-encode instance with taskset.
    # 0 (recommended) = no pinning: the admission budget already bounds total
    # SVT-AV1 parallelism, and a static core slice sized for N instances
    # strands cores whenever fewer than N are running (measured 5% slower).
    # Set it only when this container must not touch cores it was not given.
    encode_threads: int = 0
    # Which detector finds the shot boundaries.
    #   scdet         - ffmpeg's scene-change filter, run as one pass over the
    #                   source. Stages nothing and needs neither PySceneDetect
    #                   nor OpenCV. Measured against the other on three 4K
    #                   sources at scdet_threshold 2.0: boundaries identical on
    #                   one, one extra on each of the others. The whole phase
    #                   goes from 44.7s and 255s of CPU to 12.2s and 8s.
    #   pyscenedetect - the previous path: write a downscaled copy, then read it
    #                   back frame by frame with OpenCV.
    scenedetect_engine: Literal["scdet", "pyscenedetect"] = "scdet"
    # scdet score a frame must reach to count as a cut (0-100). NB ffmpeg's own
    # default of 10 is far too high for film and TV - it found 1 of 21 cuts on
    # a 4K sample. 2.0 tracks the PySceneDetect path closely. 0.8-1.5 also picks
    # up softer transitions (a single frame peaking 1.5-4.5). Below ~0.5 the
    # extra hits are broad and shallow - peak ~0.5 spread over 4-5 frames, i.e.
    # camera motion or a lighting change rather than a cut - and they start
    # displacing correct boundaries too, because the min_scene_len merge takes
    # the first candidate rather than the strongest.
    scdet_threshold: float = 2.0
    # PySceneDetect ContentDetector threshold (higher = fewer/split less).
    # Only used by scenedetect_engine=pyscenedetect.
    scenedetect_threshold: float = 27.0
    # Minimum shot length in frames (shorter segments are merged).
    min_scene_len: int = 24
    # ffmpeg scale filter used to make a small detection copy before running
    # PySceneDetect. OpenCV decodes frame-by-frame and 4K HEVC is unusably slow
    # (minutes per scene); a downscaled copy (same fps, frame numbers map 1:1)
    # is 5-10x faster. e.g. "-2:540". Empty = detect on the source directly.
    scenedetect_scale: str = "-2:540"
    # Decode the source on a GPU for that downscale pass ("auto") or never
    # ("off"). Auto tries Intel QSV first and falls back to software when there
    # is no usable device, when the codec is one the card cannot decode (AV1 on
    # a B580 exits 0 having written nothing), or when scenedetect_scale is not
    # a plain W:H that a fixed size can be derived from.
    #
    # The decode and the scale move to the GPU; the copy is still encoded with
    # x264. This does not reproduce the software copy exactly - the GPU scaler
    # is not swscale - so borderline cuts can land differently. Measured end to
    # end on three 90-second 4K clips: 46.4s -> 17.2s and 29.5s -> 20.1s on
    # 2160p, shot lists identical on two of them and 33 -> 34 on the third.
    # That is the same order as the 540p downscale this pass already does (2
    # cuts of 59 against native resolution). Below 2160p the x264 pass
    # dominates and the wall clock gets worse, though CPU still drops ~5x.
    scenedetect_hwaccel: Literal["auto", "off"] = "auto"
    # Decode the VMAF *reference* read on the GPU (VA-API on the first render
    # node): the source side of every probe score and of every verification
    # window. auto = once per job, decode a few frames after a seek both ways
    # and compare them frame by frame (framemd5); any difference keeps the
    # whole job on software. off = never. VA-API rather than QSV because the
    # QSV decoder kept different frames after a seek on an mkv with timeline
    # gaps, a DV P5 mp4 and an 8-bit H.264 WEB-DL (bit-exact pixels, wrong
    # frames); VA-API accelerates ffmpeg's own decoder and matched software
    # on all nine cases. Only this one read goes onto the
    # GPU on purpose: copying 4K frames back to the CPU tops out around 130
    # frames/s across the whole probe pool, which is about what the pool
    # already consumes, so a second read per probe would be GPU-bound and
    # slower; and the AV1 probe files decode faster on dav1d than on QSV plus
    # a download (measured: 13.5s against 10.1s wall for 120 frames). Measured
    # on 120 4K frames scored on SYCL: 39.4 -> 22.1 CPU-seconds, wall -12%,
    # decoded frames bit-exact, scores unchanged.
    #
    # Default OFF, measured. On the same 150s 4K clip at 40 cores, scores
    # identical to the last probe in all three runs:
    #   no hardware read      probing 401.3s   total 439.7s
    #   VA-API read           probing 447.7s   total 491.9s
    #   VA-API read, capped   probing 482.5s   total 528.2s
    # The read saves CPU (28.5 against 33.0 CPU-seconds for a 240-frame
    # score) but costs wall time (16.8s against 6.9s): pulling 4K frames back
    # out of the card is slower than decoding them. That only pays where the
    # CPU is the bottleneck - at 20 cores the same clip went 826s to 592s
    # with it on. At 40 it is a loss, and it doubles the card's memory
    # footprint (a score becomes two DRM clients instead of one), which is
    # what exhausted a 12GB B580 mid-episode and cost a reboot. Turn it on
    # for a CPU-starved host, not otherwise.
    reference_hwaccel: Literal["auto", "off"] = "off"
    # How many probes may score on the GPU at once (0 = auto, 6). NOT the
    # probe pool's width: the card is one device with one pool of memory and
    # each GPU score holds a SYCL context plus, with reference_hwaccel on, a
    # VA-API decode session - so one score is TWO DRM clients, ~440MB each.
    # Ten of them exhausted a 12GB B580 mid-episode (OUT_OF_DEVICE_MEMORY ->
    # DEVICE_LOST -> the job failed), and the device stayed broken for every
    # later process until the box was rebooted: the driver turns an
    # allocation failure into a lost device, which is its bug, but the
    # trigger is ours to avoid. Measured with DRM fdinfo (the interface
    # nvtop uses: sum drm-total-vram0 over /proc/*/fdinfo, deduplicated by
    # drm-client-id) on a 4K probe run - six concurrent scores peak at
    # 5.27GB across 12 clients, 44% of the card, so ten is ~8.8GB and the
    # twelve that broke it ~10.5GB before Plex and the framebuffer.
    # A wider queue buys nothing anyway: on 240-frame 4K windows throughput
    # plateaus at four (0.21/s at n=4, 0.23/s at n=10) and every job failed
    # at twelve. The cap costs 7.8% of the probe phase and is worth it: the
    # failure it prevents costs a reboot of a box that also runs Plex.
    vmaf_sycl_workers: int = 0
    # The card's memory this job may hold, in MB (0 = auto, 6000).
    #
    # vmaf_sycl_workers counts OPERATIONS; this counts bytes, which is what
    # the card actually runs out of. The two work together - the worker cap
    # stays as a ceiling on concurrency, the budget decides admission - and
    # the default is sized to land on the same place the cap does today: six
    # 4K scores with reference_hwaccel on measured 5.27GB. What it buys is
    # the cases the count cannot see: a longer probe window, a QSV probe
    # encode beside the scores, or a model that turns out to be wrong, all of
    # which change the bytes per operation without changing their number.
    #
    # Usage is measured from DRM fdinfo, so it is what the card really holds
    # rather than what was predicted - but only for THIS pid namespace. Plex
    # shares the B580 and is invisible from inside the container, so the
    # default leaves half the 12GB card unbooked rather than trying to
    # measure the other tenant. Raise it only on a card nothing else uses.
    gpu_vram_budget_mb: int = 0
    # Append one JSON line per probed shot to <logs>/probe_dataset.jsonl:
    # the shot's geometry, the probed (crf, score) points, the (q, score)
    # points when the card probed it too, both crossings, and the CRF the
    # job ended up encoding it at.
    #
    # It exists because the fit that maps one encoder's quality index to the
    # other's CRF is stuck at a residual no amount of per-job data will
    # improve: measured on a 163-shot episode, a two-parameter line is
    # already within 0.5% of its asymptote at 104 pairs (sqrt(1+1/n)), so the
    # 4.14 CRF it carries is the RELATIONSHIP, not the estimate. What could
    # improve it is a richer model - and on the same episode 94% of shots are
    # probed whole rather than sampled, which means that residual is content
    # structure a model could learn rather than measurement noise it cannot.
    #
    # A richer model needs more than one episode's 160 pairs, and every
    # episode that runs without recording them loses them for good. 160 lines
    # an episode is nothing to keep; this is the cheapest possible insurance
    # against wanting the data later.
    probe_dataset: bool = True
    # ---- the GPU probe path (opt-in, additive; "svt" leaves it switched off) ----
    # Which encoder the PROBES use. "svt" is the path everything above
    # describes. "qsv" probes on the card instead - VA-API decode into the
    # hardware AV1 encoder - and maps the quality index it finds back to an
    # SVT CRF through a per-job calibration.
    #
    # Why it can be more accurate rather than merely faster: a probe costs
    # 16.4 CPU-seconds this way against 88.3 for the SVT preset-9 probe on
    # 300 4K frames, and that budget buys back the probe window. Truncating
    # long shots to probe_max_frames is the single biggest term in today's
    # prediction error (sd 1.22 VMAF of the 1.19 measured end to end), and it
    # exists only because SVT probes are expensive.
    #
    # Measured on ten 4K windows, fitting SVT CRF against the QSV quality
    # index at the same VMAF, leave-one-out: sd 0.12 VMAF at target 91, 0.29
    # at target 94 once one outlier is removed - against 0.76 for the preset
    # gap the SVT probe path carries. The outlier is the reason for
    # gpu_probe_max_q_margin below: a very easy shot let SVT reach the target
    # at CRF 36 while the bulk sat near 20, and the line missed it by 10 CRF.
    #
    # Validated end to end on a 150s 4K clip, against the same clip on the SVT
    # path. At target 94 the calibration was REFUSED (leave-one-out 3.00 CRF
    # against the 2.00 limit) and the job fell back: delivered scores
    # identical to the SVT run, shot for shot. At target 90 it was accepted
    # (1.83 CRF over 5 anchors) and the shots it mapped delivered within 0.32
    # VMAF of the SVT path, worst case, for the same output size.
    #
    # The refusal at 94 is a property of that clip, not of the target: 12 of
    # its 22 shots cannot reach 94 at any probed CRF, so both curves are being
    # crossed at their ends where the inversion is worst. Real episodes are
    # the opposite - 4 shots of 1035 on one measured episode - which is why
    # this needs an episode-scale trial before it is worth anyone turning on.
    probe_encoder: Literal["svt", "qsv", "qsv+svt"] = "svt"
    # "qsv+svt" is the same card work used differently: the mapped CRF only
    # says WHERE to probe, and the answer still comes from SVT probes scored
    # the usual way, so a mapping that is off costs a probe instead of
    # shipping a CRF nobody measured. That is what lets it drop the refusal
    # gate "qsv" needs, and what lets the line keep learning from every shot
    # it verifies instead of from a fixed set of anchors.
    #
    # Quality indices swept on the QSV side. Same role as probe_crfs, and the
    # search over them is the same bisection; av1_qsv saturates above ~50 on
    # 4K, so there is nothing to learn past the top of this.
    gpu_probe_qs: List[int] = Field(default_factory=lambda: [14, 18, 22, 26, 30, 34, 38])
    # Shots probed BOTH ways to fit the mapping. They are ordinary probes -
    # their SVT samples are used for those shots - so this is not wasted work,
    # it is the part of the job that stays on the old path.
    gpu_probe_anchors: int = 16
    # Give up on the mapping for the whole job when the calibration's
    # leave-one-out residual exceeds this many CRF. At the measured 0.24
    # VMAF/CRF the default is ~0.5 VMAF, comfortably inside what the SVT
    # probe path already carries.
    gpu_probe_max_residual: float = 2.0
    # Per-SHOT abstention: a shot whose quality index falls further than this
    # outside the anchors' range is probed with SVT instead of trusting an
    # extrapolated line. This is what catches the easy-shot outlier above.
    gpu_probe_max_q_margin: float = 1.5
    # QSV encoder preset (0-7, lower is slower and better). 4 is what the
    # mapping above was measured at.
    gpu_probe_preset: int = 4

    @field_validator("reference_hwaccel", "scenedetect_hwaccel", mode="before")
    @classmethod
    def _yaml_reads_off_as_false(cls, v: object) -> object:
        """YAML 1.1 parses a bare `off` as the boolean False (and `on` as
        True), so `reference_hwaccel: off` in config.yaml arrives here as a
        bool and fails the Literal with a message that says nothing about
        quoting. Take it as written instead - the quotes in config.yaml are
        belt and braces, not the only thing holding this up."""
        if isinstance(v, bool):
            return "auto" if v else "off"
        return v
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
    # Added to every interpolated CRF before clamping, to trade the probe's
    # systematic bias back for size. 0 = off.
    #
    # Read probe_preset's note before setting this: the bias is content
    # dependent (measured end-to-end at -0.29, -0.18, +0.55, +0.81 and +1.52
    # VMAF on five different sources), so one constant helps some material
    # and hurts the rest. It is also the smaller half of the problem - the
    # per-shot SPREAD, which no offset can touch, comes mostly from
    # probe_max_frames truncating long shots.
    probe_crf_offset: float = 0.0
    # Pass decimal CRF values to SVT-AV1 (finer than integer CRF granularity).
    # SVT-AV1 must accept fractional --crf for this to work.
    fractional_crf: bool = False
    # After the encode, re-score this many shots from the FINISHED file against
    # the source with target_metric and log delivered vs target. Everything
    # before it trusts the probes: a CRF is picked from a fast probe encode of
    # a 120-frame window and then applied to the delivery, and nothing checks
    # that the delivery landed where the probe said. Sampled shots are spread
    # across the timeline, since the faults worth catching (a seek that starts
    # mis-landing, a shot list running out early) read low on everything after
    # a point. Diagnostic only - it never fails the job. Costs one lossless
    # window extraction per side plus one metric run per sampled shot.
    # 0 = off.
    verify_shots: int = 10
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
    # NB there is no strip_rpu knob. Profile 7/8 sources are read straight
    # from the original file: the decoder returns the base layer and ignores
    # the enhancement layer and the RPU by itself, frame-identical (framemd5,
    # software and QSV) to the dovi_split=bl remux that used to be written
    # first - three minutes and a 27GB write per 4K episode, for nothing. The
    # RPU is saved separately when save_rpu is set. A `strip_rpu` field
    # sat here for a long time, documented in config.yaml, read by nothing at
    # all; it was removed rather than wired up, because wiring it would have
    # changed the pipeline for anyone who had already set it to false and
    # (correctly) observed no effect. Unknown keys are ignored, so an existing
    # config.yaml carrying it still loads.

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
    # config.yaml's optimizer settings before the user's overrides land on
    # them: what "restore defaults" restores, and what the settings page
    # marks a field against. Excluded from dumps like builtin_presets.
    optimizer_defaults: OptimizerSettings = Field(default_factory=OptimizerSettings, exclude=True)
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


def apply_default_preset(settings: Settings) -> None:
    """Point transcode.video at whatever the default preset currently holds.

    transcode.video is the parameter set a job gets when no preset supplies
    one - most visibly preset="custom", which starts from it and applies the
    custom overrides on top.

    It used to be snapshotted from the default preset BEFORE user presets were
    loaded, so editing the default preset in the web UI updated
    transcode.presets[name] and left transcode.video on config.yaml's values.
    Measured: saving crf=19/preset=2 onto "balanced" (the default preset) and
    reloading gave presets["balanced"].crf == 19 and transcode.video.crf == 28.
    A restart did not help, because the order was wrong in load_settings rather
    than merely stale in memory.

    Called again by the preset write/delete routes so the live object tracks an
    edit without waiting for one.
    """
    name = settings.transcode.default_preset
    if name in settings.transcode.presets:
        settings.transcode.video = settings.transcode.presets[name].model_copy(deep=True)


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
    # Snapshot builtin presets BEFORE the user's overrides land on them - that
    # snapshot is what `_builtin` reports and what deleting a user preset
    # restores, so it has to be config.yaml's version.
    settings.transcode.builtin_presets = {
        name: v.model_copy(deep=True) for name, v in presets.items()
    }
    user = load_user_presets(settings)
    for name, params in user.items():
        presets[name] = params
    # ...and derive transcode.video only AFTER them. See apply_default_preset.
    apply_default_preset(settings)
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
    settings.transcode.optimizer_defaults = settings.transcode.optimizer.model_copy(deep=True)
    user_opt = usettings.get("optimizer") or {}
    if user_opt:
        # Merged OVER config.yaml, not validated on its own: a saved block used
        # to replace the whole optimizer section, so every default that
        # config.yaml changed after the save was silently pinned to whatever
        # the page had dumped that day (a production box ran 360p probes,
        # every other frame, 1200-frame windows and 40 VMAF threads that way
        # for weeks). The page now saves only the keys that differ, and an
        # old full block still merges to the same values it always had.
        try:
            settings.transcode.optimizer = OptimizerSettings.model_validate(
                {**settings.transcode.optimizer.model_dump(mode="json"), **user_opt})
        except Exception as e:  # noqa: BLE001
            logger = __import__("loguru").logger
            logger.warning("Ignoring invalid optimizer settings: {}", e)
    user_dovi = usettings.get("dovi") or {}
    if "vulkan_device" in user_dovi:
        settings.transcode.dovi.vulkan_device = str(user_dovi["vulkan_device"] or "")
    return settings


def optimizer_overrides(settings: Settings) -> Dict[str, Any]:
    """The optimizer keys whose value is not config.yaml's, JSON-shaped."""
    cur = settings.transcode.optimizer.model_dump(mode="json")
    base = settings.transcode.optimizer_defaults.model_dump(mode="json")
    return {k: v for k, v in cur.items() if base.get(k) != v}


def save_optimizer_settings(settings: Settings) -> Dict[str, Any]:
    """Persist the optimizer block as overrides only, dropping it when empty."""
    over = optimizer_overrides(settings)
    if over:
        save_user_settings(settings, {"optimizer": over})
    else:
        delete_user_setting(settings, "optimizer")
    return over


def delete_user_setting(settings: Settings, key: str) -> bool:
    """Remove one top-level key from the settings file. True if it was there."""
    p = settings.dirs.settings_file
    try:
        existing = json.loads(p.read_text()) if p.exists() else {}
    except (OSError, ValueError):
        existing = {}
    if key not in existing:
        return False
    existing.pop(key)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(existing, indent=2, ensure_ascii=False, default=str))
    tmp.replace(p)
    return True


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
        if isinstance(data.get("dovi"), dict):
            out["dovi"] = data["dovi"]
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
    # default=str so one un-encodable value cannot cost the caller its whole
    # save. Callers should hand this JSON-mode data (model_dump(mode="json"));
    # this is the guard for when they do not - a Path field added to a settings
    # model turned every optimizer save into a 500 exactly that way.
    tmp.write_text(json.dumps(existing, indent=2, ensure_ascii=False, default=str))
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
