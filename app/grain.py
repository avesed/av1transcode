"""Automatic grain handling for one job (VideoParams.grain_auto), around the optimizer engine.

The work is done by the grain service (grain/ in this repo, image av1t-grain), which has the GPU models; this module
asks it, waits for it and puts its results into the job:

  prepare()  before the encode. The service analyses the source and says on or off: on for film grain a denoiser
             should take off (noise level >= 12 on the owner's scale, Gaussian, new every frame), off for clean
             sources and for frozen or deliberate grain texture, which then encode exactly as without the switch.
             On: the service denoises the source into an intermediate (the B580's AV1 at QP 0, video only, constant
             frame rate), which gets the source's frame timestamps back here (holes and all) and becomes the engine's
             input. The engine picks its CRFs against the denoised picture; its mux still takes audio, subtitles,
             chapters and attachments from the source itself (info.path), as for a Dolby Vision P5 intermediate.
  finish()   after the encode. The output's video goes to the service, which measures per shot and brightness the
             grain it lacks against the source, writes it into the AV1 stream as film grain synthesis (grav1synth, no
             re-encode) and corrects it once against what the synthesis really adds. That stream replaces the
             output's video, on the output's own timestamps.

Failures before the encode degrade to the plain encode with a warning (an unreachable service must not stop a queue);
a failure after it fails the job, since an output denoised and left without its grain is not one to keep.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple

from loguru import logger

from app.analyzer import MediaInfo
from app.config import Settings, VideoParams


class GrainError(Exception):
    """The grain service failed, refused or could not be reached."""


class GrainContext(NamedTuple):
    """What finish() needs from prepare(): the service and the source the grain is measured against."""
    service: "GrainService"
    source: Path
    decision: dict


class GrainService:
    def __init__(self, url: str, timeout_s: float):
        self.url, self.timeout_s = url.rstrip("/"), timeout_s

    def _call(self, method: str, path: str, body: Optional[dict] = None, timeout: float = 30) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read() or b"{}").get("error", "")
            except ValueError:
                msg = ""
            raise GrainError(f"grain service {method} {path}: HTTP {e.code} {msg}".strip()) from e
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise GrainError(f"grain service at {self.url} unreachable: {e}") from e

    def run(self, kind: str, params: dict, progress_cb: Optional[Callable[[float], None]] = None,
            cancel_flag: Optional[Callable[[], bool]] = None, poll: float = 2.0) -> dict:
        """Submit a job and wait for it. progress_cb gets 0..1; a cancel cancels the service job too."""
        jid = self._call("POST", "/jobs", {"kind": kind, **params})["id"]
        deadline = time.monotonic() + self.timeout_s
        while True:
            if cancel_flag is not None and cancel_flag():
                self._cancel(jid)
                raise GrainError(f"{kind}: cancelled")
            v = self._call("GET", f"/jobs/{jid}")
            if progress_cb is not None:
                progress_cb(float(v.get("progress") or 0.0))
            if v["state"] == "done":
                return v.get("result") or {}
            if v["state"] in ("failed", "cancelled"):
                tail = " | ".join((v.get("log") or [])[-3:])
                raise GrainError(f"{kind} {v['state']}: {v.get('error')}" + (f" ({tail})" if tail else ""))
            if time.monotonic() > deadline:
                self._cancel(jid)
                raise GrainError(f"{kind}: no result after {self.timeout_s / 3600:g} h")
            time.sleep(poll)

    def _cancel(self, jid: str) -> None:
        try:
            self._call("POST", f"/jobs/{jid}/cancel", {})
        except GrainError:
            pass


# SVT-AV1's own film grain, which grain_auto replaces; in additional_video_params as "--film-grain 8",
# "--film-grain=8", "film-grain=8" or inside an svtav1-params string ("enable-overlays=1:film-grain=8")
_SVT_GRAIN_KEYS = ("film-grain", "film-grain-denoise", "fgs-table")


def _strip_svt_grain(params: str) -> Tuple[str, List[str]]:
    """additional_video_params without SVT-AV1's film grain options -> (what is left, what was taken out)."""
    toks, keep, gone, i = (params or "").split(), [], [], 0
    is_grain = lambda t: t.split("=", 1)[0].lstrip("-") in _SVT_GRAIN_KEYS
    while i < len(toks):
        t = toks[i]
        if ":" in t and "=" in t and not t.startswith("-"):          # key=value:key=value
            parts = t.split(":")
            gone += [x for x in parts if is_grain(x)]
            rest = [x for x in parts if not is_grain(x)]
            if rest:
                keep.append(":".join(rest))
        elif is_grain(t):
            if "=" not in t and i + 1 < len(toks) and not toks[i + 1].startswith("-"):
                gone.append(f"{t} {toks[i + 1]}")
                i += 1
            else:
                gone.append(t)
        else:
            keep.append(t)
        i += 1
    return " ".join(keep), gone


def without_film_grain(video: VideoParams) -> VideoParams:
    """grain_auto owns the grain: SVT-AV1's film grain stays off for its jobs, whatever the profile sets (film_grain,
    film_grain_denoise, or the options in additional_video_params). Left on, the two would stack on a grain-on
    source, the probes would pick CRFs with SVT's grain in them, and grav1synth will not write a table into a stream
    that already carries one - the job would fail after the whole encode."""
    extra, gone = _strip_svt_grain(video.additional_video_params)
    if video.film_grain:
        gone.insert(0, f"film_grain {video.film_grain}")
    if video.film_grain_denoise:
        gone.insert(0 if not video.film_grain else 1, "film_grain_denoise")
    if not gone:
        return video
    logger.warning("grain_auto: SVT-AV1's own film grain stays off with it; ignoring {}", ", ".join(gone))
    return video.model_copy(update={"film_grain": 0, "film_grain_denoise": False, "additional_video_params": extra})


def _run(cmd: Sequence[str], what: str, timeout: float = 3 * 3600) -> str:
    p = subprocess.run(list(cmd), capture_output=True, text=True, errors="replace", timeout=timeout)
    ok = (0, 1) if Path(cmd[0]).name in ("mkvmerge", "mkvpropedit") else (0,)   # 1: warnings, the file complete
    if p.returncode not in ok:
        raise GrainError(f"{what} failed ({p.returncode}): {(p.stderr or p.stdout)[-600:]}")
    return p.stdout


def _video_track_id(settings: Settings, path: Path) -> int:
    j = json.loads(_run([settings.tool_path("mkvmerge"), "-J", str(path)], "mkvmerge -J"))
    for t in j.get("tracks", []):
        if t.get("type") == "video":
            return int(t["id"])
    raise GrainError(f"no video track in {path}")


def _write_timestamps(settings: Settings, path: Path, out: Path) -> int:
    """The video's frame timestamps as an mkvmerge v2 file (ms, in display order) -> the number of lines.
    Matroska: mkvextract, exact and quick, which (v82) adds the last frame's end as one more line - mkvmerge reads
    it back as that frame's duration; anything else: the packets' pts from ffprobe, sorted, one per frame."""
    if path.suffix.lower() in (".mkv", ".mka", ".webm"):
        tid = _video_track_id(settings, path)
        _run([settings.tool_path("mkvextract"), str(path), "timestamps_v2", f"{tid}:{out}"], "mkvextract timestamps")
        return sum(1 for line in out.read_text().splitlines() if line and not line.startswith("#"))
    txt = _run([settings.tool_path("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                "packet=pts_time", "-of", "csv=p=0", str(path)], "ffprobe pts")
    pts = sorted(float(x) for x in txt.split() if x not in ("", "N/A"))
    st = _run([settings.tool_path("ffprobe"), "-v", "error", "-show_entries", "format=start_time", "-of", "csv=p=0",
               str(path)], "ffprobe start").strip()
    t0 = float(st) if st not in ("", "N/A") else (pts[0] if pts else 0.0)
    # relative to the container's start, as Matroska stores them: the video's lead over the audio (an m2ts's
    # video starting a few frames after its PCR, say) stays what the engine measures on the source
    out.write_text("# timestamp format v2\n" + "".join(f"{(t - t0) * 1000:.3f}\n" for t in pts))
    return len(pts)


def _rate(settings: Settings, path: Path) -> str:
    return _run([settings.tool_path("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=r_frame_rate", "-of", "csv=p=0", str(path)], "ffprobe rate").strip().split(",")[0]


def _set_frame_rate(settings: Settings, path: Path, rate: str) -> None:
    """Give path's video the default duration of `rate` (ffprobe's num/den). mkvmerge --timestamps sets it from the
    file's first interval in Matroska's whole milliseconds - 42 ms, 23.810 fps, for 24000/1001 - and QSV decoding
    stamps frames by it: the engine's scdet pass saw a duplicated timestamp every 144 frames and refused the file
    (the timestamps themselves were right). Players show it as the frame rate, too."""
    num, _, den = rate.partition("/")
    try:
        ns = round(1e9 * float(den or 1) / float(num))
    except (ValueError, ZeroDivisionError):
        raise GrainError(f"no frame rate for {path.name} ({rate!r})") from None
    _run([settings.tool_path("mkvpropedit"), "-q", str(path), "--edit", "track:v1", "--set", f"default-duration={ns}"],
         "mkvpropedit default duration")


def _frames(settings: Settings, path: Path) -> int:
    txt = _run([settings.tool_path("ffprobe"), "-v", "error", "-select_streams", "v:0", "-count_packets",
                "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)], "ffprobe count")
    return int(txt.strip().split(",")[0])


def _unlink(*paths: Path) -> None:
    for p in paths:
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink()
        except OSError:
            pass


def prepare(settings: Settings, info: MediaInfo, video: VideoParams, p5: bool, encode_input: Path, work_dir: Path,
            tmp_files: List[Path], stage_cb: Optional[Callable[[str], None]] = None,
            progress_cb: Optional[Callable[[float, Optional[dict]], None]] = None,
            cancel_flag: Optional[Callable[[], bool]] = None) -> tuple:
    """-> (the engine's input, GrainContext or None). None means off: encode encode_input as it is."""
    cfg = settings.transcode.grain
    if video.engine != "optimizer":
        logger.warning("grain_auto needs the optimizer engine; encoding {} without it", info.path.name)
        return encode_input, None
    if p5:
        logger.warning("grain_auto: Dolby Vision profile 5 is converted shot by shot inside the engine and cannot "
                       "be denoised first; encoding {} without it", info.path.name)
        return encode_input, None
    svc = GrainService(cfg.url, cfg.timeout_hours * 3600)

    def stage(name: str) -> None:
        if stage_cb is not None:
            stage_cb(name)
        if progress_cb is not None:
            progress_cb(0.0, None)

    def prog(p: float) -> None:
        if progress_cb is not None:
            progress_cb(round(p * 100, 1), None)

    stage("grain_analyse")
    try:
        dec = svc.run("analyse", {"source": str(encode_input)}, prog, cancel_flag)
    except GrainError as e:
        if cancel_flag is not None and cancel_flag():
            raise
        logger.warning("grain_auto: analysis failed, encoding {} as it is: {}", info.path.name, e)
        return encode_input, None
    f = dec.get("features") or {}
    logger.info("grain_auto: {} -> {} ({}); level {}, kurtosis {}, frozen {}", info.path.name,
                "on" if dec.get("on") else "off", dec.get("why"), dec.get("level"), f.get("kurtosis"), f.get("fresh_corr"))
    if not dec.get("on"):
        return encode_input, None

    stage("denoising")
    ns = time.time_ns()
    dn = work_dir / f"{info.path.stem}.grain_dn_{ns}.mkv"
    base = work_dir / f"{info.path.stem}.grain_base_{ns}.mkv"
    ts = work_dir / f"{info.path.stem}.grain_ts_{ns}.txt"
    tmp_files += [dn, ts] + ([] if cfg.keep_intermediate else [base])
    t0 = time.monotonic()
    try:
        r = svc.run("denoise", {"source": str(encode_input), "out": str(dn), "model": cfg.model}, prog, cancel_flag)
        n_ts = _write_timestamps(settings, encode_input, ts)
        n_dn = _frames(settings, dn)
        if n_ts not in (n_dn, n_dn + 1):                 # + 1: mkvextract's end line
            raise GrainError(f"the denoised video has {n_dn} frames, the source's timestamps {n_ts}")
        # the source's own frame timestamps on the denoised frames: a hole in the source stays a hole, so the
        # engine maps frames to the same timeline slots and the mux keeps sync
        _run([settings.tool_path("mkvmerge"), "-q", "-o", str(base), "--timestamps", f"0:{ts}", str(dn)],
             "mkvmerge timestamps")
        _set_frame_rate(settings, base, _rate(settings, encode_input))
    except GrainError as e:
        _unlink(dn, base, ts)
        if cancel_flag is not None and cancel_flag():
            raise
        logger.warning("grain_auto: denoising failed, encoding {} as it is: {}", info.path.name, e)
        return encode_input, None
    _unlink(dn, ts)
    logger.info("grain_auto: {} frames denoised with {} in {:.0f} s -> {}", n_dn, r.get("model"),
                time.monotonic() - t0, base.name)
    return base, GrainContext(svc, encode_input, dec)


def finish(settings: Settings, ctx: GrainContext, output: Path, shots: Sequence[int], work_dir: Path,
           stage_cb: Optional[Callable[[str], None]] = None,
           progress_cb: Optional[Callable[[float, Optional[dict]], None]] = None,
           cancel_flag: Optional[Callable[[], bool]] = None) -> dict:
    """Put the measured grain into output's video, in place. Raises GrainError."""
    if stage_cb is not None:
        stage_cb("grain")
    if progress_cb is not None:
        progress_cb(0.0, None)
    if not shots or sum(shots) <= 0:
        raise GrainError("the engine reported no shots to build the grain table on")
    ns = time.time_ns()
    vid = work_dir / f"{output.stem}.grain_v_{ns}.ivf"
    out_ivf = work_dir / f"{output.stem}.grain_g_{ns}.ivf"
    ts = work_dir / f"{output.stem}.grain_ots_{ns}.txt"
    gwork = work_dir / f"grain_work_{ns}"
    tmp = output.with_name(output.stem + f".grain_{ns}.mkv")
    try:
        _write_timestamps(settings, output, ts)
        n_out = _frames(settings, output)
        if n_out != sum(shots):
            raise GrainError(f"the output has {n_out} frames, its shots {sum(shots)}")
        _run([settings.tool_path("ffmpeg"), "-nostdin", "-v", "error", "-y", "-i", str(output), "-map", "0:v:0",
              "-c", "copy", "-f", "ivf", str(vid)], "ffmpeg ivf")
        rate = _rate(settings, output)
        res = ctx.service.run("grain", {"source": str(ctx.source), "video": str(vid), "shots": list(shots),
                                        "fps": rate or "24000/1001", "out": str(out_ivf), "work": str(gwork),
                                        "strength": settings.transcode.grain.strength},
                              (lambda p: progress_cb(round(p * 100, 1), None)) if progress_cb else None, cancel_flag)
        if _frames(settings, out_ivf) != n_out:
            raise GrainError("the grain-synthesis stream lost frames")
        # the output's other tracks, chapters, attachments, tags and title first (-D: no video), the grain stream on
        # the output's own timestamps second, then the video put back in front; finish_metadata sets the video
        # track's flags, language, colour and HDR afterwards as for any output
        oid = _video_track_id(settings, output)
        _run([settings.tool_path("mkvmerge"), "-q", "-o", str(tmp), "-D", str(output), "--timestamps", f"0:{ts}",
              str(out_ivf), "--track-order", f"1:0,{_track_order_rest(settings, output, oid)}"], "mkvmerge grain mux")
        _set_frame_rate(settings, tmp, rate)
        if _frames(settings, tmp) != n_out:
            raise GrainError("the grain mux lost frames")
        tmp.replace(output)
    finally:
        _unlink(vid, out_ivf, ts, gwork, tmp)
    t = (res.get("target") or {}).get("y")
    logger.info("grain_auto: film grain synthesis written into {} (luma target per brightness bin {})", output.name, t)
    fl = res.get("flicker") or {}
    src_sw, auto_sw = (fl.get("source") or {}).get("fine_swing"), (fl.get("auto") or {}).get("fine_swing")
    if src_sw and auto_sw:
        # logged, not acted on (the owner picked flicker-vetoed versions twice in blind tests, test-08 / test-09)
        logger.info("grain_auto: grain flicker (fine-band swing on flat static picture) {:.3f} against the source's "
                    "{:.3f} ({:.2f}x)", auto_sw, src_sw, auto_sw / src_sw)
    hard = [r for r in fl.get("shots") or [] if r.get("fine_ratio")]
    if hard:
        # measured on the hardest shots only (most flat picture times grain to synthesise), highest ratio first
        logger.info("grain_auto: flicker on the {} hardest shots, highest first: {}", len(hard), ", ".join(
            f"shot {r['shot']} {r['fine_ratio']:.2f}x ({r['source'].get('fine_swing')} -> "
            f"{r['auto'].get('fine_swing')})" for r in hard[:3]))
    return res


def _track_order_rest(settings: Settings, output: Path, video_id: int) -> str:
    """mkvmerge --track-order for the output's non-video tracks (file 0), in their original order."""
    j = json.loads(_run([settings.tool_path("mkvmerge"), "-J", str(output)], "mkvmerge -J"))
    ids = [int(t["id"]) for t in j.get("tracks", []) if int(t["id"]) != video_id]
    return ",".join(f"0:{i}" for i in ids) or "0:0"
