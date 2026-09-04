"""What the container can actually see of a GPU, for the settings page.

Every switch on that page that says "GPU" is only as good as the hardware
behind it, and the failure modes are all silent: a render node that was not
passed through, an ffmpeg built without the SYCL libvmaf, a Vulkan loader
that quietly picked llvmpipe. So the page shows what is really there, read
the same way the pipeline will use it - through ffmpeg - rather than from
sysfs. The probes are cheap (well under two seconds together) and cached, and
the self-check is the optimizer's own SYCL preflight: one pair scored on the
GPU and on the CPU, and the two compared.
"""
from __future__ import annotations

import glob
import re
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

from app.config import Settings

# testsrc2, blurred against itself: the exact pair the optimizer's preflight
# scores. A flat frame or a frame scored against itself returns the same
# number from any backend, working or not.
_PAIR = "testsrc2=s=256x256:d=1:r=30"
_TTL = 300.0
_lock = threading.Lock()
_cache: Dict[str, Any] = {}


def _run(args: List[str], timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, errors="replace",
                          timeout=timeout)


def render_nodes() -> List[str]:
    return sorted(glob.glob("/dev/dri/renderD*"))


def parse_vulkan_listing(text: str) -> List[str]:
    """Device names from ffmpeg's `-v verbose -init_hw_device vulkan` output:

        [Vulkan @ 0x...] GPU listing:
        [Vulkan @ 0x...]     0: Intel(R) Arc(tm) B580 Graphics (BMG G21) (discrete) (0xe20b)
    """
    out: List[str] = []
    listing = False
    for line in text.splitlines():
        body = re.sub(r"^\[Vulkan @ [^\]]+\]\s*", "", line)
        if body.startswith("GPU listing:"):
            listing = True
            continue
        if listing:
            m = re.match(r"\s*(\d+):\s+(.+?)\s*$", body)
            if not m:
                break
            out.append(m.group(2))
    return out


def parse_sycl(text: str) -> Dict[str, Any]:
    """The SYCL backend announces its device on stderr, and refuses an index
    it does not have with the count in the message:

        libvmaf INFO SYCL: using device: Intel(R) Arc(TM) B580 Graphics
        libvmaf ERROR SYCL: device_index 7 out of range (1 GPUs)
    """
    m = re.search(r"SYCL: using device: (.+)", text)
    if m:
        return {"device": m.group(1).strip(), "error": None}
    m = re.search(r"SYCL: (device_index \d+ out of range \((\d+) GPUs?\))", text)
    if m:
        return {"device": None, "count": int(m.group(2)), "error": m.group(1)}
    m = re.search(r"vmaf_sycl_state_init\(\d+\) failed[^\n]*", text)
    if m:
        return {"device": None, "error": m.group(0).strip()}
    return {"device": None, "error": None}


def _score_cmd(ffmpeg: str, model: str, device: Optional[int], log: str) -> List[str]:
    sycl = f"sycl_device={device}:" if device is not None else ""
    lavfi = ("[0:v]boxblur=2,format=yuv420p10le[d];[1:v]format=yuv420p10le[r];"
             f"[d][r]libvmaf=model={model}:{sycl}log_fmt=json:log_path={log}")
    return [ffmpeg, "-hide_banner", "-loglevel", "info", "-y",
            "-f", "lavfi", "-i", _PAIR, "-f", "lavfi", "-i", _PAIR,
            "-lavfi", lavfi, "-an", "-sn", "-dn", "-f", "null", "-"]


def _model_cfg(settings: Settings) -> str:
    return f"path={settings.transcode.optimizer.vmaf_model}"


def probe(settings: Settings, force: bool = False) -> Dict[str, Any]:
    """Everything the GPU section shows, cached for a few minutes.

    `sycl.device` is the name the libvmaf backend reports for the CONFIGURED
    index (or 0 when scoring is off), `sycl.count` how many it can see;
    `vulkan.devices` is ffmpeg's own listing, which is what libplacebo picks
    from; `qsv` and `vaapi` are whether a hardware decode context opens.
    """
    with _lock:
        now = time.time()
        if not force and _cache and now - _cache.get("checked_at", 0) < _TTL:
            return dict(_cache)
        try:
            ffmpeg = settings.tool_path("ffmpeg")
        except FileNotFoundError:
            ffmpeg = None
        nodes = render_nodes()
        out: Dict[str, Any] = {"render_nodes": nodes, "ffmpeg": bool(ffmpeg),
                               "sycl_built": False,
                               "sycl": {"device": None, "count": None, "error": None},
                               "vulkan": {"devices": [], "error": None},
                               "qsv": False, "vaapi": False, "checked_at": now}
        if ffmpeg:
            try:
                h = _run([ffmpeg, "-hide_banner", "-h", "filter=libvmaf"], timeout=15)
                out["sycl_built"] = "sycl_device" in h.stdout
            except (OSError, subprocess.SubprocessError):
                pass
            if out["sycl_built"]:
                dev = max(0, int(settings.transcode.optimizer.vmaf_sycl_device))
                try:
                    r = _run(_score_cmd(ffmpeg, _model_cfg(settings), dev, "/dev/null"), timeout=60)
                    parsed = parse_sycl(r.stdout + r.stderr)
                    if parsed.get("device") is None and parsed.get("count") is None:
                        # ask for an index no box has, to learn the count
                        r2 = _run(_score_cmd(ffmpeg, _model_cfg(settings), 999, "/dev/null"), timeout=60)
                        parsed.setdefault("count", parse_sycl(r2.stdout + r2.stderr).get("count"))
                    out["sycl"].update({k: parsed.get(k) for k in ("device", "count", "error")})
                except (OSError, subprocess.SubprocessError) as e:
                    out["sycl"]["error"] = str(e)
            try:
                r = _run([ffmpeg, "-hide_banner", "-v", "verbose", "-init_hw_device", "vulkan",
                          "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1", "-frames:v", "1",
                          "-f", "null", "-"], timeout=30)
                out["vulkan"]["devices"] = parse_vulkan_listing(r.stderr + r.stdout)
                if r.returncode != 0 and not out["vulkan"]["devices"]:
                    out["vulkan"]["error"] = (r.stderr.strip().splitlines() or ["vulkan init failed"])[-1]
            except (OSError, subprocess.SubprocessError) as e:
                out["vulkan"]["error"] = str(e)
            for key, dev_args in (("qsv", ["qsv=hw"]),
                                  ("vaapi", [f"vaapi=va:{nodes[0]}"] if nodes else None)):
                if not dev_args:
                    continue
                try:
                    r = _run([ffmpeg, "-hide_banner", "-v", "error", "-init_hw_device", *dev_args,
                              "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1", "-frames:v", "1",
                              "-f", "null", "-"], timeout=30)
                    out[key] = r.returncode == 0
                except (OSError, subprocess.SubprocessError):
                    out[key] = False
        _cache.clear()
        _cache.update(out)
        return dict(out)


def selfcheck(settings: Settings, device: int) -> Dict[str, Any]:
    """Score the preflight pair on SYCL device `device` and on the CPU.

    The same test the optimizer runs before trusting the GPU for a job: the
    backend has to announce itself (an ignored sycl_device would agree
    perfectly) and the two scores have to sit within 1e-3 of each other.
    """
    import json
    import tempfile

    ffmpeg = settings.tool_path("ffmpeg")
    model = _model_cfg(settings)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        gpu_log, cpu_log = f"{d}/gpu.json", f"{d}/cpu.json"
        g = _run(_score_cmd(ffmpeg, model, device, gpu_log), timeout=120)
        c = _run(_score_cmd(ffmpeg, model, None, cpu_log), timeout=120)
        result: Dict[str, Any] = {"device_index": device, "elapsed": round(time.time() - t0, 2),
                                  "announced": "vmaf-sycl" in (g.stdout + g.stderr)}
        result.update(parse_sycl(g.stdout + g.stderr))
        try:
            gpu = json.load(open(gpu_log))["pooled_metrics"]["vmaf"]["mean"]
            cpu = json.load(open(cpu_log))["pooled_metrics"]["vmaf"]["mean"]
        except (OSError, ValueError, KeyError):
            result.update({"gpu": None, "cpu": None, "delta": None, "ok": False})
            if not result.get("error"):
                result["error"] = (g.stderr.strip().splitlines() or ["scoring produced no result"])[-1]
            return result
    delta = abs(gpu - cpu)
    result.update({"gpu": gpu, "cpu": cpu, "delta": delta,
                   "ok": bool(result["announced"] and delta <= 1e-3 and not result.get("error"))})
    return result
