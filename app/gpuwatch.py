"""A separate process that watches the GPU, so that what the GPU's driver does to a caller never stops a job.

2026-10-16, Dark S02E08: the B580's GT0 reset at 03:30 during S02E07's probes. At 05:07 the optimizer's VRAM budget
read /proc/*/fdinfo, as it did before every GPU operation. In the kernel that read went xe_drm_client_fdinfo ->
show_meminfo -> ttm_bo_release -> dma_resv_copy_fences and spun there. That was an RCU stall, which hung 21 zero-copy
ffmpegs behind it. At 05:13 the kernel killed the thread ("exited with irqs disabled"). Python never learnt of it: the
probe scheduler waited for that shot's result, and the job sat at "probing 99.9%" for ten hours with no process
running.

So everything that touches the driver's state from this app goes through a child process:
  - the VRAM reading, once a second: on xe the driver's own device query (DRM_XE_DEVICE_QUERY_MEM_REGIONS), which
    reads a counter and touches no buffer object - the 10-16 spin was a buffer object's last reference dropped inside
    the fdinfo read. It is the whole card's figure (the grain service's container included, which fdinfo from here
    never saw); measured 10-17: 24 MB idle, 124 MB during the watcher's own test, back within the second. Other
    drivers keep the fdinfo sum;
  - the kernel log (the host's, mounted read-only), for this card's errors;
  - a small real test every few minutes: a 3-frame av1_vaapi encode and one SYCL score of the preflight pair, each
    with a hard timeout.
The child writes a status file each second. The app only reads that file, and a regular file read cannot hang in
the GPU driver. A child that stops writing - stuck in the kernel, as the fdinfo reader was - means the GPU is not
usable; so do recent kernel errors on this card and a failed or hung test. Then every GPU entry point takes its CPU
path (gpu_usable()). grain_auto jobs wait in GrainService.run until the GPU is usable again: owner 2026-10-16, "等
GPU 恢复".

Usable again needs a test that STARTED after the last error and passed. After a wedge that means after the rebind.
A child stuck in the kernel is abandoned, and a new one is started at most once every RESTART_EVERY_S.
  python3 -m app.gpuwatch --out FILE --node /dev/dri/renderD129 [--kernlog F] [--ffmpeg F --model CFG --sycl N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

STATUS_FILE = os.path.join(tempfile.gettempdir(), "av1t-gpuwatch.json")
# the status file is written every LOOP_S; older than STALE_S and the child is taken to be stuck
LOOP_S = 1.0
STALE_S = 15.0
# a test that has not finished after this long is a failed test - the 10-16 hang was a read that never returned
TEST_HUNG_S = 180.0
# kernel errors on this card keep the GPU out of use for at least this long, and until a later test passes
COOLOFF_S = 600.0
RESTART_EVERY_S = 600.0
# how long usable() waits for a fresh watcher's first verdict
SETTLE_S = 90.0

# kernel log lines that say this card is in trouble. "xe 0000:c6:00.0: [drm] ..." lines carry the PCI address; an
# RCU stall report does not, and on this box the only ones (10-16) came from the xe driver - a stall anywhere is
# reason enough to keep off the GPU for a while. Hung-task reports are left out: an NFS stall on the NAS makes them
# too, and would take the GPU away for nothing.
_WEDGED = re.compile(r"wedged", re.I)
_CARD_ERRORS = re.compile(r"Timedout job|job timed out|reset started|exec queue reset detected|CAT error|"
                          r"Fault response|G2H|-ETIME|\bBUG\b|Oops", re.I)
_BOX_ERRORS = re.compile(r"rcu_preempt detected (expedited )?stalls")


def render_pdev(node: str) -> Optional[str]:
    """The PCI address behind a render node, e.g. "0000:c6:00.0".

    fdinfo tags every DRM client with the device it belongs to, and this box has four render nodes (two 3090s, the
    B580, and the iGPU); without the address a memory reading would sum clients of cards this job never touches.
    """
    name = os.path.basename(node)
    try:
        return os.path.basename(os.path.realpath(f"/sys/class/drm/{name}/device"))
    except OSError:
        return None


def drm_vram_used_mb(pdev: Optional[str], root: str = "/proc") -> Optional[float]:
    """VRAM this process tree holds on `pdev`, in MB, or None if unreadable.

    The interface nvtop reads: every open DRM file exposes drm-total-vram0 in /proc/<pid>/fdinfo/<fd>, and a client
    that dup()s or forks its fd shows the same allocation under several fds, so the sum has to be deduplicated by
    drm-client-id or it counts the same bytes many times over.

    It sees only what this PID namespace can see. Inside the container that is our own ffmpegs - Plex's share of
    the card is invisible here, which is why the budget has to leave room for it rather than measure it.

    Run it in the watcher process only: reading a DRM fdinfo goes into the driver (see the module docstring).
    """
    total, seen = 0.0, set()
    try:
        pids = [d for d in os.listdir(root) if d.isdigit()]
    except OSError:
        return None
    found = False
    for pid in pids:
        d = f"{root}/{pid}/fdinfo"
        try:
            fds = os.listdir(d)
        except OSError:
            continue                      # the process exited, or is not ours
        for fd in fds:
            try:
                with open(f"{d}/{fd}") as fh:
                    text = fh.read(4096)
            except OSError:
                continue
            if "drm-client-id" not in text:
                continue
            found = True
            cid = dev = None
            vram = 0.0
            for line in text.splitlines():
                k, _, v = line.partition(":")
                v = v.strip()
                if k == "drm-client-id":
                    cid = v
                elif k == "drm-pdev":
                    dev = v
                elif k.startswith("drm-total-vram"):
                    vram += kib_to_mb(v)
            if cid is None or cid in seen:
                continue
            if pdev and dev and dev != pdev:
                continue
            seen.add(cid)
            total += vram
    return total if found or total == 0.0 else None


def kib_to_mb(v: str) -> float:
    """One fdinfo size field ("1234 KiB", "12 MiB", bare bytes) in MB."""
    parts = v.split()
    if not parts:
        return 0.0
    try:
        n = float(parts[0])
    except ValueError:
        return 0.0
    unit = (parts[1] if len(parts) > 1 else "KiB").lower()
    scale = {"b": 1 / 1048576.0, "kib": 1 / 1024.0, "mib": 1.0, "gib": 1024.0}.get(unit, 1 / 1024.0)
    return n * scale


# DRM_IOCTL_XE_DEVICE_QUERY: _IOWR('d', DRM_COMMAND_BASE + 0x00, struct drm_xe_device_query) - 40 bytes:
# u64 extensions, u32 query, u32 size, u64 data, u64 reserved[2]
_XE_QUERY = (3 << 30) | (40 << 16) | (ord("d") << 8) | 0x40
_XE_QUERY_MEM_REGIONS = 1
_XE_MEM_REGION_SIZE = 88       # struct drm_xe_mem_region: u16 class, u16 instance, u32 min_page, u64 total, used,
                               # cpu-visible size, cpu-visible used, u64 reserved[6]
_XE_MEM_VRAM = 1               # DRM_XE_MEM_REGION_CLASS_VRAM


def parse_mem_regions(raw: bytes) -> Optional[float]:
    """struct drm_xe_query_mem_regions (u32 count, u32 pad, regions) -> the VRAM regions' used bytes, in MB."""
    import struct
    if len(raw) < 8:
        return None
    n = struct.unpack_from("I", raw)[0]
    used, seen = 0, False
    for i in range(n):
        off = 8 + i * _XE_MEM_REGION_SIZE
        if off + 24 > len(raw):
            return None
        cls, _inst, _page, _total, u = struct.unpack_from("HHIQQ", raw, off)
        if cls == _XE_MEM_VRAM:
            used, seen = used + u, True
    return used / 1048576.0 if seen else None


class XeVram:
    """VRAM in use on the whole card, from the xe driver's memory-region query on one render node fd kept open
    (an open per reading would make and drop a DRM client every second). None when the query fails; the fd is
    reopened on the next reading."""

    def __init__(self, node: str):
        self.node, self.fd = node, None

    def read(self) -> Optional[float]:
        import ctypes
        import fcntl
        import struct
        try:
            if self.fd is None:
                self.fd = os.open(self.node, os.O_RDWR | os.O_CLOEXEC)
            q = bytearray(struct.pack("QIIQQQ", 0, _XE_QUERY_MEM_REGIONS, 0, 0, 0, 0))
            fcntl.ioctl(self.fd, _XE_QUERY, q, True)                 # size=0: the kernel says how much it needs
            size = struct.unpack_from("QII", q)[2]
            buf = ctypes.create_string_buffer(size)
            q = bytearray(struct.pack("QIIQQQ", 0, _XE_QUERY_MEM_REGIONS, size, ctypes.addressof(buf), 0, 0))
            fcntl.ioctl(self.fd, _XE_QUERY, q, True)
            return parse_mem_regions(buf.raw)
        except OSError:
            if self.fd is not None:
                try:
                    os.close(self.fd)
                except OSError:
                    pass
            self.fd = None
            return None


def render_driver(node: str) -> Optional[str]:
    """The kernel driver behind a render node ("xe", "i915", "nvidia" ...)."""
    try:
        return os.path.basename(os.path.realpath(f"/sys/class/drm/{os.path.basename(node)}/device/driver"))
    except OSError:
        return None


def classify(line: str, pdev: Optional[str]) -> Optional[str]:
    """A kernel log line -> "wedged", "error" or None (nothing to do with this card)."""
    if pdev and pdev in line and "xe" in line:
        if _WEDGED.search(line):
            return "wedged"
        if _CARD_ERRORS.search(line):
            return "error"
        return None
    if _BOX_ERRORS.search(line):
        return "error"
    return None


class KernelLog:
    """Follows the host's kern.log from where it was when the watcher started: rotation (a new file under the same
    name, or the file shrinking) is followed by reopening; a missing file is retried quietly."""

    def __init__(self, path: Optional[str]):
        self.path, self._fh, self._ino, self._buf = path, None, None, ""
        self._open(at_end=True)

    def _open(self, at_end: bool) -> None:
        if not self.path:
            return
        try:
            fh = open(self.path, "r", errors="replace")
            st = os.fstat(fh.fileno())
        except OSError:
            self._fh = None
            return
        if at_end:
            fh.seek(0, os.SEEK_END)
        self._fh, self._ino, self._buf = fh, st.st_ino, ""

    def lines(self) -> List[str]:
        if not self.path:
            return []
        if self._fh is None:
            self._open(at_end=False)
            if self._fh is None:
                return []
        out: List[str] = []
        try:
            chunk = self._fh.read(1 << 20)
            if chunk:
                self._buf += chunk
                *done, self._buf = self._buf.split("\n")
                out.extend(done)
            st = os.stat(self.path)
            if st.st_ino != self._ino or st.st_size < self._fh.tell():
                self._fh.close()
                self._open(at_end=False)      # rotated: the new file from its start
        except OSError:
            self._fh = None
        return out


class Health:
    """The state machine: ok / degraded / down, from kernel lines and test results (pure, for the tests)."""

    def __init__(self, now: float):
        self.started = now
        self.last_error_at: Optional[float] = None
        self.last_error = ""
        self.wedged_at: Optional[float] = None
        self.test_pass_started: Optional[float] = None    # start time of the latest passing test
        self.test_fail: Optional[Tuple[float, str]] = None
        self.test_running_since: Optional[float] = None

    def kernel(self, kind: str, line: str, now: float) -> None:
        self.last_error_at, self.last_error = now, line.strip()[-200:]
        if kind == "wedged":
            self.wedged_at = now

    def test_started(self, now: float) -> None:
        self.test_running_since = now

    def test_done(self, started: float, ok: bool, why: str, now: float) -> None:
        self.test_running_since = None
        if ok:
            self.test_pass_started, self.test_fail = started, None
        else:
            self.test_fail = (now, why)

    def state(self, now: float) -> Tuple[str, str]:
        if self.test_running_since is not None and now - self.test_running_since > TEST_HUNG_S:
            return "down", f"the GPU test has not returned for {now - self.test_running_since:.0f}s"
        passed_after = (lambda t: t is not None and self.test_pass_started is not None
                        and self.test_pass_started > t)
        if self.wedged_at is not None and not passed_after(self.wedged_at):
            return "down", "kernel: the device is wedged (needs a rebind)"
        if self.test_fail is not None:
            return "down", f"GPU test failed: {self.test_fail[1]}"
        if self.last_error_at is not None:
            if now - self.last_error_at < COOLOFF_S or not passed_after(self.last_error_at):
                return "degraded", f"kernel: {self.last_error}"
        if self.test_pass_started is None:
            return "starting", "no GPU test has passed yet"
        return "ok", ""

    def test_due(self, now: float, every: float) -> bool:
        if self.test_running_since is not None:
            return False
        if self.test_pass_started is None and self.test_fail is None:
            return True                                   # the first one, straight away
        last = max(self.test_pass_started or 0.0, self.test_fail[0] if self.test_fail else 0.0)
        if self.last_error_at is not None and now - self.last_error_at >= COOLOFF_S \
                and (self.test_pass_started is None or self.test_pass_started <= self.last_error_at):
            return now - last >= 60.0                     # after an error: re-test once a minute past the cool-off
        if self.test_fail is not None:
            return now - last >= 60.0
        return now - last >= every


def _run(args: List[str], timeout: float) -> Tuple[bool, str]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"{os.path.basename(args[0])} timed out after {timeout:.0f}s"
    except OSError as e:
        return False, str(e)
    if p.returncode != 0:
        tail = (p.stderr.strip().splitlines() or [f"exit {p.returncode}"])[-1]
        return False, tail[:200]
    return True, ""


def gpu_test(ffmpeg: str, node: str, model: Optional[str], sycl: int) -> Tuple[bool, str]:
    """A 3-frame av1_vaapi encode on the render node, then (scoring on the GPU) the SYCL preflight pair. Seconds."""
    ok, why = _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
                    "-init_hw_device", f"vaapi=va:{node}", "-filter_hw_device", "va",
                    "-f", "lavfi", "-i", "testsrc2=s=256x256:d=0.1:r=30",
                    "-vf", "format=nv12,hwupload", "-c:v", "av1_vaapi", "-f", "null", "-"], 60)
    if not ok:
        return False, f"VA-API encode: {why}"
    if model and sycl >= 0:
        from app.gpu import _score_cmd
        with tempfile.TemporaryDirectory() as d:
            log = f"{d}/s.json"
            ok, why = _run(_score_cmd(ffmpeg, model, sycl, log), 120)
            if not ok:
                return False, f"SYCL score: {why}"
            try:
                json.load(open(log))["pooled_metrics"]["vmaf"]["mean"]
            except (OSError, ValueError, KeyError):
                return False, "SYCL score: no result"
    return True, ""


def _write(path: str, status: Dict[str, Any]) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(status, fh)
    os.replace(tmp, path)


def child_main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="app.gpuwatch")
    ap.add_argument("--out", default=STATUS_FILE)
    ap.add_argument("--node", required=True)
    ap.add_argument("--kernlog", default="")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--model", default="")
    ap.add_argument("--sycl", type=int, default=-1)
    ap.add_argument("--test-every", type=float, default=600.0)
    a = ap.parse_args(argv)
    pdev = render_pdev(a.node)
    xe = XeVram(a.node) if render_driver(a.node) == "xe" else None
    parent = os.getppid()
    klog = KernelLog(a.kernlog or None)
    h = Health(time.time())
    lock = threading.Lock()
    seq = 0

    def test() -> None:
        started = time.time()
        ok, why = gpu_test(a.ffmpeg, a.node, a.model or None, a.sycl)
        with lock:
            h.test_done(started, ok, why, time.time())

    while True:
        if os.getppid() != parent:
            return                                        # the app is gone; so is the reason to watch
        now = time.time()
        for line in klog.lines():
            kind = classify(line, pdev)
            if kind:
                with lock:
                    h.kernel(kind, line, now)
        with lock:
            if h.test_due(now, a.test_every):
                h.test_started(now)
                threading.Thread(target=test, daemon=True).start()
            state, why = h.state(now)
            snap = {"last_error_at": h.last_error_at, "last_error": h.last_error, "wedged_at": h.wedged_at,
                    "test_pass_started": h.test_pass_started, "test_fail": h.test_fail,
                    "test_running_since": h.test_running_since}
        # xe: the device query (no buffer objects touched); else the fdinfo sum, whose xe read hung on 10-16
        vram = xe.read() if xe is not None else drm_vram_used_mb(pdev)
        seq += 1
        _write(a.out, {"pid": os.getpid(), "t": time.time(), "seq": seq, "state": state, "why": why,
                       "vram_mb": vram, "vram_from": "xe query" if xe is not None else "fdinfo", "pdev": pdev,
                       "node": a.node, "kernlog": bool(klog.path), **snap})
        time.sleep(LOOP_S)


# --- in the app ------------------------------------------------------------------------------------------------


class GpuWatch:
    """Runs and supervises the watcher process; answers usable() from its status file. Never waits on the child."""

    def __init__(self, args: List[str], out: str = STATUS_FILE):
        self.args, self.out = args, out
        self._proc: Optional[subprocess.Popen] = None
        self._started_at = 0.0
        self._lock = threading.Lock()
        self._last: Tuple[Optional[bool], str] = (None, "")

    def start(self) -> None:
        self._spawn()
        threading.Thread(target=self._supervise, daemon=True, name="gpuwatch-supervisor").start()

    def _spawn(self) -> None:
        try:
            os.unlink(self.out)                           # an old child's file must not pass for this one
        except OSError:
            pass
        self._proc = subprocess.Popen([sys.executable, "-m", "app.gpuwatch", "--out", self.out, *self.args],
                                      stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                      start_new_session=True)
        self._started_at = time.monotonic()
        logger.info("gpuwatch: watcher process {} started", self._proc.pid)

    def _supervise(self) -> None:
        while True:
            time.sleep(5.0)
            with self._lock:
                dead = self._proc is not None and self._proc.poll() is not None
                st = self._read()
                stale = st is None or time.time() - float(st.get("t", 0)) > STALE_S
                up_for = time.monotonic() - self._started_at
                if dead or (stale and up_for > 60.0 and up_for > RESTART_EVERY_S):
                    why = f"exited ({self._proc.returncode})" if dead else "stopped writing (stuck)"
                    logger.warning("gpuwatch: watcher process {} {}; starting another", self._proc.pid, why)
                    if not dead:
                        try:                              # one stuck in the kernel may not die; it is left behind
                            self._proc.kill()
                        except OSError:
                            pass
                    self._spawn()

    def _read(self) -> Optional[Dict[str, Any]]:
        try:
            with open(self.out) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def status(self) -> Dict[str, Any]:
        st = self._read()
        if st is None:
            return {"state": "starting", "why": "the watcher has not reported yet", "usable": False}
        age = time.time() - float(st.get("t", 0))
        if age > STALE_S:
            st = dict(st, state="down", why=f"the GPU watcher has not reported for {age:.0f}s (stuck in the driver?)")
        st["usable"] = st.get("state") == "ok"
        st["age_s"] = round(age, 1)
        return st

    def usable(self) -> Tuple[bool, str]:
        st = self.status()
        # right after the start (the app's, or a replaced child's) the first test takes a few seconds; a job that
        # asked then would run all of it on the CPU, so the answer waits for it, up to SETTLE_S
        while st.get("state") == "starting" and time.monotonic() - self._started_at < SETTLE_S:
            time.sleep(0.5)
            st = self.status()
        ok = bool(st["usable"])
        why = "" if ok else str(st.get("why") or st.get("state"))
        with self._lock:
            if ok != self._last[0]:
                if ok:
                    logger.info("gpuwatch: the GPU is usable{}", "" if self._last[0] is None else " again")
                elif st.get("state") != "starting":
                    logger.warning("gpuwatch: GPU not usable ({}); GPU work goes to the CPU, grain_auto waits", why)
                self._last = (ok, why)
        return ok, why

    def vram_mb(self) -> Optional[float]:
        st = self._read()
        if st is None or time.time() - float(st.get("t", 0)) > 5.0:
            return None
        v = st.get("vram_mb")
        return float(v) if v is not None else None


_WATCH: Optional[GpuWatch] = None


def start(settings) -> Optional[GpuWatch]:
    """From the app's start-up: the watcher for the first render node, when there is one and it is switched on."""
    global _WATCH
    opt = settings.transcode.optimizer
    nodes = sorted(f"/dev/dri/{n}" for n in (os.listdir("/dev/dri") if os.path.isdir("/dev/dri") else [])
                   if n.startswith("renderD"))
    if not opt.gpu_watch or not nodes:
        logger.info("gpuwatch: off ({})", "gpu_watch: false" if not opt.gpu_watch else "no render node")
        return None
    try:
        ffmpeg = settings.tool_path("ffmpeg")
    except FileNotFoundError:
        ffmpeg = "ffmpeg"
    args = ["--node", nodes[0], "--kernlog", opt.gpu_watch_kernel_log or "", "--ffmpeg", ffmpeg,
            "--test-every", str(float(opt.gpu_watch_test_s))]
    if int(opt.vmaf_sycl_device) >= 0:
        args += ["--model", f"path={opt.vmaf_model}", "--sycl", str(int(opt.vmaf_sycl_device))]
    _WATCH = GpuWatch(args)
    _WATCH.start()
    return _WATCH


def gpu_usable() -> Tuple[bool, str]:
    """(usable, why) for every GPU entry point. Without a watcher (switched off, no render node, tests) the GPU is
    taken as usable, which is how the app behaved before."""
    return (True, "") if _WATCH is None else _WATCH.usable()


def vram_mb() -> Optional[float]:
    return None if _WATCH is None else _WATCH.vram_mb()


def active() -> bool:
    return _WATCH is not None


def wait_usable(cancel_flag: Optional[Callable[[], bool]], what: str, poll: float = 30.0) -> None:
    """Block until the GPU is usable (grain_auto's steps, owner 2026-10-16: "等 GPU 恢复"); a cancel ends the wait."""
    ok, why = gpu_usable()
    if ok:
        return
    logger.warning("grain_auto: waiting for the GPU before {} ({})", what, why)
    t0 = last_log = time.monotonic()
    while True:
        if cancel_flag is not None and cancel_flag():
            raise InterruptedError(f"cancelled while waiting for the GPU before {what}")
        time.sleep(poll)
        ok, why = gpu_usable()
        if ok:
            logger.info("grain_auto: the GPU is usable again after {:.0f} min; {} goes on",
                        (time.monotonic() - t0) / 60, what)
            return
        if time.monotonic() - last_log > 1800:
            logger.warning("grain_auto: still waiting for the GPU before {} ({:.0f} min; {})",
                           what, (time.monotonic() - t0) / 60, why)
            last_log = time.monotonic()


if __name__ == "__main__":
    child_main()
