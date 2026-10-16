"""The GPU watcher (app/gpuwatch.py): its reading of the kernel log, its verdict, and what the app does with it."""
import json
import os
import time

import pytest

from app import gpuwatch as gw

PDEV = "0000:c6:00.0"
# 2026-10-16, from the box's kern.log
RESET = "2026-10-16T03:29:59.772789-04:00 homeserver kernel: xe 0000:c6:00.0: [drm] Tile0: GT0: reset started"
TIMEDOUT = ("2026-10-16T03:29:59.786318-04:00 homeserver kernel: xe 0000:c6:00.0: [drm] Tile0: GT0: Timedout job: "
            "seqno=4294967185, lrc_seqno=4294967185, guc_id=15, flags=0x20 in ffmpeg-real [2759510]")
WEDGED = ("2026-10-14T19:34:11.850906-04:00 homeserver kernel: xe 0000:c6:00.0: [drm] *ERROR* CRITICAL: Xe has "
          "declared device 0000:c6:00.0 as wedged.")
RCU = "2026-10-16T05:07:26.044000-04:00 homeserver kernel: rcu: INFO: rcu_preempt detected stalls on CPUs/tasks:"
QUIET = [
    "2026-10-14T19:38:51.486159-04:00 homeserver kernel: xe 0000:c6:00.0: [drm] Registered 4 planes with drm panic",
    "2026-10-16T05:09:30.718586-04:00 homeserver kernel: INFO: task fc0:3341038 blocked for more than 122 seconds.",
    "2026-10-16T05:00:00.000000-04:00 homeserver kernel: nvidia 0000:81:00.0: Timedout job something",
    "homeserver kernel: Call Trace:",
]


def test_kernel_lines_are_sorted_into_wedged_errors_and_noise():
    assert gw.classify(WEDGED, PDEV) == "wedged"
    assert gw.classify(RESET, PDEV) == "error"
    assert gw.classify(TIMEDOUT, PDEV) == "error"
    assert gw.classify(RCU, PDEV) == "error"
    assert [gw.classify(line, PDEV) for line in QUIET] == [None] * len(QUIET)


def test_an_error_keeps_the_gpu_out_until_the_cooloff_and_a_later_test():
    h = gw.Health(0.0)
    assert h.state(1.0)[0] == "starting" and h.test_due(1.0, 600)
    h.test_started(1.0)
    h.test_done(1.0, True, "", 3.0)
    assert h.state(4.0) == ("ok", "")
    h.kernel("error", RESET, 100.0)
    assert h.state(101.0)[0] == "degraded"
    assert not h.test_due(200.0, 600)                       # still in the cool-off
    assert h.state(100.0 + gw.COOLOFF_S + 1)[0] == "degraded"   # cooled off, but nothing has passed since
    t = 100.0 + gw.COOLOFF_S + 61
    assert h.test_due(t, 600)
    h.test_started(t)
    h.test_done(t, True, "", t + 2)
    assert h.state(t + 3) == ("ok", "")


def test_a_wedge_stays_down_until_a_test_passes_after_it():
    h = gw.Health(0.0)
    h.test_started(0.0)
    h.test_done(0.0, True, "", 1.0)
    h.kernel("wedged", WEDGED, 10.0)
    assert h.state(10.0 + gw.COOLOFF_S * 3)[0] == "down"
    h.test_started(5000.0)
    h.test_done(5000.0, False, "VA-API encode: vaInitialize failed", 5001.0)
    assert h.state(5002.0) == ("down", "kernel: the device is wedged (needs a rebind)")   # the wedge says more
    assert h.test_due(5062.0, 600)                          # a failed test is retried once a minute
    h.test_started(5062.0)
    h.test_done(5062.0, True, "", 5064.0)                   # the rebind happened
    assert h.state(5065.0) == ("ok", "")


def test_a_test_that_never_returns_is_a_failed_test():
    h = gw.Health(0.0)
    h.test_started(10.0)
    assert h.state(10.0 + gw.TEST_HUNG_S - 1)[0] == "starting"
    assert h.state(10.0 + gw.TEST_HUNG_S + 1)[0] == "down"


def test_the_kernel_log_is_followed_from_the_end_and_across_rotation(tmp_path):
    log = tmp_path / "kern.log"
    log.write_text("old line before the watcher\n")
    k = gw.KernelLog(str(log))
    assert k.lines() == []
    with open(log, "a") as fh:
        fh.write(RESET + "\n" + "half a li")
    assert k.lines() == [RESET]
    with open(log, "a") as fh:
        fh.write("ne\n")
    assert k.lines() == ["half a line"]
    os.rename(log, tmp_path / "kern.log.1")                 # logrotate: a new file under the old name
    log.write_text(TIMEDOUT + "\n")
    assert k.lines() == []                                  # the rotation is seen
    assert k.lines() == [TIMEDOUT]                          # and the new file is read from its start
    assert gw.KernelLog(str(tmp_path / "missing.log")).lines() == []


def test_a_watcher_that_stops_writing_means_no_gpu(tmp_path):
    out = tmp_path / "status.json"
    w = gw.GpuWatch([], out=str(out))
    w._started_at = time.monotonic() - 1000                 # past the start-up wait
    out.write_text(json.dumps({"t": time.time(), "state": "ok", "why": "", "vram_mb": 123.0}))
    assert w.usable() == (True, "") and w.vram_mb() == 123.0
    out.write_text(json.dumps({"t": time.time() - 60, "state": "ok", "why": "", "vram_mb": 123.0}))
    ok, why = w.usable()
    assert not ok and "has not reported" in why
    assert w.vram_mb() is None
    out.write_text(json.dumps({"t": time.time(), "state": "degraded", "why": "kernel: reset started"}))
    assert w.usable() == (False, "kernel: reset started")


class _Fake:
    def __init__(self, ok, why="kernel: reset started"):
        self.ok, self.why = ok, why

    def usable(self):
        return (self.ok, "" if self.ok else self.why)

    def vram_mb(self):
        return 42.0

    def status(self):
        return {"state": "ok" if self.ok else "down"}


@pytest.fixture()
def watch(monkeypatch):
    f = _Fake(True)
    monkeypatch.setattr(gw, "_WATCH", f)
    return f


def test_without_a_watcher_the_gpu_is_taken_as_usable(monkeypatch):
    monkeypatch.setattr(gw, "_WATCH", None)
    assert gw.gpu_usable() == (True, "") and gw.vram_mb() is None and not gw.active()
    gw.wait_usable(None, "nothing")                         # returns at once


def test_waiting_for_the_gpu_ends_when_it_is_back_or_on_a_cancel(watch, monkeypatch):
    watch.ok = False
    calls = []
    monkeypatch.setattr(gw.time, "sleep", lambda s: (calls.append(s), setattr(watch, "ok", len(calls) == 3)))
    gw.wait_usable(None, "the denoise step", poll=30)
    assert calls == [30, 30, 30]
    watch.ok = False                                        # and now it stays down: only the cancel ends the wait
    with pytest.raises(InterruptedError):
        gw.wait_usable(lambda: len(calls) >= 5, "the denoise step")
    assert len(calls) == 5
