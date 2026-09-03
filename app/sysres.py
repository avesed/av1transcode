"""CPU and memory limits as seen from INSIDE this cgroup, not the whole box.

`os.cpu_count()` and /proc/meminfo answer for the machine, which is the wrong
question for a process running under a limit: a container capped at 8 CPUs
still reads 32, and MemTotal counts memory other tenants already hold. The
optimizer sizes its encoder pool from these numbers, so reading the machine
instead of our own slice is wrong in both directions - it over-commits against
whatever else shares the box, and under-uses a limit that happens to be
generous.

Everything here degrades to the machine-wide answer when there is no limit, so
a bare-metal install behaves as before.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_CGROUP_ROOT = Path("/sys/fs/cgroup")
_GB = 1024.0 ** 3


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except (OSError, ValueError):
        return None


def _cgroup_v2_dir() -> Optional[Path]:
    """The cgroup v2 directory holding OUR controller files.

    With a private cgroup namespace (docker's default on modern kernels) our
    cgroup is mounted at the root, so the files sit directly in /sys/fs/cgroup.
    With the host namespace we see the whole tree and have to follow the path
    from /proc/self/cgroup instead.
    """
    if not (_CGROUP_ROOT / "cgroup.controllers").exists():
        return None
    rel = ""
    for line in (_read(Path("/proc/self/cgroup")) or "").splitlines():
        if line.startswith("0::"):
            rel = line[3:].strip()
            break
    if rel and rel != "/":
        nested = _CGROUP_ROOT / rel.lstrip("/")
        if nested.is_dir():
            return nested
    return _CGROUP_ROOT


def _meminfo_gb(key: str) -> Optional[float]:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(key):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except (OSError, ValueError):
        return None
    return None


def cpu_budget() -> float:
    """Cores this process may actually use.

    The minimum of the cgroup CPU quota and the affinity mask. Quota is what
    `docker run --cpus` sets, and nothing in Python reports it: os.cpu_count()
    ignores it entirely and sched_getaffinity only sees `--cpuset-cpus`.
    """
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
        else (os.cpu_count() or 1)
    quota: Optional[float] = None
    v2 = _cgroup_v2_dir()
    if v2 is not None:
        raw = _read(v2 / "cpu.max")            # "<quota|max> <period>"
        if raw:
            parts = raw.split()
            if len(parts) == 2 and parts[0] != "max":
                try:
                    quota = int(parts[0]) / int(parts[1])
                except (ValueError, ZeroDivisionError):
                    quota = None
    else:
        q = _read(_CGROUP_ROOT / "cpu" / "cpu.cfs_quota_us")
        p = _read(_CGROUP_ROOT / "cpu" / "cpu.cfs_period_us")
        try:
            if q and p and int(q) > 0:
                quota = int(q) / int(p)
        except (ValueError, ZeroDivisionError):
            quota = None
    if quota and quota > 0:
        return max(1.0, min(quota, float(affinity)))
    return float(max(1, affinity))


def memory_limit_gb() -> Optional[float]:
    """Hard memory ceiling for this cgroup, or None when it is unlimited.

    A limit larger than the machine's own RAM is treated as unlimited: that is
    how "max" shows up on cgroup v1 (a sentinel near 2^63).
    """
    total = _meminfo_gb("MemTotal:") or 0.0
    raw: Optional[str]
    v2 = _cgroup_v2_dir()
    if v2 is not None:
        raw = _read(v2 / "memory.max")
        if raw in (None, "max"):
            return None
    else:
        raw = _read(_CGROUP_ROOT / "memory" / "memory.limit_in_bytes")
        if raw is None:
            return None
    try:
        limit = int(raw) / _GB
    except (TypeError, ValueError):
        return None
    if limit <= 0 or (total and limit >= total):
        return None
    return limit


def memory_in_use_gb() -> Optional[float]:
    """Memory currently charged to this cgroup (page cache included)."""
    v2 = _cgroup_v2_dir()
    path = (v2 / "memory.current") if v2 is not None \
        else (_CGROUP_ROOT / "memory" / "memory.usage_in_bytes")
    raw = _read(path)
    try:
        return int(raw) / _GB if raw is not None else None
    except (TypeError, ValueError):
        return None


def memory_available_gb() -> float:
    """Memory we could allocate right now without pushing anything to swap.

    Inside a limited cgroup that is the headroom below the limit; otherwise the
    kernel's own MemAvailable, which already discounts what other processes
    hold and adds back reclaimable page cache. MemTotal is the wrong number for
    this and is only the last resort.
    """
    limit = memory_limit_gb()
    if limit is not None:
        used = memory_in_use_gb() or 0.0
        return max(0.5, limit - used)
    avail = _meminfo_gb("MemAvailable:")
    if avail is not None:
        return max(0.5, avail)
    return max(0.5, (_meminfo_gb("MemTotal:") or 8.0) * 0.75)
