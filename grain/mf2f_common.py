"""Shared pieces for the MF2F-trained student: the fresh-grain curve (same definition as mf2f_prep.py) and the noise
map (as mf2f_train.py), on the GPU.
  curve = fresh_curve(pairs)          pairs: iterable of (f0, f1) luma tensors in 10-bit codes
  nm = noise_map(y_centre, curve, K)  (1, 1, H/2, W/2), packed resolution
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

EDGES = [0, 16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256]
CENT = [(a + b) / 2 for a, b in zip(EDGES, EDGES[1:])]
_K = {}


def blur(x, sg):
    key = (sg, x.device)
    if key not in _K:
        r = int(math.ceil(3 * sg))
        t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
        k = torch.exp(-t * t / (2 * sg * sg))
        _K[key] = (k / k.sum(), r)
    k, r = _K[key]
    sh = x.shape
    x = x.reshape(-1, 1, *sh[-2:])
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return x.reshape(sh)


def fresh_curve(pairs, min_px=2000):
    """Std of (f1 - f0) / sqrt 2 on static flat pixels per 8-bit luma bin; None where too few pixels."""
    acc = [[0.0, 0] for _ in EDGES[:-1]]
    for f0, f1 in pairs:
        b0, b1 = blur(f0, 2.0), blur(f1, 2.0)
        gy, gx = torch.gradient(b0)
        m = (blur(torch.sqrt(gx * gx + gy * gy), 2.0) < 1.5) & ((b1 - b0).abs() < 0.5)
        d = (f1 - f0) / math.sqrt(2)
        idx = b0 / 4
        for k, (lo, hi) in enumerate(zip(EDGES, EDGES[1:])):
            mm = m & (idx >= lo) & (idx < hi)
            n = int(mm.sum())
            if n:
                dd = d[mm]
                acc[k][0] += float(((dd - dd.mean()) ** 2).sum())
                acc[k][1] += n
    return [round(math.sqrt(s / n), 3) if n >= min_px else None for s, n in acc]


def fill(c, default=None):
    ok = [i for i, v in enumerate(c) if v is not None]
    if not ok:
        return default
    return [c[min(ok, key=lambda j: abs(j - i))] for i in range(len(c))]


def noise_map(yc, curve, K):
    """yc (H, W) centre luma in codes -> (1, 1, H/2, W/2) map K x std(brightness) / 876 (torch interp on GPU)."""
    lum = F.avg_pool2d(blur(yc, 2.0)[None, None], 2)[0, 0] / 4
    xs = torch.tensor(CENT, device=yc.device, dtype=torch.float32)
    ys = torch.tensor(curve, device=yc.device, dtype=torch.float32)
    i = torch.searchsorted(xs, lum.clamp(xs[0], xs[-1]).contiguous()).clamp(1, len(CENT) - 1)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    s = y0 + (y1 - y0) * ((lum.clamp(xs[0], xs[-1]) - x0) / (x1 - x0))
    return (K * s / 876.0)[None, None]


def grain_std(yc, curve):
    """yc (H, W) luma in codes -> (H, W) the file's fresh grain std (codes) at each pixel's brightness."""
    lum = blur(yc, 2.0) / 4
    xs = torch.tensor(CENT, device=yc.device, dtype=torch.float32)
    ys = torch.tensor(curve, device=yc.device, dtype=torch.float32)
    lc = lum.clamp(xs[0], xs[-1]).contiguous()
    i = torch.searchsorted(xs, lc).clamp(1, len(CENT) - 1)
    return ys[i - 1] + (ys[i] - ys[i - 1]) * ((lc - xs[i - 1]) / (xs[i] - xs[i - 1]))


def limit_removal(src, out, sig, c=1.3, win=9, ds=1):
    """Never remove much more than the grain (owner, test-07: v3 wiped shadow detail and dark make-up): where the local
    rms of (src - out) over win x win exceeds c x the grain std there, the removal is scaled down to that, so texture
    taken along with the grain comes back. src, out, sig (H, W) in codes."""
    r = src - out
    if ds > 1:                                           # the local rms is smooth: ds x ds means first, window win / ds
        e = F.avg_pool2d((r * r)[None, None], ds, ds, ceil_mode=True)
        k = max(1, (win // ds) | 1)
        e = F.avg_pool2d(e, k, 1, k // 2, count_include_pad=False)
        rms = torch.sqrt(F.interpolate(e, size=r.shape, mode="bilinear", align_corners=False)[0, 0])
        sc = (c * sig / rms.clamp(min=1e-3)).clamp(max=1.0)
        return src - r * sc
    rms = torch.sqrt(F.avg_pool2d((r * r)[None, None], win, 1, win // 2, count_include_pad=False)[0, 0])
    return src - r * (c * sig / rms.clamp(min=1e-3)).clamp(max=1.0)
