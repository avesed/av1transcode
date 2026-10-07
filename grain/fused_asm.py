"""Fused Triton (PyTorch XPU) version of student_v3.assemble_fast() fed with LOW-resolution flows.

  packed, conf = assemble_fused(Y, U, V, lowflows, f, tau=6.0)

Y (1, 7, H, W), U / V (1, 7, H/2, W/2) float32 codes on xpu, H and W multiples of 8; lowflows: 6 tensors
(1, 2, H/f, W/f) float32 in low-res pixels for the neighbours [0, 1, 2, 4, 5, 6] (None = zero flow); f in 1, 2, 4.
Equals, within fp32 rounding, assemble_fast(Y, U, V, [F.interpolate(lf, (H, W), "bilinear", align_corners=False) * f ...], tau):
packed (1, 7, 6, H/2, W/2) in [0, 1] and conf (1, 6, H/2, W/2).

Pass 1 (one launch per frame): each program takes a BH x BW tile of packed pixels; for the four full-res luma positions
of a packed pixel it upsamples the low-res flow on the fly (bilinear, align_corners=False, like F.interpolate),
reproduces warp()'s grid normalisation round trip and grid_sample's bicubic (A = -0.75, border padding: every tap index
clamped) at that position, writes the 4 pixel_unshuffle channels / 1023 and d = avg_pool2d(wy - Yc, 2); the chroma flow
is the 2 x 2 mean of those four upsampled flows / 2 (= avg_pool2d(full_flow, 2) / 2) and U / V are sampled bilinearly
(border padding: the coordinate itself clipped). Full-res flows and warped luma never touch global memory.
Pass 2: separable 7-tap Gaussian (sigma 1, reflect padding) of d, then exp(-|.| / tau) -> conf; two small kernels.

PyTorch's true division by a Python scalar on XPU multiplies by the fp32 reciprocal (RECIP=True reproduces that);
the kernels are launched with enable_fp_fusion=False (measured faster here; the match is 0.0009 vs 0.0005 codes).
Measured 2026-10-07, B580, ST 4K frame, f = 4 (fused_asm_test.py): 3.3 ms (of which blur + exp 0.5 ms) against 47.1 ms for
6 x F.interpolate (4.6) + assemble_fast (42.3); packed max |diff| 0.0009 codes, mean 0.00003, conf max 1.4e-5; peak
1.19 GiB allocated (0.47 GiB above the resident inputs). f = 2: 3.5 ms, f = 1: 3.2 ms. Tile 8 x 32 / 4 warps was the best
of the sweep (register-heavy bicubic: bigger tiles, 2 warps or grf_mode 256 are all slower).
"""
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _clampi(i, n):
    return tl.minimum(tl.maximum(i, 0), n - 1)


@triton.jit
def _pos(coord, flow, inv_m1, m1, RECIP: tl.constexpr):
    """warp(): g = (coord + flow) / (size - 1) * 2 - 1; grid_sample (align_corners=True): ((g + 1) / 2) * (size - 1)."""
    s = coord + flow
    if RECIP:
        g = s * inv_m1 * 2.0 - 1.0
    else:
        g = s / m1 * 2.0 - 1.0
    return (g + 1.0) * 0.5 * m1


@triton.jit
def _cubic_coeffs(t):
    """get_cubic_upsample_coefficients with A = -0.75: cubic_convolution2(t + 1), cubic_convolution1(t),
    cubic_convolution1(1 - t), cubic_convolution2(2 - t)."""
    a = t + 1.0
    c0 = ((-0.75 * a + 3.75) * a - 6.0) * a + 3.0
    c1 = (1.25 * t - 2.25) * t * t + 1.0
    u = 1.0 - t
    c2 = (1.25 * u - 2.25) * u * u + 1.0
    b = u + 1.0
    c3 = ((-0.75 * b + 3.75) * b - 6.0) * b + 3.0
    return c0, c1, c2, c3


@triton.jit
def _bicubic(p, ix, iy, H, W):
    """grid_sample bicubic, padding_mode=border: 4 x 4 taps around floor(ix), floor(iy), each clamped into the image."""
    fx = tl.floor(ix)
    fy = tl.floor(iy)
    tx = ix - fx
    ty = iy - fy
    x0 = tl.minimum(tl.maximum(fx, -8.0), W + 8.0).to(tl.int32)      # finite before the int conversion
    y0 = tl.minimum(tl.maximum(fy, -8.0), H + 8.0).to(tl.int32)
    cx0, cx1, cx2, cx3 = _cubic_coeffs(tx)
    cy0, cy1, cy2, cy3 = _cubic_coeffs(ty)
    xa = _clampi(x0 - 1, W)
    xb = _clampi(x0, W)
    xc = _clampi(x0 + 1, W)
    xd = _clampi(x0 + 2, W)
    ra = _clampi(y0 - 1, H) * W
    rb = _clampi(y0, H) * W
    rc = _clampi(y0 + 1, H) * W
    rd = _clampi(y0 + 2, H) * W
    r0 = tl.load(p + ra + xa) * cx0 + tl.load(p + ra + xb) * cx1 + tl.load(p + ra + xc) * cx2 + tl.load(p + ra + xd) * cx3
    r1 = tl.load(p + rb + xa) * cx0 + tl.load(p + rb + xb) * cx1 + tl.load(p + rb + xc) * cx2 + tl.load(p + rb + xd) * cx3
    r2 = tl.load(p + rc + xa) * cx0 + tl.load(p + rc + xb) * cx1 + tl.load(p + rc + xc) * cx2 + tl.load(p + rc + xd) * cx3
    r3 = tl.load(p + rd + xa) * cx0 + tl.load(p + rd + xb) * cx1 + tl.load(p + rd + xc) * cx2 + tl.load(p + rd + xd) * cx3
    return r0 * cy0 + r1 * cy1 + r2 * cy2 + r3 * cy3


@triton.jit
def _bilinear(p, ix, iy, H, W):
    """grid_sample bilinear, padding_mode=border: the coordinate is clipped to [0, size - 1] first."""
    ix = tl.minimum(tl.maximum(ix, 0.0), (W - 1).to(tl.float32))
    iy = tl.minimum(tl.maximum(iy, 0.0), (H - 1).to(tl.float32))
    fx = tl.floor(ix)
    fy = tl.floor(iy)
    x0 = fx.to(tl.int32)
    y0 = fy.to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    nw = (x1.to(tl.float32) - ix) * (y1.to(tl.float32) - iy)
    ne = (ix - fx) * (y1.to(tl.float32) - iy)
    sw = (x1.to(tl.float32) - ix) * (iy - fy)
    se = (ix - fx) * (iy - fy)
    x0 = _clampi(x0, W)
    y0 = _clampi(y0, H)
    x1 = _clampi(x1, W)
    y1 = _clampi(y1, H)
    r0 = y0 * W
    r1 = y1 * W
    return tl.load(p + r0 + x0) * nw + tl.load(p + r0 + x1) * ne + tl.load(p + r1 + x0) * sw + tl.load(p + r1 + x1) * se


@triton.jit
def _upflow(fl, xf, yf, hl, wl, F: tl.constexpr):
    """(fx, fy) = F.interpolate(lowflow, (H, W), "bilinear", align_corners=False)[.., yf, xf] * F; xf, yf int tensors."""
    if F == 1:
        o = yf * wl + xf
        fx = tl.load(fl + o)
        fy = tl.load(fl + hl * wl + o)
        return fx, fy
    else:
        sx = tl.maximum((xf.to(tl.float32) + 0.5) * (1.0 / F) - 0.5, 0.0)
        sy = tl.maximum((yf.to(tl.float32) + 0.5) * (1.0 / F) - 0.5, 0.0)
        x1 = tl.minimum(sx.to(tl.int32), wl - 1)
        y1 = tl.minimum(sy.to(tl.int32), hl - 1)
        x1p = tl.where(x1 < wl - 1, 1, 0)
        y1p = tl.where(y1 < hl - 1, 1, 0)
        lx1 = sx - x1.to(tl.float32)
        lx0 = 1.0 - lx1
        ly1 = sy - y1.to(tl.float32)
        ly0 = 1.0 - ly1
        o00 = y1 * wl + x1
        o01 = o00 + x1p
        o10 = o00 + y1p * wl
        o11 = o10 + x1p
        fx = ly0 * (lx0 * tl.load(fl + o00) + lx1 * tl.load(fl + o01)) + ly1 * (lx0 * tl.load(fl + o10) + lx1 * tl.load(fl + o11))
        q = fl + hl * wl
        fy = ly0 * (lx0 * tl.load(q + o00) + lx1 * tl.load(q + o01)) + ly1 * (lx0 * tl.load(q + o10) + lx1 * tl.load(q + o11))
        return fx * F, fy * F


@triton.jit
def _asm_kernel(y_ptr, u_ptr, v_ptr, yc_ptr, fl_ptr, out_ptr, d_ptr,
                H, W, Hh, Wh, hl, wl, inv_wm1, inv_hm1, inv_whm1, inv_hhm1,
                F: tl.constexpr, HAS_FLOW: tl.constexpr, CENTRE: tl.constexpr, RECIP: tl.constexpr,
                BH: tl.constexpr, BW: tl.constexpr):
    pid = tl.program_id(0)
    nx = tl.cdiv(Wh, BW)
    ty = pid // nx
    tx = pid - ty * nx
    yo = ty * BH + tl.arange(0, BH)[:, None]
    xo = tx * BW + tl.arange(0, BW)[None, :]
    m = (yo < Hh) & (xo < Wh)
    HW = Hh * Wh
    ob = out_ptr + yo * Wh + xo
    xi0 = 2 * xo
    yi0 = 2 * yo
    if CENTRE:
        y00 = tl.load(y_ptr + yi0 * W + xi0, mask=m, other=0.0)
        y01 = tl.load(y_ptr + yi0 * W + xi0 + 1, mask=m, other=0.0)
        y10 = tl.load(y_ptr + (yi0 + 1) * W + xi0, mask=m, other=0.0)
        y11 = tl.load(y_ptr + (yi0 + 1) * W + xi0 + 1, mask=m, other=0.0)
        u = tl.load(u_ptr + yo * Wh + xo, mask=m, other=0.0)
        v = tl.load(v_ptr + yo * Wh + xo, mask=m, other=0.0)
    else:
        xf0 = xi0.to(tl.float32)
        xf1 = xf0 + 1.0
        yf0 = yi0.to(tl.float32)
        yf1 = yf0 + 1.0
        if HAS_FLOW:
            fx00, fy00 = _upflow(fl_ptr, xi0, yi0, hl, wl, F)
            fx01, fy01 = _upflow(fl_ptr, xi0 + 1, yi0, hl, wl, F)
            fx10, fy10 = _upflow(fl_ptr, xi0, yi0 + 1, hl, wl, F)
            fx11, fy11 = _upflow(fl_ptr, xi0 + 1, yi0 + 1, hl, wl, F)
        else:
            fx00 = tl.zeros([BH, BW], tl.float32)
            fy00 = fx00
            fx01 = fx00
            fy01 = fx00
            fx10 = fx00
            fy10 = fx00
            fx11 = fx00
            fy11 = fx00
        wm1 = (W - 1).to(tl.float32)
        hm1 = (H - 1).to(tl.float32)
        y00 = _bicubic(y_ptr, _pos(xf0, fx00, inv_wm1, wm1, RECIP), _pos(yf0, fy00, inv_hm1, hm1, RECIP), H, W)
        y01 = _bicubic(y_ptr, _pos(xf1, fx01, inv_wm1, wm1, RECIP), _pos(yf0, fy01, inv_hm1, hm1, RECIP), H, W)
        y10 = _bicubic(y_ptr, _pos(xf0, fx10, inv_wm1, wm1, RECIP), _pos(yf1, fy10, inv_hm1, hm1, RECIP), H, W)
        y11 = _bicubic(y_ptr, _pos(xf1, fx11, inv_wm1, wm1, RECIP), _pos(yf1, fy11, inv_hm1, hm1, RECIP), H, W)
        yc00 = tl.load(yc_ptr + yi0 * W + xi0, mask=m, other=0.0)
        yc01 = tl.load(yc_ptr + yi0 * W + xi0 + 1, mask=m, other=0.0)
        yc10 = tl.load(yc_ptr + (yi0 + 1) * W + xi0, mask=m, other=0.0)
        yc11 = tl.load(yc_ptr + (yi0 + 1) * W + xi0 + 1, mask=m, other=0.0)
        dd = (((y00 - yc00) + (y01 - yc01)) + (y10 - yc10)) + (y11 - yc11)
        tl.store(d_ptr + yo * Wh + xo, dd * 0.25, mask=m)
        fcx = ((((fx00 + fx01) + fx10) + fx11) * 0.25) * 0.5
        fcy = ((((fy00 + fy01) + fy10) + fy11) * 0.25) * 0.5
        whm1 = (Wh - 1).to(tl.float32)
        hhm1 = (Hh - 1).to(tl.float32)
        px = _pos(xo.to(tl.float32), fcx, inv_whm1, whm1, RECIP)
        py = _pos(yo.to(tl.float32), fcy, inv_hhm1, hhm1, RECIP)
        u = _bilinear(u_ptr, px, py, Hh, Wh)
        v = _bilinear(v_ptr, px, py, Hh, Wh)
    s = 1.0 / 1023.0
    tl.store(ob, y00 * s, mask=m)
    tl.store(ob + HW, y01 * s, mask=m)
    tl.store(ob + 2 * HW, y10 * s, mask=m)
    tl.store(ob + 3 * HW, y11 * s, mask=m)
    tl.store(ob + 4 * HW, u * s, mask=m)
    tl.store(ob + 5 * HW, v * s, mask=m)


@triton.jit
def _blur_kernel(x_ptr, o_ptr, H, W, k0, k1, k2, k3, k4, k5, k6, inv_tau,
                 AXIS: tl.constexpr, EXP: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr):
    """7-tap filter along AXIS (1 = x, 0 = y) with reflect padding on a stack of (H, W) planes (program_id(2));
    EXP: write exp(-|.| * inv_tau)."""
    yo = tl.program_id(1) * BH + tl.arange(0, BH)[:, None]
    xo = tl.program_id(0) * BW + tl.arange(0, BW)[None, :]
    m = (yo < H) & (xo < W)
    base = x_ptr + tl.program_id(2).to(tl.int64) * H * W
    acc = tl.zeros([BH, BW], tl.float32)
    for s in tl.static_range(-3, 4):
        if s == -3:
            w = k0
        elif s == -2:
            w = k1
        elif s == -1:
            w = k2
        elif s == 0:
            w = k3
        elif s == 1:
            w = k4
        elif s == 2:
            w = k5
        else:
            w = k6
        if AXIS == 1:
            c = xo + s
            outside = (c < 0) | (c >= W)
            v = tl.load(base + yo * W + c, mask=m & (c >= 0) & (c < W), other=0.0)      # contiguous: block load
            cr = tl.where(c < 0, -c, 2 * (W - 1) - c)
            cr = _clampi(cr, W)
            v += tl.load(base + yo * W + cr, mask=m & outside, other=0.0)               # reflected border taps only
        else:
            r = yo + s
            r = tl.where(r < 0, -r, r)
            r = tl.where(r >= H, 2 * (H - 1) - r, r)
            r = _clampi(r, H)
            v = tl.load(base + r * W + xo, mask=m, other=0.0)
        acc += v * w
    if EXP:
        acc = tl.exp(-tl.abs(acc) * inv_tau)
    tl.store(o_ptr + tl.program_id(2).to(tl.int64) * H * W + yo * W + xo, acc, mask=m)


def _inv(n):
    """fp32(1) / fp32(n): what PyTorch XPU multiplies by for tensor / python_scalar."""
    return float(np.float32(1.0) / np.float32(n))


_KERNELS = {}


def _gauss7(device):
    """student_v3._blur's 7 weights for sigma 1, computed on the device the same way once (no per-call sync)."""
    if device not in _KERNELS:
        t = torch.arange(-3, 4, device=device, dtype=torch.float32)
        k = torch.exp(-t * t / 2.0)
        _KERNELS[device] = (k / k.sum()).tolist()
    return _KERNELS[device]


def _opts(t, grf_mode=None):
    """launch options only the XPU backend knows (grf_mode, sanitize_overflow); CUDA (the 3090) gets none."""
    if t.device.type != "xpu":
        return {}
    return {"sanitize_overflow": False, **({"grf_mode": grf_mode} if grf_mode is not None else {})}


def blur_conf(d, tau=6.0, BH=16, BW=64, num_warps=4):
    """d (N, H, W) -> exp(-|blur_sigma1(d)| / tau) with student_v3._blur's weights and reflect padding."""
    N, H, W = d.shape
    k = _gauss7(d.device)
    tmp = torch.empty_like(d)
    out = torch.empty_like(d)
    grid = (triton.cdiv(W, BW), triton.cdiv(H, BH), N)
    _blur_kernel[grid](d, tmp, H, W, *k, 0.0, AXIS=1, EXP=False, BH=BH, BW=BW, num_warps=num_warps,
                       enable_fp_fusion=False, **_opts(d))
    _blur_kernel[grid](tmp, out, H, W, *k, _inv(tau), AXIS=0, EXP=True, BH=BH, BW=BW, num_warps=num_warps,
                       enable_fp_fusion=False, **_opts(d))
    return out


def assemble_fused(Y, U, V, lowflows, f, tau=6.0, BH=8, BW=32, num_warps=4, recip=True, fp_fusion=False, grf_mode="default"):
    """See the module docstring. Returns (packed (1, 7, 6, H/2, W/2), conf (1, 6, H/2, W/2))."""
    B, nf, H, W = Y.shape
    assert B == 1 and nf % 2 == 1 and len(lowflows) == nf - 1, (Y.shape, len(lowflows))
    assert H % 8 == 0 and W % 8 == 0 and f in (1, 2, 4), (H, W, f)
    assert Y.is_contiguous() and U.is_contiguous() and V.is_contiguous() and Y.dtype == torch.float32
    Hh, Wh = H // 2, W // 2
    hl, wl = H // f, W // f
    c = nf // 2
    packed = torch.empty(1, nf, 6, Hh, Wh, device=Y.device, dtype=torch.float32)
    d = torch.empty(nf - 1, Hh, Wh, device=Y.device, dtype=torch.float32)
    grid = (triton.cdiv(Wh, BW) * triton.cdiv(Hh, BH),)
    inv = (_inv(W - 1), _inv(H - 1), _inv(Wh - 1), _inv(Hh - 1))
    k = 0
    for j in range(nf):
        if j == c:
            _asm_kernel[grid](Y[0, j], U[0, j], V[0, j], Y[0, c], d, packed[0, j], d, H, W, Hh, Wh, hl, wl, *inv,
                              F=f, HAS_FLOW=False, CENTRE=True, RECIP=recip, BH=BH, BW=BW, num_warps=num_warps,
                              enable_fp_fusion=fp_fusion, **_opts(Y, grf_mode))
            continue
        lf = lowflows[k]
        if lf is not None:
            assert lf.shape == (1, 2, hl, wl), (lf.shape, (1, 2, hl, wl))
            lf = lf.contiguous().float()
        _asm_kernel[grid](Y[0, j], U[0, j], V[0, j], Y[0, c], d if lf is None else lf, packed[0, j], d[k],
                          H, W, Hh, Wh, hl, wl, *inv,
                          F=f, HAS_FLOW=lf is not None, CENTRE=False, RECIP=recip, BH=BH, BW=BW, num_warps=num_warps,
                          enable_fp_fusion=fp_fusion, **_opts(Y, grf_mode))
        k += 1
    conf = blur_conf(d, tau)
    return packed, conf[None]
