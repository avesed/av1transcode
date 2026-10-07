"""Light motion alignment for the v3 student: SpyNet (as in RVRT, weights taken from the RVRT checkpoint) on luma, and
warping of neighbour frames onto the centre frame with a per-pixel confidence.

  sp = load_spynet(dev)                      # frozen
  flow = sp_flow(sp, ref_y, sup_y)           # (B, 2, H, W) in pixels: sup(x + flow(x)) ~ ref(x); luma in codes
  out = warp(img, flow)                      # img (B, C, H, W)
SpyNet sees luma scaled to [0, 1] (limited range) on all three channels, pre-blurred 1 px (grain is not motion), and
works at up to 1920 px wide (4K is averaged down 2x and the flow scaled back up).
"""
import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            nn.Conv2d(8, 32, 7, 1, 3), nn.ReLU(inplace=False), nn.Conv2d(32, 64, 7, 1, 3), nn.ReLU(inplace=False),
            nn.Conv2d(64, 32, 7, 1, 3), nn.ReLU(inplace=False), nn.Conv2d(32, 16, 7, 1, 3), nn.ReLU(inplace=False),
            nn.Conv2d(16, 2, 7, 1, 3))

    def forward(self, x):
        return self.basic_module(x)


def warp(img, flow, mode="bilinear"):
    """img (B, C, H, W), flow (B, 2, H, W) pixels: out(x) = img(x + flow(x)), border padding."""
    B, C, H, W = img.shape
    gy, gx = torch.meshgrid(torch.arange(H, device=img.device, dtype=img.dtype), torch.arange(W, device=img.device, dtype=img.dtype), indexing="ij")
    grid = torch.stack(((gx + flow[:, 0]) / (W - 1) * 2 - 1, (gy + flow[:, 1]) / (H - 1) * 2 - 1), -1)
    return F.grid_sample(img, grid, mode=mode, padding_mode="border", align_corners=True)


class SpyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, ref, supp):
        h, w = ref.shape[-2:]
        hf, wf = int(math.ceil(h / 32) * 32), int(math.ceil(w / 32) * 32)
        ref = F.interpolate(ref, size=(hf, wf), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(hf, wf), mode="bilinear", align_corners=False)
        ref, supp = [(ref - self.mean) / self.std], [(supp - self.mean) / self.std]
        for _ in range(5):
            ref.insert(0, F.avg_pool2d(ref[0], 2, 2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(supp[0], 2, 2, count_include_pad=False))
        flow = ref[0].new_zeros([ref[0].size(0), 2, ref[0].size(2) // 2, ref[0].size(3) // 2])
        for lv in range(len(ref)):
            up = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            if up.size(2) != ref[lv].size(2):
                up = F.pad(up, [0, 0, 0, 1], mode="replicate")
            if up.size(3) != ref[lv].size(3):
                up = F.pad(up, [0, 1, 0, 0], mode="replicate")
            flow = self.basic_module[lv](torch.cat([ref[lv], warp(supp[lv], up), up], 1)) + up
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
        flow[:, 0] *= w / wf
        flow[:, 1] *= h / hf
        return flow


def load_spynet(dev, path=None):
    """SpyNet weights: the RVRT checkpoint (keys spynet.*) or a SpyNet-only state dict (grain service, SPYNET env)."""
    path = path or os.environ.get("SPYNET", "/t/rvrt/rvrt_denoise.pth")
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("params", sd)
    if any(k.startswith("spynet.") for k in sd):
        sd = {k[len("spynet."):]: v for k, v in sd.items() if k.startswith("spynet.")}
    sub = {k: v for k, v in sd.items() if "mean" not in k and "std" not in k}
    m = SpyNet()
    missing = m.load_state_dict(sub, strict=False)
    assert not [k for k in missing.missing_keys if not k.endswith(("mean", "std"))], missing
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval().to(dev)


def _blur1(x):
    t = torch.arange(-3, 4, device=x.device, dtype=x.dtype)
    k = torch.exp(-t * t / 2)
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (3, 3, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, 3, 3), mode="reflect"), k.view(1, 1, -1, 1))


def sp_flow(sp, ref_y, sup_y, max_w=1920, norm=None, native=False):
    """ref_y, sup_y (B, H, W) luma codes -> flow (B, 2, H, W): sup warped by it matches ref.
    norm "lift": the pair is stretched by its joint 0.5 / 99.5 percentiles and passed through a square root, so that
    dark scenes (HDR shadows sit in the bottom 15 % of the code range) reach SpyNet with usable contrast."""
    H, W = ref_y.shape[-2:]
    f = 1
    while W // f > max_w:
        f *= 2
    if norm == "lift":
        both = torch.stack([ref_y, sup_y]).flatten()[::97].float()
        lo, hi = torch.quantile(both, 0.005), torch.quantile(both, 0.995)
        sc = lambda y: ((y - lo) / (hi - lo).clamp(min=16)).clamp(0, 1).sqrt()
    else:
        sc = lambda y: (y - 64) / 876                    # unchanged default (clamped after the blur)
    prep = lambda y: _blur1(sc(F.avg_pool2d(y[:, None], f) if f > 1 else y[:, None])).clamp(0, 1).repeat(1, 3, 1, 1)
    dt = ref_y.device.type
    with torch.autocast(dt, dtype=torch.float16, enabled=dt == "cuda" or (dt == "xpu" and os.environ.get("SPY_FP16_XPU") == "1")):
        fl = sp(prep(ref_y), prep(sup_y)).float()
    if native:                                           # at the computed scale (pixels of it) + the factor to full size
        return fl, f
    if f > 1:
        fl = F.interpolate(fl, size=(H, W), mode="bilinear", align_corners=False) * f
    return fl
