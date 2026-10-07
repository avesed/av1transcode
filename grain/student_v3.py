"""v3 student: one U-Net over a motion-ALIGNED 7-frame stack.

Inputs (all at packed half resolution, as student_v2): the centre frame and six neighbours warped onto it (each
6 channels: Y pixel-unshuffled to 4 + U + V), one alignment confidence per neighbour (1 = the warped neighbour
matches the centre's picture, 0 = occlusion / flow failure). 7 x 6 + 6 = 48 channels. No noise map: with the
neighbours aligned, the frame-to-frame differences show the grain level directly (and synthetic training data has no
measured curve). Output: residual correction of the centre frame.
Conv + bias, ReLU, PixelShuffle only (INT8 / OpenVINO friendly like v2); the last convolution starts at zero, so the
untrained network returns the centre frame.
kp=True (chroma by kernel prediction): the last convolution also gives nf logits per pixel; chroma = the softmax-weighted
sum of the nf aligned frames' chroma (neighbour weights multiplied by their confidence) minus the usual residual. With
one shared U-Net, chroma never learnt from the plain residual: luma took all the features and v3a / v3b passed chroma
through (|out - in| 0.012 codes) while a chroma-only overfit of the same net did learn; averaging the aligned chroma is
the right prior for chroma grain that changes every frame.
ctau > 0 (with kp): each neighbour's chroma weight is also multiplied by exp(-(|U_j - U_c| + |V_j - V_c|) / ctau) on
1 px-blurred chroma in codes, a fixed temporal bilateral term: v3e (kp only) smeared colour edges where alignment is
not exact (grain-free input: chroma edges 56.7 dB, colour shift 0.1 code), costing 2.2 dB of chroma on light grain.

  from student_v3 import load_v3, assemble
  net = load_v3(torch.load(ckpt))
  frames, conf, _ = assemble(Y, U, V, flows)   # see assemble()
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from student_v2 import pack, unpack, SCALE
from align import warp


def cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.ReLU(inplace=True))


class StudentV3(nn.Module):
    def __init__(self, c=(64, 128, 256), nf=7, kp=False, ctau=0.0):
        super().__init__()
        c0, c1, c2 = c
        self.nf, self.kp, self.ctau = nf, kp, ctau
        cin = 6 * nf + (nf - 1)
        self.inc = nn.Sequential(cbr(cin, c0 * 2), cbr(c0 * 2, c0), cbr(c0, c0))
        self.down1 = nn.Sequential(cbr(c0, c1, 2), cbr(c1, c1), cbr(c1, c1))
        self.down2 = nn.Sequential(cbr(c1, c2, 2), cbr(c2, c2), cbr(c2, c2), cbr(c2, c2))
        self.up2 = nn.Sequential(nn.Conv2d(c2, c1 * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.dec1 = nn.Sequential(cbr(c1, c1), cbr(c1, c1))
        self.up1 = nn.Sequential(nn.Conv2d(c1, c0 * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.out = nn.Sequential(cbr(c0, c0), cbr(c0, c0), nn.Conv2d(c0, 6 + (nf if kp else 0), 3, 1, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
        if kp:
            self.init_kp()

    def init_kp(self, centre=3.0):
        """chroma logits: zero weights, the centre frame favoured (logit +3: weight 0.77 among 7 confident frames)."""
        with torch.no_grad():
            self.out[-1].weight[6:].zero_()
            self.out[-1].bias[6:].zero_()
            self.out[-1].bias[6 + self.nf // 2] = centre

    def forward(self, frames, conf):
        """frames (B, nf, 6, h, w) packed, aligned, centre at nf // 2, values in [0, 1]; conf (B, nf - 1, h, w) in
        [0, 1]. h, w multiples of 4. -> (B, 6, h, w)."""
        s = SCALE.to(frames)
        ctr = frames[:, self.nf // 2]
        x = torch.cat([((frames - 0.5) * s[:, None]).flatten(1, 2), conf * 2 - 1], 1)
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x1 = self.dec1(self.up2(x2) + x1)
        x0 = self.up1(x1) + x0
        r = self.out(x0)
        if not self.kp:
            return ctr - r / s
        c = self.nf // 2
        cf = torch.cat([conf[:, :c], torch.ones_like(conf[:, :1]), conf[:, c:]], 1)
        if self.ctau > 0:
            d = _blur((frames[:, :, 4:6] - frames[:, c:c + 1, 4:6]) * 1023, 1.0)     # differences first: exact in fp16
            cf = cf * torch.exp(-d.abs().sum(2) / self.ctau)
        w = torch.softmax(r[:, 6:] + torch.log(cf.clamp(min=1e-4)), 1)
        uv = (w[:, :, None] * frames[:, :, 4:6]).sum(1)
        return torch.cat([ctr[:, :4], uv], 1) - r[:, :6] / s


def load_v3(ck):
    """checkpoint dict -> StudentV3 (or a light StudentV3L, ck["net"] = "v3l-<arch>") with its weights (eval mode, CPU)."""
    if str(ck.get("net", "")).startswith("v3l-"):
        from student_v3l import StudentV3L
        net = StudentV3L(ck["net"][4:], tuple(ck["channels"]), lkp=ck.get("lkp", False))
        net.load_state_dict(ck["state"])
        return net.eval()
    net = StudentV3(tuple(ck["channels"]), kp=ck.get("kp", False), ctau=ck.get("ctau", 0.0))
    net.load_state_dict(ck["state"])
    return net.eval()


def _blur(x, sg=2.0):
    r = int(3 * sg + 0.5)
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t * t / (2 * sg * sg))
    k = k / k.sum()
    sh = x.shape
    x = x.reshape(-1, 1, *sh[-2:])
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1)).reshape(sh)


def assemble_fast(Y, U, V, flows, tau=6.0):
    """Inference version of assemble(): the confidence from one blur of the difference (blur is linear, so
    |blur(w) - blur(c)| = |blur(w - c)|) at packed resolution (sigma 1 there ~ sigma 2 at full), no neighbour mean.
    Returns (packed frames, confidences, None). B580 4K: 74 ms -> see xpu_prof.py."""
    B, nf, H, W = Y.shape
    c = nf // 2
    ys, us, vs, confs = [], [], [], []
    k = 0
    for j in range(nf):
        if j == c:
            ys.append(Y[:, j]), us.append(U[:, j]), vs.append(V[:, j])
            continue
        fl = flows[k]
        k += 1
        wy = warp(Y[:, j:j + 1], fl, mode="bicubic")[:, 0]
        flc = F.avg_pool2d(fl, 2) / 2
        us.append(warp(U[:, j:j + 1], flc)[:, 0]), vs.append(warp(V[:, j:j + 1], flc)[:, 0]), ys.append(wy)
        confs.append(torch.exp(-_blur(F.avg_pool2d((wy - Y[:, c])[:, None], 2)[:, 0], 1.0).abs() / tau))
    sc = lambda a: a / 1023.0
    packed = torch.stack([pack(sc(ys[j]), sc(us[j]), sc(vs[j])) for j in range(nf)], 1)
    return packed, torch.stack(confs, 1), None


def assemble(Y, U, V, flows, tau=6.0):
    """Y (B, nf, H, W), U / V (B, nf, H/2, W/2) in codes, flows: list of nf - 1 tensors (B, 2, H, W) in full-res
    pixels for the non-centre frames in order (F(centre -> n)). Returns packed aligned frames (B, nf, 6, H/2, W/2)
    in [0, 1], confidences (B, nf - 1, H/2, W/2), and the aligned neighbours' mean luma (B, H, W) for the
    independence loss. Confidence = exp(-|blur2(warped Y) - blur2(centre Y)| / tau)."""
    B, nf, H, W = Y.shape
    c = nf // 2
    ys, us, vs, confs, nb = [], [], [], [], []
    k = 0
    for j in range(nf):
        if j == c:
            ys.append(Y[:, j]), us.append(U[:, j]), vs.append(V[:, j])
            continue
        fl = flows[k]
        k += 1
        wy = warp(Y[:, j:j + 1], fl, mode="bicubic")[:, 0]
        flc = F.avg_pool2d(fl, 2) / 2
        wu = warp(U[:, j:j + 1], flc)[:, 0]
        wv = warp(V[:, j:j + 1], flc)[:, 0]
        ys.append(wy), us.append(wu), vs.append(wv)
        cf = torch.exp(-(_blur(wy) - _blur(Y[:, c])).abs() / tau)
        confs.append(F.avg_pool2d(cf[:, None], 2)[:, 0])
        nb.append(wy)
    sc = lambda a: a / 1023.0
    packed = torch.stack([pack(sc(ys[j]), sc(us[j]), sc(vs[j])) for j in range(nf)], 1)
    return packed, torch.stack(confs, 1), torch.stack(nb, 1).mean(1)
