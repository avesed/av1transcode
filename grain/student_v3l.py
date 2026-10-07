"""Light v3 students for speed (owner 2026-10-07: 4.8-6 fps at 4K is too slow). Same inputs and outputs as StudentV3
(packed aligned frames + confidences in, packed frame out, kernel-predicted chroma), but the work moved off the packed
full resolution, where v3g spends most of its 2.9 TFLOP per 4K frame (48 -> 128 -> 64 channels there):
  thin  packed full resolution: one 48 -> W0 convolution and a W0 head; the U-Net body at 1/2 and 1/4
  s2d   a second space-to-depth at the input (each position covers 4 x 4 luma): nothing runs at packed full resolution;
        output through PixelShuffle
lkp=True: luma by kernel prediction too (per packed luma channel, softmax weights over the aligned frames times
confidence) minus the residual. Random-init light students sat on the identity plateau under distillation (s2d at 2e-4:
synthetic flat 46.25 dB, the noisy input's, for 2000+ steps); averaging aligned frames denoises from the first step and
the gradient reaches the body through the logits, as it did for chroma.
Kept to Conv + ReLU + PixelShuffle (+ the kp softmax), like StudentV3.
  net = StudentV3L(arch, widths, lkp=False)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from student_v2 import SCALE
from student_v3 import _blur


def cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.ReLU(inplace=True))


class StudentV3L(nn.Module):
    def __init__(self, arch="thin", w=(24, 64, 128), nf=7, lkp=False):
        super().__init__()
        self.arch, self.nf, self.kp, self.ctau, self.lkp = arch, nf, True, 0.0, lkp
        cin = 6 * nf + (nf - 1)
        w0, w1, w2 = w
        nout = 6 + nf + (4 * nf if lkp else 0)
        if arch == "thin":
            self.inc = cbr(cin, w0)
            self.down1 = nn.Sequential(cbr(w0, w1, 2), cbr(w1, w1), cbr(w1, w1))
            self.down2 = nn.Sequential(cbr(w1, w2, 2), cbr(w2, w2), cbr(w2, w2), cbr(w2, w2))
            self.up2 = nn.Sequential(nn.Conv2d(w2, w1 * 4, 3, 1, 1), nn.PixelShuffle(2))
            self.dec1 = nn.Sequential(cbr(w1, w1), cbr(w1, w1))
            self.up1 = nn.Sequential(nn.Conv2d(w1, w0 * 4, 3, 1, 1), nn.PixelShuffle(2))
            self.out = nn.Sequential(cbr(w0, w0), nn.Conv2d(w0, nout, 3, 1, 1))
        else:                                            # s2d
            self.inc = nn.Sequential(cbr(cin * 4, w1), cbr(w1, w1))
            self.down1 = nn.Sequential(cbr(w1, w2, 2), cbr(w2, w2), cbr(w2, w2))
            self.up1 = nn.Sequential(nn.Conv2d(w2, w1 * 4, 3, 1, 1), nn.PixelShuffle(2))
            self.dec = nn.Sequential(cbr(w1, w1), cbr(w1, w1))
            self.out = nn.Sequential(nn.Conv2d(w1, nout * 4, 3, 1, 1), nn.PixelShuffle(2))
        with torch.no_grad():                            # kp logits: zero weights, the centre frame favoured
            last = self.out[-1] if arch == "thin" else self.out[0]
            centre = lambda c: (c - 6) % nf == nf // 2      # every logit group (chroma, then 4 luma) favours the centre
            if arch == "thin":
                last.weight[6:].zero_(); last.bias[6:].zero_()
                for c in range(6, nout):
                    if centre(c):
                        last.bias[c] = 3.0
            else:                                        # PixelShuffle: channel c of the output = conv channels 4c..4c+3
                for c in range(6, nout):
                    last.weight[4 * c:4 * c + 4].zero_(); last.bias[4 * c:4 * c + 4] = 3.0 if centre(c) else 0.0

    def forward(self, frames, conf):
        s = SCALE.to(frames)
        c = self.nf // 2
        ctr = frames[:, c]
        x = torch.cat([((frames - 0.5) * s[:, None]).flatten(1, 2), conf * 2 - 1], 1)
        if self.arch == "thin":
            x0 = self.inc(x)
            x1 = self.down1(x0)
            x2 = self.down2(x1)
            x1 = self.dec1(self.up2(x2) + x1)
            r = self.out(self.up1(x1) + x0)
        else:
            x1 = self.inc(F.pixel_unshuffle(x, 2))
            x2 = self.down1(x1)
            r = self.out(self.dec(self.up1(x2) + x1))
        cf = torch.cat([conf[:, :c], torch.ones_like(conf[:, :1]), conf[:, c:]], 1)
        lcf = torch.log(cf.clamp(min=1e-4))
        nf = self.nf
        wgt = torch.softmax(r[:, 6:6 + nf] + lcf, 1)
        uv = (wgt[:, :, None] * frames[:, :, 4:6]).sum(1)
        if self.lkp:
            B, _, h, w = r.shape
            wy = torch.softmax(r[:, 6 + nf:].reshape(B, 4, nf, h, w) + lcf[:, None], 2)      # (B, 4, nf, h, w)
            y = (wy * frames[:, :, :4].transpose(1, 2)).sum(2)                            # (B, 4, h, w)
        else:
            y = ctr[:, :4]
        return torch.cat([y, uv], 1) - r[:, :6] / s
