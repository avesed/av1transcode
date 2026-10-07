"""Student denoiser for 10-bit YUV 4:2:0 video, FastDVDnet-shaped and INT8-friendly (Conv+bias, ReLU, PixelShuffle only;
no BatchNorm: its running statistics blew up the small model in eval mode).
Each frame is packed losslessly to half resolution: Y pixel-unshuffled to 4 channels plus U and V = 6 channels at
H/2 x W/2, so the whole net runs at a quarter of the pixels. Five frames in; stage 1 denoises the three overlapping
triplets with one shared block, stage 2 fuses them. A noise-level map (sigma / 876, as the teacher got) is the extra
input channel. The output is a residual on the centre frame.
  from student import Student, pack, unpack"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def pack(y, u, v):
    """y (B, H, W), u/v (B, H/2, W/2) in [0, 1] -> (B, 6, H/2, W/2)."""
    return torch.cat([F.pixel_unshuffle(y[:, None], 2), u[:, None], v[:, None]], 1)


def unpack(x):
    """(B, 6, H/2, W/2) -> y (B, H, W), u, v (B, H/2, W/2)."""
    return F.pixel_shuffle(x[:, :4], 2)[:, 0], x[:, 4], x[:, 5]


SCALE = torch.tensor([2., 2., 2., 2., 8., 8.]).view(1, 6, 1, 1)


def cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.ReLU(inplace=True))


class DenBlock(nn.Module):
    """Small U-Net: 3 frames (6 ch each) + noise map -> 6 ch residual-corrected centre frame."""
    def __init__(self, c=(32, 64, 128), fin=3):
        super().__init__()
        c0, c1, c2 = c
        self.inc = nn.Sequential(cbr(6 * fin + 1, c0 * 2), cbr(c0 * 2, c0))
        self.down1 = nn.Sequential(cbr(c0, c1, 2), cbr(c1, c1))
        self.down2 = nn.Sequential(cbr(c1, c2, 2), cbr(c2, c2), cbr(c2, c2))
        self.up2 = nn.Sequential(nn.Conv2d(c2, c1 * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.dec1 = nn.Sequential(cbr(c1, c1), cbr(c1, c1))
        self.up1 = nn.Sequential(nn.Conv2d(c1, c0 * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.out = nn.Sequential(cbr(c0, c0), nn.Conv2d(c0, 6, 3, 1, 1))
        nn.init.zeros_(self.out[-1].weight)               # start as the identity: residual 0
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, f0, f1, f2, nm):
        # Inputs centred and scaled per channel: luma x2, chroma x8, noise map x8, so all span roughly +-1. Centred
        # values keep low-precision formats (bf16 / fp16 / int8) fine enough for grain of 1-2 codes. Unscaled, chroma
        # (+-0.05 around 0.5) was drowned by luma (+-0.4) and the net never learned it. The residual is scaled back.
        s = SCALE.to(f1)
        x0 = self.inc(torch.cat([(f0 - 0.5) * s, (f1 - 0.5) * s, (f2 - 0.5) * s, nm * 8], 1))
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x1 = self.dec1(self.up2(x2) + x1)
        x0 = self.up1(x1) + x0
        return f1 - self.out(x0) / s


class Student(nn.Module):
    def __init__(self, c=(32, 64, 128)):
        super().__init__()
        self.b1 = DenBlock(c)
        self.b2 = DenBlock(c)

    def forward(self, x, nm):
        """x (B, 5, 6, h, w) packed frames, nm (B, 1, h, w); h, w multiples of 4. -> (B, 6, h, w)."""
        f = [x[:, i] for i in range(5)]
        a = self.b1(f[0], f[1], f[2], nm)
        b = self.b1(f[1], f[2], f[3], nm)
        c = self.b1(f[2], f[3], f[4], nm)
        return self.b2(a, b, c, nm)
