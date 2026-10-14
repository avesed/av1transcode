"""Quantize a v3 student's U-Net trunk for the B580, once, offline -> a .pt2 that dn_v3.py loads with INT8_TRUNK.

torchao pt2e with the XPU inductor quantizer: weights int8 per channel (symmetric), activations uint8 per tensor
(affine 0-255). 14 of the 17 convolutions read post-ReLU activations, whose zero point comes out exactly 0, so oneDNN
runs them without zero-point compensation; the three that can see negative inputs (the first and the two after the
skip adds) keep theirs (made symmetric they were 0.5 ms faster per tile but up to 59 codes off). Calibration: captured
network inputs (dn_v3_int8 CAPTURE_DIR .pt files: frames, conf), every other one; the rest is held out and reported
against the fp16 network the production path runs.
  python3 quantize_int8.py CKPT CAPTURE_DIR[:CAPTURE_DIR ...] OUT.pt2      (grain image with torchao)
"""
import dataclasses, glob, sys, time
import torch
import torch.nn as nn
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from torchao.quantization.pt2e.quantizer.xpu_inductor_quantizer import (XPUInductorQuantizer,
                                                                        get_default_xpu_inductor_quantization_config)
from student_v3 import load_v3

CK, CAP, OUT = sys.argv[1:4]
dev = torch.device("xpu")
files = sorted(f for d in CAP.split(":") for f in glob.glob(f"{d}/*.pt"))
calib, held = files[0::2], files[1::2]


class Trunk(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return self.n.trunk(x)


net = load_v3(torch.load(CK, map_location="cpu")).to(dev).eval()
load = lambda p: [t.to(dev).float() for t in (lambda d: (d["frames"], d["conf"]))(torch.load(p))]
base = get_default_xpu_inductor_quantization_config()
act = dataclasses.replace(base.input_activation, dtype=torch.uint8, quant_min=0, quant_max=255)
cfg = dataclasses.replace(base, input_activation=act, output_activation=act)
dyn = {"x": {2: 4 * torch.export.Dim("h4", min=16, max=400), 3: 4 * torch.export.Dim("w4", min=16, max=700)}}
with torch.no_grad():
    x0 = net.pre(*load(calib[0]))
    m = torch.export.export(Trunk(net), (x0,), dynamic_shapes=dyn).module()
    q = XPUInductorQuantizer()
    q.set_global(cfg)
    m = prepare_pt2e(m, q)
    t = time.time()
    for p in calib:
        m(net.pre(*load(p)))
    torch.xpu.synchronize()
    m = convert_pt2e(m)
    zps = [n.args[2] for n in m.graph.nodes if "quantize_per_tensor" in str(n.target) and "dequant" not in str(n.target)]
    print(f"calibrated on {len(calib)} tiles in {time.time() - t:.0f} s; activation zero points {zps}", flush=True)
    torch.export.save(torch.export.export(m, (x0,), dynamic_shapes=dyn), OUT)
    q8 = torch.compile(torch.export.load(OUT).module(), dynamic=False, options={"freezing": True})
    fp16 = torch.compile(net, dynamic=False)
    errs = []
    for p in held:
        f, c = load(p)
        with torch.autocast("xpu", dtype=torch.float16):
            ref = fp16(f.half(), c.half()).float()
        errs.append(((net.post(f, c, q8(net.pre(f, c))) - ref).abs() * 1023).flatten())
    e = torch.cat(errs)
    print(f"{OUT}: {len(held)} held-out tiles vs the fp16 network: mean {float(e.mean()):.3f} codes, "
          f"p99 {float(torch.quantile(e[::97], 0.99)):.2f}, max {float(e.max()):.1f}", flush=True)
