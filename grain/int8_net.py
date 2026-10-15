"""The v3 student with its U-Net trunk in int8 on the B580: the offline .pt2 from quantize_int8.py, loaded and compiled
with the graph changes that make it fast and keep it right (2026-10-14, kernel study in ~/av1t-verify/kernopt/).

  net8 = load_int8_net("v3g_trunk_u8.pt2", net)      # net: the loaded StudentV3 (pre / post and its nf, kp, ctau)
  out = net8(frames.float(), conf.float())           # fp32 in and out, NO autocast around it

B580, 572 x 992 tile (a quarter of a 4K frame): 7.1 ms against 15.2 for the fp16 network as dn_v3 ran it and 11.8 with
FAST_FP16; output within mean 0.57 / p99 3.6 / max 15 10-bit codes of the fp16 network.

Two settings are silent when missing, both made here:
  * Inductor's freezing, globally (torch.compile(options={"freezing": True}) does not take): it is what lowers the
    graph onto oneDNN's int8 convolution with prepacked weights. Without it the convs take ~68 ms a tile.
  * the torchao XPU quantizer module imported in the process: the import registers the qconv weight-prepack and
    fusion patterns. A process that only loads the .pt2 compiles the decomposed dequantize -> fp32 conv path.
Also: a deep-copied exported module must be recompile()d, or it too compiles to that path; and a second torch.compile
wrapper of the same module object reuses the first one's Dynamo cache, so config changes made in between do not apply.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._inductor.config as icfg
from torch._inductor.custom_graph_pass import CustomGraphPass
from torchao.quantization.pt2e.quantizer.xpu_inductor_quantizer import XPUInductorQuantizer  # noqa: F401 (see above)

Q_ = torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ = torch.ops.quantized_decomposed.dequantize_per_tensor.default
QCONV = (torch.ops.onednn.qconv_pointwise.default, torch.ops.onednn.qconv_pointwise.tensor)


def rewire_skips(g, expect=2):
    """add(pixel_shuffle(up), relu) reads the skip's dequantized uint8 copy instead of the fp32 ReLU output: the
    producing convs then requantize in-kernel (no fp32 write, no separate quantize pass) and the fused
    pixel-shuffle + add + quantize reads 1 byte an element. -0.9 ms a tile, +0.013 codes mean. expect: the trunk has
    two such adds, the mixed network's int8 middle one."""
    n_done = 0
    for n in g.graph.nodes:
        if n.target is torch.ops.aten.add.Tensor:
            skip = n.args[1]
            qn = next(u for u in skip.users if u.target is Q_)
            dq = next(u for u in qn.users if u.target is DQ)
            n.args = (n.args[0], dq)
            n_done += 1
    assert n_done == expect, f"expected {expect} skip adds, found {n_done}"
    g.graph.lint()
    g.recompile()
    return g


class Bf16Out(CustomGraphPass):
    """Inductor post-grad PRE pass: the qconvs that write fp32 (up2 and up1 before their pixel shuffle, the last conv)
    write bf16 instead, their bias staying fp32. int8-mixed-bf16 by autocast was wrong (mean 72 codes off): autocast
    casts the conv bias to bf16 too, and the B580's oneDNN qconv reads a bf16 bias as garbage. -0.5 ms a tile."""
    def __call__(self, graph):
        if isinstance(graph, torch.fx.GraphModule):
            graph = graph.graph
        for n in graph.nodes:
            if n.target in QCONV and n.args[13] is torch.float32:
                a = list(n.args)
                a[13] = torch.bfloat16
                n.args = tuple(a)
                if isinstance(n.meta.get("val"), torch.Tensor):
                    n.meta["val"] = n.meta["val"].to(torch.bfloat16)

    def uuid(self):
        return "grain-int8-bf16out-v1"


def pad_first_conv(g, cin_pad=64):
    """the first conv's int8 weight (128, 48, 3, 3) zero-padded to cin_pad input channels (the input is padded in
    Int8Net: the pad channels quantize to exactly the zero point and meet zero weights, so the int32 sums are
    bit-identical). The 48-channel conv ran at 30 TMAC/s against 85-95 for the rest. The export's shape guard, which
    pins the channel count, goes."""
    conv = next(n for n in g.graph.nodes if n.target is torch.ops.aten.conv2d.default)
    wnode = conv.args[1].args[0]
    assert wnode.op == "get_attr", wnode
    parts = wnode.target.split(".")
    parent = g
    for p in parts[:-1]:
        parent = getattr(parent, p)
    w = getattr(parent, parts[-1])
    assert w.dtype == torch.int8 and w.shape[1] < cin_pad, (w.dtype, w.shape)
    new = torch.zeros(w.shape[0], cin_pad, *w.shape[2:], dtype=w.dtype, device=w.device)
    new[:, :w.shape[1]] = w
    if parts[-1] in parent._buffers:
        parent._buffers[parts[-1]] = new
    elif parts[-1] in parent._parameters:
        parent._parameters[parts[-1]] = nn.Parameter(new, requires_grad=False)
    else:
        setattr(parent, parts[-1], new)
    for n in list(g.graph.nodes):
        if n.op == "call_module" and "_guards_fn" in str(n.target) and not n.users:
            g.graph.erase_node(n)
    g.graph.lint()
    g.recompile()
    return g


class Int8Net(nn.Module):
    """StudentV3.forward with the trunk replaced: pre and post are the fp32 network's own, all in one compiled graph
    (splitting them out cost 0.5-0.8 ms a tile)."""
    def __init__(self, net, trunk, cin_pad=0):
        super().__init__()
        self.trunk, self.cin_pad = trunk, cin_pad
        self.pre, self.post = net.pre, net.post        # bound methods: the net's weights are not part of this graph

    def forward(self, frames, conf):
        x = self.pre(frames, conf)
        if self.cin_pad:
            x = F.pad(x, (0, 0, 0, 0, 0, self.cin_pad - x.shape[1]))
        return self.post(frames, conf, self.trunk(x).float())


def load_int8_net(path, net, pad_in=64):
    """-> the compiled int8 network, (frames, conf) fp32 -> (B, 6, H, W) fp32. net must not have been pad_input()ed
    (the trunk is padded here). No .eval(): the exported module has no train() / eval(), and nothing in it depends on
    the mode."""
    assert not getattr(net, "cin_pad", 0), "load_int8_net pads the trunk itself"
    icfg.freezing = True
    icfg.post_grad_custom_pre_pass = Bf16Out()
    trunk = copy.deepcopy(torch.export.load(path).module())
    trunk.recompile()
    trunk = rewire_skips(trunk)
    if pad_in:
        trunk = pad_first_conv(trunk, pad_in)
    return torch.compile(Int8Net(net, trunk, cin_pad=pad_in), dynamic=False)


class Middle(nn.Module):
    """the trunk between the input convs (inc) and the output convs (out): what the mixed network runs in int8
    (quantize_int8.py mid exports it; its input x0 is inc's ReLU output, its output up1's before the skip add)."""
    def __init__(self, n):
        super().__init__()
        self.down1, self.down2, self.up2, self.dec1, self.up1 = n.down1, n.down2, n.up2, n.dec1, n.up1   # only these in the .pt2

    def forward(self, x0):
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x1 = self.dec1(self.up2(x2) + x1)
        return self.up1(x1)


class MixedNet(nn.Module):
    """inc and out in fp16 (autocast, the first conv padded to 64 input channels), the middle in int8, one graph. The
    int8 part stays outside autocast: autocast around the qconvs gave the bf16-bias garbage (Bf16Out)."""
    def __init__(self, net, mid):
        super().__init__()
        self.net, self.mid = net, mid

    def forward(self, frames, conf):
        n = self.net
        with torch.autocast(frames.device.type, dtype=torch.float16):
            x0 = n.inc(n.pre(frames, conf))
        m = self.mid(x0.float())
        with torch.autocast(frames.device.type, dtype=torch.float16):
            return n.post(frames, conf, n.out(m + x0)).float()


def load_mixed_net(path, net):
    """-> the compiled mixed-precision network, (frames, conf) fp32 -> (B, 6, H, W) fp32, from quantize_int8.py's
    middle .pt2. All of int8's loss sat in the three input and three output convs (per-layer sensitivity on the v3
    validation windows, 2026-10-15): with those in fp16 it scores as the fp16 network (synthetic grain 49.25 dB,
    grain-free 60.58; all-int8 48.85 / 58.51), mean 0.05 codes from it; B580 4K tile 8.9 ms (all-int8 6.9, FAST_FP16
    11.1). net: the loaded StudentV3, not pad_input()ed."""
    assert not getattr(net, "cin_pad", 0), "load_mixed_net pads the input conv itself"
    icfg.freezing = True
    icfg.post_grad_custom_pre_pass = Bf16Out()
    mid = copy.deepcopy(torch.export.load(path).module())
    mid.recompile()
    mid = rewire_skips(mid, expect=1)
    return torch.compile(MixedNet(net.pad_input(64), mid), dynamic=False)
