"""Denoise a file with the v3 student (aligned 7-frame stack), at one or more linear strengths ALPHA.

Per frame t: SpyNet flows between consecutive frames are computed once (forward j -> j+1 and backward j -> j-1, on
1 px-blurred luma, half resolution above 1920 px, fp16) and chained to reach t-3 .. t+3; at the clip ends the
nearest existing frame is used. Network in fp16 autocast. Output = source - ALPHA x (source - network output).
Writes OUTDIR/v3_<ALPHA>.mkv (lossless H.264 10-bit 4:2:0, LOSSLESS=ffv1 for FFV1, LOSSLESS=av1hw for the B580's AV1
at QP 0 with RENDER_NODE; the source's colour tags), or with DN_PREFIX viewing copies like
dn_mf2f.py.
Env LIMIT=c: the luma removal is capped at c x the file's fresh grain std (mf2f_common.limit_removal; curve from 24
frame pairs spread over the file). v3g removed 1.4-1.7x the grain on Westworld's textured pixels and wiped shadow
detail and dark make-up (owner, test-07); at 1.3 that came back while CIA (0.6-0.9x) was left alone. LIMWIN: the
window (px) of that local rms (default 9).
Speed / device: DEV=xpu runs on the B580 (SpyNet in fp16 there too), FLOWMW (default 1920) is the widest the flows are
computed at (960: quarter resolution at 4K, half at 1080p; detail and synthetic PSNR unchanged within 0.15 dB),
FASTASM=1 uses assemble_fast(), FUSED=1 the fused Triton assembly (XPU), TILES=2 runs the network on 2 x 2 tiles (bounded memory at 4K).
COMPILE=1 compiles the network (torch.compile; the first frames are slow).
Decoding and writing run in their own threads (raw frames cross as 16-bit, one copy each way); PROF=1 prints where the
time went.
  python3 dn_v3.py SRC OUTDIR CKPT ALPHA [ALPHA ...]   (vLLM CUDA image; PYTHONPATH has teachers and mf2f)
"""
import json, os, queue, subprocess, sys, threading, time
import numpy as np
import torch
import torch.nn.functional as F
from student_v2 import pack, unpack
from student_v3 import load_v3, assemble, assemble_fast
from align import load_spynet, sp_flow, warp
from mf2f_common import fill, fresh_curve, grain_std, limit_removal

SRC, OUT, CK = sys.argv[1], sys.argv[2], sys.argv[3]
ALPHAS = [float(x) for x in sys.argv[4:]]
os.makedirs(OUT, exist_ok=True)
dev = torch.device(os.environ.get("DEV", "cuda"))
if dev.type == "xpu":
    os.environ.setdefault("SPY_FP16_XPU", "1")
FLOWMW = int(os.environ.get("FLOWMW", "1920"))
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", "0"))   # grain service: "PROGRESS <frames>" lines
TILES = int(os.environ.get("TILES", "1"))
LIMWIN = int(os.environ.get("LIMWIN", "9"))
ASM = assemble_fast if os.environ.get("FASTASM") == "1" else assemble
FUSED = os.environ.get("FUSED") == "1"                # fused_asm.assemble_fused (Triton, XPU): low-res flows straight in,
if FUSED:                                             # upsampled and sampled inside; B580 4K 47 -> 3.3 ms, max 0.001 code off
    from fused_asm import assemble_fused
ck = torch.load(CK, map_location="cpu")
net = load_v3(ck).to(dev)
sp = load_spynet(dev)
LIMFN = limit_removal
FAST = ASM is assemble_fast
if os.environ.get("COMPILE") == "1":                  # Inductor / Triton: B580 4K network 88 -> 57 ms (v3g), 44 -> 26 (student);
    net = torch.compile(net, dynamic=False, mode=os.environ.get("COMPILE_MODE") or None)   # assemble / limiter: no gain
if os.environ.get("COMPILE_SP") == "1":               # SpyNet (its first frames compile: measure on long clips)
    sp = torch.compile(sp, dynamic=False, mode=os.environ.get("COMPILE_MODE") or None)
s = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                               "stream=width,height,r_frame_rate,color_transfer,color_primaries,color_space,color_range",
                               "-of", "json", SRC], capture_output=True, text=True).stdout)["streams"][0]
w, h = s["width"], s["height"]
fs = w * h * 3 // 2
tags = []
for opt, k in (("-color_primaries", "color_primaries"), ("-color_trc", "color_transfer"), ("-colorspace", "color_space"),
               ("-color_range", "color_range")):
    if s.get(k):
        tags += [opt, s[k]]
dec = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", SRC, "-map", "0:v:0", "-fps_mode", "passthrough",
                        "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-"], stdout=subprocess.PIPE)
LIMIT = float(os.environ.get("LIMIT", "0"))
curve = None
if LIMIT:
    nfr = int(json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                                         "stream=nb_read_packets", "-of", "json", SRC], capture_output=True, text=True).stdout)
              ["streams"][0]["nb_read_packets"])
    picks = sorted({p + d for p in np.linspace(2, nfr - 4, 24).astype(int) for d in (0, 1)})
    sel = "+".join(f"eq(n\\,{p})" for p in picks)
    raw = np.frombuffer(subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", SRC, "-map", "0:v:0", "-vf", f"select='{sel}'",
                                        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-"],
                                       capture_output=True).stdout, np.uint16)
    py = [torch.from_numpy(raw[i * fs:i * fs + w * h].astype(np.float32).reshape(h, w)).to(dev) for i in range(len(raw) // fs)]
    curve = fill(fresh_curve([(py[i], py[i + 1]) for i in range(0, len(py) - 1, 2)]), default=[2.0] * 11)
    del py, raw
    print(f"{os.path.basename(SRC)}: removal capped at {LIMIT:g} x fresh grain std {curve}", flush=True)
PREFIX = os.environ.get("DN_PREFIX")
X265 = {"color_primaries": "colorprim", "color_transfer": "transfer", "color_space": "colormatrix"}
if PREFIX:
    xp = ":".join([f"{X265[k]}={s[k]}" for k in X265 if s.get(k) and s[k] != "unknown"] + ["range=limited", "log-level=error"])
    enc_args = ["-c:v", "libx265", "-preset", "fast", "-tune", "grain", "-crf", "10", "-x265-params", xp,
                "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1", "-an", "-movflags", "+faststart"]
    keys = [None] + ALPHAS
    name = lambda a: f"{PREFIX}_{keys.index(a)}"
    ext = "mp4"
else:
    pre_args = []
    if os.environ.get("LOSSLESS") == "av1hw":          # the grain service's intermediate: the B580's AV1 encoder at QP 0 (VDEnc,
        pre_args = ["-vaapi_device", os.environ.get("RENDER_NODE", "/dev/dri/renderD128")]   # beside the compute work);
        enc_args = ["-vf", "format=p010,hwupload", "-c:v", "av1_vaapi", "-rc_mode", "CQP", "-qp", "0"]  # 49.6 dB vs the
        # exact frames, the final encode 0.1-0.26 dB / 0.6-1 VMAF below the lossless route at 4-8% fewer bits (owner
        # 2026-10-07: "没什么区别就都用av1吧"); a 4K hour is ~90 GB instead of 500-700 GB lossless
    elif os.environ.get("LOSSLESS", "x264") == "x264":     # lossless H.264 (qp 0, ultrafast, 10-bit): 4K encode 18 -> 48 fps and
        enc_args = ["-c:v", "libx264", "-qp", "0", "-preset", "ultrafast", "-pix_fmt", "yuv420p10le",   # decode 22 -> 66
                    "-x264-params", "log-level=error"]        # fps vs FFV1 (16 cores), 17% bigger, PSNR inf
    else:
        enc_args = ["-c:v", "ffv1", "-level", "3", "-slices", "24"]
    keys = ALPHAS
    name = lambda a: f"v3_{a:g}"
    ext = "mkv"
if PREFIX:
    pre_args = []
encs = {a: subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", *pre_args, "-f", "rawvideo", "-pix_fmt", "yuv420p10le",
                             "-s", f"{w}x{h}", "-r", s["r_frame_rate"], *tags, "-i", "-", *enc_args, *tags,
                             f"{OUT}/{name(a)}.part.{ext}"], stdin=subprocess.PIPE)
        for a in keys}
ph, pw = (-h) % 8, (-w) % 8                       # packed size must be a multiple of 4


PROF = os.environ.get("PROF") == "1"
TM = {}
_sync = (torch.xpu.synchronize if dev.type == "xpu" else torch.cuda.synchronize)


def tick(label, t_prev):
    """PROF: device-synchronised stage times."""
    if not PROF:
        return t_prev
    _sync()
    t_now = time.time()
    TM[label] = TM.get(label, 0.0) + t_now - t_prev
    return t_now


rq = queue.Queue(maxsize=6)


def reader():
    while True:
        b = dec.stdout.read(fs * 2)
        if len(b) < fs * 2:
            rq.put(None)
            return
        rq.put(np.frombuffer(b, np.int16))           # 10-bit codes fit in int16; converted on the device


threading.Thread(target=reader, daemon=True).start()
wq = {a: queue.Queue(maxsize=6) for a in keys}


def writer(a):
    while True:
        item = wq[a].get()
        if item is None:
            return
        buf, ev = item
        ev.synchronize()                                 # the asynchronous device -> pinned host copy has landed
        encs[a].stdin.write(memoryview(buf.numpy()).cast("B"))   # no 12 MB copy holding the GIL


wthreads = [threading.Thread(target=writer, args=(a,), daemon=True) for a in keys]
for t_ in wthreads:
    t_.start()


RING = [torch.empty(w * h * 3 // 2, dtype=torch.int16, pin_memory=True) for _ in range(10)]   # > queue size + in flight
RI = [0]


def emit(a, y, u, v):
    """y, u, v codes on the device (padded) -> one asynchronous int16 copy into a pinned ring buffer -> the writer
    thread, which waits for the copy (B580 4K: the synchronous copy cost 9.5 ms of compute time per frame)."""
    planes = torch.cat([y[:h, :w].reshape(-1), u[:h // 2, :w // 2].reshape(-1), v[:h // 2, :w // 2].reshape(-1)])
    buf = RING[RI[0] % len(RING)]
    RI[0] += 1
    buf.copy_(planes.round().clamp(0, 1023).to(torch.int16), non_blocking=True)
    ev = torch.Event(device=dev)
    ev.record()
    wq[a].put((buf, ev))


def read():
    arr = rq.get()
    if arr is None:
        return None
    a = torch.from_numpy(arr).to(dev).float()
    y, u, v = a[:w * h].view(h, w), a[w * h:w * h * 5 // 4].view(h // 2, w // 2), a[w * h * 5 // 4:].view(h // 2, w // 2)
    if ph or pw:
        y = F.pad(y[None, None], (0, pw, 0, ph), mode="replicate")[0, 0]
        u = F.pad(u[None, None], (0, pw // 2, 0, ph // 2), mode="replicate")[0, 0]
        v = F.pad(v[None, None], (0, pw // 2, 0, ph // 2), mode="replicate")[0, 0]
    return y, u, v


fr, fwd, bwd = {}, {}, {}
n_in, n_out, eof, t0 = 0, 0, False, time.time()


def run(frames, conf):
    """network in fp16, whole frame or TILES x TILES tiles with 32 px of overlap (packed resolution)."""
    with torch.autocast(dev.type, dtype=torch.float16):
        if TILES == 1:
            return net(frames, conf).float()
        Hp, Wp = frames.shape[-2:]
        th, tw = -(-Hp // TILES // 4) * 4, -(-Wp // TILES // 4) * 4
        o = torch.empty(frames.shape[0], 6, Hp, Wp, device=dev)
        for y0 in range(0, Hp, th):
            for x0 in range(0, Wp, tw):
                a0, b0 = max(0, y0 - 32), max(0, x0 - 32)
                a1, b1 = min(Hp, y0 + th + 32), min(Wp, x0 + tw + 32)
                t = net(frames[..., a0:a1, b0:b1], conf[..., a0:a1, b0:b1]).float()
                o[..., y0:y0 + th, x0:x0 + tw] = t[..., y0 - a0:y0 - a0 + th, x0 - b0:x0 - b0 + tw]
        return o


FS = [1]


def limit_low(src, out):
    """limit_removal with everything smooth computed at quarter resolution: the local rms, the brightness the grain
    std is read at, and the resulting scale, which alone is brought back to full size (B580 4K 10 ms -> see PROF)."""
    r = src - out
    e = F.avg_pool2d((r * r)[None, None], 4, 4, ceil_mode=True)
    k = max(1, (LIMWIN // 4) | 1)
    e = F.avg_pool2d(e, k, 1, k // 2, count_include_pad=False)[0, 0]
    sig = grain_std(F.avg_pool2d(src[None, None], 4, 4, ceil_mode=True)[0, 0], curve)
    sc = (LIMIT * sig / e.sqrt().clamp(min=1e-3)).clamp(max=1.0)
    return src - r * F.interpolate(sc[None, None], size=r.shape, mode="bilinear", align_corners=False)[0, 0]


def flow(a, b):
    """F(a -> b) for consecutive a, b (cached) at the scale SpyNet ran at (FS[0] x smaller than the frame)."""
    cache = fwd if b > a else bwd
    if a not in cache:
        t_ = tick("-", time.time()) if PROF else 0.0
        fl, FS[0] = sp_flow(sp, fr[a][0][None], fr[b][0][None], max_w=FLOWMW, native=True)
        cache[a] = fl[0]
        if PROF:
            tick("  of which SpyNet calls", t_)
    return cache[a]


def chain_all(t, need):
    """need: {k: True} for k in +-1..3 -> {k: F(t -> t + k)} at full size; consecutive flows chained incrementally at
    the flow scale (one warp per step instead of re-chaining every distance at full resolution)."""
    out = {}
    for step in (1, -1):
        ks = sorted((k for k in need if k * step > 0), key=abs)
        if not ks:
            continue
        acc, j = None, t
        for d in range(1, abs(ks[-1]) + 1):
            f = flow(j, j + step)
            acc = f if acc is None else acc + warp(f[None], acc[None])[0]
            j += step
            if d * step in need:
                out[d * step] = acc
    if FUSED:
        return {k: v[None] for k, v in out.items()}
    Hf, Wf = fr[t][0].shape
    return {k: (F.interpolate(v[None], size=(Hf, Wf), mode="bilinear", align_corners=False) * FS[0] if FS[0] > 1 else v[None])
            for k, v in out.items()}


with torch.no_grad():
    while True:
        tp0 = tick("-", time.time()) if PROF else 0.0
        while not eof and n_in <= n_out + 3:
            f = read()
            if f is None:
                eof = True
                break
            fr[n_in] = f
            n_in += 1
        if n_out >= n_in:
            break
        tp0 = tick("read", tp0) if PROF else 0.0
        last = n_in - 1
        ks = [k for k in (-3, -2, -1, 1, 2, 3)]
        idx = [min(max(n_out + k, 0), last) for k in ks]
        cf_ = chain_all(n_out, {j - n_out: True for j in idx if j != n_out})
        if FUSED:
            flows = [cf_[j - n_out] if j != n_out else None for j in idx]
        else:
            flows = [cf_[j - n_out] if j != n_out else torch.zeros(1, 2, *fr[n_out][0].shape, device=dev) for j in idx]
        order = idx[:3] + [n_out] + idx[3:]
        Y = torch.stack([fr[j][0] for j in order])[None]
        U = torch.stack([fr[j][1] for j in order])[None]
        V = torch.stack([fr[j][2] for j in order])[None]
        tp = tick("flows", tp0 if PROF else 0.0)
        if FUSED:
            frames, conf = assemble_fused(Y.contiguous(), U.contiguous(), V.contiguous(), flows, FS[0])
        else:
            frames, conf, _ = ASM(Y, U, V, flows)
        tp = tick("assemble", tp)
        o = run(frames, conf)
        tp = tick("network", tp)
        c = frames[:, 3]
        if LIMIT:
            sy = fr[n_out][0]
            oy, ou, ov = unpack(o)
            oy = (limit_low(sy, oy[0] * 1023) if FAST else LIMFN(sy, oy[0] * 1023, grain_std(sy, curve), LIMIT, LIMWIN)) / 1023
            o = pack(oy[None], ou, ov)
        tp = tick("limiter", tp)
        if PREFIX:
            emit(None, *(a[0] * 1023 for a in unpack(c)))
        for a in ALPHAS:
            emit(a, *(x[0] * 1023 for x in unpack(c + a * (o - c))))
        tick("output copy", tp)
        n_out += 1
        for d in (fr, fwd, bwd):
            for j in [j for j in d if j < n_out - 4]:
                del d[j]
        if PROGRESS_EVERY and n_out % PROGRESS_EVERY == 0:
            print(f"PROGRESS {n_out}", flush=True)
        if n_out % 200 == 0:
            print(f"{os.path.basename(SRC)}: {n_out} frames  {n_out / (time.time() - t0):.1f} fps", flush=True)
dec.wait()
for a in keys:
    wq[a].put(None)
for t_ in wthreads:
    t_.join()
for a, e in encs.items():
    e.stdin.close()
    e.wait()
    os.replace(f"{OUT}/{name(a)}.part.{ext}", f"{OUT}/{name(a)}.{ext}")
print(f"{os.path.basename(SRC)}: done, {n_in} frames, alphas {ALPHAS}, {n_in / (time.time() - t0):.1f} fps", flush=True)
if PROF:
    print("per frame: " + "  ".join(f"{k} {v / max(n_in, 1) * 1000:.1f} ms" for k, v in TM.items() if k != "-"), flush=True)
