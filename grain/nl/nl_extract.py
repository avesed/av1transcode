"""Perceptual noise statistics of whole clips (GPU, CUDA).

For NC centre frames per clip:
  1. the RVRT-distilled student denoises the centre frame at several strengths (5-frame window);
  2. source and denoised frames go to display luminance (cd/m^2): YUV -> R'G'B' -> EOTF (PQ absolute, clipped at
     the display peak; SDR BT.1886 gamma 2.4 at SDR_WHITE; HLG 1000-nit system) -> luminance;
  3. a Laplacian pyramid (5-tap binomial) of source and of each denoised frame. Per band k:
       La  = expanded Gaussian level k+1 of the denoised frame          (local adaptation luminance)
       N   = (B_k(src) - B_k(den)) / La, local RMS over 3x3              (noise contrast)
       M   = local RMS over 5x5 of B_k(den) / La                         (masker contrast: picture content)
     and a 3-D histogram over (log|N|, log M, log La) is accumulated over the centres;
  4. chroma: band-passed U/V residual (codes) at chroma levels 0-2, 2-D histogram over (log|Nc|, log La);
  5. h3_dt: as 3 but N from the frame-to-frame change of the residual (strength TSTR, centre vs centre+1, / sqrt 2):
     content the denoiser removed is the same in both frames when static and cancels, independent grain stays;
  6. scalars: active area, mean luminance per centre, motion, temporal correlation of the residual per band.
The histograms let the visibility model (CSF at any ppd, masking, pooling, black level, SDR white) be fitted and
applied on the CPU afterwards without touching the video again.

  python3 nl_extract.py [clip ids...]     (vLLM CUDA image; /w = noise, /t = teachers, /nl = nl)
"""
import json, os, queue, subprocess, sys, threading, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("NL_TEACHERS", "/t"))
from student_v2 import Student, pack, unpack

W, OUT = "/w", os.environ.get("NL_OUT", "/nl/hist")
CKPT = os.environ.get("NL_CKPT", "/t/out/student/s32v2/model.pt")
STRENGTHS = [30, 70, 120]
TSTR = 70                      # strength also run on centre+1 (temporal correlation of the residual)
NC = int(os.environ.get("NL_NC", 8))
NB = 6                         # luma bands
NBC = 3                        # chroma bands
PEAK_PQ = 1000.0
SDR_WHITE = 200.0
N_EDGES = np.arange(-4.0, 0.41, 0.2)
M_EDGES = np.arange(-3.6, 0.31, 0.3)
L_EDGES = np.arange(-3.0, 3.34, 1 / 3)
C_EDGES = np.arange(-1.5, 1.51, 0.15)        # chroma residual, log10 codes
os.makedirs(OUT, exist_ok=True)
dev = torch.device(os.environ.get("NL_DEVICE", "cuda"))
torch.backends.cudnn.benchmark = True
m1, m2, c1, c2, c3 = 0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875
EN = torch.tensor(N_EDGES, device=dev, dtype=torch.float32)
EM = torch.tensor(M_EDGES, device=dev, dtype=torch.float32)
EL = torch.tensor(L_EDGES, device=dev, dtype=torch.float32)
EC = torch.tensor(C_EDGES, device=dev, dtype=torch.float32)
nN, nM, nL, nC = len(N_EDGES) + 1, len(M_EDGES) + 1, len(L_EDGES) + 1, len(C_EDGES) + 1


def eotf(e, trc):
    """Non-linear [0,1] -> cd/m^2 on the owner's display model."""
    if trc == "smpte2084":
        p = e.clamp(0, 1) ** (1 / m2)
        return (10000 * ((p - c1).clamp(min=0) / (c2 - c3 * p)) ** (1 / m1)).clamp(max=PEAK_PQ)
    if trc == "arib-std-b67":
        e = e.clamp(0, 1)
        lin = torch.where(e <= 0.5, e * e / 3, (torch.exp((e - 0.55991073) / 0.17883277) + 0.28466892) / 12)
        return 1000 * lin ** 1.2          # luminance-only OOTF approximation, per channel
    return SDR_WHITE * e.clamp(0, 1) ** 2.4


def luminance(y, u, v, trc, bt2020):
    """y (H, W) codes, u/v (H/2, W/2) codes -> display luminance (H, W) cd/m^2."""
    Y = (y - 64) / 876
    up = lambda c: F.interpolate(c[None, None], scale_factor=2, mode="bilinear", align_corners=False)[0, 0]
    Cb, Cr = (up(u) - 512) / 896, (up(v) - 512) / 896
    if bt2020:
        R, G, B = Y + 1.4746 * Cr, Y - 0.16455 * Cb - 0.57135 * Cr, Y + 1.8814 * Cb
        kr, kg, kb = 0.2627, 0.6780, 0.0593
    else:
        R, G, B = Y + 1.5748 * Cr, Y - 0.1873 * Cb - 0.4681 * Cr, Y + 1.8556 * Cb
        kr, kg, kb = 0.2126, 0.7152, 0.0722
    return kr * eotf(R, trc) + kg * eotf(G, trc) + kb * eotf(B, trc)


K5 = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32, device=dev) / 16
K3 = torch.tensor([1, 2, 1], dtype=torch.float32, device=dev) / 4


def sep(x, k):
    """Separable blur of (H, W) with kernel k, reflect padding."""
    r = len(k) // 2
    x = F.pad(x[None, None], (r, r, r, r), mode="reflect" if min(x.shape[-2:]) > r else "replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    return x[0, 0]


def down(x):
    return sep(x, K5)[::2, ::2]


def expand(x, shape):
    return F.interpolate(x[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0]


def gpyr(x, n):
    g = [x]
    for _ in range(n):
        g.append(down(g[-1]))
    return g


def lap(g, k):
    up = expand(g[k + 1], g[k].shape)
    return g[k] - up, up


def hist3(n, m, la):
    i = torch.bucketize(torch.log10(n.clamp(min=1e-8)).reshape(-1), EN)
    j = torch.bucketize(torch.log10(m.clamp(min=1e-8)).reshape(-1), EM)
    l = torch.bucketize(torch.log10(la.clamp(min=1e-8)).reshape(-1), EL)
    return torch.bincount((i * nM + j) * nL + l, minlength=nN * nM * nL).reshape(nN, nM, nL)


def hist2(c, la):
    i = torch.bucketize(torch.log10(c.clamp(min=1e-8)).reshape(-1), EC)
    l = torch.bucketize(torch.log10(la.clamp(min=1e-8)).reshape(-1), EL)
    return torch.bincount(i * nL + l, minlength=nC * nL).reshape(nC, nL)


def probe(path):
    s = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height,color_transfer,color_space,r_frame_rate", "-of", "json", path],
                                  capture_output=True, text=True).stdout)["streams"][0]
    return s


CLIPDIR = os.environ.get("NL_CLIPDIR", f"{W}/clips")


def decode(cid):
    path = f"{CLIPDIR}/{cid}.mp4"
    s = probe(path)
    w, h = s["width"], s["height"]
    raw = np.frombuffer(subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "8", "-i", path, "-map", "0:v:0",
                                        "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-"],
                                       capture_output=True, check=True).stdout, np.uint16)
    fs = w * h * 3 // 2
    T = len(raw) // fs
    return cid, s, raw[:T * fs].reshape(T, fs)


def active_box(yall):
    """Rows/cols that are ever not bar-black (letterbox / pillarbox) over the sampled frames."""
    mx = yall.amax(0)                                   # (H, W) max over frames
    rows = torch.where((mx > 68).float().mean(1) > 0.02)[0]
    cols = torch.where((mx > 68).float().mean(0) > 0.02)[0]
    if len(rows) < 64 or len(cols) < 64:
        return 0, yall.shape[1], 0, yall.shape[2]
    r0, r1 = int(rows[0]) // 2 * 2, (int(rows[-1]) + 1) // 2 * 2
    c0, c1_ = int(cols[0]) // 2 * 2, (int(cols[-1]) + 1) // 2 * 2
    return r0, r1, c0, c1_


@torch.no_grad()
def denoise(net, fr, w, h, c, s):
    """Student on frames c-2..c+2 at strength s -> denoised y (H, W) and u, v (H/2, W/2) in codes."""
    win = torch.from_numpy(fr[c - 2:c + 3].astype(np.float32)).to(dev) / 1023
    y = win[:, :w * h].reshape(5, h, w)
    u = win[:, w * h:w * h * 5 // 4].reshape(5, h // 2, w // 2)
    v = win[:, w * h * 5 // 4:].reshape(5, h // 2, w // 2)
    Pk = pack(y, u, v)
    hh, ww = Pk.shape[-2:]
    ph, pw = (-hh) % 4, (-ww) % 4
    if ph or pw:
        Pk = F.pad(Pk, (0, pw, 0, ph), mode="reflect")
    nm = torch.full((1, 1, Pk.shape[-2], Pk.shape[-1]), s / 876.0, device=dev)
    o = net(Pk[None], nm)[0][..., :hh, :ww]
    dy, du, dv = unpack(o[None])
    return dy[0] * 1023, du[0] * 1023, dv[0] * 1023


@torch.no_grad()
def measure(net, item):
    cid, s, fr = item
    w, h, trc = s["width"], s["height"], s.get("color_transfer") or "bt709"
    bt2020 = (s.get("color_space") or "").startswith("bt2020") or trc in ("smpte2084", "arib-std-b67")
    T = len(fr)
    centres = np.linspace(2, T - 4, NC).astype(int)
    ysamp = torch.from_numpy(fr[centres, :w * h].reshape(NC, h, w).astype(np.float32)).to(dev)
    r0, r1, c0, c1_ = active_box(ysamp)
    del ysamp
    H3 = {st: [torch.zeros(nN, nM, nL, dtype=torch.int64, device=dev) for _ in range(NB)] for st in STRENGTHS}
    HC = [torch.zeros(nC, nL, dtype=torch.int64, device=dev) for _ in range(NBC)]
    HD = [torch.zeros(nN, nM, nL, dtype=torch.int64, device=dev) for _ in range(NB)]
    tcorr = np.zeros((NC, NB))
    lmean, motion, lmed = [], [], []
    for ci, c in enumerate(centres):
        f = torch.from_numpy(fr[c].astype(np.float32)).to(dev)
        y = f[:w * h].reshape(h, w)
        u = f[w * h:w * h * 5 // 4].reshape(h // 2, w // 2)
        v = f[w * h * 5 // 4:].reshape(h // 2, w // 2)
        sl = (slice(r0, r1), slice(c0, c1_))
        slc = (slice(r0 // 2, r1 // 2), slice(c0 // 2, c1_ // 2))
        Ls = luminance(y, u, v, trc, bt2020)[sl] + 1e-4
        lmean.append(float(Ls.mean()))
        lmed.append(float(Ls.median()))
        prev = torch.from_numpy(fr[c - 1, :w * h].astype(np.float32)).to(dev).reshape(h, w)[sl]
        motion.append(float((F.avg_pool2d(y[sl][None, None], 4) - F.avg_pool2d(prev[None, None], 4)).abs().mean()))
        gs = gpyr(Ls, NB)
        resid_t = {}
        for st in STRENGTHS:
            dy, du, dv = denoise(net, fr, w, h, c, st)
            Ld = luminance(dy, du, dv, trc, bt2020)[sl] + 1e-4
            gd = gpyr(Ld, NB)
            for k in range(NB):
                Bs, _ = lap(gs, k)
                Bd, La = lap(gd, k)
                La = La.clamp(min=1e-4)
                nres = (Bs - Bd) / La
                n = sep(nres * nres, K3).clamp(min=0).sqrt()
                m = sep((Bd / La) ** 2, K5).clamp(min=0).sqrt()
                H3[st][k] += hist3(n, m, La)
                if st == TSTR:
                    resid_t[k] = (nres, m, La)
            if st == TSTR:
                # chroma residual, band-passed, at chroma resolution; adaptation from luma Gaussian level 1
                ru, rv = (u - du)[slc], (v - dv)[slc]
                gu, gv = gpyr(ru, NBC), gpyr(rv, NBC)
                for k in range(NBC):
                    bu, _ = lap(gu, k)
                    bv, _ = lap(gv, k)
                    e = sep(bu * bu + bv * bv, K3).clamp(min=0).sqrt()
                    la = expand(gd[k + 1], e.shape)
                    HC[k] += hist2(e, la.clamp(min=1e-4))
        # temporal correlation of the residual (strength TSTR) between centre and centre+1, per band
        f1 = torch.from_numpy(fr[c + 1].astype(np.float32)).to(dev)
        y1 = f1[:w * h].reshape(h, w)
        u1 = f1[w * h:w * h * 5 // 4].reshape(h // 2, w // 2)
        v1 = f1[w * h * 5 // 4:].reshape(h // 2, w // 2)
        L1 = luminance(y1, u1, v1, trc, bt2020)[sl] + 1e-4
        dy, du, dv = denoise(net, fr, w, h, c + 1, TSTR)
        Ld1 = luminance(dy, du, dv, trc, bt2020)[sl] + 1e-4
        g1, gd1 = gpyr(L1, NB), gpyr(Ld1, NB)
        for k in range(NB):
            B1, _ = lap(g1, k)
            Bd1, La1 = lap(gd1, k)
            n1 = (B1 - Bd1) / La1.clamp(min=1e-4)
            n0, m0, la0 = resid_t[k]
            a, b = n0.reshape(-1), n1.reshape(-1)
            tcorr[ci, k] = float((a * b).mean() / ((a * a).mean().sqrt() * (b * b).mean().sqrt() + 1e-12))
            dt = (n0 - n1) * 0.7071067811865476          # frame-to-frame residual change: independent noise only
            HD[k] += hist3(sep(dt * dt, K3).clamp(min=0).sqrt(), m0, la0)
    out = {f"h3_s{st}": torch.stack(H3[st]).cpu().numpy().astype(np.int32) for st in STRENGTHS}
    out["hc"] = torch.stack(HC).cpu().numpy().astype(np.int32)
    out["h3_dt"] = torch.stack(HD).cpu().numpy().astype(np.int32)
    np.savez_compressed(f"{OUT}/{cid}.npz", **out, tcorr=tcorr, lmean=np.array(lmean), lmed=np.array(lmed),
                        motion=np.array(motion), box=np.array([r0, r1, c0, c1_]), w=w, h=h, trc=trc, bt2020=bt2020,
                        fps=s.get("r_frame_rate", ""), frames=T, centres=centres, strengths=np.array(STRENGTHS),
                        n_edges=N_EDGES, m_edges=M_EDGES, l_edges=L_EDGES, c_edges=C_EDGES, peak_pq=PEAK_PQ,
                        sdr_white=SDR_WHITE)
    return out


@torch.no_grad()
def window_hist_dt(net, fr, w, h, trc, bt2020):
    """Inference path: a 6-frame window (6, w*h*3/2 codes) -> (histograms (NB, nN, nM, nL), motion), built exactly as
    the training extraction builds them (centre frame 2, frame 3 for the frame-to-frame change; letterbox from the
    window): h3_dt and h3_s70 as measure(), h3_dts and h3_frz (static pixels, common residual) as nl_extract2."""
    ys = torch.from_numpy(fr[:, :w * h].reshape(len(fr), h, w).astype(np.float32)).to(dev)
    r0, r1, c0, c1_ = active_box(ys)
    if r1 - r0 < 512 or c1_ - c0 < 512:                  # (near-)black window: measure the whole frame
        r0, r1, c0, c1_ = 0, h, 0, w
    del ys
    sl = (slice(r0, r1), slice(c0, c1_))
    y2 = torch.from_numpy(fr[2, :w * h].astype(np.float32)).to(dev).reshape(h, w)[sl]
    y1 = torch.from_numpy(fr[1, :w * h].astype(np.float32)).to(dev).reshape(h, w)[sl]
    motion = float((F.avg_pool2d(y2[None, None], 4) - F.avg_pool2d(y1[None, None], 4)).abs().mean())
    res = []
    for c in (2, 3):
        f = torch.from_numpy(fr[c].astype(np.float32)).to(dev)
        y = f[:w * h].reshape(h, w)
        u = f[w * h:w * h * 5 // 4].reshape(h // 2, w // 2)
        v = f[w * h * 5 // 4:].reshape(h // 2, w // 2)
        Ls = luminance(y, u, v, trc, bt2020)[sl] + 1e-4
        dy, du, dv = denoise(net, fr, w, h, c, TSTR)
        Ld = luminance(dy, du, dv, trc, bt2020)[sl] + 1e-4
        gs, gd = gpyr(Ls, NB + 2), gpyr(Ld, NB + 2)
        bands = []
        for k in range(NB):
            Bs, _ = lap(gs, k)
            Bd, La = lap(gd, k)
            La = La.clamp(min=1e-4)
            m = sep((Bd / La) ** 2, K5).clamp(min=0).sqrt() if c == 2 else None
            bands.append(((Bs - Bd) / La, m, La, gd[k + 2]))
        res.append(bands)
    H = {"h3_dt": [], f"h3_s{TSTR}": [], "h3_dts": [], "h3_frz": []}
    for k in range(NB):
        n0, m0, la0, g0 = res[0][k]
        n1, _, _, g1 = res[1][k]
        dt = (n0 - n1) * 0.7071067811865476
        ed = sep(dt * dt, K3).clamp(min=0).sqrt()
        H["h3_dt"].append(hist3(ed, m0, la0))
        H[f"h3_s{TSTR}"].append(hist3(sep(n0 * n0, K3).clamp(min=0).sqrt(), m0, la0))
        st = expand((torch.log10(g0) - torch.log10(g1)).abs(), n0.shape) < 0.005
        fz = (n0 + n1) * 0.5
        ef = sep(fz * fz, K3).clamp(min=0).sqrt()
        if st.any():
            H["h3_dts"].append(hist3(ed[st], m0[st], la0[st]))
            H["h3_frz"].append(hist3(ef[st], m0[st], la0[st]))
        else:
            z = torch.zeros(nN, nM, nL, dtype=torch.int64, device=dev)
            H["h3_dts"].append(z)
            H["h3_frz"].append(z.clone())
    return {k: torch.stack(v).cpu().numpy().astype(np.int64) for k, v in H.items()}, motion


def main():
    ck = torch.load(CKPT, map_location="cpu")
    net = Student(ck["channels"])
    net.load_state_dict(ck["state"])
    net = net.eval().to(dev)
    ids = sys.argv[1:]
    if not ids:
        ids = sorted(f[:-4] for f in os.listdir(CLIPDIR) if f.endswith(".mp4"))
    ids = [c for c in ids if not os.path.exists(f"{OUT}/{c}.npz")]
    q = queue.Queue(maxsize=1)

    def producer():
        for c in ids:
            try:
                q.put(decode(c))
            except Exception as e:  # noqa: BLE001
                print(f"{c}: decode failed: {e}", flush=True)
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    done, t0 = 0, time.time()
    while (item := q.get()) is not None:
        t1 = time.time()
        try:
            measure(net, item)
        except Exception as e:  # noqa: BLE001
            print(f"{item[0]}: failed: {e}", flush=True)
            continue
        done += 1
        print(f"{done}/{len(ids)} {item[0]} {item[1]['width']}x{item[1]['height']} {item[1].get('color_transfer')} "
              f"{time.time() - t1:.1f}s eta {(time.time() - t0) / done * (len(ids) - done) / 60:.0f} min "
              f"gpu {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
