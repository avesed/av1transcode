"""Grain-auto after the final encode: give the encode back the grain it lacks, as AV1 film grain synthesis.

The steps grainauto's blind tests ran (grainauto.grain(), ga_bins.py, ga_table.py):
  1. target   per shot and brightness bin (8-bit luma index of the encode, AV1's scaling index), the std of
              source - encode on the encode's flat pixels (9 x 9 local std < 3 codes), luma and Cb / Cr at the co-sited
              index, every STEP-th frame. That is the grain the encode does not have.
  2. table    ga_table.py: the fixed fine "D" grain shape (AR coefficients from ND.tbl) with per-shot amplitudes
              scaled to the target through the calibration cal_D.json, one table segment per shot
  3. apply    grav1synth writes the table into the AV1 frame headers (no re-encode)
  4. applied  what the grain synthesis actually adds: the result decoded with minus without film grain
  5. again    the table rebuilt with the target / applied correction (CORR), applied once more
  flicker     logged, not acted on (owner 2026-10-07): grainauto's flicker_gpu.py swing of the fine and mid bands on
              flat static picture, source vs the first table's result, measured in step 4's decode pass
Bins are measured on the GPU (ga_bins.py's numpy cost about 0.3 s per 4K frame); the source is decoded on VA-API when
RENDER_NODE is set (sequential reads only: no seek, so none of the hwaccel seek traps).
  python3 grain_apply.py SRC VIDEO.ivf SHOTS.json FPS OUT.ivf WORKDIR [CHROMA]
SHOTS.json = {"shots": [{"frames": n}, ...]} in timeline order. Prints JSON {"target": ..., "applied": ...} summaries.
"""
import json, os, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
EDGES = [0, 16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256]
STEP = int(os.environ.get("GRAIN_STEP", "3"))
# the largest believable luma grain (std, 10-bit) a bin's median over shots may ask for: grainauto's sources measured
# 3.6-10.4, Breaking Bad's 4K 23.4. A broken source read (VA-API nv12 surfaces downloaded as p010) measured 95-847,
# the brightness itself: the source and the encode were not compared frame for frame, and the table built on it was
# all 255, the strongest grain AV1 can write. So the step fails instead.
MAX_TARGET_Y = 80.0
DONOR, CAL = f"{HERE}/assets/ND.tbl", f"{HERE}/assets/cal_D.json"
GRAV1SYNTH = os.environ.get("GRAV1SYNTH", "grav1synth")
dev = torch.device(os.environ.get("DEV", "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"))
EDGES_T = torch.tensor(EDGES[1:-1], dtype=torch.float32, device=dev)          # bucketize -> bin 0..10


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def size(path):
    s = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                                   "-of", "json", path], capture_output=True, text=True).stdout)["streams"][0]
    return s["width"], s["height"]


# VA-API surface format by the source's pixel format: hwdownload must name the surface's own format (p010le on an
# 8-bit H.264 source's nv12 surfaces read as garbage, luma about half, and made the grain target the brightness)
HW_DOWNLOAD = {"yuv420p": "nv12", "yuvj420p": "nv12", "nv12": "nv12", "yuv420p10le": "p010le", "p010le": "p010le"}


def pix_fmt(path):
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=pix_fmt",
                           "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip().split(",")[0]


def reader(path, w, h, grain=True, hw=False):
    """raw 10-bit frames in decode order = display order (-fps_mode passthrough), as (Y, U, V) float tensors on dev.
    grain=False exports the film grain parameters instead of applying them; hw decodes on VA-API (4:2:0 8/10-bit;
    anything else decodes in software)."""
    fs = w * h * 3 // 2
    pre = ([] if grain else ["-export_side_data", "film_grain"])
    node = os.environ.get("RENDER_NODE")
    surf = HW_DOWNLOAD.get(pix_fmt(path)) if hw and node else None
    if surf:
        pre += ["-hwaccel", "vaapi", "-hwaccel_device", node, "-hwaccel_output_format", "vaapi"]
        vf = ["-vf", f"hwdownload,format={surf},format=yuv420p10le"]
    else:
        vf = []
    p = subprocess.Popen(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", *pre, "-i", path, "-map", "0:v:0", *vf,
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-"], stdout=subprocess.PIPE)
    try:
        while True:
            b = p.stdout.read(fs * 2)
            if len(b) < fs * 2:
                break
            a = torch.from_numpy(np.frombuffer(b, np.int16)).to(dev).float()
            yield a[:w * h].view(h, w), a[w * h:w * h * 5 // 4].view(h // 2, w // 2), a[w * h * 5 // 4:].view(h // 2, w // 2)
    finally:
        p.kill()
        p.wait()


KF = {}


def blur(x, sg):
    if sg not in KF:
        r = int(np.ceil(3 * sg))
        t = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
        k = torch.exp(-t * t / (2 * sg * sg))
        KF[sg] = (k / k.sum(), r)
    k, r = KF[sg]
    x = F.conv2d(F.pad(x[None, None], (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))[0, 0]


class Flicker:
    """flicker_gpu.py: per frame, on the source's flat static mid-brightness pixels, the fine (|Y - g1.5|) and mid
    (|g1.5 - g4|) band energy of the source and of a variant; swing = mean |E_t - E_t-1| / mean E within each shot,
    pooled over shots by frames."""
    def __init__(self, shots):
        self.bounds = np.cumsum([0] + list(shots))
        self.E = {"source": [], "auto": []}
        self.prev = None

    def add(self, src, var):
        b3 = blur(src, 3.0)
        if self.prev is None:
            self.prev = b3
            for k in self.E:
                self.E[k].append((np.nan, np.nan))
            return
        gy, gx = torch.gradient(b3)
        struct = blur(torch.sqrt(gx * gx + gy * gy), 3.0)
        m = ((struct < 1.5) & ((b3 - self.prev).abs() < 0.5) & (src > 200) & (src < 800)).float()
        self.prev = b3
        if float(m.mean()) < 0.01:
            for k in self.E:
                self.E[k].append((np.nan, np.nan))
            return
        n = m.sum().clamp(min=1)
        for k, y in (("source", src), ("auto", var)):
            g1, g4 = blur(y, 1.5), blur(y, 4.0)
            self.E[k].append((float(((y - g1).abs() * m).sum() / n), float(((g1 - g4).abs() * m).sum() / n)))

    def result(self):
        res = {}
        for k, e in self.E.items():
            a = np.array(e, dtype=np.float64)
            out = {}
            for bi, band in enumerate(("fine", "mid")):
                sw, w, lev = [], [], []
                for s0, s1 in zip(self.bounds[:-1], self.bounds[1:]):
                    x = a[s0 + 1:s1, bi] if len(a) else np.array([])
                    x = x[~np.isnan(x)]
                    if len(x) > 5:
                        sw.append(np.mean(np.abs(np.diff(x))) / np.mean(x)); w.append(len(x)); lev.append(np.mean(x))
                out[f"{band}_swing"] = round(float(np.average(sw, weights=w)), 4) if sw else None
                out[f"{band}_level"] = round(float(np.average(lev, weights=w)), 3) if sw else None
            res[k] = out
        return res


def bins(mode, shots, path_a, path_b=None, hw_a=False, flicker_src=None):
    """ga_bins.py on the GPU. mode target: A = source, B = the encode; applied: A = with grain, B = without. In applied
    mode with flicker_src, every frame also feeds a Flicker of A against that source -> (bins, flicker result)."""
    w, h = size(path_a)
    bounds, o = [], 0
    for s in shots:
        bounds.append((o, o + s))
        o += s
    nb = len(EDGES) - 1
    acc = torch.zeros(len(bounds), nb, 5, dtype=torch.float64, device=dev)    # y, ny, cb, cr, nc
    ga = reader(path_a, w, h, True, hw_a)
    gb = reader(path_b, w, h, True) if mode == "target" else reader(path_a, w, h, False)
    fl = Flicker(shots) if flicker_src else None
    gs = reader(flicker_src, w, h, True, True) if flicker_src else None
    si, t0 = 0, time.time()
    with torch.no_grad():
        for f, (a, b) in enumerate(zip(ga, gb)):
            if fl is not None:
                s_ = next(gs, None)
                if s_ is not None:
                    fl.add(s_[0], a[0])
            if f % STEP:
                continue
            while si < len(bounds) and f >= bounds[si][1]:
                si += 1
            if si >= len(bounds):
                break
            (ay, au, av), (by, bu, bv) = a, b
            ry, ru, rv = ay - by, au - bu, av - bv
            if mode == "target":
                p = F.pad(by[None, None], (4, 4, 4, 4), mode="reflect")
                m1 = F.avg_pool2d(p, 9, 1)[0, 0]
                m2 = F.avg_pool2d(p * p, 9, 1)[0, 0]
                flat = (m2 - m1 * m1).clamp(min=0).sqrt() < 3.0
            else:
                flat = torch.ones_like(by, dtype=torch.bool)
            idx = by / 4
            fl_c = flat[0::2, 0::2] & flat[1::2, 0::2] & flat[0::2, 1::2] & flat[1::2, 1::2]
            idx_c = (idx[0::2, 0::2] + idx[1::2, 0::2] + idx[0::2, 1::2] + idx[1::2, 1::2]) / 4
            k = torch.bucketize(idx, EDGES_T, right=True)[flat]
            kc = torch.bucketize(idx_c, EDGES_T, right=True)[fl_c]
            acc[si, :, 0] += torch.bincount(k, weights=(ry[flat] ** 2).double(), minlength=nb)
            acc[si, :, 1] += torch.bincount(k, minlength=nb).double()
            acc[si, :, 2] += torch.bincount(kc, weights=(ru[fl_c] ** 2).double(), minlength=nb)
            acc[si, :, 3] += torch.bincount(kc, weights=(rv[fl_c] ** 2).double(), minlength=nb)
            acc[si, :, 4] += torch.bincount(kc, minlength=nb).double()
            if f and f % 3000 == 0:
                log(f"{mode}: {f} frames, {f / (time.time() - t0):.1f} fps")
    acc = acc.cpu().numpy()
    out = {"edges": EDGES, "mode": mode, "shots": []}
    for i, (s0, s1) in enumerate(bounds):
        bb = {}
        for kk, lo in enumerate(EDGES[:-1]):
            y, ny, cb, cr, nc = acc[i, kk]
            if ny >= 4000 and nc >= 1000:
                bb[str(lo)] = {"y": round(float((y / ny) ** 0.5), 3), "cb": round(float((cb / nc) ** 0.5), 3),
                               "cr": round(float((cr / nc) ** 0.5), 3), "n": int(ny)}
        out["shots"].append({"shot": i, "frames": [s0, s1], "bins": bb})
    return (out, fl.result()) if fl is not None else out


def summary(b):
    """median over shots per bin and channel, for the log and the job report."""
    res = {}
    for ch in ("y", "cb", "cr"):
        row = []
        for lo in EDGES[:-1]:
            v = [s["bins"][str(lo)][ch] for s in b["shots"] if str(lo) in s["bins"]]
            row.append(round(float(np.median(v)), 2) if v else None)
        res[ch] = row
    return res


def table(shots_json, fps, target_json, out_tbl, chroma, corr=None):
    env = dict(os.environ, **({"CORR": corr} if corr else {}))
    p = subprocess.run([sys.executable, f"{HERE}/ga_table.py", DONOR, shots_json, fps, "build", target_json, CAL, out_tbl,
                        str(chroma)], env=env, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"ga_table build failed: {p.stderr[-800:]}")
    return p.stdout


def apply(tbl, src_ivf, out_ivf):
    p = subprocess.run([GRAV1SYNTH, "apply", "-y", "-g", tbl, "-o", out_ivf, src_ivf], capture_output=True, text=True)
    if p.returncode or not os.path.exists(out_ivf):
        raise RuntimeError(f"grav1synth apply failed: {(p.stderr or p.stdout)[-800:]}")


def run(src, video, shots_json, fps, out_ivf, work, chroma=1.0):
    os.makedirs(work, exist_ok=True)
    shots = [s["frames"] for s in json.load(open(shots_json))["shots"]]
    log(f"target: {src} against {video}, {len(shots)} shots, {sum(shots)} frames")
    tgt = bins("target", shots, src, video, hw_a=True)
    ty = [v for v in summary(tgt)["y"] if v is not None]
    if not ty:
        raise RuntimeError("no flat picture to measure the grain on in any shot")
    if max(ty) > MAX_TARGET_Y:
        raise RuntimeError(f"grain target {max(ty):.1f} (luma std, 10-bit) is not grain: the source and the encode "
                           f"do not line up (per bin {ty})")
    tj = f"{work}/target.json"
    json.dump(tgt, open(tj, "w"))
    tbl = f"{work}/grain.tbl"
    table(shots_json, fps, tj, tbl, chroma)
    tmp = f"{work}/grain0.ivf"
    apply(tbl, video, tmp)
    log("applied: measuring the first table")
    app, flick = bins("applied", shots, tmp, flicker_src=src)
    aj = f"{work}/applied0.json"
    json.dump(app, open(aj, "w"))
    os.remove(tmp)
    corr_log = table(shots_json, fps, tj, tbl, chroma, f"{tj}:{aj}")
    apply(tbl, video, out_ivf)
    res = {"target": summary(tgt), "applied0": summary(app), "flicker": flick, "correction": corr_log.strip()[-400:],
           "table": tbl}
    log(f"flicker (logged only): {json.dumps(flick)}")
    log("done")
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    r = run(a[0], a[1], a[2], a[3], a[4], a[5], float(a[6]) if len(a) > 6 else 1.0)
    print("RESULT " + json.dumps(r), flush=True)
