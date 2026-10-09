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
              flat static picture, source vs the first table's result, measured in step 4's decode pass on the
              FLICKER_SHOTS hardest shots only - most flat picture times grain to synthesise, from step 1 - and
              reported per shot. Every frame of every shot cost the whole step 46 of its 75 min on a 1080p episode.
Bins are measured on the GPU (ga_bins.py's numpy cost about 0.3 s per 4K frame); every stream is decoded on the CPU,
each in a thread of its own (prefetch). Every frame is decoded, but only the frames a step needs (every STEP-th, and
the flicker shots' every frame) leave ffmpeg, through select.
  6. strength  the corrected table's scaling times STRENGTH (the owner's 30% less, test-10)
  python3 grain_apply.py SRC VIDEO.ivf SHOTS.json FPS OUT.ivf WORKDIR [CHROMA [STRENGTH]]
SHOTS.json = {"shots": [{"frames": n}, ...]} in timeline order. Prints JSON {"target": ..., "applied": ...} summaries.
"""
import json, os, re, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
EDGES = [0, 16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256]
STEP = int(os.environ.get("GRAIN_STEP", "3"))
FLICKER_SHOTS = int(os.environ.get("GRAIN_FLICKER_SHOTS", "10"))
FLICKER_MIN_FRAMES = 24                       # a swing needs a run of frames; a 1 s shot is the shortest worth it
FLICKER_BINS = (48, 64, 80, 96, 128, 160)     # the 8-bit brightness bins inside the flicker mask's 200-800 (10-bit)
FLICKER_PACE = 12                             # see bins(): the flicker source's read lets every 12th frame through too
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


def probe(path, key):
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", f"stream={key}",
                           "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip().split(",")[0]


# dav1d on the engine's single-tile 4K encodes is bound by how many frames it has in flight, not by cores: two reads
# together take as long as one. 32 frames instead of its own 8: 38.5 -> 28.3 s for the applied pass's two reads of
# the 1458-frame Breaking Bad clip, 2.5 -> 4.5 GB per decoder (64: 24 s and 7.3 GB, 128: 24 s and 12.7 GB).
DAV1D_FRAMES = int(os.environ.get("GRAIN_DAV1D_FRAMES", "32"))


def keep(step, ranges=()):
    """select expression for frame number n: every step-th frame (step 0: none) and every frame of the [a, b) ranges."""
    terms = ([f"not(mod(n,{step}))"] if step else []) + [f"between(n,{a},{b - 1})" for a, b in ranges]
    return "+".join(terms) or "0"


def kept(total, step, ranges=()):
    """the frame numbers keep(step, ranges) lets through, in order."""
    f = set(range(0, total, step)) if step else set()
    for a, b in ranges:
        f.update(range(a, b))
    return sorted(f)


def planes(t, w, h):
    """a reader_raw frame -> (Y, U, V) float tensors on dev."""
    a = t.to(dev).float()
    return a[:w * h].view(h, w), a[w * h:w * h * 5 // 4].view(h // 2, w // 2), a[w * h * 5 // 4:].view(h // 2, w // 2)


def reader_raw(path, w, h, grain=True, hw=False, select=None):
    """raw 10-bit frames in decode order = display order (-fps_mode passthrough), as int16 CPU tensors (one per frame,
    Y U V planar; planes() puts them on dev). grain=False exports the film grain parameters instead of applying them;
    hw decodes on VA-API (4:2:0 8/10-bit; anything else decodes in software). select: a keep() expression; only those
    frames are read out, in order."""
    fs = w * h * 3 // 2
    pre = ([] if grain else ["-export_side_data", "film_grain"])
    node = os.environ.get("RENDER_NODE")
    surf = HW_DOWNLOAD.get(probe(path, "pix_fmt")) if hw and node else None
    if not surf and DAV1D_FRAMES and probe(path, "codec_name") == "av1":
        pre += ["-threads", str(len(os.sched_getaffinity(0))), "-max_frame_delay", str(DAV1D_FRAMES)]
    chain = [f"select='{select}'"] if select else []      # before hwdownload: a dropped frame never crosses PCIe
    if surf:
        pre += ["-hwaccel", "vaapi", "-hwaccel_device", node, "-hwaccel_output_format", "vaapi"]
        chain += ["hwdownload", f"format={surf}", "format=yuv420p10le"]
    vf = ["-vf", ",".join(chain)] if chain else []
    p = subprocess.Popen(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", *pre, "-i", path, "-map", "0:v:0", *vf,
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-"], stdout=subprocess.PIPE)
    try:
        while True:
            b = p.stdout.read(fs * 2)
            if len(b) < fs * 2:
                break
            yield torch.from_numpy(np.frombuffer(b, np.int16))
    finally:
        p.kill()
        p.wait()


# GRAIN_HWDEC=1 decodes on VA-API instead. Off: on the 7.0.0-34 kernel a 4K HEVC decode on the B580 beside the
# bins' compute delivered wrong pictures - half of one frame, then the rest of its GOP through the references (frames
# 1302-1347 of the Breaking Bad clip, luma off by 20-117 codes on average, target 4.3 -> 67) - with nothing in the
# kernel log, and only while the compute ran. Exact alone, and exact under synthetic loads; read in turn it got away
# with it. GRAIN_PREFETCH: frames each stream decodes ahead in a thread of its own (0 = read in turn, as before).
HWDEC = os.environ.get("GRAIN_HWDEC", "0") == "1"
PREFETCH = int(os.environ.get("GRAIN_PREFETCH", "4"))


def prefetch(gen, n=None):
    """gen's items made by a thread of its own, up to n ahead. A 4K episode's grain step read two or three 25 MB frame
    streams in turn on one Python thread - pipe read, int16 to float, copy to the GPU, then the bins - and that thread
    sat at 100% of a core with the decoders waiting on it. Each stream's pipe reads now overlap the others' and the
    bins. CPU work only (reader_raw): planes() puts each frame on the GPU from the consuming thread."""
    import queue
    import threading
    n = PREFETCH if n is None else n
    if n <= 0:
        yield from gen
        return
    q, end, err, stop = queue.Queue(maxsize=n), object(), [], threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.5)
                return True
            except queue.Full:
                pass
        return False

    def work():
        try:
            for item in gen:
                if not put(item):
                    break
        except BaseException as e:                       # noqa: BLE001 - raised again on the consumer's side
            err.append(e)
        finally:
            gen.close()                                  # the reader's finally kills its ffmpeg
            put(end)
    threading.Thread(target=work, daemon=True).start()
    try:
        while True:
            item = q.get()
            if item is end:
                if err:
                    raise err[0]
                return
            yield item
    finally:
        stop.set()


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
    """flicker_gpu.py, per shot: per frame, on the source's flat static mid-brightness pixels, the fine (|Y - g1.5|)
    and mid (|g1.5 - g4|) band energy of the source and of a variant; swing = mean |E_t - E_t-1| / mean E within the
    shot (its first frame has no previous one to call the picture static against). Pooled over shots by frames."""
    def __init__(self):
        self.E = {}
        self.prev, self.cur = None, None

    def add(self, shot, src, var):
        b3 = blur(src, 3.0)
        if shot != self.cur:
            self.cur, self.prev = shot, b3
            self.E[shot] = {"source": [], "auto": []}
            return
        gy, gx = torch.gradient(b3)
        struct = blur(torch.sqrt(gx * gx + gy * gy), 3.0)
        m = ((struct < 1.5) & ((b3 - self.prev).abs() < 0.5) & (src > 200) & (src < 800)).float()
        self.prev = b3
        if float(m.mean()) < 0.01:
            return
        n = m.sum().clamp(min=1)
        for k, y in (("source", src), ("auto", var)):
            g1, g4 = blur(y, 1.5), blur(y, 4.0)
            self.E[shot][k].append((float(((y - g1).abs() * m).sum() / n), float(((g1 - g4).abs() * m).sum() / n)))

    @staticmethod
    def _bands(e):
        a = np.array(e, dtype=np.float64).reshape(-1, 2)
        out = {}
        for bi, band in enumerate(("fine", "mid")):
            x = a[:, bi]
            if len(x) > 5:
                out[f"{band}_swing"] = round(float(np.mean(np.abs(np.diff(x))) / np.mean(x)), 4)
                out[f"{band}_level"] = round(float(np.mean(x)), 3)
        return out, len(a)

    def result(self, bounds, scores):
        res = {"source": {}, "auto": {}, "shots": []}
        pooled = {k: {} for k in ("source", "auto")}
        for shot, e in self.E.items():
            row = {"shot": shot, "frames": list(bounds[shot]), "score": round(scores.get(shot, 0.0), 1)}
            for k in ("source", "auto"):
                row[k], n = self._bands(e[k])
                row["measured"] = n
                for key, v in row[k].items():
                    pooled[k].setdefault(key, []).append((v, n))
            s_, a_ = row["source"].get("fine_swing"), row["auto"].get("fine_swing")
            row["fine_ratio"] = round(a_ / s_, 3) if s_ and a_ else None
            res["shots"].append(row)
        for k in ("source", "auto"):
            for key, vw in pooled[k].items():
                res[k][key] = round(float(np.average([v for v, _ in vw], weights=[w for _, w in vw])), 4)
        res["shots"].sort(key=lambda r: -(r["fine_ratio"] or 0))
        return res


def hardest(tgt, k=FLICKER_SHOTS):
    """the k shots where flicker would show most: flat picture inside the flicker mask's brightness, per frame
    measured, times the luma grain to synthesise there squared -> {shot: score}, at least FLICKER_MIN_FRAMES long."""
    scores = {}
    for s in tgt["shots"]:
        s0, s1 = s["frames"]
        sampled = len(range(s0 - s0 % STEP + (STEP if s0 % STEP else 0), s1, STEP)) or 1
        if s1 - s0 < FLICKER_MIN_FRAMES:
            continue
        sc = sum(b["n"] / sampled * b["y"] ** 2 for lo, b in s["bins"].items() if int(lo) in FLICKER_BINS) / 1e3
        if sc > 0:
            scores[s["shot"]] = sc
    return dict(sorted(scores.items(), key=lambda t: -t[1])[:k])


def bins(mode, shots, path_a, path_b=None, hw_a=False, flicker_src=None, flicker_shots=None):
    """ga_bins.py on the GPU, on every STEP-th frame. mode target: A = source, B = the encode; applied: A = with
    grain, B = without it (path_b, the encode it was written into; else A with its grain exported). In applied mode
    with flicker_src, every frame of the flicker_shots ({shot: score}) also feeds a Flicker of A against that source
    -> (bins, flicker result). hw_a: A may decode on VA-API (only with GRAIN_HWDEC=1)."""
    w, h = size(path_a)
    bounds, o = [], 0
    for s in shots:
        bounds.append((o, o + s))
        o += s
    total = o
    nb = len(EDGES) - 1
    acc = torch.zeros(len(bounds), nb, 5, dtype=torch.float64, device=dev)    # y, ny, cb, cr, nc
    fl_ranges = [bounds[i] for i in sorted(flicker_shots or {})] if flicker_src else []
    fl_frames = set(kept(total, 0, fl_ranges))
    ga = prefetch(reader_raw(path_a, w, h, True, HWDEC and hw_a, keep(STEP, fl_ranges)))
    if mode == "target" or path_b:
        # target: the encode, which has no grain yet; applied: the encode the grain was written into (path_b), the
        # very pictures path_a decodes to with its grain left out
        gb = prefetch(reader_raw(path_b, w, h, True, False, keep(STEP)))
    else:
        gb = prefetch(reader_raw(path_a, w, h, False, False, keep(STEP)))  # path_a, its grain exported, not applied
    fl = Flicker() if fl_ranges else None
    # the flicker source also lets every FLICKER_PACE-th frame through, and the loop pulls it along frame by frame: ffmpeg
    # held back the last few frames of a selected run until its next selected frame (or the end of the file), and a
    # source read nobody pulls between flicker shots stops decoding once its queue is full - on S01E02 of Good Girls the
    # loop sat waiting while the source decoded tens of thousands of frames, 16 of the applied pass's 33 minutes
    gs = zip(kept(total, FLICKER_PACE, fl_ranges),
             prefetch(reader_raw(flicker_src, w, h, True, HWDEC, keep(FLICKER_PACE, fl_ranges)))) if fl is not None else None
    gs_at = next(gs, None) if gs is not None else None
    si, fi, t0, done = 0, 0, time.time(), 0
    with torch.no_grad():
        for f in kept(total, STEP, fl_ranges):
            a = next(ga, None)
            if a is None:
                break
            a = planes(a, w, h)
            while gs_at is not None and gs_at[0] < f:
                gs_at = next(gs, None)
            if f in fl_frames:
                s_ = planes(gs_at[1], w, h) if gs_at is not None and gs_at[0] == f else None
                while fi < len(fl_ranges) and f >= fl_ranges[fi][1]:
                    fi += 1
                if s_ is not None:
                    fl.add(bounds.index(fl_ranges[fi]), s_[0], a[0])
            if f % STEP:
                continue
            b = next(gb, None)
            if b is None:
                break
            b = planes(b, w, h)
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
            done += 1
            if done % 1000 == 0:
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
    if not flicker_src:
        return out
    return out, (fl.result(bounds, flicker_shots) if fl is not None else {"source": {}, "auto": {}, "shots": []})


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


# where a failed apply leaves its table, input and full log (the job's work dir is removed with the job): the
# service's /cache volume, the latest failure only. S01E03 of A Good Girl's Guide failed in here on 2026-10-08 with
# "[av1_metadata] Failed to write unit 1 (type 6)" and nothing else kept to reproduce it with.
DEBUG_DIR = os.environ.get("GRAIN_DEBUG_DIR", "/cache/grain-debug" if os.path.isdir("/cache") else "")


def apply(tbl, src_ivf, out_ivf):
    # --replace: without it grav1synth skips a stream that already has grain headers, exit 0 and no output
    p = subprocess.run([GRAV1SYNTH, "apply", "-y", "--replace", "-g", tbl, "-o", out_ivf, src_ivf], capture_output=True,
                       text=True)
    if p.returncode or not os.path.exists(out_ivf):
        out = (p.stderr or "") + (p.stdout or "")
        kept = ""
        if DEBUG_DIR:
            import shutil
            shutil.rmtree(DEBUG_DIR, ignore_errors=True)
            os.makedirs(DEBUG_DIR, exist_ok=True)
            shutil.copy(tbl, f"{DEBUG_DIR}/grain.tbl")
            shutil.copy(src_ivf, f"{DEBUG_DIR}/input.ivf")         # a 4K hour is 3-5 GB; only the latest is kept
            open(f"{DEBUG_DIR}/grav1synth.log", "w").write(f"rc {p.returncode}\n{out}")
            kept = f" (kept in {DEBUG_DIR})"
        # on one line, the first lines that name the problem: the service shows a job's last log lines, 300
        # characters each, and the last ones were the bitstream filter's summary of it
        first = [l.strip() for l in out.splitlines() if re.search(r"error|invalid|range|match|fail", l, re.I)][:3]
        why = " | ".join(first) or out.strip()[-200:].replace("\n", " | ")
        raise RuntimeError(f"grav1synth apply failed{kept}: {why}")


def scaled(tbl_in, tbl_out, k):
    """the table with every sY / sCb / sCr scaling value times k (each point's x kept): the grain's amplitude times k,
    since AV1 adds scaling(y) * grain >> scaling_shift."""
    out = []
    for line in open(tbl_in):
        t = line.split()
        if t and t[0] in ("sY", "sCb", "sCr"):
            pts = [int(x) for x in t[2:2 + 2 * int(t[1])]]
            pts[1::2] = [min(255, int(round(v * k))) for v in pts[1::2]]
            line = "\t" + " ".join([t[0], t[1]] + [str(x) for x in pts]) + "\n"
        out.append(line)
    open(tbl_out, "w").write("".join(out))


def run(src, video, shots_json, fps, out_ivf, work, chroma=1.0, strength=1.0):
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
    hard = hardest(tgt)
    log(f"applied: measuring the first table; flicker on the {len(hard)} hardest shots {sorted(hard)}")
    app, flick = bins("applied", shots, tmp, video, flicker_src=src, flicker_shots=hard)
    aj = f"{work}/applied0.json"
    json.dump(app, open(aj, "w"))
    os.remove(tmp)
    corr_log = table(shots_json, fps, tj, tbl, chroma, f"{tj}:{aj}")
    if strength != 1.0:
        # the measured grain times strength, on the corrected table: test-10 (2026-10-07), the owner picked 30% less
        # grain on two clips of four and called the other two the same
        scaled(tbl, f"{work}/grain_s.tbl", strength)
        tbl = f"{work}/grain_s.tbl"
    apply(tbl, video, out_ivf)
    res = {"target": summary(tgt), "applied0": summary(app), "flicker": flick, "correction": corr_log.strip()[-400:],
           "strength": strength, "table": tbl}
    log(f"flicker (logged only): {json.dumps(flick)}")
    log("done")
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    r = run(a[0], a[1], a[2], a[3], a[4], a[5], float(a[6]) if len(a) > 6 else 1.0, float(a[7]) if len(a) > 7 else 1.0)
    print("RESULT " + json.dumps(r), flush=True)
