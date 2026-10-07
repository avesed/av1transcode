"""Grain-auto analysis: is a source film grain the denoiser should take off (on), or not (off)?

Two measurements, the same ones grainauto's blind tests were decided by (test-03 .. test-09):
  level     perceived noise on the owner's 1-50 scale, nl/nl_level.py's physical visibility model (CSF, masking,
            display model) on sampled windows. The stacked CNN / ARNIQA heads are left out: on the 14 sources tested
            the physical level never came near the only threshold that uses it (12; lowest 13.7).
  features  on PAIRS consecutive frame pairs spread over the file, against a probe denoise (the RVRT-distilled v2
            student at sigma 60, 5-frame window), on flat static mid-brightness pixels:
              kurtosis    excess kurtosis of the source frame difference: 0 for Gaussian film grain, large for sparse /
                          spiky deliberate texture
              fresh_corr  correlation of the residual's high-pass at t and t+1: 0 for grain new every frame, 1 frozen
            grainauto's ga_features.py read every 12th pair of the whole file; medians over PAIRS sampled pairs are what
            a full episode can afford.
Rules (grainauto RULES): level < 12 -> clean (off); kurtosis > 3.0 or fresh_corr > 0.35 -> texture (off: the old VMAF
target raise went, test-09: +25-73% bits for "same" or worse); otherwise grain (on).
  python3 analyse.py SRC [--pairs 48] [--json OUT]   (grain service image; DEV xpu|cuda|cpu)
"""
import argparse, json, math, os, subprocess, sys, tempfile
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from student_v2 import Student, pack, unpack

RULES = {"clean_level": 12.0, "texture_kurtosis": 3.0, "texture_frozen": 0.35}
WEIGHTS = os.environ.get("GRAIN_WEIGHTS", "/weights")
dev = torch.device(os.environ.get("DEV", "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"))
KS = {}


def blur(x, sg):
    """separable Gaussian, reflect padding (as ga_features.py)."""
    if sg not in KS:
        r = int(math.ceil(3 * sg))
        t = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
        k = torch.exp(-t * t / (2 * sg * sg))
        KS[sg] = (k / k.sum(), r)
    k, r = KS[sg]
    x = F.conv2d(F.pad(x[None, None], (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))[0, 0]


def probe(path):
    s = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height,r_frame_rate:format=duration", "-of", "json", path],
                                  capture_output=True, text=True).stdout)
    v = s["streams"][0]
    return {"w": v["width"], "h": v["height"], "fps": v["r_frame_rate"], "duration": float(s["format"]["duration"])}


def level(path):
    """physical nl_level on the default sample windows -> its JSON (level is 'level')."""
    with tempfile.NamedTemporaryFile(suffix=".json") as tf:
        env = dict(os.environ, NL_TEACHERS=HERE, NL_CKPT=f"{WEIGHTS}/s32v2.pt",
                   PYTHONPATH=":".join(x for x in (f"{HERE}/nl", HERE, os.environ.get("PYTHONPATH", "")) if x))
        p = subprocess.run([sys.executable, f"{HERE}/nl/nl_level.py", path, "--device", dev.type, "--json", tf.name],
                           env=env, capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(f"nl_level failed: {p.stderr[-800:]}")
        return json.load(open(tf.name))


def features(path, info, pairs=48):
    w, h = info["w"], info["h"]
    fs = w * h * 3 // 2
    ck = torch.load(f"{WEIGHTS}/s32v2.pt", map_location="cpu")
    net = Student(ck["channels"])
    net.load_state_dict(ck["state"])
    net = net.eval().to(dev)
    num, _, den = info["fps"].partition("/")
    fps = float(num) / float(den or 1)
    dur = info["duration"]
    times = np.linspace(dur * 0.03, max(dur * 0.97 - 6 / fps, 0), pairs) if dur > 4 else [0.0]
    acc = {"kurtosis": [], "fresh_corr": [], "y_std": [], "share": []}
    nm = None
    with torch.no_grad():
        for t in times:
            raw = np.frombuffer(subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(t - 2 / fps, 0):.3f}", "-i", path,
                                                "-map", "0:v:0", "-frames:v", "6", "-fps_mode", "passthrough", "-f", "rawvideo",
                                                "-pix_fmt", "yuv420p10le", "-"], capture_output=True).stdout, np.uint16)
            if len(raw) < 6 * fs:
                continue
            fr = torch.from_numpy(raw[:6 * fs].astype(np.float32).reshape(6, fs)).to(dev)
            Y = fr[:, :w * h].view(6, h, w)
            U = fr[:, w * h:w * h * 5 // 4].view(6, h // 2, w // 2)
            V = fr[:, w * h * 5 // 4:].view(6, h // 2, w // 2)
            x = pack(Y / 1023, U / 1023, V / 1023)                     # (6, 6, h/2, w/2)
            hh, ww = x.shape[-2:]
            ph, pw = (-hh) % 4, (-ww) % 4
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode="reflect")
            if nm is None or nm.shape[-2:] != x.shape[-2:]:
                nm = torch.full((1, 1, x.shape[-2], x.shape[-1]), 60 / 876.0, device=dev)
            den_y = []
            for c in (2, 3):                                           # frames t and t+1, each with its 5-frame window
                o = net(x[c - 2:c + 3][None], nm)[0][..., :hh, :ww]
                den_y.append(unpack(o[None])[0][0] * 1023)
            ay0, ay1, by0, by1 = Y[2], Y[3], den_y[0], den_y[1]
            b3a, b3b = blur(by0, 3.0), blur(by1, 3.0)
            gy, gx = torch.gradient(b3a)
            struct = blur(torch.sqrt(gx * gx + gy * gy), 3.0)
            m = (struct < 1.5) & ((b3b - b3a).abs() < 0.5) & (by0 > 200) & (by0 < 800)
            if m.sum() < 20000:
                continue
            ry = ay0 - by0
            h0 = ry - blur(ry, 1.5)
            r1 = ay1 - by1
            h1 = r1 - blur(r1, 1.5)
            x0, x1 = h0[m], h1[m]
            x0, x1 = x0 - x0.mean(), x1 - x1.mean()
            acc["fresh_corr"].append(float((x0 * x1).sum() / torch.sqrt((x0 * x0).sum() * (x1 * x1).sum()).clamp(min=1e-6)))
            d = ((ay1 - ay0) / math.sqrt(2))[m]
            d = d - d.mean()
            acc["kurtosis"].append(float((d ** 4).mean() / (d ** 2).mean().clamp(min=1e-6) ** 2 - 3))
            acc["y_std"].append(float(ry[m].std()))
            acc["share"].append(float(m.float().mean()))
    out = {"pairs_used": len(acc["kurtosis"])}
    for k, v in acc.items():
        out[k] = round(float(np.median(v)), 4) if v else None
    return out


def decide(lv, f):
    if lv < RULES["clean_level"]:
        return "clean", f"level {lv:.1f} < {RULES['clean_level']}"
    if f["pairs_used"] == 0:
        return "clean", "no flat static mid-brightness pixels to measure the grain on"
    if (f["kurtosis"] or 0) > RULES["texture_kurtosis"] or (f["fresh_corr"] or 0) > RULES["texture_frozen"]:
        return "texture", (f"kurtosis {f['kurtosis']} (> {RULES['texture_kurtosis']}) or frozen {f['fresh_corr']} "
                           f"(> {RULES['texture_frozen']})")
    return "grain", f"level {lv:.1f}, kurtosis {f['kurtosis']}, frozen {f['fresh_corr']}: Gaussian, fresh"


def analyse(path, pairs=48):
    info = probe(path)
    lv = level(path)
    f = features(path, info, pairs)
    cls, why = decide(float(lv["level"]), f)
    return {"class": cls, "on": cls == "grain", "why": why, "level": round(float(lv["level"]), 1),
            "level_windows_p10_p50_p90": lv.get("window_p10_p50_p90"), "features": f, "rules": RULES, "probe": info}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--pairs", type=int, default=48)
    ap.add_argument("--json")
    A = ap.parse_args()
    r = analyse(A.src, A.pairs)
    print(json.dumps(r), flush=True)
    if A.json:
        json.dump(r, open(A.json, "w"), indent=1)
