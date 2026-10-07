"""Perceived noise level (owner's 1-50 scale) of a video, for a given viewing condition.

  python3 nl_level.py VIDEO [--ppd 40] [--samples N] [--model /nl/model_v1.json] [--device cuda|xpu|cpu] [--json OUT]

ppd = display pixels per degree of visual angle at the screen centre (the video is assumed shown full-screen on a
display of --display-w x --display-h pixels). The owner rated at about 28-47 ppd (32" 4K at 30-50 cm); a 65" 4K TV
at 2-3 m is about 90-140 ppd. Black level / ambient (--black, cd/m^2) and SDR white (--sdr-white) describe the display.

How: N sample windows of 6 frames spread over the video (seek + decode); per window the RVRT-distilled student
denoises frames 2 and 3; the frame-to-frame change of the denoising residual (independent grain only) is turned into
display-luminance Laplacian bands, a histogram over (noise contrast, picture contrast, adaptation luminance) is
pooled over windows, and the fitted visibility model (CSF at the given ppd, texture masking, Minkowski pooling)
maps it to the 1-50 scale. Also reports the level of each window (per-scene spread).
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("NL_TEACHERS", "/t"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--ppd", type=float, default=None, help="pixels per degree (default: the model's rating condition)")
    ap.add_argument("--samples", type=int, default=0, help="sample windows (default: 1 per 40 s, 8..96)")
    ap.add_argument("--model", default=os.path.join(HERE, "model_v2.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"))
    ap.add_argument("--black", type=float, default=None, help="black level + reflections, cd/m^2")
    ap.add_argument("--sdr-white", type=float, default=None)
    ap.add_argument("--display-w", type=int, default=3840)
    ap.add_argument("--display-h", type=int, default=2160)
    ap.add_argument("--json", default=None)
    ap.add_argument("--stack", action="store_true", help="owner-head stack: physical model + ridge heads on CNN v4 tile "
                    "features and ARNIQA embeddings (better match to the owner at the rating condition, ~+1 min/episode)")
    ap.add_argument("--stack-file", default=os.path.join(HERE, "stack_v2.npz"))
    A = ap.parse_args()
    os.environ["NL_DEVICE"] = A.device
    import nl_extract as X                      # display model, pyramids, histograms, student wrapper
    from vismodel import Hist, Hist2
    M = json.load(open(A.model))
    disp = dict(M["disp"])
    if A.ppd is not None:
        disp["ppd"] = A.ppd
    if A.black is not None:
        disp["Lb"] = A.black
    if A.sdr_white is not None:
        disp["sdr_white"] = A.sdr_white
    vismodel_mod = sys.modules["vismodel"]
    vismodel_mod.DISPLAY_W, vismodel_mod.DISPLAY_H = A.display_w, A.display_h

    t0 = time.time()
    s = X.probe(A.video)
    w, h = s["width"], s["height"]
    trc = s.get("color_transfer") or "bt709"
    bt2020 = (s.get("color_space") or "").startswith("bt2020") or trc in ("smpte2084", "arib-std-b67")
    dur = float(json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json",
                                           A.video], capture_output=True, text=True).stdout)["format"]["duration"])
    n = A.samples or int(np.clip(dur / 40, 8, 96))
    times = np.linspace(dur * 0.02, dur * 0.98 - 1.0, n) if dur > 8 else np.linspace(0, max(dur - 0.5, 0), min(n, 4))
    ck = torch.load(X.CKPT, map_location="cpu")
    net = X.Student(ck["channels"])
    net.load_state_dict(ck["state"])
    net = net.eval().to(X.dev)
    fs = w * h * 3 // 2
    per_window, acc = [], None
    if A.stack:
        S = np.load(A.stack_file)
        import stack_feats as SF
        sf = SF.StackFeatures(X.dev, w, h, trc, bt2020, ppd_scale=disp["ppd"] / M["disp"]["ppd"])
    for t in times:
        raw = np.frombuffer(subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t:.3f}", "-i", A.video,
                                            "-map", "0:v:0", "-frames:v", "6", "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-"],
                                           capture_output=True).stdout, np.uint16)
        if len(raw) < 6 * fs:
            continue
        fr = raw[:6 * fs].reshape(6, fs)
        hs, mot = X.window_hist_dt(net, fr, w, h, trc, bt2020)
        per_window.append((float(t), hs, mot))
        if A.stack:
            sf.add(fr)
        acc = {k: v.copy() for k, v in hs.items()} if acc is None else {k: acc[k] + hs[k] for k in acc}

    def level(hists, motion):
        arr = dict(hists, n_edges=X.N_EDGES, m_edges=X.M_EDGES, l_edges=X.L_EDGES, w=w, h=h, trc=trc,
                   lmean=np.array([0.0]), tcorr=np.zeros((1, X.NB)), motion=np.array([motion]))
        var = M["variant"]
        H = Hist2(["v"], var, arrays={"v": arr}) if isinstance(var, list) else Hist(["v"], var, arrays={"v": arr})
        v = H.ev(M["th"], H.prep(**disp), bands=M.get("bands"))
        g = M["th"].get("link", 0.0)
        v = v if abs(g) < 1e-3 else (10 ** (g * v) - 1) / g
        return float(M["ab"][0] + M["ab"][1] * v[0])
    if acc is None:
        raise SystemExit("no decodable sample windows")
    mot_all = float(np.mean([m for _, _, m in per_window]))
    phys = level(acc, mot_all)
    out = {"video": A.video, "level": round(phys, 2), "motion": round(mot_all, 2), "ppd": disp["ppd"], "black": disp["Lb"],
           "sdr_white": disp["sdr_white"], "windows": len(per_window), "seconds": round(time.time() - t0, 1),
           "per_window": [{"t": round(t, 2), "level": round(level(hh, mm), 2), "motion": round(mm, 2)} for t, hh, mm in per_window]}
    if A.stack:
        cf, emb = sf.result()
        rc = float(S["cnnf_b0"] + ((cf - S["cnnf_mean"]) / S["cnnf_std"]) @ S["cnnf_w"])
        ra = float(S["arn_b0"] + ((emb - S["arn_mean"]) / S["arn_std"]) @ S["arn_w"])
        b = S["b"]
        out["level_phys"] = out["level"]
        out["level"] = round(float(b[0] + b[1] * phys + b[2] * rc + b[3] * ra), 2)
        out["stack_parts"] = {"phys": round(phys, 2), "cnn_head": round(rc, 2), "arniqa_head": round(ra, 2)}
    lv = np.array([p["level"] for p in out["per_window"]])
    out["window_p10_p50_p90"] = [round(float(x), 2) for x in np.percentile(lv, [10, 50, 90])]
    out["level_on_scale"] = round(float(np.clip(out["level"], 1, 50)), 2)     # the owner's scale ends at 1 and 50
    print(json.dumps({k: v for k, v in out.items() if k != "per_window"}))
    if A.json:
        json.dump(out, open(A.json, "w"), indent=1)


if __name__ == "__main__":
    main()
