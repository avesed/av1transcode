"""grainauto step (from noise/build_tbl.py, paths and frame rate made explicit).
AV1 film grain table with a fixed grain shape and amplitudes that match a measured target.

The shape (AR coefficients and p line) is the element-wise median over the segments of a donor table.
  cal OUT.tbl                                           - flat scaling 64 (luma and chroma), for measuring k
  build TARGET.json CAL_APPLIED.json OUT.tbl [CHROMA]   - one segment per shot

In build mode:
  - k = applied std per scaling unit, from the cal measurement (median of the mid bins);
  - luma: sY at the bin centres = target / k; 0 where the target is below 1.0 code or the bin is unmeasured
    above the brightest measured bin; unmeasured bins below it take the nearest measured value;
  - chroma: Cb and Cr per brightness bin the same way (floor 0.3 code), times CHROMA (default 1.0); with\n    p ... 128 192 256 the chroma scaling index is the co-sited luma.

  python3 ga_table.py DONOR.tbl SHOTS.json FPS MODE ...   (host, stdlib; FPS like 24000/1001)
"""
import json, os, random, statistics as st, sys

DONOR, SHOTS, FPS, MODE = sys.argv[1:5]
sys.argv = sys.argv[:1] + sys.argv[2:]                    # keep the old positions of the mode arguments below
rep = json.load(open(SHOTS))
num, _, den = FPS.partition("/")
FR = float(den or 1) / float(num) * 1e7                   # 10 MHz ticks per frame
segs, cur = [], None
for line in open(DONOR):
    t = line.split()
    if not t:
        continue
    if t[0] == "E":
        cur = {}
        segs.append(cur)
    elif cur is not None:
        cur[t[0]] = t[1:]
p_line = segs[0]["p"]
med = lambda key: [int(round(st.median(int(s[key][i]) for s in segs if key in s))) for i in range(len(segs[0][key]))]
cY, cCb, cCr = med("cY"), med("cCb"), med("cCr")
EDGES = [0, 16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256]
CENT = [(a + b) // 2 for a, b in zip(EDGES, EDGES[1:])]


def entry(f0, f1, sy, scb, scr, seed):
    pts = lambda v: f"{len(v)} " + " ".join(f"{x} {y}" for x, y in v)
    return (f"E {round(f0 * FR)} {round(f1 * FR)} 1 {seed} 1\n\tp {' '.join(p_line)}\n\tsY {pts(sy)}\n\tsCb {pts(scb)}\n"
            f"\tsCr {pts(scr)}\n\tcY {' '.join(map(str, cY))}\n\tcCb {' '.join(map(str, cCb))}\n\tcCr {' '.join(map(str, cCr))}\n")


rng = random.Random(7)
total = sum(s["frames"] for s in rep["shots"])
out = ["filmgrn1\n"]
if MODE == "cal":
    out.append(entry(0, total, [(0, 64), (255, 64)], [(0, 64), (255, 64)], [(0, 64), (255, 64)], rng.randrange(1, 65535)))
    open(sys.argv[4], "w").write("".join(out))
    print("cal table written")
    sys.exit(0)
target, cal, OUT = json.load(open(sys.argv[4])), json.load(open(sys.argv[5])), sys.argv[6]
CH = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
# optional closed-loop correction: CORR=target.json:applied.json (same table family) -> per channel and bin,
# median-over-shots target / applied, clipped to [0.25, 4], multiplies the scaling at that bin
CORR = {}
if os.environ.get("CORR"):
    tj, aj = (json.load(open(x)) for x in os.environ["CORR"].split(":"))
    for ch in ("y", "cb", "cr"):
        f = []
        for e in EDGES[:-1]:
            tv = [s_["bins"][str(e)][ch] for s_ in tj["shots"] if str(e) in s_["bins"]]
            av = [s_["bins"][str(e)][ch] for s_ in aj["shots"] if str(e) in s_["bins"]]
            f.append(min(4.0, max(0.25, st.median(tv) / st.median(av))) if tv and av and st.median(av) > 0.05 else 1.0)
        CORR[ch] = f
    print("correction " + "  ".join(f"{ch}: " + " ".join(f"{x:.2f}" for x in v) for ch, v in CORR.items()))
mid = [str(e) for e in (32, 48, 64, 80, 96)]
ky = st.median(s["bins"][b]["y"] for s in cal["shots"] for b in mid if b in s["bins"]) / 64
kcb = st.median(s["bins"][b]["cb"] for s in cal["shots"] for b in mid if b in s["bins"]) / 64
kcr = st.median(s["bins"][b]["cr"] for s in cal["shots"] for b in mid if b in s["bins"]) / 64
print(f"k luma {ky:.4f}  k cb {kcb:.4f}  k cr {kcr:.4f}  (std per scaling unit)")
f0 = 0
for s, shot in zip(target["shots"], rep["shots"]):
    f1 = f0 + shot["frames"]
    bins = s["bins"]
    meas = [i for i, e in enumerate(EDGES[:-1]) if str(e) in bins]

    def curve(ch, k, floor, gain=1.0):
        """Scaling points at the bin centres: target / k; 0 below the floor or above the brightest measured bin;
        unmeasured bins below it take the nearest measured one."""
        pts = []
        for i, e in enumerate(EDGES[:-1]):
            if str(e) in bins:
                v = bins[str(e)][ch]
            elif meas and i < max(meas):
                v = bins[str(EDGES[min(meas, key=lambda j: abs(j - i))])][ch]
            else:
                v = 0.0
            g = gain * (CORR[ch][i] if CORR else 1.0)
            pts.append((CENT[i], 0 if v < floor else min(255, int(round(g * v / k)))))
        return [(0, pts[0][1])] + pts + [(255, pts[-1][1])]

    # with p ... 128 192 256 the chroma scaling index is the co-sited luma, so chroma follows brightness too
    assert p_line[6:] == ["128", "192", "256", "128", "192", "256"], p_line
    # AV1 allows 14 luma points but only 10 per chroma plane: keep the endpoints and the centres 24..176
    # (the dropped 8 / 208 / 240 sit next to an endpoint or in the unmeasured, zero top)
    # near black: hold the first bin's value up to 16, so the ramp to the 24 point cannot lift 0-16
    # luma: 0 8 16 24 40 .. 240 255 = 14 points; chroma: 0 16 40 .. 176 255 = 10 points
    hold = lambda c: [c[0], c[1], (16, c[1][1])] + c[2:]
    trim = lambda c: [c[0], (16, c[1][1])] + [q for q in c[1:-1] if 40 <= q[0] <= 176] + [c[-1]]
    sy, scb, scr = hold(curve("y", ky, 1.0)), trim(curve("cb", kcb, 0.3, CH)), trim(curve("cr", kcr, 0.3, CH))
    assert len(sy) <= 14 and len(scb) <= 10 and len(scr) <= 10
    out.append(entry(f0, f1, sy, scb, scr, rng.randrange(1, 65535)))
    f0 = f1
open(OUT, "w").write("".join(out))
print(f"wrote {OUT}: {len(rep['shots'])} segments, chroma x{CH}")
