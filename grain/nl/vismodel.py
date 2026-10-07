"""Noise-visibility model on the histograms of nl_extract.py (CPU, numpy).

Per clip and luma band k, a histogram over (noise contrast n, masker contrast m, adaptation luminance La) gives the
fraction of pyramid coefficients in each bin (n, m are Weber contrasts: band amplitude / local mean luminance).
Display / viewing parameters: ppd (display pixels per degree), Lb (black level + reflections, cd/m^2), sdr_white.

Per bin, in this order (all optional, chosen by `opt`):
    La' = La (* sdr_white/200 for SDR) + Lb;  n, m *= La/(La+Lb)            black level lowers contrast in the dark
    units:   weber  n                                                         (contrast)
             pq     n * La' * dPQ/dL * 876                                    (10-bit PQ code values, ~JND-uniform)
             pu21   n * La' * dPU21/dL                                        (PU21 units)
             pow    n * (La'/100)^gamma
    thresh:  n_u = sqrt(max(n_u^2 - (tau * u(La') / S)^2, 0))               supra-threshold: subtract CSF threshold
    csfw:    n_u *= (S / S_ref)^alpha                                         CSF as a weight (alpha=1: threshold units)
    mask:    E = n_u^2 / (1 + 10^logk * (m / 0.01)^q)                         texture masking by the picture
    pool:    P_k = (sum_bins w E^(beta/2))^(1/beta),  V = (sum_k P_k^beta_b)^(1/beta_b), returned as log10 V
S = castleCSF(rho_k, La') with rho_k the CVVDP band frequency at ppd_v = ppd / (display pixels per video pixel).
"""
import glob, json, os
import numpy as np

HIST = os.environ.get("NL_HIST", "/nl/hist")
_here = os.path.dirname(os.path.abspath(__file__))
LUT = json.load(open(os.environ.get("NL_CSF", os.path.join(_here, "csf_lut_weber.json"))))
LUT_L = np.log10(np.array(LUT["L_bkg"]))
LUT_R = np.log10(np.array(LUT["rho"]))
LUT_S = {0: np.array(LUT["o0_c1"]), 5: np.array(LUT["o5_c1"])}       # (nL, nrho) log10 sensitivity
DISPLAY_W, DISPLAY_H = 3840, 2160
PQ = (0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875)
PU21 = [0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627, 0.09150303166, 0.9099517204, 596.3148142]


def csf(rho, La, omega=0):
    """log10 S at spatial freq rho (cpd) and luminance La (cd/m^2), bilinear in log-log, clamped to the LUT."""
    lr = np.clip(np.log10(rho), LUT_R[0], LUT_R[-1])
    ll = np.clip(np.log10(La), LUT_L[0], LUT_L[-1])
    S = LUT_S[omega]
    i = np.clip(np.searchsorted(LUT_L, ll) - 1, 0, len(LUT_L) - 2)
    j = np.clip(np.searchsorted(LUT_R, lr) - 1, 0, len(LUT_R) - 2)
    tl = (ll - LUT_L[i]) / (LUT_L[i + 1] - LUT_L[i])
    tr = (lr - LUT_R[j]) / (LUT_R[j + 1] - LUT_R[j])
    return ((1 - tl) * (1 - tr) * S[i, j] + tl * (1 - tr) * S[i + 1, j] + (1 - tl) * tr * S[i, j + 1]
            + tl * tr * S[i + 1, j + 1])


def pq_encode(L):
    m1, m2, c1, c2, c3 = PQ
    y = (np.clip(L, 0, 10000) / 10000) ** m1
    return ((c1 + c2 * y) / (1 + c3 * y)) ** m2


def pu21_encode(L):
    p = PU21
    L = np.clip(L, 0.005, 10000)
    return p[6] * (((p[0] + p[1] * L ** p[3]) / (1 + p[2] * L ** p[3])) ** p[4] - p[5])


def unit_log(La, units, gamma=0.0):
    """log10 of d(unit)/d(lnL) at La: multiplies a Weber contrast into the unit."""
    if units == "weber":
        return np.zeros_like(La)
    if units == "pow":
        return gamma * np.log10(La / 100.0)
    f = {"pq": lambda L: pq_encode(L) * 876, "pu21": pu21_encode}[units]
    e = 1.02
    return np.log10(np.maximum((f(La * e) - f(La / e)) / (2 * np.log(e)), 1e-12))


def band_freqs(ppd_v, nb):
    """CVVDP lpyr convention: band 0 at ppd/2, band k>=1 at 0.3228 * 2^-(k-1) * ppd/2."""
    return np.array([1.0] + [0.3228 * 2.0 ** -(k - 1) for k in range(1, nb)]) * ppd_v / 2


def centres(edges, step):
    return np.r_[edges[0] - step / 2, (edges[:-1] + edges[1:]) / 2, edges[-1] + step / 2]


def load_npz(cid, hist_dir=None):
    return np.load(f"{hist_dir or HIST}/{cid}.npz")


def split_variant(variant, hist_dir=None):
    """'dir:variant' (e.g. '/nl/hist2:h3_frz') -> (dir, variant)."""
    if ":" in variant:
        d, v = variant.rsplit(":", 1)
        return d, v
    return hist_dir, variant


class Hist:
    """Sparse histogram entries of many clips, for one variant (h3_s30/h3_s70/h3_s120/h3_dt)."""

    def __init__(self, ids, variant="h3_s70", hist_dir=None, arrays=None):
        hist_dir, variant = split_variant(variant, hist_dir)
        self.ids = list(ids)
        rows = []
        self.meta = {}
        for ci, cid in enumerate(self.ids):
            z = arrays[cid] if arrays is not None else load_npz(cid, hist_dir)
            H = z[variant].astype(np.float64)
            nb = H.shape[0]
            if ci == 0:
                self.nc = centres(z["n_edges"], 0.2)
                self.mc = centres(z["m_edges"], 0.3)
                self.lc = centres(z["l_edges"], 1 / 3)
            tot = H.reshape(nb, -1).sum(1)
            k, a, b, c = np.nonzero(H)
            w = H[k, a, b, c] / tot[k]
            rows.append(np.c_[np.full(len(k), ci), k, a, b, c, w])
            w_, h_ = int(z["w"]), int(z["h"])
            self.meta[cid] = {"w": w_, "h": h_, "trc": str(z["trc"]), "scale": min(DISPLAY_W / w_, DISPLAY_H / h_),
                              "lmean": float(np.mean(z["lmean"])), "tcorr": z["tcorr"].mean(0), "motion": float(np.mean(z["motion"]))}
        E = np.concatenate(rows)
        self.clip = E[:, 0].astype(int)
        self.band = E[:, 1].astype(int)
        self.ln = self.nc[E[:, 2].astype(int)]
        self.lm = self.mc[E[:, 3].astype(int)]
        self.li = E[:, 4].astype(int)
        self.ll = self.lc[self.li]
        self.w = E[:, 5]
        self.nb = nb
        self.sdr = np.array([self.meta[c]["trc"] not in ("smpte2084", "arib-std-b67") for c in self.ids])
        self.scale = np.array([self.meta[c]["scale"] for c in self.ids])
        self.motion = np.array([self.meta[c]["motion"] for c in self.ids])

    def subset(self, idx):
        sub = object.__new__(Hist)
        sub.__dict__.update(self.__dict__)
        idx = list(idx)
        keep = np.isin(self.clip, idx)
        remap = -np.ones(len(self.ids), int)
        remap[idx] = np.arange(len(idx))
        sub.ids = [self.ids[i] for i in idx]
        for f in ("band", "ln", "lm", "ll", "li", "w"):
            setattr(sub, f, getattr(self, f)[keep])
        sub.clip = remap[self.clip[keep]]
        sub.sdr, sub.scale, sub.motion = self.sdr[idx], self.scale[idx], self.motion[idx]
        return sub

    def prep(self, ppd=40.0, Lb=0.005, sdr_white=200.0, omega=0, units="pq", thresh=False, csfw=False, **_):
        """Parameter-independent per-entry arrays for one display / option configuration."""
        La = 10 ** self.ll
        La = np.where(self.sdr[self.clip], La * (sdr_white / 200.0), La)
        latt = np.log10(La / (La + Lb))
        La2 = La + Lb
        P = {"units": units, "thresh": thresh, "csfw": csfw, "La2": La2}
        P["ul"] = unit_log(La2, units) if units != "pow" else None
        P["ln0"] = self.ln + latt
        P["lm"] = self.lm + latt
        if thresh or csfw:
            ppd_v = ppd / self.scale[self.clip]
            rho = band_freqs(1.0, self.nb)[self.band] * ppd_v
            P["lS"] = csf(rho, La2, omega)
        P["key"] = self.clip * self.nb + self.band
        return P

    def ev(self, th, P, bands=None, return_bands=False):
        ul = P["ul"] if P["ul"] is not None else th.get("gamma", 0.0) * np.log10(P["La2"] / 100.0)
        ln = P["ln0"] + ul
        if P["thresh"] or P["csfw"]:
            lS = P["lS"] + th.get("sens", 0.0)
        if P["thresh"]:
            lt = np.log10(th.get("tau", 1.0)) + ul - lS
            e2 = 10 ** (2 * ln) - 10 ** (2 * lt)
            ln = np.where(e2 > 0, 0.5 * np.log10(np.maximum(e2, 1e-300)), -30.0)
        if P["csfw"]:
            ln = ln + th.get("alpha", 1.0) * (lS - 2.0)
        latt = -np.log10(1 + 10 ** (th.get("logk", -9.0) + th.get("q", 2.0) * (P["lm"] + 2.0)))   # masking (energy)
        lE = 2 * ln + latt
        if "lg0" in th:
            # luminance gate: grain on near-black is not seen (display / dark adaptation); g = L^n / (L^n + L0^n)
            gn = th.get("gn", 2.0)
            lE = lE - 2 * np.log10(1 + 10 ** (gn * (th["lg0"] - np.log10(P["La2"]))))
        beta = th.get("beta", 2.0)
        x = 0.5 * beta * lE
        n = len(self.ids)
        key = P["key"]                                                # non-decreasing (entries built in order)
        if "starts" not in P:
            P["starts"] = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
            P["ukey"] = key[P["starts"]]
        mx = np.full(n * self.nb, -np.inf)
        mx[P["ukey"]] = np.maximum.reduceat(x, P["starts"])
        if "beta_l" not in th:
            s = np.bincount(key, weights=self.w * 10 ** (x - mx[key]), minlength=n * self.nb)
            with np.errstate(divide="ignore", invalid="ignore"):
                logP = (np.log10(np.maximum(s, 1e-300)) + mx) / beta
            if th.get("nu", 0.0):
                # nu = 1: masking acts as a measurement weight (mean over unmasked area), not as perceptual masking
                cov = np.bincount(key, weights=self.w * 10 ** latt, minlength=n * self.nb)
                logP = logP - th["nu"] * np.log10(np.maximum(cov, 1e-12)) / beta
        else:
            # region pooling: mean within each adaptation-luminance bin, then Minkowski beta_l over bins with the
            # bin's area share to the power eta (eta < 1: small regions count relatively more)
            nl_ = len(self.lc)
            k2 = key * nl_ + self.li
            s2 = np.bincount(k2, weights=self.w * 10 ** (x - mx[key]), minlength=n * self.nb * nl_)
            A = np.bincount(k2, weights=self.w, minlength=n * self.nb * nl_)
            ok = A > 1e-9
            with np.errstate(divide="ignore", invalid="ignore"):
                lpl = np.where(ok, (np.log10(np.maximum(s2, 1e-300) / np.maximum(A, 1e-300))
                                    + np.repeat(mx, nl_)) / beta, -np.inf)
            bl, eta = th["beta_l"], th.get("eta", 1.0)
            z = np.where(ok, bl * lpl + eta * np.log10(np.maximum(A, 1e-300)), -np.inf).reshape(n * self.nb, nl_)
            zm = z.max(1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                logP = (np.log10(np.nansum(10 ** (z - zm), 1)) + zm[:, 0]) / bl
        logP = logP.reshape(n, self.nb)
        if bands is not None:
            logP = logP[:, bands]
        bb = th.get("beta_b", 2.0)
        m2 = logP.max(1, keepdims=True)
        v = np.log10((10 ** (bb * (logP - m2))).sum(1)) / bb + m2[:, 0]
        if th.get("gm", 0.0):
            # motion masking: mean |frame difference| (4x4-pooled luma codes) -> lower visibility, 1 parameter
            v = v - th["gm"] * np.log10(1 + self.motion / 10.0)
        return (v, logP) if return_bands else v

    def V(self, th, bands=None, return_bands=False, **disp):
        return self.ev(th, self.prep(**disp), bands=bands, return_bands=return_bands)


class Hist2:
    """Two variants (e.g. h3_dt: independent grain, h3_s70: all residual incl. frozen grain) pooled together:
    V = log10(10^v_a + 10^(logw) * 10^v_b) (energy-like sum of the two pooled visibilities), shared parameters."""

    def __init__(self, ids, variants, hist_dir=None, arrays=None, _parts=None):
        self.parts = _parts or [Hist(ids, v, hist_dir=hist_dir, arrays=arrays) for v in variants]
        self.ids = self.parts[0].ids

    def subset(self, idx):
        return Hist2(None, None, _parts=[p.subset(idx) for p in self.parts])

    def prep(self, **disp):
        return [p.prep(**disp) for p in self.parts]

    def ev(self, th, P, bands=None, return_bands=False):
        va = self.parts[0].ev(th, P[0], bands=bands)
        thb = dict(th)
        thb.update({k[:-2]: v for k, v in th.items() if k.endswith("_b")})     # part-b overrides, e.g. logk_b
        vb = self.parts[1].ev(thb, P[1], bands=bands)
        lw = th.get("logw", 0.0)
        m = np.maximum(va, vb + lw)
        return m + np.log10(10 ** (va - m) + 10 ** (vb + lw - m))
