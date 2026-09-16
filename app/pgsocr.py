"""HDMV PGS (bitmap) subtitles -> plain-text SRT, in-process.

Plex has no soft-subtitle target for a bitmap track, so when it picks one it
re-encodes the whole picture to burn it in: 0.2-0.9x real time at ~660% CPU,
16 threads inside the overlay's scale, once per frame. drop_empty_subtitles
(config.OptimizerSettings) stopped that happening for tracks with NOTHING in
them. This is the other half: a track that really does carry subtitles gets a
text copy beside it, so the player has something to send instead.

WHAT THIS IS NOT. It is not a replacement for the image track - that is kept,
untouched - and it is not a general subtitle converter. It reads HDMV PGS and
nothing else, and it is run on ENGLISH tracks only, because the image ships
one tesseract model (eng) and pointing it at Chinese or Thai glyphs does not
fail, it produces confident garbage that would be written into the output as a
subtitle track. See ShotEncoder._ocr_companions for where that is enforced.

THE PIPELINE, and what each step was measured to be worth (Agents of
S.H.I.E.L.D. S05E05, 711 cues, scored against that file's own SDH text track):

  parse     .sup -> cues. A cue is one COMPOSITION display set plus the ERASE
            display set that follows it, so the cue count is the number of
            composition display sets and never the number of display sets.
  render    RLE -> an RGBA canvas, via the palette bound at the END segment of
            the display set that carries it (object and palette ids are reused
            by every display set in a track, so binding later hands every cue
            the last bitmap in the file).
  preprocess  luma*alpha, inverted, 2x bilinear, NO binarisation (see
            signal_map: these are white glyphs with a BLACK OUTLINE, and the
            naive "composite over white" leaves a stencil whose interior
            matches the background).
  ocr       tesseract --psm 6, fed a PGM on stdin. No PNG files: the prototype
            wrote one per cue because its stages were separate processes.
  post      six measured repairs (see POST_RULES), each justified by a counted
            error, none of them hardcoded to one show.
  gates     structural self-consistency only (see measure/evaluate). There is
            no ground truth in production, so nothing here claims an accuracy.

Measured end to end at 0.1276% CER case-folded with 689/711 cues exact.

stdlib + numpy only. The image has numpy and NO PIL, and this module must
import in the test environment too, where there is no tesseract at all.
"""

from __future__ import annotations

import os
import re
import struct
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from loguru import logger

# Segment types, and the 90 kHz clock every PGS timestamp is counted in.
PDS, ODS, PCS, WDS, END = 0x14, 0x15, 0x16, 0x17, 0x80
TICKS_PER_SECOND = 90000.0

# Where a packaged English word list lives, for the out-of-vocabulary gate.
# Missing is not an error: the gate is skipped and says so (see measure()).
LEXICON_PATHS = ("/usr/share/dict/american-english", "/usr/share/dict/words")


class PgsError(Exception):
    """A .sup this module cannot read. Always caught by ocr_track()."""


# ---- segment layer --------------------------------------------------------


def segments(buf: bytes):
    """Yield (offset, pts_ticks, type, payload) over a .sup.

    Strict: a desync or a truncated payload raises rather than silently
    dropping the rest of the track. A subtitle quietly missing from the output
    is the failure mode this whole feature exists to avoid.
    """
    off, n = 0, len(buf)
    while off < n:
        if off + 13 > n:
            raise PgsError(f"short segment header at {off}: {n - off}B left")
        if buf[off:off + 2] != b"PG":
            raise PgsError(f"lost sync at {off}: {buf[off:off + 4].hex()}")
        pts = struct.unpack_from(">I", buf, off + 2)[0]
        stype = buf[off + 10]
        size = struct.unpack_from(">H", buf, off + 11)[0]
        endo = off + 13 + size
        if endo > n:
            raise PgsError(f"truncated payload at {off}: declared {size}, "
                           f"have {n - off - 13}")
        yield off, pts, stype, buf[off + 13:endo]
        off = endo


def parse_pcs(p: bytes) -> dict:
    """Presentation Composition Segment: the screen, the palette it uses, and
    the objects it puts on screen (none at all = an erase)."""
    w, h = struct.unpack_from(">HH", p, 0)
    n_obj = p[10]
    objs, off = [], 11
    for _ in range(n_obj):
        oid, win, flags = struct.unpack_from(">HBB", p, off)
        x, y = struct.unpack_from(">HH", p, off + 4)
        off += 8
        if flags & 0x80:              # object_cropped_flag: four more shorts
            off += 8
        objs.append({"oid": oid, "win": win, "x": x, "y": y})
    if off != len(p):
        raise PgsError(f"PCS payload not fully consumed: {off} of {len(p)}")
    return {"screen_w": w, "screen_h": h, "pal_id": p[9], "objs": objs}


def parse_wds(p: bytes) -> Dict[int, dict]:
    """Window Definition Segment: where on screen each window sits."""
    n, off, wins = p[0], 1, {}
    for _ in range(n):
        wid = p[off]
        x, y, w, h = struct.unpack_from(">HHHH", p, off + 1)
        wins[wid] = {"x": x, "y": y, "w": w, "h": h}
        off += 9
    if off != len(p):
        raise PgsError(f"WDS payload not fully consumed: {off} of {len(p)}")
    return wins


def parse_pds(p: bytes) -> Tuple[int, Dict[int, tuple]]:
    """Palette Definition Segment.

    palette_id(1), palette_version(1), then 5-byte entries of
    (entry_id, Y, Cr, Cb, alpha). ENTRIES START AT OFFSET 2: starting at 1
    reads (entry_id, Cb) as (Y, alpha) and renders grey text on white, which
    tesseract reads as nothing at all.
    """
    if (len(p) - 2) % 5:
        raise PgsError(f"PDS payload {len(p)}B is not 2 + 5n")
    out = {}
    for i in range(2, len(p), 5):
        out[p[i]] = (p[i + 1], p[i + 2], p[i + 3], p[i + 4])
    return p[0], out


# ---- pixel layer ----------------------------------------------------------


def rle_decode(data: bytes, width: int, height: int) -> np.ndarray:
    """PGS run-length -> a (height, width) array of palette indices.

    A nonzero byte is one pixel of that index; 0x00 introduces a run. Rows
    shorter than the declared width are left padded with index 0, which is the
    transparent entry in every track measured - a row that decodes short is a
    damaged object, not a differently-shaped one.
    """
    rows, row = [], bytearray()
    i, n = 0, len(data)
    while i < n:
        b = data[i]
        i += 1
        if b:
            row.append(b)
            continue
        if i >= n:
            break
        b2 = data[i]
        i += 1
        if b2 == 0:                                  # end of line
            rows.append(row)
            row = bytearray()
            continue
        kind = b2 & 0xC0
        if kind == 0x00:                             # short run of index 0
            cnt, col = b2 & 0x3F, 0
        elif kind == 0x40:                           # long run of index 0
            cnt = ((b2 & 0x3F) << 8) | data[i]
            i += 1
            col = 0
        elif kind == 0x80:                           # short run of colour
            cnt = b2 & 0x3F
            col = data[i]
            i += 1
        else:                                        # long run of colour
            cnt = ((b2 & 0x3F) << 8) | data[i]
            i += 1
            col = data[i]
            i += 1
        row += bytes((col,)) * cnt
    if row:
        rows.append(row)
    img = np.zeros((height, width), np.uint8)
    for y, r in enumerate(rows[:height]):
        px = np.frombuffer(bytes(r[:width]), np.uint8)
        img[y, :px.size] = px
    return img


def palette_rgba(pal: Dict[int, tuple]) -> np.ndarray:
    """A 256x4 uint8 LUT: YCrCb limited-range -> RGB with BT.709 coefficients.

    Indices the PDS never mentions stay fully transparent, which is what makes
    an undersized palette render as a hole rather than as black text.
    """
    lut = np.zeros((256, 4), np.uint8)
    for idx, (y, cr, cb, a) in pal.items():
        yy = 1.16438356 * (y - 16)
        r = yy + 1.79274107 * (cr - 128)
        g = yy - 0.21324861 * (cb - 128) - 0.53290933 * (cr - 128)
        b = yy + 2.11240179 * (cb - 128)
        lut[idx] = np.clip([r, g, b, a], 0, 255).astype(np.uint8)
    return lut


# ---- display-set layer ----------------------------------------------------


class Cue(NamedTuple):
    """One subtitle: when it appears, when it is erased, and what to draw.

    `objs` holds (placement, width, height, rle_bytes) rather than a rendered
    array, so the parse of a whole track costs the size of the .sup and not
    the size of its bitmaps - a 26.5MB track renders to far more than that.
    """
    index: int
    start: float
    end: float
    objs: List[tuple]
    pal: Dict[int, tuple]
    wins: Dict[int, dict]


def parse_sup(buf: bytes) -> Tuple[List[Cue], Counter]:
    """.sup bytes -> (cues, notes). Notes are counted, never raised: a track
    with one damaged object still delivers every other subtitle.

    Timing: start = the PTS of a composition display set, end = the PTS of the
    erase display set that follows it. A composition never erased (the last
    cue of a truncated track) is given the median duration of the cues that
    were closed, rather than a made-up constant.
    """
    palettes: Dict[int, dict] = {}
    objects: Dict[int, tuple] = {}
    partial: Dict[int, dict] = {}
    windows: Dict[int, dict] = {}
    cur = None
    cues: List[Cue] = []
    notes: Counter = Counter()
    open_i = -1
    starts: List[int] = []
    ends: List[Optional[int]] = []

    for off, pts, stype, p in segments(buf):
        notes[f"seg_{stype:02x}"] += 1
        if stype == PCS:
            if cur is not None:
                notes["pcs_without_end"] += 1
            cur = dict(pts=pts, **parse_pcs(p))
        elif cur is None:
            notes["segment_outside_display_set"] += 1
            continue
        elif stype == WDS:
            windows = parse_wds(p)
        elif stype == PDS:
            pid, entries = parse_pds(p)
            palettes.setdefault(pid, {}).update(entries)
        elif stype == ODS:
            oid = struct.unpack_from(">H", p, 0)[0]
            seqf = p[3]
            if seqf & 0x80:                       # first (or only) fragment
                w, h = struct.unpack_from(">HH", p, 7)
                partial[oid] = {"w": w, "h": h, "data": bytearray(p[11:])}
            elif oid in partial:                  # continuation
                partial[oid]["data"] += p[4:]
            else:
                notes["ods_continuation_without_first"] += 1
                continue
            if seqf & 0x40:                       # last fragment: bind it
                o = partial.pop(oid)
                objects[oid] = (o["w"], o["h"], bytes(o["data"]))
        elif stype == END:
            # BIND HERE. Every display set reuses object id 0 and palette id
            # 0, so resolving these references any later would give every cue
            # the last bitmap and the last palette in the file.
            if cur["objs"]:
                # A composition that REPLACES one still on screen ends it
                # here, at the new cue's own PTS. Left open, it would fall to
                # the median duration below and run past the subtitle that
                # replaced it, putting two cues on the screen at once.
                if open_i >= 0 and ends[open_i] is None:
                    ends[open_i] = pts
                bound = []
                for co in cur["objs"]:
                    if co["oid"] not in objects:
                        notes["composition_object_with_no_bitmap"] += 1
                        continue
                    w, h, data = objects[co["oid"]]
                    bound.append((co, w, h, data))
                cues.append(Cue(len(cues), pts / TICKS_PER_SECOND, 0.0, bound,
                                dict(palettes.get(cur["pal_id"], {})),
                                dict(windows)))
                starts.append(pts)
                ends.append(None)
                open_i = len(cues) - 1
            elif open_i >= 0 and ends[open_i] is None:
                ends[open_i] = pts
                open_i = -1
            else:
                notes["erase_with_no_open_cue"] += 1
            cur = None
    if cur is not None:
        notes["unterminated_display_set"] += 1
    if partial:
        notes["unfinished_object_sequences"] += len(partial)

    closed = [e - s for s, e in zip(starts, ends) if e is not None and e > s]
    fallback = (sorted(closed)[len(closed) // 2] if closed
                else int(2.0 * TICKS_PER_SECOND))
    out = []
    for i, c in enumerate(cues):
        e = ends[i]
        if e is None or e <= starts[i]:
            notes["cue_without_erase"] += 1
            e = starts[i] + fallback
        out.append(c._replace(end=e / TICKS_PER_SECOND))
    return out, notes


def render(cue: Cue) -> np.ndarray:
    """Composite one cue's objects onto their window(s) -> an RGBA canvas."""
    wins = [cue.wins[o["win"]] for o, _w, _h, _d in cue.objs
            if o["win"] in cue.wins]
    if not wins:
        # A track with no WDS still places objects: fall back to the
        # composition's own coordinates rather than dropping the cue.
        wins = [{"x": o["x"], "y": o["y"], "w": w, "h": h}
                for o, w, h, _d in cue.objs]
    if not wins:
        return np.zeros((1, 1, 4), np.uint8)
    x0 = min(w["x"] for w in wins)
    y0 = min(w["y"] for w in wins)
    x1 = max(w["x"] + w["w"] for w in wins)
    y1 = max(w["y"] + w["h"] for w in wins)
    canvas = np.zeros((max(y1 - y0, 1), max(x1 - x0, 1), 4), np.uint8)
    lut = palette_rgba(cue.pal)
    for co, ow, oh, data in cue.objs:
        rgba = lut[rle_decode(data, ow, oh)]
        dx, dy = co["x"] - x0, co["y"] - y0
        ch, cw = canvas.shape[:2]
        if dx < 0 or dy < 0 or dy + oh > ch or dx + ow > cw:
            # Clip BOTH ends: a composition may place an object outside its
            # own window (once in the 71-file batch). Clamping only the
            # overflow is not enough - for a negative dx, "cw - dx" GROWS the
            # slice instead of shrinking it and the canvas is then indexed
            # from its end, which raises IndexError out of a module that
            # promises a reason instead (see ocr_track).
            sx, sy = max(0, -dx), max(0, -dy)
            dx, dy = dx + sx, dy + sy
            sh, sw = min(oh - sy, ch - dy), min(ow - sx, cw - dx)
            if sh <= 0 or sw <= 0:
                continue
            rgba, oh, ow = rgba[sy:sy + sh, sx:sx + sw], sh, sw
        dst = canvas[dy:dy + oh, dx:dx + ow]
        m = rgba[:, :, 3] > 0             # later objects draw over earlier
        dst[m] = rgba[m]
    return canvas


# ---- preprocessing --------------------------------------------------------


def signal_map(rgba: np.ndarray) -> np.ndarray:
    """-> float32 in [0,1] where 1 means "this pixel is glyph".

    luma * alpha. These cues are white glyphs with a BLACK OUTLINE on a
    transparent ground (measured on S05E05: opaque-dark pixels 18.2% against
    opaque-bright 14.5%), so the glyph core (white, opaque) goes to 1 while
    the outline (black, opaque) and the background (alpha 0) both collapse to
    0. That collapse is what makes the letters solid.

    The naive alternative - composite over white - leaves white glyphs on a
    white page bounded by dark outlines, i.e. a hollow stencil whose interior
    matches the background, and tesseract reads the outlines as the letters.
    """
    rgb = rgba[..., :3].astype(np.float32)
    a = rgba[..., 3].astype(np.float32) / 255.0
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return (lum / 255.0) * a


def upscale(img: np.ndarray, f: int) -> np.ndarray:
    """Bilinear upscale of a float image. numpy only - there is no PIL."""
    if f == 1:
        return img
    h, w = img.shape
    yy = np.clip((np.arange(h * f) + 0.5) / f - 0.5, 0, h - 1)
    xx = np.clip((np.arange(w * f) + 0.5) / f - 0.5, 0, w - 1)
    y0 = np.floor(yy).astype(np.intp)
    x0 = np.floor(xx).astype(np.intp)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (yy - y0).astype(np.float32)[:, None]
    wx = (xx - x0).astype(np.float32)[None, :]
    top = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
    bot = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def otsu(gray: np.ndarray) -> int:
    """Otsu's threshold, for the one fallback rung that binarises."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    omega = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    denom = omega * (total - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(denom > 0,
                         (mu[-1] * omega - mu * total) ** 2 / denom, 0)
    return int(np.argmax(sigma))


def preprocess(rgba: np.ndarray, scale: int = 2, binarise: bool = False,
               pad: int = 0) -> np.ndarray:
    """RGBA cue -> uint8 grayscale, DARK TEXT ON LIGHT, ready for tesseract.

    Binarisation is OFF by default: measured, keeping the antialiasing beats
    Otsu on this material, and Otsu on a cue that is nearly all background
    picks a threshold inside the noise.
    """
    s = upscale(signal_map(rgba), scale)
    gray = np.clip((1.0 - s) * 255.0, 0, 255).astype(np.uint8)
    if binarise:
        gray = np.where(gray > otsu(gray), 255, 0).astype(np.uint8)
    if pad:
        gray = np.pad(gray, pad, mode="constant", constant_values=255)
    return gray


def to_pgm(gray: np.ndarray) -> bytes:
    """Binary PGM (P5). This is what goes down tesseract's stdin: no PNG
    encoder, no temp file, no zlib pass over every cue."""
    h, w = gray.shape
    return b"P5\n%d %d\n255\n" % (w, h) + gray.tobytes()


# ---- the OCR engine -------------------------------------------------------


class TesseractEngine:
    """tesseract over a pipe, one process per cue.

    THE SEAM. Anything with `.name` and

        recognise(rgba) -> (text, rung_name)

    works here; ocr_track() enters the engine through exactly that one call,
    and nothing else about this class is required. That is what lets a GPU
    vision-language model be dropped in later without touching the parser, the
    post-pass, the SRT assembly or the gates - it would receive the ORIGINAL
    RGBA canvas, colour and antialiasing intact, rather than a bitmap
    preprocessed for tesseract. `rung_name` is how a retry is reported instead
    of hidden; an engine with no ladder returns "primary" for everything.
    """

    name = "tesseract"

    def __init__(self, binary: str = "tesseract", lang: str = "eng",
                 psm: int = 6, scale: int = 2, tessdata: Optional[str] = None,
                 timeout: float = 60.0) -> None:
        self.binary = binary
        self.lang = lang
        self.psm = psm
        self.scale = scale
        self.tessdata = tessdata
        self.timeout = timeout

    def _cmd(self, psm: int) -> List[str]:
        cmd = [self.binary, "-", "-", "--psm", str(psm), "-l", self.lang]
        if self.tessdata:
            cmd += ["--tessdata-dir", self.tessdata]
        return cmd

    def _run(self, gray: np.ndarray, psm: int) -> str:
        # OMP_THREAD_LIMIT=1: tesseract links OpenMP and otherwise starts a
        # thread pool per process. With a pool of workers that is cores^2
        # threads and the box thrashes (measured: see ocr_track's bound).
        env = dict(os.environ, OMP_THREAD_LIMIT="1")
        p = subprocess.run(self._cmd(psm), input=to_pgm(gray),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=self.timeout, env=env)
        if p.returncode != 0:
            raise PgsError(f"tesseract rc={p.returncode}: "
                           f"{p.stderr.decode('utf-8', 'replace')[:200]}")
        return p.stdout.decode("utf-8", "replace")

    def recognise(self, rgba: np.ndarray) -> Tuple[str, str]:
        """-> (raw text, which rung produced it).

        The ladder only ever runs for a cue that came back EMPTY, and the rung
        is reported so that a silent fallback can never be mistaken for a
        clean read - a track that needed the ladder often is a track whose
        preprocessing does not suit it, which is gate `fallback_rate`.
        """
        txt = self._run(preprocess(rgba, self.scale), self.psm)
        if clean_text(txt):
            return txt, "primary"
        # a short cue that psm 6 read as a block of text
        txt = self._run(preprocess(rgba, self.scale), 7)
        if clean_text(txt):
            return txt, "psm7"
        # a cue whose contrast the antialiasing hid
        txt = self._run(preprocess(rgba, self.scale, binarise=True), self.psm)
        if clean_text(txt):
            return txt, "otsu"
        # bigger, single line, with a margin to sit in
        txt = self._run(preprocess(rgba, max(self.scale, 4), pad=20), 7)
        if clean_text(txt):
            return txt, "big-psm7"
        return "", "empty"


# ---- post-processing ------------------------------------------------------


def clean_text(raw: str) -> str:
    """Whitespace discipline only: strip each line, drop blank lines, collapse
    runs of spaces. Line breaks WITHIN a cue are preserved - they are the
    speaker split, and a player renders them."""
    lines = []
    for ln in raw.replace("\f", "\n").splitlines():
        ln = re.sub(r"[ \t ]+", " ", ln).strip()
        if ln:
            lines.append(ln)
    return "\n".join(lines)


def _lev(a: str, b: str) -> int:
    """Plain Levenshtein, for the acronym gazetteer. Short strings only."""
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


INITIALISM_RE = re.compile(r"(?:[A-Za-z]{0,2}\.){3,}")
WELLFORMED_RE = re.compile(r"^(?:[A-Z]\.){3,}$")
_BAR_APOST = ("'ve", "'m", "'ll", "'d", "'re", "'s", "'VE", "'M", "'LL")
_QUOTE_STARTS = ('"', "“", "”")


def load_lexicon(paths: Sequence[str] = LEXICON_PATHS) -> set:
    """The first word list that exists, lower-cased. Empty when there is none,
    which every rule and gate below degrades to a no-op for."""
    for p in paths:
        try:
            words = Path(p).read_text("utf-8", errors="replace").split()
        except OSError:
            continue
        lex = set()
        for w in words:
            w = w.strip().lower()
            if w:
                lex.add(w)
                lex.add(w.replace("'s", ""))
        if lex:
            return lex
    return set()


def build_doc_context(texts: Sequence[str], lexicon: Optional[set] = None) -> dict:
    """Evidence gathered from the WHOLE track, for the rules that need it.

    The gazetteer holds initialisms this track spells correctly SOMEWHERE, so
    nothing is hardcoded to one show: an episode with no well-formed instance
    gets an empty gazetteer and that rule never fires.
    """
    gaz: Counter = Counter()
    for t in texts:
        for tok in re.findall(r"\S+", t):
            # Tolerate the one stroke confusion the rule exists to fix before
            # scanning: '|' is not [A-Za-z], so it breaks the initialism match
            # itself. Measured on S05E05, both well-formed instances of the
            # acronym read "S.H.|.E.L.D.", so scanning raw finds nothing.
            # WELLFORMED_RE still demands one upper-case letter per segment,
            # so a damaged form can never seed the gazetteer.
            tok = tok.replace("|", "I").replace("1", "I")
            for m in INITIALISM_RE.finditer(tok):
                s = m.group(0)
                if WELLFORMED_RE.match(s):
                    gaz["".join(ch for ch in s if ch.isalpha())] += 1
    return {"initialisms": gaz,
            "lexicon": lexicon if lexicon is not None else set()}


EMPTY_DOC = {"initialisms": Counter(), "lexicon": set()}


def _rx(pattern, repl):
    def f(txt, doc):
        return pattern.subn(repl, txt)
    return f


def _bar_to_I(txt: str, doc: dict) -> Tuple[str, int]:
    """Case-aware '|' repair.

    A blanket rule always emits a capital I. That is right when the stroke was
    the pronoun, an initialism letter or a contraction ("|'ve"), and wrong
    when it was a lowercase 'l' inside a word ("bu|let"). This decides per
    occurrence from the surrounding characters, and asks the lexicon only for
    the genuinely ambiguous word-initial case.

    Measured on the 711-cue set: 99 occurrences, all 99 genuinely 'I', and
    this version substitutes the same 99 characters as the blanket rule while
    being strictly safer where '|' stands for an 'l'.
    """
    lex = doc.get("lexicon") or set()
    n = 0
    out_lines = []
    for line in txt.split("\n"):
        s, res, i = line, [], 0
        while i < len(s):
            if s[i] != "|":
                res.append(s[i])
                i += 1
                continue
            a = i
            while a > 0 and (s[a - 1].isalnum() or s[a - 1] == "|"):
                a -= 1
            b = i
            while b + 1 < len(s) and (s[b + 1].isalnum() or s[b + 1] == "|"):
                b += 1
            left, right = s[a:i], s[i + 1:b + 1]
            tail = s[i + 1:i + 4]
            if not left and not right:
                rep = "I"                      # a standalone stroke
            elif not left and any(tail.startswith(x) for x in _BAR_APOST):
                rep = "I"                      # |'ve, |'m, |'ll
            elif left and left.replace("|", "").islower():
                rep = "l"                      # inside a lowercase word
            elif (i > 0 and s[i - 1] == ".") or (i + 1 < len(s) and s[i + 1] == "."):
                rep = "I"                      # initialism letter
            elif not left and right and right.islower():
                # word-initial before lowercase is ambiguous ('|n' is "in",
                # '|ook' is "look"): only a word real with 'l' and NOT real
                # with 'i' becomes 'l'.
                if lex and ("l" + right).lower() in lex \
                        and ("i" + right).lower() not in lex:
                    rep = "l"
                else:
                    rep = "I"
            else:
                rep = "I"
            res.append(rep)
            n += 1
            i += 1
        out_lines.append("".join(res))
    return "\n".join(out_lines), n


def _dotted_acronym(txt: str, doc: dict) -> Tuple[str, int]:
    """Repair stylised dotted initialisms (the S.H.I.E.L.D. shape).

    Two conservative mechanisms: a lone 'l' or '1' as a segment of an
    otherwise upper-case initialism is an 'I'; and a DAMAGED token within edit
    distance 2 of an initialism this same track spells correctly is snapped to
    it. Guards exclude ellipses, two-segment abbreviations ("p.m.") and
    ordinary words: at least 3 dotted segments, at least 2 letters, at least
    one upper-case letter, and no segment longer than 2 letters.
    """
    gaz = doc.get("initialisms") or Counter()
    n = 0

    def fix(m):
        nonlocal n
        s = m.group(0)
        # Dots beyond the initialism's own final dot are an ellipsis and must
        # survive: the truth track writes "S.H.I.E.L.D..." at the end of a
        # sentence, so eating it trades one error for another.
        trail = len(s) - len(s.rstrip("."))
        body = s[:len(s) - trail]
        core = body.split(".")
        letters = [c for c in body if c.isalpha()]
        if len(core) < 3 or len(letters) < 2:
            return s
        if not any(c.isupper() for c in letters):
            return s
        if any(len(x) > 2 for x in core):
            return s
        extra = "." * max(trail - 1, 0)
        fixed = ["I" if x in ("l", "1") else x for x in core]
        cand = "".join(fixed).upper()
        plain = "".join(x + "." for x in fixed)
        # ONLY A DAMAGED TOKEN IS SNAPPED. A token that already reads as a
        # well-formed initialism is a reading, not damage, and the gazetteer
        # never gets to rewrite it into a DIFFERENT one - the distance bound
        # alone is wide enough to span two real-world initialisms, and the
        # result is ASCII, the right length and the right cue count, so no
        # gate downstream can see it. Measured over the recorded raw OCR of
        # all 73 real tracks: of the 50 snaps that change an acronym this
        # refuses exactly 2, and both were corruptions of a correct read
        # ("S.H.I.E.L.D." -> "S.H.I.E.L.", "I.E.L.D." -> "E.L.D."), while all
        # 48 repairs of a damaged form ("S.H.I.LE.L.D.") still happen. It is
        # also what stops "U.S.A." becoming "U.S.S.R." on a track that says
        # both - and the srt carrying that error is the track that TAKES the
        # default flag, so it is the one the viewer is shown.
        if gaz and not WELLFORMED_RE.match(plain):
            best, bestd = None, 99
            for g, _cnt in gaz.most_common():
                d = _lev(cand, g)
                if d < bestd:
                    best, bestd = g, d
            if best is not None and bestd <= 2 and abs(len(cand) - len(best)) <= 2:
                new = "".join(c + "." for c in best) + extra
                if new != s:
                    n += 1
                return new
        new = plain + extra
        if new != s:
            n += 1
        return new

    return INITIALISM_RE.sub(fix, txt), n


def _pair_dash(txt: str, doc: dict) -> Tuple[str, int]:
    """Restore the leading speaker dash on the sibling line of a dialogue pair.

    In this authoring convention a two-line cue has a dash on BOTH lines (two
    speakers) or on neither (one speaker running over), so a dash on exactly
    one line is a dropped dash rather than a style. Measured on the 711-cue
    set: the truth track has 79 both-dash cues and ZERO one-sided ones, so the
    rule has no legitimate case to damage.
    """
    lines = txt.split("\n")
    if len(lines) != 2:
        return txt, 0
    d = [ln.lstrip().startswith("-") for ln in lines]
    if sum(d) != 1:
        return txt, 0
    i = 1 if d[0] else 0
    other = lines[i]
    lines[i] = "- " + (other[1:] if other[:1] in _QUOTE_STARTS else other).lstrip()
    return "\n".join(lines), 1


# ORDER MATTERS: dash_space normalises "-X" to "- X" and smart_quotes folds
# the curly quotes, both before pair_dash inspects the start of each line.
POST_RULES = [
    ("bar_to_I", _bar_to_I),
    ("dash_space", _rx(re.compile(r"(?m)^-(?=\S)"), "- ")),
    ("smart_quotes", _rx(re.compile("[‘’]"), "'")),
    ("smart_quotes", _rx(re.compile("[“”]"), '"')),
    ("dotted_acronym", _dotted_acronym),
    ("pair_dash", _pair_dash),
]


def postprocess(raw: str, doc: Optional[dict] = None) -> Tuple[str, Counter]:
    """Every measured repair, in order. -> (text, per-rule hit counts)."""
    doc = EMPTY_DOC if doc is None else doc
    txt = clean_text(raw)
    hits: Counter = Counter()
    for name, fn in POST_RULES:
        txt, n = fn(txt, doc)
        if n:
            hits[name] += n
    return clean_text(txt), hits          # a substitution can strand space


# ---- SRT assembly ---------------------------------------------------------


def ts(sec: float) -> str:
    ms = int(round(max(sec, 0.0) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt_text(records: Sequence[Tuple[float, float, str]]) -> str:
    """records: (start, end, text), empty text already removed."""
    return "\n".join(f"{i}\n{ts(st)} --> {ts(en)}\n{txt}\n"
                     for i, (st, en, txt) in enumerate(records, 1))


# ---- gates ----------------------------------------------------------------

# Calibrated on the one track where a ground truth exists (S05E05, 711 cues)
# and then checked against a 71-file batch over the real library, where 70 of
# 71 passed. A gate is only worth having if a good file clears it with room,
# so each bound records what that file measured.
#
#   empty_rate         cues blank after the whole fallback ladder, over the
#                      PGS composition display sets. Shield 0.0; a blank cue
#                      is a silently dropped subtitle. cue_coverage is the
#                      SAME measurement the other way up (the two sum to 1),
#                      so it is reported by measure() and not gated: bounded
#                      at 0.995 beside this one at 0.010, coverage always
#                      failed first and the 1% tolerance documented here was
#                      dead - a track with 0.6% blank cues was refused while
#                      the log blamed a gate nobody had tuned.
#   timing_parse_sane  starts strictly increasing and no non-positive
#                      duration. Either failing means the .sup mis-parsed, so
#                      this one is hard.
#   overlap_rate       cues overlapping the next. Shield 0/711. NOT folded
#                      into timing_parse_sane: four files in the batch carry a
#                      single "English SDH" label card laid over the previous
#                      cue (1/54 .. 1/203, at most 0.0185), which is a
#                      cosmetic artefact in the SOURCE and not a parse defect.
#                      A genuinely slipped parse overlaps far more than one
#                      pair in fifty.
#   bar_rate           residual '|' per 1000 chars. A bar that survived the
#                      post-pass means the image was not read as text at all.
#   fallback_rate      cues that needed a retry rung. Shield 0.0 (711/711
#                      primary). Rises when the preprocessing does not suit
#                      the track.
#   nonascii_rate      the eng model has no business emitting non-ASCII, and
#                      the post-pass folds the curly quotes it can emit.
#                      Shield 0 of 24933 characters, truth likewise.
#   chars_per_cue      median characters per cue. Catches a track that OCR'd
#                      to near-nothing while keeping the right cue count.
#   oov_rate           share of word tokens not in the system word list. The
#                      only signal here that tracks recognition quality
#                      without a truth, so it carries the loosest bound and is
#                      applied only above MIN_TOKENS words: at ~139 tokens one
#                      unusual word moves the rate by 0.7 points. Tokens are
#                      clitic-normalised first - the word list indexes no
#                      contraction, so without that "let's" and "that's"
#                      counted as OOV and the gate was measuring the LEXICON
#                      rather than the OCR (it failed two correct files).
THRESHOLDS = {
    "empty_rate": ("<=", 0.010),
    "timing_parse_sane": ("==", True),
    "overlap_rate": ("<=", 0.020),
    "bar_rate": ("<=", 1.0),
    "fallback_rate": ("<=", 0.050),
    "nonascii_rate": ("<=", 0.005),
    "chars_per_cue": (">=", 8.0),
    "oov_rate": ("<=", 0.060),
}
MIN_TOKENS = 300

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_CLITIC_RE = re.compile(r"'(s|t|re|ll|ve|d|m|em)$")


def measure(records: Sequence[Tuple[float, float, str]], n_cues: int,
            rungs: Counter, lexicon: set) -> dict:
    """Raw signals for one track. No pass/fail here; evaluate() does that."""
    body = "\n".join(t for _s, _e, t in records)
    chars = sorted(len(t) for _s, _e, t in records)
    total = sum(chars)
    n_out = len(records)
    starts_inc = all(records[i][0] < records[i + 1][0] for i in range(n_out - 1))
    overlaps = sum(1 for i in range(n_out - 1)
                   if records[i][1] > records[i + 1][0] + 1e-6)
    bad_dur = sum(1 for s, e, _t in records if e - s <= 0)
    freq: Counter = Counter()
    for _s, _e, t in records:
        for w in _TOKEN_RE.findall(t):
            freq[_CLITIC_RE.sub("", w.lower().strip("'-"))] += 1
    ntok = sum(freq.values())
    oov = sum(c for w, c in freq.items() if w and w not in lexicon)
    used = sum(rungs.values()) or 1
    return {
        "pgs_cues": n_cues,
        "srt_cues": n_out,
        "empty_cues": n_cues - n_out,
        "cue_coverage": round(n_out / max(n_cues, 1), 6),
        "empty_rate": round((n_cues - n_out) / max(n_cues, 1), 6),
        "timing_parse_sane": bool(starts_inc and bad_dur == 0),
        "overlaps": overlaps,
        "overlap_rate": round(overlaps / max(n_out, 1), 6),
        "chars_total": total,
        "chars_per_cue": float(chars[len(chars) // 2]) if chars else 0.0,
        "bars": body.count("|"),
        "bar_rate": round(1000.0 * body.count("|") / max(total, 1), 4),
        "fallback_rate": round(1.0 - rungs.get("primary", 0) / used, 6),
        "nonascii": sum(1 for ch in body if ord(ch) > 127),
        "nonascii_rate": round(sum(1 for ch in body if ord(ch) > 127)
                               / max(total, 1), 6),
        "tokens": ntok,
        "oov_rate": round(oov / ntok, 6) if ntok else 0.0,
        "lexicon_entries": len(lexicon),
    }


def evaluate(sig: dict, thresholds: Optional[dict] = None) -> Tuple[bool, list]:
    """-> (ok, gates). A gate that cannot be measured is SKIPPED, not failed:
    oov_rate below MIN_TOKENS words, and every gate needing a word list when
    the image has none. Skipping says so in the gate's own record rather than
    quietly passing."""
    gates = []
    for name, (op, bound) in (thresholds or THRESHOLDS).items():
        skip = ""
        if name == "oov_rate":
            if sig.get("tokens", 0) < MIN_TOKENS:
                skip = f"only {sig.get('tokens', 0)} tokens (< {MIN_TOKENS})"
            elif not sig.get("lexicon_entries"):
                skip = "no word list in this image"
        val = sig.get(name)
        if skip:
            gates.append({"name": name, "value": val, "op": op,
                          "threshold": bound, "pass": True, "skipped": skip})
            continue
        if val is None:
            ok = False
        elif op == ">=":
            ok = val >= bound
        elif op == "<=":
            ok = val <= bound
        else:
            ok = val == bound
        gates.append({"name": name, "value": val, "op": op,
                      "threshold": bound, "pass": bool(ok), "skipped": ""})
    return all(g["pass"] for g in gates), gates


# ---- the runner -----------------------------------------------------------


class OcrResult(NamedTuple):
    """What one track's OCR produced. `text` is None whenever no SRT should be
    written - a failure, a refusal, or a gate that did not hold - and `why`
    then says which, for the one log line."""
    text: Optional[str]
    why: str
    cues: int = 0
    written: int = 0
    seconds: float = 0.0
    signals: Optional[dict] = None
    gates: Optional[list] = None


def default_workers(cpus: Optional[int] = None) -> int:
    """How many cues to OCR at once.

    One tesseract process per cue, each pinned to a single OpenMP thread, so
    the pool is CPU-bound and the right bound is the CPU budget this container
    actually has - not the host's core count, which is 64 on a box whose
    av1transcode is limited to 40.

    Capped at 16 because that is where the measured curve stops paying. One
    711-cue track, 32 cores, wall seconds against workers: 1 -> 177.9,
    2 -> 90.0, 4 -> 46.0, 8 -> 24.8, 16 -> 16.3, 32 -> 12.8. The CPU the child
    processes actually burn is flat to 8 (159.4, 159.6, 159.0, 158.4 seconds)
    and then climbs: 168.5 at 16, 186.9 at 32. So 32 workers buy 3.5 seconds
    of wall over 16 and cost 11% more CPU to do it, on a box that has just
    finished an encode and may still be busy with the next one - and this runs
    in the mux, minutes at the end of a job measured in hours.
    """
    if cpus is None:
        try:
            cpus = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cpus = os.cpu_count() or 1
    return max(1, min(16, int(cpus)))


def ocr_track(sup: Path, engine=None, workers: Optional[int] = None,
              lexicon: Optional[set] = None,
              budget: float = 900.0) -> OcrResult:
    """One .sup -> the text of an SRT, or a reason there is none.

    THIS NEVER RAISES. A missing tesseract, a damaged .sup, a cue that hangs,
    a gate that does not hold - every one of them comes back as text=None and
    the caller leaves that track alone. An encode that ran for three hours
    must not be failed by a subtitle nicety.
    """
    t0 = time.perf_counter()
    try:
        cues, notes = parse_sup(Path(sup).read_bytes())
    except (PgsError, OSError, ValueError, IndexError, struct.error) as e:
        return OcrResult(None, f"could not parse {Path(sup).name}: {e}")
    if not cues:
        return OcrResult(None, "the track carries no composition display set")
    damaged = {k: v for k, v in notes.items() if not k.startswith("seg_")}
    if damaged:
        logger.warning("pgs ocr: {} parsed with notes {}", Path(sup).name,
                       damaged)

    engine = engine or TesseractEngine()
    n = workers or default_workers()
    raws: List[Optional[str]] = [None] * len(cues)
    rungs: Counter = Counter()
    lock = threading.Lock()
    deadline = t0 + budget
    stop = False

    def one(cue: Cue):
        nonlocal stop
        if stop:
            return
        if time.perf_counter() > deadline:
            stop = True
            return
        txt, rung = engine.recognise(render(cue))
        with lock:
            raws[cue.index] = txt
            rungs[rung] += 1

    try:
        # Threads, not processes: the work per cue is a numpy render and then
        # an external process, and waiting on that process holds no GIL. It
        # also keeps ONE parse of the track - handing 711 bitmaps to a process
        # pool would pickle the whole .sup across the pipe.
        with ThreadPoolExecutor(max_workers=n) as ex:
            list(ex.map(one, cues))
    except (OSError, ValueError, IndexError, struct.error,
            subprocess.SubprocessError, PgsError) as e:
        return OcrResult(None, f"OCR failed: {e}")
    if stop or any(r is None for r in raws):
        done = sum(1 for r in raws if r is not None)
        return OcrResult(None, f"OCR ran out of its {budget:.0f}s budget after "
                               f"{done} of {len(cues)} cues")

    lexicon = load_lexicon() if lexicon is None else lexicon
    doc = build_doc_context([clean_text(r or "") for r in raws], lexicon)
    records = []
    for cue in cues:
        txt, _hits = postprocess(raws[cue.index] or "", doc)
        if txt:                        # an empty cue is never emitted
            records.append((cue.start, cue.end, txt))
    if not records:
        return OcrResult(None, "every cue came back empty", len(cues), 0,
                         time.perf_counter() - t0)

    sig = measure(records, len(cues), rungs, lexicon)
    ok, gates = evaluate(sig)
    took = time.perf_counter() - t0
    if not ok:
        bad = ", ".join(f"{g['name']}={g['value']} {g['op']} {g['threshold']}"
                        for g in gates if not g["pass"])
        return OcrResult(None, f"gates failed: {bad}", len(cues), len(records),
                         took, sig, gates)
    return OcrResult(srt_text(records), "ok", len(cues), len(records), took,
                     sig, gates)
