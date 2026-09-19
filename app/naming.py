"""Output names that read the way Sonarr/Radarr would name the result.

Library names come from a naming format such as
`{Series.CleanTitleYear}.S{season:00}E{episode:00}.{Episode.CleanTitle}.
{Quality.Full}.{MediaInfo.Simple}.{MediaInfo.VideoDynamicRangeType}`, so a
source is `...Bluray-2160p.Remux.h265.DTS-HD.MA.DV.HDR10.mkv`, and once the
AV1 replaces it Sonarr renames it `...Remux.AV1.DTS-HD.MA.HDR10.mkv`: the
codec becomes AV1, and Dolby Vision and HDR10+ drop out of the dynamic range
because the encode keeps neither. Doing the same here gives the output the
name Sonarr would give it. Only those two parts change - quality, audio and
release group describe things the encode copies, so they stay.
"""
from __future__ import annotations

import re
from typing import Tuple

_START = r"(?<![A-Za-z0-9])"
_END = r"(?![A-Za-z0-9])"
# {MediaInfo VideoCodec} as Sonarr/Radarr write it (h264/x264/AVC, h265/x265/
# HEVC, MPEG2, VC1, VP9, XviD, DivX) plus the scene spellings H.264/H265.
_CODEC = re.compile(_START + r"(?:[hx]\.?26[45]|avc|hevc|mpeg-?2|vc-?1|vp9|xvid|divx)" + _END, re.I)
_RESOLUTION = re.compile(_START + r"\d{3,4}[pi]" + _END, re.I)
# A run of dynamic-range tokens: Sonarr's "DV.HDR10", "DV.HDR10Plus", "DV.HLG",
# "DV.SDR", "HDR10Plus", and the scene "DV.HDR10+", "DoVi", "DV-HDR".
_DR = r"(?:dv|dovi|hdr10(?:\+|plus)|hdr10|hdr|hlg|pq|sdr)"
_DR_RUN = re.compile(_START + _DR + r"(?:[ ._-]" + _DR + r")*" + r"(?![A-Za-z0-9+])", re.I)
# What the output cannot be, so a run naming one of these is rewritten.
_NOT_KEPT = re.compile(r"(?i)^(?:dv|dovi|hdr10(?:\+|plus))$")
_AV1_TWICE = re.compile(_START + r"AV1[ ._-]AV1" + _END)


def dynamic_range(color_trc: str, has_hdr_metadata: bool) -> str:
    """{MediaInfo VideoDynamicRangeType} of the encoded file: Sonarr calls PQ
    with mastering display or content light metadata HDR10, bare PQ "PQ",
    HLG "HLG", and writes nothing for SDR."""
    if color_trc == "smpte2084":
        return "HDR10" if has_hdr_metadata else "PQ"
    if color_trc == "arib-std-b67":
        return "HLG"
    return ""


def av1_stem(stem: str, dynamic_range: str) -> Tuple[str, bool]:
    """The source's file stem renamed for the AV1 output.

    Returns (stem, codec_found). With no codec token to replace, the name
    cannot say AV1 by itself and the caller marks it some other way.
    Dynamic-range tokens are only looked for after the resolution (or the
    codec, when there is no resolution), which keeps episode titles out of
    reach; a run is rewritten only when it names DV or HDR10+.
    """
    res = _RESOLUTION.search(stem)
    first_codec = _CODEC.search(stem, res.end() if res else 0)
    codec_found = first_codec is not None
    if res:
        anchor = res.end()
    elif first_codec:
        anchor = first_codec.start()
    else:
        return stem, False

    out = stem[:anchor] + _CODEC.sub("AV1", stem[anchor:])
    while _AV1_TWICE.search(out):           # "HEVC.x265" -> one AV1
        out = _AV1_TWICE.sub("AV1", out)

    pos = anchor
    while True:
        m = _DR_RUN.search(out, pos)
        if not m:
            break
        tokens = re.split(r"[ ._-]", m.group(0))
        if not any(_NOT_KEPT.match(t) for t in tokens):
            pos = m.end()
            continue
        start, end = m.start(), m.end()
        if not dynamic_range:               # SDR: the run goes, with one separator
            if start > 0 and out[start - 1] in " ._-":
                start -= 1
            elif end < len(out) and out[end] in " ._-":
                end += 1
        out = out[:start] + dynamic_range + out[end:]
        pos = start + len(dynamic_range)
    return out, codec_found
