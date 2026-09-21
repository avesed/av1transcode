"""Release groups' and sites' advertising in an output's metadata.

The mux carries a source's title, global tags, track names and attachments
over as they are, so whatever a release signed its files with ends up in the
AV1 too. A survey of 336 sources, two from each of the library's 168 shows
(2026-09-21), found it in four places and nowhere else:

  title        release names with the group in them: "Alien.Earth.S01E01.
               2160p.DSNP.WEB-DL.DV.P5[Ben The Men]", "...x265-RARBG",
               "PSArips.com | Percy.Jackson...-PSA". Wrong on the output as
               well: it is not DV, not x265.
  global tags  COMMENT "[哔嘀影视-bdys.me]", "ZmWeb", "AilMWeb"; COPYRIGHT
               "ZmPT", "HHWEB", "Panda"; UPLOADER "3MWEB"; Group "HHWEB";
               Muxer "Rainbow Island".
  track names  "BTM", "BTM SDH", "BTM DDP5.1 Atmos" on every Ben The Men mp4
               (ffmpeg makes the mp4 name a NAME tag), "JAKET789 DIY简英字幕".
  attachments  "BEN.THE.MEN.TORRENTS.jpg".

Chapter names carried none - "Intro", "Credits", "Scene 12" - so chapters are
left alone. What counts as a signature is a URL, or one of the configured
transcode.metadata.ad_tokens.
"""
from __future__ import annotations

import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Optional, Pattern

from loguru import logger

from app.config import Settings

# A link, or a bare domain on one of the TLDs sites sign with. Bare domains
# only on these: "DDP5.1" and "H.265" have dots too.
_URL = (r"(?:https?://|www\.)[^\s\]\)】|]+"
        r"|(?<![\w.])[\w-]{2,}(?:\.[\w-]+)*\.(?:com|net|org|cc|me|tv|io|xyz|top|info|vip|cn|club)(?!\w)")
# What makes a string a scene release name rather than a title (with no
# spaces and a dot, see release_name)
_QUALITY = re.compile(r"(?<![0-9a-z])(?:2160p|1080p|720p|576p|480p|web-?dl|webrip"
                      r"|blu-?ray|bdrip|remux|hdtv|x26[45]|h\.?26[45]|hevc)(?![0-9a-z])", re.I)
_BRACKETED = re.compile(r"[\[(【（{][^\[\]()【】（）{}]*[\])】）}]")
_SEPARATORS = " \t-_|/.:·@~,;"
# Tags that only ever say who released or muxed the source, never anything
# about the programme: the source muxer's ENCODER (libebml, Lavf, HandBrake -
# the output is written by mkvmerge), the mp4 box fields ffmpeg turns into
# tags, and the fields groups sign in. Every value of these in the survey was
# a signature or a stamp.
_JUNK_GLOBAL = {"ENCODER", "ENCODED_BY", "COMMENT", "COPYRIGHT", "UPLOADER",
                "GROUP", "MUXER", "MAJOR_BRAND", "MINOR_VERSION",
                "COMPATIBLE_BRANDS"}
_JUNK_TRACK = {"HANDLER_NAME", "VENDOR_ID"}
# ffmpeg's own stamp on a track it wrote ("Lavc63.1.101 srt"); the video
# track's "SVT-AV1 v4.2.0" is ours and stays
_LAV_STAMP = re.compile(r"^Lav[cf]\d")
_STATISTICS = {"BPS", "DURATION", "NUMBER_OF_FRAMES", "NUMBER_OF_BYTES",
               "_STATISTICS_WRITING_APP", "_STATISTICS_WRITING_DATE_UTC",
               "_STATISTICS_TAGS"}
_FONT_SUFFIXES = (".ttf", ".otf", ".ttc", ".woff", ".woff2", ".pfb", ".pfm")


def ad_pattern(tokens: Iterable[str]) -> Pattern[str]:
    """One pattern for a URL or any of `tokens`, as a whole word.

    A token's words may be joined by space, dot, dash or nothing: "Ben The
    Men" is also "BEN.THE.MEN" (a release name, an attachment) and
    "BenTheMen".
    """
    alts = []
    for token in tokens:
        words = [re.escape(w) for w in re.split(r"[\s._-]+", token.strip()) if w]
        if words:
            alts.append(r"[\s._-]*".join(words))
    parts = [_URL]
    if alts:
        # longest first, so "PSArips" is not cut to "PSA"
        alts.sort(key=len, reverse=True)
        parts.insert(0, r"(?<![0-9a-z])(?:" + "|".join(alts) + r")(?![0-9a-z])")
    return re.compile("|".join(f"(?:{p})" for p in parts), re.I)


def release_name(text: str) -> bool:
    """Whether `text` is a scene release name: no spaces once a bracketed
    group is taken off, dotted, and naming a resolution, source or codec.

    "The Night Agent (2023) S01E01 (2160p)" is a title, not a release name.
    """
    bare = re.sub(r"\[[^\]]*\]", "", text).strip()
    return (bool(bare) and not re.search(r"\s", bare) and "." in bare
            and bool(_QUALITY.search(bare)))


def strip_ads(text: str, ads: Pattern[str]) -> str:
    """`text` with every signature taken out, and whatever separated it.

    A bracketed part holding one goes whole: "English [www.x.com]" is
    "English", "BTM DDP5.1 Atmos" is "DDP5.1 Atmos", and "BTM" is "".
    """
    out = _BRACKETED.sub(lambda m: " " if ads.search(m.group(0)) else m.group(0), text)
    out = ads.sub(" ", out)
    # "A | BTM | B" leaves "A |  | B": one separator where there were two
    out = re.sub(r"\s*([-|/·])(?:\s*[-|/·])+\s*", r" \1 ", out)
    return re.sub(r"\s+", " ", out).strip(_SEPARATORS)


def _drop_tag(name: str, value: str, target: str, ads: Pattern[str]) -> bool:
    """Whether one SimpleTag goes. `target` is "global", "track" or "other"."""
    name = name.upper()
    if name in _STATISTICS:
        return False
    if target == "global" and name in _JUNK_GLOBAL:
        return True
    if target == "track" and (name in _JUNK_TRACK
                              or (name == "ENCODER" and _LAV_STAMP.match(value))):
        return True
    return bool(ads.search(value)) or (target == "global" and release_name(value))


def clean_tags(root: ET.Element, ads: Pattern[str]) -> bool:
    """Take the signatures out of a Matroska tags tree. True if it changed.

    A SimpleTag goes whole (nested ones with it), and a Tag left with none
    goes too.
    """
    changed = False
    for tag in list(root.findall("Tag")):
        targets = tag.find("Targets")
        if targets is None or not any(
                child.tag.endswith("UID") for child in targets):
            target = "global"
        elif targets.find("TrackUID") is not None:
            target = "track"
        else:
            target = "other"            # a chapter's, an edition's, an attachment's
        for simple in list(tag.findall("Simple")):
            if _drop_tag(simple.findtext("Name") or "", simple.findtext("String") or "",
                         target, ads):
                tag.remove(simple)
                changed = True
        if not tag.findall("Simple"):
            root.remove(tag)
            changed = True
    return changed


def ad_edits(settings: Settings, output: Path, scratch: Path) -> List[List[str]]:
    """mkvpropedit arguments taking the signatures out of `output`, one group
    per thing edited. Reads the header only: mkvmerge -J, and mkvextract, which
    seeks to the tags."""
    ads = ad_pattern(settings.transcode.metadata.ad_tokens)
    try:
        proc = subprocess.run([settings.tool_path("mkvmerge"), "-J", str(output)],
                              capture_output=True, text=True, timeout=120)
        ident = json.loads(proc.stdout or "{}")
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        logger.warning("could not read {} for release ads ({}); left as it is",
                       output.name, e)
        return []
    groups: List[List[str]] = []
    title = ((ident.get("container") or {}).get("properties") or {}).get("title")
    if title and (ads.search(title) or release_name(title)):
        groups.append(["--edit", "info", "--delete", "title"])
    for track in ident.get("tracks") or []:
        props = track.get("properties") or {}
        name, uid = props.get("track_name"), props.get("uid")
        if not name or uid is None:
            continue
        clean = "" if release_name(name) else strip_ads(name, ads)
        if clean == name:
            continue
        edit = ["--edit", f"track:={uid}"]
        groups.append(edit + (["--set", f"name={clean}"] if clean else ["--delete", "name"]))
    for att in ident.get("attachments") or []:
        name = att.get("file_name") or ""
        uid = (att.get("properties") or {}).get("uid")
        # never a font: the ASS subtitles are drawn with them
        font = ("font" in (att.get("content_type") or "")
                or name.lower().endswith(_FONT_SUFFIXES))
        if uid is not None and not font and ads.search(name):
            groups.append(["--delete-attachment", f"={uid}"])
    tags = _clean_tags_file(settings, output, scratch, ads)
    if tags is not None:
        groups.append(["--tags", f"all:{tags}"])
    return groups


def _clean_tags_file(settings: Settings, output: Path, scratch: Path,
                     ads: Pattern[str]) -> Optional[str]:
    """The output's tags with the signatures taken out, as a file for
    mkvpropedit's --tags all: ("" to remove them all), or None when nothing
    needs to change or they could not be read."""
    raw = scratch / "tags.xml"
    try:
        subprocess.run([settings.tool_path("mkvextract"), str(output), "tags", str(raw)],
                       capture_output=True, text=True, timeout=120)
        if not raw.exists() or raw.stat().st_size == 0:
            return None                 # no tags at all
        tree = ET.parse(raw)
    except (OSError, subprocess.SubprocessError, ET.ParseError) as e:
        logger.warning("could not read the tags of {} ({}); left as they are",
                       output.name, e)
        return None
    root = tree.getroot()
    if not clean_tags(root, ads):
        return None
    if not root.findall("Tag"):
        return ""
    clean = scratch / "tags.clean.xml"
    try:
        tree.write(clean, encoding="utf-8", xml_declaration=True)
    except OSError as e:
        logger.warning("could not write the cleaned tags of {} ({}); left as "
                       "they are", output.name, e)
        return None
    return str(clean)
