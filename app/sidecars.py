"""Carry a source's sidecar subtitles over to its output's name.

Plex and Jellyfin pair an external subtitle with a video by file name: the
subtitle's name has to start with the video's - "<stem>.zh.srt" for
"<stem>.mkv". Jellyfin does exactly that and nothing looser
(MediaInfoResolver.GetExternalFiles: the video's own folder, its name, then a
delimiter). Plex measured on the production library: a folder holding one
episode and a subtitle named for another release of it did not list that
subtitle; the one named for the episode's file was listed, from the folder
itself and from its subs/ folder.

The output is renamed (naming.av1_stem turns "HEVC" into "AV1") and written to
another folder (<source_dir>/av1/ by default), so every sidecar the source had -
Bazarr's downloads, a translation aisub wrote - stops showing up for the AV1
version, and stays behind as an orphan once the source is gone. carry() copies
them to the output's name, in the same place relative to the output as they
were to the source:

    S1/Show.S01E01.2160p.HEVC.mkv            S1/av1/Show.S01E01.2160p.AV1.mkv
    S1/Show.S01E01.2160p.HEVC.en.srt     ->  S1/av1/Show.S01E01.2160p.AV1.en.srt
    S1/subs/Show.S01E01.2160p.HEVC.zh.srt -> S1/av1/subs/Show.S01E01.2160p.AV1.zh.srt

A copy, so both versions keep their subtitles while both exist; with
transcode.delete_source the source-side files go once the source itself is
deleted (remove_sources), which makes it a move. The output keeps the
source's timing - audio and embedded subtitles are copied with their
timestamps and the new video is synced to the source's lead over them
(optimizer mux, --sync) - so a subtitle timed for one is timed for the other.

Nothing here can fail a job: a file that cannot be read or written is
reported and left where it was.
"""
from __future__ import annotations

import filecmp
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# What a player takes as an external subtitle. .idx/.sub travel as a pair,
# and they share a stem, so both are carried.
SUBTITLE_EXTS = frozenset({".srt", ".ass", ".ssa", ".vtt", ".sub", ".idx", ".sup", ".smi"})
# Folders beside a video that hold its subtitles (matched case-insensitively,
# written back under the name the source's folder has).
SUBTITLE_DIRS = frozenset({"subs", "subtitles", "sub"})
# A source this applies to: a video file. A disc (BDMV folder, .iso, .mpls)
# has no sidecars named after it.
VIDEO_EXTS = frozenset({".mkv", ".mp4", ".m4v", ".avi", ".mov", ".ts", ".m2ts", ".webm", ".wmv"})

COPIED, SAME, KEPT, FAILED = "copied", "same", "kept", "failed"


@dataclass
class Carried:
    source: Path        # the sidecar named after the source
    dest: Path          # where the output's copy is
    status: str         # COPIED | SAME (that file was already there) | KEPT (a different one was) | FAILED
    error: str = ""

    def line(self) -> str:
        what = {COPIED: "copied", SAME: "already there", KEPT: "kept the different file already at",
                FAILED: "failed"}[self.status]
        tail = f" ({self.error})" if self.error else ""
        return f"{self.source} -> {what} {self.dest}{tail}"


def applies_to(source: Path) -> bool:
    return source.suffix.lower() in VIDEO_EXTS and source.is_file()


def _named_for(name: str, stem: str) -> bool:
    """`name` is a sidecar of the video called `stem`: its name, then a dot."""
    return len(name) > len(stem) + 1 and name[:len(stem)].lower() == stem.lower() and name[len(stem)] == "."


def find(source: Path) -> List[Tuple[Path, str]]:
    """The sidecar subtitles of `source`: (path, folder relative to the source's).

    The folder is "" for the source's own, else the name of the subtitle
    folder beside it. A subtitle whose name starts with a LONGER video stem in
    the same folder belongs to that video ("Show.S01E01.part2.en.srt" next to
    "Show.S01E01.mkv" and "Show.S01E01.part2.mkv") and is left out.
    """
    home = source.parent
    stem = source.stem
    try:
        entries = list(home.iterdir())
    except OSError:
        return []
    others = [entry.stem for entry in entries
              if entry != source and entry.suffix.lower() in VIDEO_EXTS and len(entry.stem) > len(stem)
              and _named_for(entry.name, stem)]
    places = [(home, "")]
    places += sorted((entry, entry.name) for entry in entries
                     if entry.name.lower() in SUBTITLE_DIRS and entry.is_dir())
    found: List[Tuple[Path, str]] = []
    for folder, rel in places:
        try:
            names = sorted(child.name for child in folder.iterdir())
        except OSError:
            continue
        for name in names:
            path = folder / name
            if Path(name).suffix.lower() not in SUBTITLE_EXTS or not _named_for(name, stem):
                continue
            if any(_named_for(name, other) for other in others):
                continue
            if path.is_file():
                found.append((path, rel))
    return found


def _copy(src: Path, dest: Path) -> None:
    """Copy beside the target and rename onto it, so a player never reads half a file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(f".{dest.name}.{os.getpid()}.part")
    try:
        shutil.copyfile(src, part)
        try:
            shutil.copystat(src, part)
        except OSError:
            pass            # a NAS that refuses the timestamps still gets the file
        os.replace(part, dest)
    finally:
        try:
            part.unlink()
        except FileNotFoundError:
            pass


def carry(source: Path, output: Path) -> List[Carried]:
    """Copy each sidecar of `source` to `output`'s name (see the module docstring).

    A file already at the destination is never replaced: the same bytes are
    SAME, different bytes KEPT (someone put a subtitle there - it stays, and
    so does the source's, even with delete_source).
    """
    out: List[Carried] = []
    for path, rel in find(source):
        folder = output.parent / rel if rel else output.parent
        dest = folder / (output.stem + path.name[len(source.stem):])
        try:
            if dest.exists():
                same = filecmp.cmp(path, dest, shallow=False)
                out.append(Carried(path, dest, SAME if same else KEPT))
                continue
            _copy(path, dest)
            out.append(Carried(path, dest, COPIED))
        except OSError as e:
            out.append(Carried(path, dest, FAILED, str(e)))
    return out


def remove_sources(carried: List[Carried]) -> List[Path]:
    """After delete_source took the source: drop the sidecars whose copy is in place.

    What was KEPT or FAILED stays - its subtitle exists nowhere else.
    Returns what was removed.
    """
    removed: List[Path] = []
    for item in carried:
        if item.status not in (COPIED, SAME):
            continue
        try:
            item.source.unlink()
            removed.append(item.source)
        except FileNotFoundError:
            pass
        except OSError:
            continue
    return removed
