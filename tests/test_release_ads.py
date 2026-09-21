import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from app import release_ads
from app.config import Metadata, Settings

ADS = release_ads.ad_pattern(Metadata().ad_tokens)


# Everything here was read off the library's own sources (see release_ads).
@pytest.mark.parametrize("title", [
    "Alien.Earth.S01E01.2160p.DSNP.WEB-DL.DV.P5[Ben The Men]",
    "Masters.Of.The.Air.S01E01.2160p[Ben The Men]",
    "Chernobyl.S01E01.1.23.45.2160p.UHD.BluRay.x265.10bit.HDR.DTS-HD.MA.5.1-SWTYBLZ",
    "The.Rookie.S01E01.1080p.WEBRip.x265-RARBG",
    "PSArips.com | Percy.Jackson.and.the.Olympians.S02E03.2160p.10bit.HDR.DV.WEBRip.6CH.x265.HEVC-PSA",
])
def test_release_titles_are_ads(title):
    assert ADS.search(title) or release_ads.release_name(title)


@pytest.mark.parametrize("title", [
    "Shameless (US) - S09E07 - Down Like the Titanic",
    "Better Call Saul (S03E01) Mabel",
    "Common People",
    "Bête Noire",
    "The Night Agent (2023) S01E01 (2160p) ",
    "Fingers and Toes / Пальцы рук и ног (1x01)",
    "Spoiler: Dexys Midnight Runners Get a Royalty Payment",
    "Capitolo 1 : L'arrivo",
    "12:00",
    "1",
    "Arcane League of Legends",
    "Kung Fu Panda",
])
def test_episode_titles_are_not(title):
    assert not ADS.search(title) and not release_ads.release_name(title)


@pytest.mark.parametrize("name, clean", [
    ("BTM", ""),
    ("BTM SDH", "SDH"),
    ("BTM DDP5.1 Atmos", "DDP5.1 Atmos"),
    ("BTM EU Forced", "EU Forced"),
    ("BTM DD 2CH", "DD 2CH"),
    ("JAKET789 DIY简英字幕", "DIY简英字幕"),
    ("English [www.example.com]", "English"),
    ("[哔嘀影视-bdys.me]", ""),
    ("English | BTM | SDH", "English | SDH"),
])
def test_signatures_come_out_of_track_names(name, clean):
    assert release_ads.strip_ads(name, ADS) == clean


@pytest.mark.parametrize("name", [
    "DTS-HD Master Audio / 5.1 / 48 kHz / 3394 kbps / 24-bit",
    "English [Dolby Digital Plus with Dolby Atmos 5.1]",
    "Commentary by Creators Peter Gould, Vince Gilligan, Actors Bob Odenkirk",
    "MVO [WinMedia]",
    "Original | English (United States) | (SDH)",
    "Latin American | Dub | SDH",
    "中文（简体）",
    "Português (Brasil) (SDH)",
    "H.265 / DDP5.1",
])
def test_track_names_without_a_signature_are_left_alone(name):
    assert release_ads.strip_ads(name, ADS) == name


def _tags(*tags):
    """A Matroska tags tree: (targets dict, [(name, value), ...]) per Tag."""
    root = ET.Element("Tags")
    for targets, simple in tags:
        tag = ET.SubElement(root, "Tag")
        t = ET.SubElement(tag, "Targets")
        for k, v in targets.items():
            ET.SubElement(t, k).text = str(v)
        for name, value in simple:
            s = ET.SubElement(tag, "Simple")
            ET.SubElement(s, "Name").text = name
            ET.SubElement(s, "String").text = value
    return root


def _pairs(root):
    return [[(s.findtext("Name"), s.findtext("String")) for s in tag.findall("Simple")]
            for tag in root.findall("Tag")]


def test_clean_tags_drops_signatures_and_stamps_and_keeps_the_rest():
    stats = [("BPS", "3394149"), ("DURATION", "00:57:58.218000000"),
             ("NUMBER_OF_FRAMES", "326083"), ("NUMBER_OF_BYTES", "1475698876"),
             ("_STATISTICS_WRITING_APP", "mkvmerge v82.0 ('I'm The President') 64-bit"),
             ("_STATISTICS_TAGS", "BPS DURATION NUMBER_OF_FRAMES NUMBER_OF_BYTES")]
    root = _tags(
        ({"TargetTypeValue": 70}, [("IMDB", "tt1586680"), ("TMDB", "tv/34307"),
                                   ("ENCODER", "Lavf63.1.101")]),
        ({"TargetTypeValue": 60}, [("PART_NUMBER", "9"), ("TOTAL_PARTS", "14")]),
        ({}, [("COMMENT", "[哔嘀影视-bdys.me]"), ("COPYRIGHT", "ZmPT"),
              ("UPLOADER", "3MWEB"), ("Group", "HHWEB"), ("Muxer", "Rainbow Island"),
              ("MAJOR_BRAND", "isom"), ("COMPATIBLE_BRANDS", "isomdby1iso2mp41"),
              ("DESCRIPTION", "While attempting to steal a new piece of technology"),
              ("TITLE", "Chernobyl.S01E01.2160p.UHD.BluRay.x265-SWTYBLZ")]),
        ({"TrackUID": 1}, [("ENCODER", "SVT-AV1 v4.2.0"),
                           ("ENCODER_SETTINGS", "preset=4 / crf=21-33"), *stats]),
        ({"TrackUID": 2}, [("HANDLER_NAME", "SoundHandler"), ("NAME", "BTM DDP5.1 Atmos"),
                           ("ENCODER", "Lavc63.1.101 flac"), *stats]),
        ({"TrackUID": 3}, [("HANDLER_NAME", "SubtitleHandler")]),
    )
    assert release_ads.clean_tags(root, ADS) is True
    assert _pairs(root) == [
        [("IMDB", "tt1586680"), ("TMDB", "tv/34307")],
        [("PART_NUMBER", "9"), ("TOTAL_PARTS", "14")],
        [("DESCRIPTION", "While attempting to steal a new piece of technology")],
        [("ENCODER", "SVT-AV1 v4.2.0"), ("ENCODER_SETTINGS", "preset=4 / crf=21-33"),
         *stats],
        stats,
    ]
    assert release_ads.clean_tags(root, ADS) is False


def test_ad_edits_names_each_thing_to_change(tmp_path, monkeypatch):
    settings = Settings()
    ident = {
        "container": {"properties": {"title": "Twisted.Metal.S01E02.2160p.HDR[Ben The Men]"}},
        "tracks": [
            {"type": "video", "properties": {"uid": 11}},
            {"type": "audio", "properties": {"uid": 22, "track_name": "BTM DDP5.1 Atmos"}},
            {"type": "subtitles", "properties": {"uid": 33, "track_name": "BTM"}},
            {"type": "subtitles", "properties": {"uid": 44, "track_name": "English SDH"}},
        ],
        "attachments": [
            {"file_name": "BEN.THE.MEN.TORRENTS.jpg", "content_type": "image/jpeg",
             "properties": {"uid": 55}},
            {"file_name": "cover.jpg", "content_type": "image/jpeg", "properties": {"uid": 66}},
            # a font never goes, whatever it is called
            {"file_name": "BTM-Sans.ttf", "content_type": "font/ttf", "properties": {"uid": 77}},
        ],
    }
    tags = ('<?xml version="1.0"?><Tags><Tag><Targets/>'
            "<Simple><Name>COMMENT</Name><String>ZmWeb</String></Simple>"
            "</Tag></Tags>")

    def fake_run(cmd, **kw):
        if cmd[1] == "-J":
            return type("R", (), {"returncode": 0, "stdout": json.dumps(ident)})()
        Path(cmd[-1]).write_text(tags)
        return type("R", (), {"returncode": 0, "stdout": ""})()

    monkeypatch.setattr(release_ads.subprocess, "run", fake_run)
    monkeypatch.setattr(Settings, "tool_path", lambda self, n: f"/usr/bin/{n}")
    assert release_ads.ad_edits(settings, tmp_path / "o.mkv", tmp_path) == [
        ["--edit", "info", "--delete", "title"],
        ["--edit", "track:=22", "--set", "name=DDP5.1 Atmos"],
        ["--edit", "track:=33", "--delete", "name"],
        ["--delete-attachment", "=55"],
        ["--tags", "all:"],             # nothing left: every tag goes
    ]


def test_nothing_is_edited_on_a_clean_file(tmp_path, monkeypatch):
    settings = Settings()
    ident = {"container": {"properties": {"title": "Shameless (US) - S09E07 - Title"}},
             "tracks": [{"type": "audio", "properties": {"uid": 1, "track_name": "English"}}],
             "attachments": []}

    def fake_run(cmd, **kw):
        if cmd[1] == "-J":
            return type("R", (), {"returncode": 0, "stdout": json.dumps(ident)})()
        return type("R", (), {"returncode": 0, "stdout": ""})()     # no tags at all

    monkeypatch.setattr(release_ads.subprocess, "run", fake_run)
    monkeypatch.setattr(Settings, "tool_path", lambda self, n: f"/usr/bin/{n}")
    assert release_ads.ad_edits(settings, tmp_path / "o.mkv", tmp_path) == []


@pytest.mark.skipif(any(shutil.which(t) is None for t in
                        ("ffmpeg", "ffprobe", "mkvmerge", "mkvextract", "mkvpropedit")),
                    reason="needs a real ffmpeg, ffprobe and mkvtoolnix")
def test_finish_metadata_takes_the_ads_out_of_a_real_file(tmp_path):
    """Built the way a Ben The Men mp4 comes out of the mux, and read back
    with mkvmerge and mkvextract: the signatures are gone, and the statistics,
    the ids, the synopsis and the font are all still there."""
    from app import transcoder

    ivf = tmp_path / "v.ivf"
    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                        "-i", "testsrc=size=160x120:rate=5:duration=1", "-c:v", "libsvtav1",
                        "-preset", "12", str(ivf)], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        pytest.skip("needs an ffmpeg with libsvtav1")
    tone = tmp_path / "tone.flac"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "sine=duration=1", "-c:a", "flac", str(tone)], check=True)
    tags = tmp_path / "tags.xml"
    tags.write_text(
        '<?xml version="1.0"?><Tags>'
        "<Tag><Targets><TargetTypeValue>70</TargetTypeValue></Targets>"
        "<Simple><Name>IMDB</Name><String>tt1586680</String></Simple></Tag>"
        "<Tag><Targets><TargetTypeValue>50</TargetTypeValue></Targets>"
        "<Simple><Name>COMMENT</Name><String>[哔嘀影视-bdys.me]</String></Simple>"
        "<Simple><Name>DESCRIPTION</Name><String>A synopsis.</String></Simple>"
        "</Tag></Tags>")
    logo = tmp_path / "BEN.THE.MEN.TORRENTS.jpg"
    logo.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    font = tmp_path / "font.ttf"
    font.write_bytes(bytes(range(256)))
    out = tmp_path / "out.mkv"
    subprocess.run(["mkvmerge", "-q", "-o", str(out),
                    "--title", "Dark.Matter.S01E01.2160p.ATVP.WEB-DL.DV.HDR10.PLUS[Ben The Men]",
                    str(ivf), "--track-name", "0:BTM DDP5.1 Atmos", str(tone),
                    "--global-tags", str(tags),
                    "--attachment-mime-type", "image/jpeg", "--attach-file", str(logo),
                    "--attachment-mime-type", "font/ttf", "--attach-file", str(font)],
                   check=True)
    assert transcoder.finish_metadata(Settings(), out) is True

    ident = json.loads(subprocess.run(["mkvmerge", "-J", str(out)], check=True,
                                      capture_output=True, text=True).stdout)
    assert ident["container"]["properties"].get("title") is None
    assert [t["properties"].get("track_name") for t in ident["tracks"]] == [None, "DDP5.1 Atmos"]
    assert [a["file_name"] for a in ident["attachments"]] == ["font.ttf"]
    extracted = tmp_path / "back.xml"
    subprocess.run(["mkvextract", str(out), "tags", str(extracted)], check=True,
                   capture_output=True)
    names = [s.findtext("Name") for s in ET.parse(extracted).getroot().iter("Simple")]
    assert "COMMENT" not in names
    assert {"IMDB", "DESCRIPTION", "BPS", "NUMBER_OF_FRAMES"} <= set(names)
