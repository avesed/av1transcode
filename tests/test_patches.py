"""Static checks of the vendored FFmpeg patches and the Dockerfile that applies
them. A patch that is vendored but never applied, or a re-cut that loses a
hunk, otherwise only shows up after a full image build - or at run time."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PATCHES = ROOT / "patches"
DOCKERFILE = (ROOT / "Dockerfile").read_text()
KEEP_GRAPH = "ffmpeg-n9.0.1-keep-graph-on-equivalent-hwframes.patch"
HEVC_KEEP = "ffmpeg-n9.0.1-hevc-keep-hwaccel-on-equivalent-sps.patch"
# the order they must be applied in: zerocopy is cut on top of libvmaf-sycl,
# and each later one was verified at --fuzz=0 on top of the ones before it
APPLY_ORDER = [
    "ffmpeg-n9.0.1-libvmaf-sycl.patch",
    "ffmpeg-n9.0.1-libvmaf-sycl-zerocopy.patch",
    "ffmpeg-n9.0.1-keep-decoder-until-cleanup.patch",
    KEEP_GRAPH,
    HEVC_KEEP,
]
# built file the Dockerfile greps -> (source the hunk is in, its marker), per patch
BUILD_MARKERS = {
    KEEP_GRAPH: {
        "/usr/local/bin/ffmpeg": ("fftools/ffmpeg_filter.c", "Keeping filter graph"),
        "/usr/local/lib/libavfilter.a": ("libavfilter/vf_hwdownload.c",
                                         "Accepting frames from an equivalent hwframe context"),
    },
    HEVC_KEEP: {
        "/usr/local/lib/libavcodec.a": ("libavcodec/hevc/hevcdec.c",
                                        "Keeping hwaccel %s for a new SPS"),
    },
}


def _hunks(text):
    """(file, body line) for every hunk line, checking each hunk's header counts
    against its body the way patch(1) reads them."""
    lines, path, i = text.splitlines(), None, 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.startswith("+++ "):
            path = line.split()[1].removeprefix("b/")
            continue
        m = re.match(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@", line)
        if not m:
            continue
        old, new = int(m.group(1) or 1), int(m.group(2) or 1)
        while old or new:
            assert i < len(lines), f"{path}: hunk {line} ends early"
            body = lines[i]
            i += 1
            if body.startswith("\\"):  # "\ No newline at end of file"
                continue
            tag = body[:1] or " "
            assert tag in " +-", f"{path}: hunk {line} has a stray line {body!r}"
            old -= tag in " -"
            new -= tag in " +"
            assert old >= 0 and new >= 0, f"{path}: hunk {line} counts too few lines"
            yield path, body


def test_dockerfile_applies_every_ffmpeg_patch():
    vendored = sorted(p.name for p in PATCHES.glob("ffmpeg-*.patch"))
    copied = re.findall(r"^COPY patches/(\S+\.patch) /build/$", DOCKERFILE, re.M)
    applied = re.findall(r"^\s*patch -p1 --fuzz=0 < /build/(\S+\.patch) && \\$",
                         DOCKERFILE, re.M)
    assert KEEP_GRAPH in vendored and HEVC_KEEP in vendored
    assert sorted(APPLY_ORDER) == vendored
    assert copied == APPLY_ORDER
    assert applied == APPLY_ORDER  # each one, exactly once, in this order
    readme = (PATCHES / "README.md").read_text()
    for name in vendored:
        assert DOCKERFILE.index(f"COPY patches/{name}") < DOCKERFILE.index(f"< /build/{name}")
        assert f"## `{name}`" in readme


def test_vendored_patches_are_well_formed():
    for p in PATCHES.glob("*.patch"):
        assert list(_hunks(p.read_text())), p.name


@pytest.mark.parametrize("patch", sorted(BUILD_MARKERS))
def test_build_greps_for_what_the_patch_adds(patch):
    hunks = list(_hunks((PATCHES / patch).read_text()))
    applied = DOCKERFILE.index(f"< /build/{patch}")
    for built, (source, marker) in BUILD_MARKERS[patch].items():
        check = f'grep -aq "{marker}" {built} && \\'
        assert check in DOCKERFILE and DOCKERFILE.index(check) > applied
        # added by this hunk, and not already upstream text a stock build carries
        assert any(f == source and b.startswith("+") and marker in b for f, b in hunks)
        assert not any(marker in b for f, b in hunks if not b.startswith("+"))
    # the optimizer's rebuild detector keys on this line: the patch must neither
    # remove it nor reuse its text for what it keeps
    assert not any("Reconfiguring filter graph" in b for _, b in hunks)


def test_hevc_keep_line_is_logged_at_info():
    """Production reads run at -loglevel info, so a kept SPS change leaves a
    trace there only if its line is at INFO; VERBOSE would be filtered out."""
    added = [b for f, b in _hunks((PATCHES / HEVC_KEEP).read_text())
             if f == "libavcodec/hevc/hevcdec.c" and b.startswith("+")]
    logs = [b for b in added if "Keeping hwaccel %s for a new SPS" in b]
    assert len(logs) == 1
    assert "av_log(" in logs[0] and "AV_LOG_INFO" in logs[0]
