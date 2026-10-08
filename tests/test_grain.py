"""grain_auto: the grain service client (app/grain.py) and its place in run_full_transcode.

A fake grain service stands in for the GPU one: its "denoise" re-encodes the source's video at a constant frame rate
(what the real one writes, holes and all lost) and its "grain" hands the video back unchanged, so these tests check
what this repo does with the service's files - timestamps, tracks, frame counts, failure handling - on real media.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from app import grain
from app.config import load_settings

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("mkvmerge")),
                                reason="needs ffmpeg and mkvmerge")


@pytest.fixture()
def settings(tmp_path):
    s = load_settings()
    s.dirs.work = tmp_path / "work"
    s.dirs.work.mkdir()
    return s


def run(*cmd):
    subprocess.run([str(c) for c in cmd], check=True, capture_output=True)


def timestamps(path: Path) -> list:
    out = Path(tempfile.mkdtemp()) / "ts.txt"            # not beside path: that may be the work dir under test
    tid = next(t["id"] for t in json.loads(subprocess.run(["mkvmerge", "-J", str(path)], capture_output=True,
                                                           text=True).stdout)["tracks"] if t["type"] == "video")
    run("mkvextract", path, "timestamps_v2", f"{tid}:{out}")
    return [float(x) for x in out.read_text().splitlines() if x and not x.startswith("#")]


def rate(path: Path) -> str:
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                           "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()


def tracks(path: Path) -> list:
    return [t["type"] for t in json.loads(subprocess.run(["mkvmerge", "-J", str(path)], capture_output=True,
                                                         text=True).stdout)["tracks"]]


@pytest.fixture()
def holey_source(tmp_path):
    """24 fps video with a 3-frame hole after frame 23, plus an audio track: what the timeline-slot engine keeps."""
    raw = tmp_path / "raw.mkv"
    run("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=24:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2.2", "-frames:v", "48", "-c:v", "ffv1", "-c:a", "flac", raw)
    ts = tmp_path / "hole.txt"
    stamps = [i * 1000 / 24 for i in range(24)] + [(i + 3) * 1000 / 24 for i in range(24, 48)]
    ts.write_text("# timestamp format v2\n" + "".join(f"{t:.3f}\n" for t in stamps))
    src = tmp_path / "src.mkv"
    run("mkvmerge", "-q", "-o", src, "--timestamps", f"0:{ts}", raw)
    run("mkvpropedit", "-q", src, "--edit", "track:v1", "--set", "default-duration=41666667")   # as a remux has it
    return src


class FakeService:
    """The grain service's HTTP API with scripted results. denoise: CFR re-encode of the source's video
    (drop_frame=True loses one); grain: the video handed back; fail: kinds that fail."""
    def __init__(self, on=True, drop_frame=False, fail=()):
        self.on, self.drop_frame, self.fail, self.calls = on, drop_frame, set(fail), []
        svc = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj, code=200):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                if self.path == "/jobs":
                    svc.calls.append(body)
                    return self._send({"id": f"{len(svc.calls):04x}"})
                self._send({})

            def do_GET(self):
                job = svc.calls[int(self.path.rsplit("/", 1)[1], 16) - 1]
                self._send(svc.result(job))
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def result(self, job):
        kind = job["kind"]
        if kind in self.fail:
            return {"state": "failed", "error": f"{kind} broke", "log": ["boom"]}
        if kind == "analyse":
            return {"state": "done", "progress": 1, "result": {"class": "grain" if self.on else "clean", "on": self.on,
                                                               "why": "scripted", "level": 30, "features": {}}}
        if kind == "denoise" and not Path(job["out"]).exists():
            n = "-frames:v 47" if self.drop_frame else ""
            subprocess.run(f"ffmpeg -v error -i {job['source']} -map 0:v:0 -fps_mode passthrough -f rawvideo "
                           f"-pix_fmt yuv420p - | "
                           f"ffmpeg -v error -y -f rawvideo -pix_fmt yuv420p -s 160x120 -r 24 -i - {n} -c:v ffv1 "
                           f"{job['out']}", shell=True, check=True)
            return {"state": "running", "progress": 0.5}           # one poll in flight before it is done
        if kind == "denoise":
            return {"state": "done", "progress": 1, "result": {"out": job["out"], "model": job.get("model")}}
        if kind == "grain":
            shutil.copy(job["video"], job["out"])
            return {"state": "done", "progress": 1, "result": {"target": {"y": [1.0] * 11},
                                                               "flicker": {"source": {"fine_swing": 0.03},
                                                                           "auto": {"fine_swing": 0.033}}}}
        return {"state": "failed", "error": "unknown kind"}

    def close(self):
        self.httpd.shutdown()


@pytest.fixture()
def service(settings):
    svcs = []

    def make(**kw):
        s = FakeService(**kw)
        settings.transcode.grain.url = s.url
        svcs.append(s)
        return s
    yield make
    for s in svcs:
        s.close()


def _info(src):
    from app.analyzer import MediaInfo
    return MediaInfo(path=src)


def _video(engine="optimizer"):
    from app.config import VideoParams
    return VideoParams(engine=engine, grain_auto=True)


def test_off_encodes_the_input_as_it_is(settings, service, holey_source):
    service(on=False)
    tmp = []
    out, ctx = grain.prepare(settings, _info(holey_source), _video(), False, holey_source, settings.dirs.work, tmp)
    assert out == holey_source and ctx is None
    assert list(settings.dirs.work.iterdir()) == []


def test_an_unreachable_service_degrades_to_the_plain_encode(settings, holey_source):
    settings.transcode.grain.url = "http://127.0.0.1:9"
    out, ctx = grain.prepare(settings, _info(holey_source), _video(), False, holey_source, settings.dirs.work, [])
    assert out == holey_source and ctx is None


def test_av1an_and_p5_never_ask_the_service(settings, service, holey_source):
    s = service()
    for video, p5 in ((_video("av1an"), False), (_video(), True)):
        out, ctx = grain.prepare(settings, _info(holey_source), video, p5, holey_source, settings.dirs.work, [])
        assert out == holey_source and ctx is None
    assert s.calls == []


def test_on_gives_the_denoised_video_the_sources_timestamps(settings, service, holey_source):
    """The service writes constant-rate frames; the base must carry the source's own timestamps, hole included,
    or the engine's timeline slots and the mux's sync go wrong after it."""
    s = service()
    tmp, stages, prog = [], [], []
    out, ctx = grain.prepare(settings, _info(holey_source), _video(), False, holey_source, settings.dirs.work, tmp,
                             stages.append, lambda p, _st=None: prog.append(p))
    assert ctx is not None and out != holey_source and out.exists()
    assert stages == ["grain_analyse", "denoising"]
    assert 50.0 in prog                                      # the service's progress reached the job
    assert [k["kind"] for k in s.calls] == ["analyse", "denoise"]
    assert timestamps(out) == pytest.approx(timestamps(holey_source), abs=1.0)
    assert rate(holey_source) == "24/1" and rate(out) == "24/1"   # not mkvmerge's 1000/42 from the timestamps file
    assert tracks(out) == ["video"]                          # audio and the rest still come from the source (info.path)
    assert out in tmp                                        # removed with the job's other intermediates
    assert sorted(p.name for p in settings.dirs.work.iterdir()) == [out.name]


def test_a_frame_lost_in_denoising_degrades_and_leaves_nothing(settings, service, holey_source):
    service(drop_frame=True)
    out, ctx = grain.prepare(settings, _info(holey_source), _video(), False, holey_source, settings.dirs.work, [])
    assert out == holey_source and ctx is None
    assert list(settings.dirs.work.iterdir()) == []


@pytest.fixture()
def av1_output(tmp_path, holey_source):
    """An engine output: AV1 video on the source's timestamps, its audio, and a subtitle track."""
    v = tmp_path / "v.mkv"
    run("ffmpeg", "-v", "error", "-y", "-i", holey_source, "-map", "0:v:0", "-c:v", "libsvtav1", "-preset", "12",
        "-fps_mode", "passthrough", v)
    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:00,500 --> 00:00:01,000\nhello\n")
    out = tmp_path / "out.av1.mkv"
    run("mkvmerge", "-q", "-o", out, v, "-D", holey_source, "--language", "0:eng", srt)
    return out


def test_finish_puts_the_grain_stream_back_on_the_outputs_timestamps(settings, service, holey_source, av1_output):
    s = service()
    before, before_rate = timestamps(av1_output), rate(av1_output)
    ctx = grain.GrainContext(grain.GrainService(s.url, 60), holey_source, {})
    res = grain.finish(settings, ctx, av1_output, [24, 24], settings.dirs.work)
    assert res["flicker"]["auto"]["fine_swing"] == 0.033
    job = s.calls[-1]
    assert job["kind"] == "grain" and job["shots"] == [24, 24] and job["source"] == str(holey_source)
    assert job["strength"] == 0.7                                # the owner's 30% less grain (test-10)
    assert tracks(av1_output) == ["video", "audio", "subtitles"]
    assert timestamps(av1_output) == pytest.approx(before, abs=1.0)
    assert rate(av1_output) == before_rate == "24/1"
    assert list(settings.dirs.work.iterdir()) == []


def test_finish_refuses_shots_that_do_not_cover_the_output(settings, service, holey_source, av1_output):
    s = service()
    ctx = grain.GrainContext(grain.GrainService(s.url, 60), holey_source, {})
    with pytest.raises(grain.GrainError, match="frames"):
        grain.finish(settings, ctx, av1_output, [24, 20], settings.dirs.work)
    assert tracks(av1_output) == ["video", "audio", "subtitles"]


def test_a_failed_grain_step_fails_the_job_and_drops_the_output(settings, monkeypatch, tmp_path, holey_source):
    """Denoised and left without its grain, the output is not one to keep."""
    from app import optimizer, transcoder
    from app.analyzer import MediaInfo
    from app.decisions import TranscodePlan
    from app.optimizer import MuxReport
    from app.transcoder import TranscodeError

    plan = TranscodePlan()
    plan.params = _video()
    out = tmp_path / "out.av1.mkv"
    base = settings.dirs.work / "base.mkv"

    def prepare(*a, **_kw):
        base.write_bytes(b"denoised")
        a[6].append(base)
        return base, grain.GrainContext(None, holey_source, {})

    def encode(_s, _i, _p, src, output, _tempdir, **_kw):
        assert Path(src) == base                         # the engine read the denoised intermediate
        Path(output).write_bytes(b"encoded")
        return MuxReport(shots=(48,))

    def finish(*_a, **_kw):
        raise grain.GrainError("grav1synth broke")
    monkeypatch.setattr(grain, "prepare", prepare)
    monkeypatch.setattr(grain, "finish", finish)
    monkeypatch.setattr(optimizer, "run_shot_transcode", encode)
    with pytest.raises(TranscodeError, match="grain synthesis failed"):
        transcoder.run_full_transcode(settings, MediaInfo(path=holey_source), plan, holey_source, out)
    assert not out.exists()
    assert not base.exists()


def test_svt_film_grain_is_taken_out_of_every_place_a_profile_can_set_it():
    from app.config import VideoParams
    v = VideoParams(engine="optimizer", grain_auto=True, film_grain=10, film_grain_denoise=True,
                    additional_video_params="--sharpness 1 --film-grain 8 film-grain-denoise=1 --fgs-table /t/a.tbl "
                                            "enable-overlays=1:film-grain=5:tune=0 --film-grain=4 --enable-qm 1")
    out = grain.without_film_grain(v)
    assert out.film_grain == 0 and not out.film_grain_denoise
    assert out.additional_video_params == "--sharpness 1 enable-overlays=1:tune=0 --enable-qm 1"
    assert out.grain_auto and v.film_grain == 10                 # a copy: the profile itself is untouched
    clean = VideoParams(engine="optimizer", grain_auto=True, additional_video_params="--sharpness 1")
    assert grain.without_film_grain(clean) is clean


def test_grain_auto_jobs_encode_without_svt_film_grain(settings, monkeypatch, tmp_path, holey_source):
    """Both engines read plan.params: with grain_auto on, it reaches them with SVT's film grain off, also when the
    service says off for the source."""
    from app import optimizer, transcoder
    from app.analyzer import MediaInfo
    from app.decisions import TranscodePlan
    from app.optimizer import MuxReport

    plan = TranscodePlan()
    plan.params = _video().model_copy(update={"film_grain": 8, "additional_video_params": "--film-grain 8 --tune 0"})
    seen = {}
    monkeypatch.setattr(grain, "prepare", lambda *a, **_k: (a[4], None))

    def encode(_s, _i, p, _src, output, _tempdir, **_kw):
        seen["params"] = p.params
        Path(output).write_bytes(b"encoded")
        return MuxReport(shots=(48,))
    monkeypatch.setattr(optimizer, "run_shot_transcode", encode)
    try:
        transcoder.run_full_transcode(settings, MediaInfo(path=holey_source), plan, holey_source, tmp_path / "o.mkv")
    except Exception:                    # the stand-in output is not a real file; only what the engine got matters
        pass
    assert seen["params"].film_grain == 0
    assert seen["params"].additional_video_params == "--tune 0"


@pytest.fixture()
def hdr10_source(tmp_path):
    """An HDR10 HEVC source: its mastering display and MaxCLL in SEI, as WEB-DL and remux sources carry them."""
    src = tmp_path / "hdr.mkv"
    run("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=24", "-frames:v", "24",
        "-pix_fmt", "yuv420p10le", "-c:v", "libx265", "-x265-params",
        "log-level=error:hdr10=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:max-cll=1000,400:"
        "master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)", src)
    return src


def test_the_denoised_base_carries_the_sources_hdr10_metadata(settings, service, hdr10_source):
    """The base is encoded from raw frames, so it has no HDR10 metadata of its own; the engine's encode takes it from
    its input, and without it the output's AV1 stream had none (only its container did)."""
    if not shutil.which("mkvpropedit"):
        pytest.skip("needs mkvpropedit")
    service()
    out, ctx = grain.prepare(settings, _info(hdr10_source), _video(), False, hdr10_source, settings.dirs.work, [])
    assert ctx is not None
    p = next(t for t in json.loads(subprocess.run(["mkvmerge", "-J", str(out)], capture_output=True,
                                                  text=True).stdout)["tracks"] if t["type"] == "video")["properties"]
    assert p["max_content_light"] == 1000 and p["max_frame_light"] == 400
    assert p["max_luminance"] == pytest.approx(1000) and p["min_luminance"] == pytest.approx(0.0001)
    assert p["chromaticity_coordinates"].startswith("0.68")              # red x first, as mkvmerge lists them
    sd = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream_side_data=side_data_type", "-of", "csv=p=0", str(out)], capture_output=True,
                        text=True).stdout
    assert "Mastering display metadata" in sd and "Content light level metadata" in sd
