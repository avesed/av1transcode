"""Grain-auto service for av1transcode: analysis, denoise and AV1 grain synthesis on the GPU, as jobs over HTTP.

av1transcode (profile switch grain_auto) asks it, in order, for:
  analyse  {"source"}                         -> {"class": grain|texture|clean, "on": bool, "why", "level", ...}
  denoise  {"source", "out", "model"?}        -> out: the denoised video alone (B580 AV1 at QP 0, constant frame rate;
                                                 the caller puts the source's timestamps back)
  grain    {"source", "video", "shots", "fps", "out", "work", "strength"?}
                                              -> out: VIDEO (an AV1 ivf) with a measured film grain table written in
API (JSON):
  GET  /health                       {"ok": true, "device", "busy": job id or null, "models": [...]}
  POST /jobs {"kind": ..., ...}      {"id"}
  GET  /jobs/<id>                    {"id", "kind", "state": queued|running|done|failed|cancelled, "progress": 0..1,
                                      "result", "error", "log": last lines}
  POST /jobs/<id>/cancel
One job runs at a time (one GPU); each runs its work in a subprocess, which a cancel terminates. Paths must lie under
GRAIN_ROOTS (default /media) and are the caller's own: this container mounts the same volumes under the same names.
  python3 server.py   (grain service image; GRAIN_PORT 8790, GRAIN_WEIGHTS /weights, RENDER_NODE /dev/dri/renderD128)
"""
import collections, json, os, re, shutil, signal, subprocess, sys, tempfile, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("GRAIN_PORT", "8790"))
WEIGHTS = os.environ.get("GRAIN_WEIGHTS", "/weights")
ROOTS = [os.path.realpath(r) for r in os.environ.get("GRAIN_ROOTS", "/media").split(":") if r]
MODELS = {"v3g": "v3g.pt", "v3s2": "v3s2.pt"}         # v3g: the blind-tested one (test-08/09); v3s2: half width, ~1.6x faster
DEFAULT_MODEL = os.environ.get("GRAIN_MODEL", "v3g")
DEV = os.environ.get("DEV", "xpu")
JOBS = {}
QUEUE = collections.deque()
LOCK = threading.Condition()


def allowed(path, must_exist=False):
    real = os.path.realpath(path)
    if not os.path.isabs(path) or not any(real == r or real.startswith(r + os.sep) for r in ROOTS):
        raise ValueError(f"{path} is not under {ROOTS}")
    if must_exist and not os.path.exists(real):
        raise ValueError(f"{path} does not exist")
    return real


class Job:
    def __init__(self, kind, params):
        self.id, self.kind, self.params = uuid.uuid4().hex[:12], kind, params
        self.state, self.progress, self.result, self.error = "queued", 0.0, None, None
        self.log = collections.deque(maxlen=40)
        self.proc, self.cancelled, self.t0 = None, False, time.time()

    def view(self):
        return {"id": self.id, "kind": self.kind, "state": self.state, "progress": round(self.progress, 4),
                "result": self.result, "error": self.error, "log": list(self.log), "seconds": round(time.time() - self.t0, 1)}


def run_proc(job, cmd, env=None, on_line=None):
    """run one step of a job; every output line goes to the job log (and on_line); a cancel terminates it."""
    job.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                start_new_session=True)
    for line in job.proc.stdout:
        line = line.rstrip()
        if not line or re.search(r"Warning|warn\(|USDT|non-writable|from_numpy|nbits|stepping|^[EW]\d{4} ", line):
            continue
        if not line.startswith("PROGRESS "):
            job.log.append(line[-300:])
        if on_line:
            on_line(line)
    rc = job.proc.wait()
    job.proc = None
    if job.cancelled:
        raise RuntimeError("cancelled")
    if rc:
        raise RuntimeError(f"{os.path.basename(cmd[1])} exited {rc}: " + " | ".join(list(job.log)[-4:]))


def py_env(**extra):
    env = dict(os.environ, GRAIN_WEIGHTS=WEIGHTS, DEV=DEV,
               PYTHONPATH=":".join(x for x in (HERE, os.environ.get("PYTHONPATH", "")) if x), **extra)
    return env


def probe_frames(path):
    """frame count for progress: the stream's own count when the container has one, else duration x rate."""
    s = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=nb_frames,r_frame_rate,width:format=duration", "-of", "json", path],
                                  capture_output=True, text=True).stdout)
    v = s["streams"][0]
    if str(v.get("nb_frames", "")).isdigit():
        return int(v["nb_frames"]), v["width"]
    num, _, den = v["r_frame_rate"].partition("/")
    return int(float(s["format"]["duration"]) * float(num) / float(den or 1)), v["width"]


def do_analyse(job):
    src = allowed(job.params["source"], True)
    with tempfile.NamedTemporaryFile(suffix=".json") as tf:
        run_proc(job, [sys.executable, f"{HERE}/analyse.py", src, "--json", tf.name], py_env())
        return json.load(open(tf.name))


def do_denoise(job):
    src = allowed(job.params["source"], True)
    out = allowed(job.params["out"])
    model = job.params.get("model") or DEFAULT_MODEL
    if model not in MODELS:
        raise ValueError(f"model {model}: not one of {list(MODELS)}")
    total, width = probe_frames(src)
    tmp = tempfile.mkdtemp(prefix=".grain_dn_", dir=os.path.dirname(out))
    env = py_env(COMPILE="1", COMPILE_SP="1", FUSED="1", FLOWMW="960", FASTASM="1", TILES="2" if width > 1920 else "1",
                 LIMIT="1.3", LIMWIN="25", LOSSLESS="av1hw", PROGRESS_EVERY="50",
                 SPYNET=f"{WEIGHTS}/spynet.pt", RENDER_NODE=os.environ.get("RENDER_NODE", "/dev/dri/renderD128"))

    def on_line(line):
        if line.startswith("PROGRESS "):
            job.progress = min(0.999, int(line.split()[1]) / max(total, 1))
    try:
        run_proc(job, [sys.executable, f"{HERE}/dn_v3.py", src, tmp, f"{WEIGHTS}/{MODELS[model]}", "1"], env, on_line)
        os.replace(f"{tmp}/v3_1.mkv", out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"out": out, "model": model, "frames": total}


def do_grain(job):
    p = job.params
    src, video = allowed(p["source"], True), allowed(p["video"], True)
    out, work = allowed(p["out"]), allowed(p["work"])
    os.makedirs(work, exist_ok=True)
    shots = [int(n) for n in p["shots"]]
    sj = f"{work}/shots.json"
    json.dump({"shots": [{"frames": n} for n in shots]}, open(sj, "w"))
    marks = {"target:": 0.05, "applied:": 0.55, "done": 1.0}
    result = {}

    def on_line(line):
        for k, v in marks.items():
            if k in line:
                job.progress = max(job.progress, v)
        if line.startswith("RESULT "):
            result.update(json.loads(line[7:]))
    run_proc(job, [sys.executable, f"{HERE}/grain_apply.py", src, video, sj, p["fps"], out, work, str(p.get("chroma", 1.0)),
                   str(float(p.get("strength", 1.0)))],
             py_env(RENDER_NODE=os.environ.get("RENDER_NODE", "/dev/dri/renderD128")), on_line)
    return result


KINDS = {"analyse": do_analyse, "denoise": do_denoise, "grain": do_grain}


def worker():
    while True:
        with LOCK:
            while not QUEUE:
                LOCK.wait()
            job = QUEUE.popleft()
        if job.cancelled:
            continue
        job.state = "running"
        try:
            job.result = KINDS[job.kind](job)
            job.progress, job.state = 1.0, "done"
        except Exception as e:                           # noqa: BLE001 - reported to the caller
            job.state = "cancelled" if job.cancelled else "failed"
            job.error = f"{type(e).__name__}: {e}"


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, fmt, *args):                    # quiet: the caller logs what matters
        pass

    def do_GET(self):
        if self.path == "/health":
            busy = next((j.id for j in JOBS.values() if j.state == "running"), None)
            return self._send(200, {"ok": True, "device": DEV, "busy": busy, "models": list(MODELS),
                                    "default_model": DEFAULT_MODEL})
        m = re.fullmatch(r"/jobs/([0-9a-f]+)", self.path)
        if m and m.group(1) in JOBS:
            return self._send(200, JOBS[m.group(1)].view())
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._send(400, {"error": "bad JSON"})
        if self.path == "/jobs":
            kind = body.get("kind")
            if kind not in KINDS:
                return self._send(400, {"error": f"kind must be one of {list(KINDS)}"})
            try:                                          # paths checked now, so a bad request fails at once
                for k in ("source", "video"):
                    if k in body:
                        allowed(body[k], True)
                for k in ("out", "work"):
                    if k in body:
                        allowed(body[k])
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            job = Job(kind, body)
            JOBS[job.id] = job
            with LOCK:
                QUEUE.append(job)
                LOCK.notify()
            for old in [j for j in JOBS.values() if j.state in ("done", "failed", "cancelled") and time.time() - j.t0 > 86400]:
                JOBS.pop(old.id, None)
            return self._send(200, {"id": job.id})
        m = re.fullmatch(r"/jobs/([0-9a-f]+)/cancel", self.path)
        if m and m.group(1) in JOBS:
            job = JOBS[m.group(1)]
            job.cancelled = True
            if job.proc is not None:
                try:
                    os.killpg(job.proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if job.state == "queued":
                job.state = "cancelled"
            return self._send(200, job.view())
        self._send(404, {"error": "not found"})


if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    print(f"grain service on :{PORT}, device {DEV}, roots {ROOTS}, default model {DEFAULT_MODEL}", flush=True)
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()
