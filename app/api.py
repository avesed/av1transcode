from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app import config, db
from app.config import Settings
from app.queue import TranscodeManager

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(settings: Settings, store: "db.JobStore", manager: "TranscodeManager") -> FastAPI:
    app = FastAPI(title="AV1 Transcode Archive", version="0.1.0")
    router = APIRouter(prefix="/api")

    # ---------- helpers ----------
    def _auth(request: Request) -> None:
        if settings.web.api_key:
            key = request.headers.get("X-API-Key")
            if key != settings.web.api_key:
                raise HTTPException(401, "invalid api key")

    def _job_out(job: dict) -> dict:
        return {
            "id": job["id"],
            "source": job["source"],
            "status": job["status"],
            "progress": job["progress"],
            "stage": job["stage"],
            "progress_fps": job.get("progress_fps"),
            "progress_done": job.get("progress_done"),
            "progress_total": job.get("progress_total"),
            "error": job["error"],
            "meta": db.loads(job.get("meta")),
            "params": db.loads(job.get("params")),
            "output_path": job.get("output_path"),
            "rpu_path": job.get("rpu_path"),
            "size_before": job.get("size_before"),
            "size_after": job.get("size_after"),
            "retries": job.get("retries"),
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
        }

    # ---------- status ----------
    @router.get("/health")
    def health() -> dict:
        return {"status": "ok", "time": time.time()}

    @router.get("/status")
    def status() -> dict:
        counts = store.count_by_status()
        return {
            "jobs": counts,
            "queue_len": counts.get("pending", 0),
            "workers_running": manager.running,
            "dirs": {k: str(v) for k, v in settings.dirs.model_dump().items()},
            "system": {
                "input": str(settings.dirs.input),
                "output": str(settings.dirs.output),
                "rpu": str(settings.dirs.rpu),
            },
        }

    # ---------- jobs ----------
    @router.get("/jobs")
    def list_jobs(status: Optional[str] = None, limit: int = Query(100, le=500)):
        hits = store.list(status=status, limit=limit)
        return [_job_out(j) for j in hits]

    @router.get("/jobs/{jid}")
    def get_job(jid: str):
        j = store.get(jid)
        if not j:
            raise HTTPException(404, "job not found")
        return _job_out(j)

    @router.get("/jobs/{jid}/log")
    def get_log(jid: str):
        log_p = settings.dirs.logs / f"job_{jid}.log"
        if not log_p.exists():
            raise HTTPException(404, "no log")
        return FileResponse(log_p, media_type="text/plain")

    @router.post("/jobs")
    def create_job(request: Request, body: dict):
        _auth(request)
        src = body.get("path")
        preset = body.get("preset", "")
        overrides = body.get("overrides") or {}
        if not src:
            raise HTTPException(400, "path required")
        if not isinstance(overrides, dict):
            raise HTTPException(400, "overrides must be an object")
        jid = manager.enqueue_file(src, preset=preset, overrides=overrides)
        if not jid:
            raise HTTPException(400, "could not enqueue (invalid path or already active)")
        return {"job_id": jid}

    # Cancelling is a mutation like any other: it aborts work in flight and
    # unlinks the output file of a job that had already finished encoding. It
    # was the one write path that never asked for the key.
    @router.post("/jobs/{jid}/cancel")
    def cancel_job(jid: str, request: Request):
        _auth(request)
        if not manager.cancel(jid):
            job = store.get(jid)
            if not job:
                raise HTTPException(404, "job not found")
            # finished jobs are left as they are; say so rather than "not found"
            raise HTTPException(409, f"job already {job['status']}")
        return {"cancelled": jid}

    @router.post("/cancel")
    def cancel_all(request: Request):
        _auth(request)
        return {"cancelled": manager.cancel()}

    @router.post("/jobs/prune")
    def prune_jobs(request: Request, body: Optional[dict] = None):
        _auth(request)
        statuses = (body or {}).get("statuses")
        n = store.prune(statuses)
        from app.logger import prune_job_logs

        keep = {j["id"] for j in store.list(status=None, limit=100000)}
        removed_logs = prune_job_logs(settings, keep_ids=keep)
        return {"pruned": n, "logs_removed": removed_logs}

    # ---------- presets ----------
    @router.get("/presets")
    def presets():
        builtin = settings.transcode.builtin_presets
        return {
            name: {**v.model_dump(), "_builtin": name in builtin}
            for name, v in settings.transcode.presets.items()
        }

    @router.put("/presets/{name}")
    def put_preset(name: str, request: Request, body: dict):
        _auth(request)
        from app.config import VideoParams

        try:
            params = VideoParams.model_validate(body)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"invalid preset params: {e}")
        user = config.load_user_presets(settings)
        user[name] = params
        config.save_user_presets(settings, user)
        settings.transcode.presets[name] = params
        # transcode.video is derived from the default preset, so editing THAT
        # preset has to re-derive it or preset="custom" keeps starting from the
        # values this call just replaced.
        config.apply_default_preset(settings)
        logger.info("Saved user preset '{}'", name)
        return {"name": name, "params": params.model_dump(mode="json"),
                "_builtin": name in settings.transcode.builtin_presets}

    @router.delete("/presets/{name}")
    def delete_preset(name: str, request: Request):
        _auth(request)
        user = config.load_user_presets(settings)
        removed = user.pop(name, None)
        config.save_user_presets(settings, user)
        if name in settings.transcode.presets:
            del settings.transcode.presets[name]
        # restore builtin definition if one exists
        if name in settings.transcode.builtin_presets:
            settings.transcode.presets[name] = settings.transcode.builtin_presets[name]
        config.apply_default_preset(settings)
        logger.info("Deleted preset '{}' (had user override: {})", name, removed is not None)
        return {"deleted": name}

    # ---------- parallelism settings ----------
    @router.get("/workers")
    def get_workers():
        return {
            "concurrency": settings.workers.concurrency,
            "av1an_workers": settings.workers.av1an_workers,
            "active_worker_count": manager.worker_count(),
        }

    @router.put("/settings/workers")
    def put_workers(request: Request, body: dict):
        _auth(request)
        concurrency = int(body.get("concurrency", settings.workers.concurrency))
        av1an_workers = int(body.get("av1an_workers", settings.workers.av1an_workers))
        if concurrency < 1:
            raise HTTPException(422, "concurrency must be >= 1")
        if av1an_workers < 0:
            raise HTTPException(422, "av1an_workers must be >= 0")
        # persist (merge keeps delete_source intact)
        config.save_user_settings(settings, {"workers": {
            "concurrency": concurrency, "av1an_workers": av1an_workers,
        }})
        # apply live
        settings.workers.concurrency = concurrency
        settings.workers.av1an_workers = av1an_workers
        manager.set_concurrency(concurrency)
        logger.info("Updated workers: {}", {"concurrency": concurrency,
                                            "av1an_workers": av1an_workers})
        return {"ok": True, "workers": {
            "concurrency": concurrency, "av1an_workers": av1an_workers}}

    # ---------- optimizer engine settings ----------
    @router.get("/settings/optimizer")
    def get_optimizer():
        return settings.transcode.optimizer.model_dump()

    @router.get("/settings/optimizer/defaults")
    def get_optimizer_defaults():
        """config.yaml's values, and which keys the user has moved off them."""
        return {"defaults": settings.transcode.optimizer_defaults.model_dump(mode="json"),
                "overrides": config.optimizer_overrides(settings)}

    @router.delete("/settings/optimizer")
    def delete_optimizer(request: Request):
        """Back to config.yaml: drop every override and forget the saved block."""
        _auth(request)
        settings.transcode.optimizer = settings.transcode.optimizer_defaults.model_copy(deep=True)
        config.delete_user_setting(settings, "optimizer")
        logger.info("Optimizer settings restored to config.yaml defaults")
        return {"ok": True, "optimizer": settings.transcode.optimizer.model_dump(mode="json")}

    @router.put("/settings/optimizer")
    def put_optimizer(request: Request, body: dict):
        _auth(request)
        from app.config import OptimizerSettings

        # Merge onto the current settings rather than validating the body on
        # its own: model_validate fills anything absent with its DEFAULT, so a
        # form that posts a subset of the fields silently reset the rest. The
        # settings page has always posted a subset, and every field added since
        # was being wiped on each save.
        merged = {**settings.transcode.optimizer.model_dump(), **(body or {})}
        try:
            params = OptimizerSettings.model_validate(merged)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"invalid optimizer settings: {e}")
        settings.transcode.optimizer = params
        # Only the keys that differ from config.yaml are written (JSON-shaped,
        # so the Path fields encode). The block used to be the full dump, and
        # a full dump pins every default at the moment of saving.
        over = config.save_optimizer_settings(settings)
        logger.info("Updated optimizer settings; overrides now {}", over)
        return {"ok": True, "optimizer": params.model_dump(), "overrides": over}

    # ---------- GPU ----------
    _GPU_KEYS = ("vmaf_sycl_device", "vmaf_sycl_min_width", "scenedetect_hwaccel",
                 "reference_hwaccel", "vmaf_zero_copy", "vmaf_sycl_workers",
                 "gpu_vram_budget_mb")

    def _gpu_settings() -> dict:
        o = settings.transcode.optimizer
        return {"vmaf_sycl_device": o.vmaf_sycl_device,
                "vmaf_sycl_min_width": o.vmaf_sycl_min_width,
                "scenedetect_hwaccel": o.scenedetect_hwaccel,
                "reference_hwaccel": o.reference_hwaccel,
                "vmaf_zero_copy": o.vmaf_zero_copy,
                "vmaf_sycl_workers": o.vmaf_sycl_workers,
                "gpu_vram_budget_mb": o.gpu_vram_budget_mb,
                "vulkan_device": settings.transcode.dovi.vulkan_device}

    @router.get("/gpu")
    def get_gpu(refresh: bool = False):
        """What the container sees, and what the current settings make of it."""
        from app import gpu

        status = gpu.probe(settings, force=refresh)
        return {"status": status, "settings": _gpu_settings()}

    @router.post("/gpu/selfcheck")
    def gpu_selfcheck(request: Request, body: Optional[dict] = None):
        """The optimizer's own preflight: one pair on the GPU, one on the CPU."""
        _auth(request)
        from app import gpu

        dev = int((body or {}).get("device", settings.transcode.optimizer.vmaf_sycl_device))
        if dev < 0:
            dev = 0
        try:
            return gpu.selfcheck(settings, dev)
        except FileNotFoundError as e:
            raise HTTPException(500, str(e))

    @router.put("/settings/gpu")
    def put_gpu(request: Request, body: dict):
        _auth(request)
        from app.config import OptimizerSettings

        body = body or {}
        merged = {**settings.transcode.optimizer.model_dump(),
                  **{k: body[k] for k in _GPU_KEYS if k in body}}
        try:
            params = OptimizerSettings.model_validate(merged)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"invalid GPU settings: {e}")
        settings.transcode.optimizer = params
        over = config.save_optimizer_settings(settings)
        if "vulkan_device" in body:
            vk = str(body.get("vulkan_device") or "").strip()
            settings.transcode.dovi.vulkan_device = vk
            config.save_user_settings(settings, {"dovi": {"vulkan_device": vk}})
        logger.info("Updated GPU settings: {}", _gpu_settings())
        return {"ok": True, "settings": _gpu_settings(), "overrides": over}

    # ---------- safety settings (delete_source) ----------
    @router.get("/settings/safety")
    def get_safety():
        return {"delete_source": settings.transcode.delete_source}
    @router.put("/settings/delete_source")
    def put_delete_source(request: Request, body: dict):
        """Enable/disable source deletion after success.

        Enabling is destructive (source files get deleted), so the caller
        must explicitly confirm with confirm=true.
        """
        _auth(request)
        enabled = bool(body.get("enabled"))
        if enabled and not body.get("confirm"):
            raise HTTPException(422, "confirm=true required to enable delete_source")
        settings.transcode.delete_source = enabled
        config.save_user_settings(settings, {"delete_source": enabled})
        if enabled:
            logger.warning("delete_source ENABLED (destructive: source files will be deleted after success)")
        else:
            logger.info("delete_source disabled")
        return {"ok": True, "delete_source": enabled}

    # ---------- directory browser ----------
    def _browse(p: Path) -> list[dict]:
        items = []
        for child in p.iterdir():
            try:
                if child.name.startswith("."):
                    continue
                items.append({
                    "name": child.name,
                    "dir": child.is_dir(),
                    "size": child.stat().st_size if child.is_file() else None,
                })
            except OSError:
                continue
        items.sort(key=lambda e: (not e["dir"], e["name"].lower()))
        return items

    # Keyed even though it is a GET. The other reads here return this app's own
    # state; this one walks the HOST filesystem from any absolute path, so with
    # the container's mounts it enumerates the media library and everything
    # else the process can see. That is not the same kind of read.
    @router.get("/browse")
    def browse(request: Request, path: str = "/"):
        _auth(request)
        p = Path(path)
        if not p.is_absolute():
            raise HTTPException(400, "path must be absolute")
        try:
            if p.exists() and p.is_file():
                # file: return its parent so the UI can navigate around it
                p = p.parent
            if not p.is_dir():
                raise HTTPException(404, "no such directory")
            entries = _browse(p)
        except OSError as e:
            raise HTTPException(403, f"cannot read: {e}")
        return {"path": str(p), "parent": str(p.parent) if p != p.parent else None,
                "entries": entries}

    # ---------- static / index ----------
    @router.get("/", include_in_schema=False)
    def index():
        html = STATIC_DIR / "index.html"
        return HTMLResponse(html.read_text() if html.exists() else "")

    # map / to router root GET / as well
    @app.get("/", include_in_schema=False)
    def _root():
        html = STATIC_DIR / "index.html"
        return HTMLResponse(html.read_text() if html.exists() else "")

    app.include_router(router)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app