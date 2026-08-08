from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app import config, db, decisions
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

    @router.post("/jobs/{jid}/cancel")
    def cancel_job(jid: str):
        ok = manager.cancel(jid)
        if not ok:
            raise HTTPException(404, "job not found")
        return {"cancelled": jid}

    @router.post("/cancel")
    def cancel_all():
        return {"cancelled": manager.cancel()}

    @router.post("/jobs/prune")
    def prune_jobs(request: Request, body: Optional[dict] = None):
        _auth(request)
        statuses = (body or {}).get("statuses")
        n = store.prune(statuses)
        return {"pruned": n}

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
        logger.info("Saved user preset '{}'", name)
        return {"name": name, "params": params.model_dump(),
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

    @router.get("/browse")
    def browse(path: str = "/"):
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