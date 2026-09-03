from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger

from app import __version__, analyzer, decisions, db
from app.config import Settings, load_settings

app = typer.Typer(help="AV1 auto-transcoding archive system (Python scheduler + frontend)")


def _build_manager(settings: Settings):
    from app.queue import TranscodeManager

    store = db.JobStore(settings)
    manager = TranscodeManager(settings, store)
    return manager, store


@app.command()
def run(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="config.yaml path"),
    no_watch: bool = typer.Option(False, "--no-watch", help="disable directory watching"),
    no_web: bool = typer.Option(False, "--no-web", help="disable API server"),
) -> None:
    """Run the full scheduler: watcher + workers (+ web server by default)."""
    settings = load_settings(config)
    from app.logger import setup_logging, cleanup_old_job_logs

    setup_logging(settings)
    settings.ensure_dirs()

    from app.queue import TranscodeManager

    store = db.JobStore(settings)
    manager = TranscodeManager(settings, store)

    # Reclaim what a killed process could not: this runs before manager.start(),
    # so every temp tree still on disk belongs to a job that is already over.
    from app.transcoder import sweep_stale_work

    try:
        n = sweep_stale_work(settings)
        if n:
            logger.info("Reclaimed {} stale work file(s)/dir(s) from {}",
                        n, settings.dirs.work)
    except OSError as e:
        logger.warning("stale work sweep failed: {}", e)

    # prune old job logs at startup and periodically
    try:
        n = cleanup_old_job_logs(settings)
        if n:
            logger.info("Cleaned up {} old job log(s)", n)
    except Exception:  # noqa: BLE001
        logger.debug("job log cleanup failed", exc_info=True)

    def _log_cleanup_loop() -> None:
        while True:
            time.sleep(6 * 3600)
            try:
                n = cleanup_old_job_logs(settings)
                if n:
                    logger.info("Cleaned up {} old job log(s)", n)
            except Exception:  # noqa: BLE001
                pass

    threading.Thread(target=_log_cleanup_loop, daemon=True).start()

    if settings.watcher.enabled and not no_watch:
        from app.watcher import FileWatcher

        # enqueue_new_file, not enqueue_file: the watcher must not re-submit a
        # file this system has already handled. See TranscodeManager.
        watcher = FileWatcher(settings, manager.enqueue_new_file)
        watcher.start()
    else:
        watcher = None
        logger.info("Watcher disabled")

    manager.start()

    if settings.web.enabled and not no_web:
        import uvicorn
        from app.api import create_app

        web_app = create_app(settings, store, manager)
        logger.info("Starting web UI on http://{}:{}", settings.web.host, settings.web.port)
        try:
            uvicorn.run(web_app, host=settings.web.host, port=settings.web.port,
                        log_level="warning")
        except KeyboardInterrupt:
            pass
        finally:
            manager.stop()
            if watcher:
                watcher.stop()
    else:
        logger.info("Running headless (no web UI). Ctrl-C to stop.")
        try:
            import time as _t
            while True:
                _t.sleep(2)
        except KeyboardInterrupt:
            pass
        finally:
            manager.stop()
            if watcher:
                watcher.stop()


@app.command()
def scan(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    preset: str = typer.Option("", help="preset name to use"),
) -> None:
    """Analyze all files in the input dir and enqueue those needing transcode."""
    settings = load_settings(config)
    settings.ensure_dirs()
    store = db.JobStore(settings)
    from app.queue import TranscodeManager

    manager = TranscodeManager(settings, store)
    files = sorted(settings.dirs.input.rglob("*"))
    n = 0
    for p in files:
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") not in settings.watcher.extensions:
            continue
        jid = manager.enqueue_file(str(p), preset=preset)
        if jid:
            n += 1
    typer.echo(f"Enqueued {n} file(s).")


def _parse_custom(items: Optional[List[str]]) -> dict:
    """Parse --custom key=value pairs into an overrides dict for VideoParams."""
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            typer.echo(f"warning: ignoring --custom '{item}' (expected key=value)")
            continue
        key, _, val = item.partition("=")
        key = key.strip()
        val = val.strip()
        if key in ("crf", "preset", "film_grain", "passes", "keyint",
                   "extra_split_sec", "min_scene_len", "tune",
                   "probes", "probing_rate", "vmaf_threads"):
            out[key] = int(val)
        elif key in ("film_grain_denoise",):
            out[key] = val.lower() in ("1", "true", "yes", "on")
        else:
            out[key] = val
    return out


@app.command()
def process(
    file: str = typer.Argument(..., help="path to a video file"),
    preset: str = typer.Option("", "-p", "--preset"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    single: bool = typer.Option(False, "-s", "--single", help="process synchronously in foreground"),
    custom: Optional[List[str]] = typer.Option(
        None, "--custom", help="override VideoParams, repeatable: --custom crf=24 --custom preset=3"),
) -> None:
    """Submit one file for processing (or run it synchronously with --single)."""
    settings = load_settings(config)
    settings.ensure_dirs()

    from app.queue import TranscodeManager

    store = db.JobStore(settings)
    manager = TranscodeManager(settings, store)
    overrides = _parse_custom(custom)
    if single:
        manager._process(manager.enqueue_file(file, preset=preset, overrides=overrides))
    else:
        jid = manager.enqueue_file(file, preset=preset, overrides=overrides)
        manager.start()
        typer.echo(f"Submitted job {jid}. Use: {sys.argv[0]} status")


@app.command()
def status(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Show job summary."""
    settings = load_settings(config)
    store = db.JobStore(settings)
    counts = store.count_by_status()
    for s, c in counts.items():
        typer.echo(f"{s}: {c}")
    typer.echo("")
    for j in store.list(status=None, limit=15):
        pct = round(float(j["progress"] or 0))
        print(f"  {j['id'][:8]} {j['status']:<9} {pct:>3}%  {Path(j['source']).name}")


@app.command()
def cancel(
    job_id: Optional[str] = typer.Argument(None, help="job id, or omit to cancel all pending"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    settings = load_settings(config)
    manager, store = _build_manager(settings)
    n = manager.cancel(job_id)
    typer.echo(f"Cancelled {n} job(s).")


@app.command()
def presets(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Show current encode presets."""
    settings = load_settings(config)
    for name, v in settings.transcode.presets.items():
        print(f"[{name}]")
        print("  crf={} preset={} film_grain={} passes={}".format(
            v.crf, v.preset, v.film_grain, v.passes))
        if v.additional_video_params:
            print(f"  extra: {v.additional_video_params}")


@app.command()
def check(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Verify all required tools are present."""
    settings = load_settings(config)
    import shutil

    ok = True
    for tool, bin in settings.tools.model_dump().items():
        found = shutil.which(bin)
        if found:
            print(f"  [ok] {tool}: {found}")
        else:
            print(f"  [MISSING] {tool}: {bin}")
            ok = False
    if not ok:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()