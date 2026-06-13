"""FastAPI lifespan — auto-drains for the metrics + webhook workers.

When `enable_background_drains=True` (default) the API server kicks the
background loops at boot:

  * webhook dispatch      every 30s   per active tenant
  * metrics ingest        every 1h    per active tenant

The legacy publish_jobs drain was removed (Etapa A): publishing now goes
through Zernio, not the per-platform worker. `publish_interval_s` is kept
on the signatures for back-compat but no longer starts a loop.

Each loop iterates `TenantsRepo.list_all()` and runs the drain serially
for each tenant. Per-tenant errors are logged and swallowed so a single
broken integration doesn't kill the loop. Shutdown cancels + awaits the
tasks before returning.

Tests call `create_app(enable_background_drains=False)` so the loops
never start. The drain functions themselves are exercised by their own
test suites.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import structlog
from fastapi import FastAPI

from nexoclip.db import Database, TenantsRepo

_log = structlog.get_logger(__name__)

# Default cadences. Tests + ops can override via `LifespanIntervals` if
# tighter timing matters.
_DEFAULT_PUBLISH_INTERVAL_S = 60.0
_DEFAULT_WEBHOOK_INTERVAL_S = 30.0
_DEFAULT_METRICS_INTERVAL_S = 3600.0
# Channel polling is comparatively expensive (a yt-dlp listing per watch
# plus a full pipeline run per new VOD) — keep it slow.
_DEFAULT_CHANNEL_POLL_INTERVAL_S = 900.0


async def _webhook_loop(db: Database, interval_s: float) -> None:
    """Drain webhook subscriptions for every tenant on a loop."""
    from nexoclip.webhooks import run_webhook_dispatch

    while True:
        try:
            await asyncio.sleep(interval_s)
            tenants = await TenantsRepo(db).list_all()
            for tenant in tenants:
                try:
                    await run_webhook_dispatch(tenant.id, db)
                except Exception as e:
                    _log.warning(
                        "webhook_drain_failed",
                        tenant_id=tenant.id,
                        error=str(e),
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning("webhook_loop_iteration_failed", error=str(e))


async def _metrics_loop(db: Database, interval_s: float) -> None:
    """Pull engagement metrics for every tenant on a loop."""
    from nexoclip.metrics import run_metrics_ingest

    while True:
        try:
            await asyncio.sleep(interval_s)
            tenants = await TenantsRepo(db).list_all()
            for tenant in tenants:
                try:
                    await run_metrics_ingest(tenant.id, db)
                except Exception as e:
                    _log.warning(
                        "metrics_drain_failed",
                        tenant_id=tenant.id,
                        error=str(e),
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning("metrics_loop_iteration_failed", error=str(e))


async def _channel_poll_loop(
    db: Database,
    dispatcher: object,
    output_dir: Path,
    interval_s: float,
) -> None:
    """Poll every channel watch for new VODs and auto-ingest them."""
    from nexoclip.channels import make_channel_ingest_callback, poll_channel_watches
    from nexoclip.jobs import JobDispatcher

    assert isinstance(dispatcher, JobDispatcher)
    ingest_callback = make_channel_ingest_callback(
        db, dispatcher, output_dir=output_dir
    )
    while True:
        try:
            await asyncio.sleep(interval_s)
            await poll_channel_watches(db, ingest_callback=ingest_callback)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning("channel_poll_loop_iteration_failed", error=str(e))


@asynccontextmanager
async def background_drains_lifespan(
    app: FastAPI,
    *,
    publish_interval_s: float = _DEFAULT_PUBLISH_INTERVAL_S,
    webhook_interval_s: float = _DEFAULT_WEBHOOK_INTERVAL_S,
    metrics_interval_s: float = _DEFAULT_METRICS_INTERVAL_S,
    channel_poll_interval_s: float = _DEFAULT_CHANNEL_POLL_INTERVAL_S,
) -> AsyncIterator[None]:
    """Start the background loops on boot; cancel + await on shutdown."""
    db: Database = app.state.db
    started_at = _dt.datetime.now(_dt.UTC).isoformat()
    _log.info(
        "lifespan_starting",
        started_at=started_at,
        publish_interval_s=publish_interval_s,
        webhook_interval_s=webhook_interval_s,
        metrics_interval_s=metrics_interval_s,
        channel_poll_interval_s=channel_poll_interval_s,
    )
    tasks = [
        asyncio.create_task(
            _webhook_loop(db, webhook_interval_s), name="nexoclip-webhook-loop"
        ),
        asyncio.create_task(
            _metrics_loop(db, metrics_interval_s), name="nexoclip-metrics-loop"
        ),
    ]
    # The channel-poll loop needs the pipeline dispatcher + output dir; only
    # start it when the app wired a dispatcher (the dashboard app does).
    dispatcher = getattr(app.state, "job_dispatcher", None)
    if dispatcher is not None:
        from nexoclip.settings import get_settings

        output_dir = Path(get_settings().default_output_dir)
        tasks.append(
            asyncio.create_task(
                _channel_poll_loop(
                    db, dispatcher, output_dir, channel_poll_interval_s
                ),
                name="nexoclip-channel-poll-loop",
            )
        )
    try:
        yield
    finally:
        _log.info("lifespan_shutting_down")
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await task
