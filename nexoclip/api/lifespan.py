"""FastAPI lifespan — auto-drains for the publish + metrics + webhook workers.

When `enable_background_drains=True` (default) the API server kicks four
background loops at boot:

  * publish_jobs drain    every 60s   per active tenant
  * webhook dispatch      every 30s   per active tenant
  * metrics ingest        every 1h    per active tenant
  * Instagram refresh     every 6h    per active tenant   (Wave 2)

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

import structlog
from fastapi import FastAPI

from nexoclip.db import Database, TenantsRepo

_log = structlog.get_logger(__name__)

# Default cadences. Tests + ops can override via `LifespanIntervals` if
# tighter timing matters.
_DEFAULT_PUBLISH_INTERVAL_S = 60.0
_DEFAULT_WEBHOOK_INTERVAL_S = 30.0
_DEFAULT_METRICS_INTERVAL_S = 3600.0
# IG long-lived token refresh — Wave 2. Tokens expire ~60 days
# after issue; we refresh inside a 14-day window so we have
# multiple opportunities before lapse. 6h is plenty of headroom
# (and cheap — one Graph API call per IG-connected tenant).
_DEFAULT_IG_REFRESH_INTERVAL_S = 6 * 3600.0


async def _publish_loop(db: Database, interval_s: float) -> None:
    """Drain `publish_jobs` for every tenant on a loop."""
    from nexoclip.publish import run_publish_jobs

    while True:
        try:
            await asyncio.sleep(interval_s)
            tenants = await TenantsRepo(db).list_all()
            for tenant in tenants:
                try:
                    await run_publish_jobs(tenant.id, db)
                except Exception as e:
                    _log.warning(
                        "publish_drain_failed",
                        tenant_id=tenant.id,
                        error=str(e),
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # The outer try should never fire (tenant fetch is cheap), but if
            # it ever does we'd rather log + restart the loop than crash.
            _log.warning("publish_loop_iteration_failed", error=str(e))


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


async def _instagram_refresh_loop(db: Database, interval_s: float) -> None:
    """Refresh near-expiry IG long-lived tokens per tenant.

    Wave 2. See nexoclip.integrations.instagram.refresh for the
    per-tenant logic; this loop just iterates.
    """
    from nexoclip.integrations.instagram.refresh import run_instagram_refresh

    while True:
        try:
            await asyncio.sleep(interval_s)
            tenants = await TenantsRepo(db).list_all()
            for tenant in tenants:
                try:
                    refreshed = await run_instagram_refresh(tenant.id, db)
                    if refreshed:
                        _log.info(
                            "instagram_refresh_drain",
                            tenant_id=tenant.id,
                            refreshed=refreshed,
                        )
                except Exception as e:
                    _log.warning(
                        "instagram_refresh_drain_failed",
                        tenant_id=tenant.id,
                        error=str(e),
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning("instagram_refresh_loop_iteration_failed", error=str(e))


@asynccontextmanager
async def background_drains_lifespan(
    app: FastAPI,
    *,
    publish_interval_s: float = _DEFAULT_PUBLISH_INTERVAL_S,
    webhook_interval_s: float = _DEFAULT_WEBHOOK_INTERVAL_S,
    metrics_interval_s: float = _DEFAULT_METRICS_INTERVAL_S,
    ig_refresh_interval_s: float = _DEFAULT_IG_REFRESH_INTERVAL_S,
) -> AsyncIterator[None]:
    """Start the four drain loops on boot; cancel + await on shutdown."""
    db: Database = app.state.db
    started_at = _dt.datetime.now(_dt.UTC).isoformat()
    _log.info(
        "lifespan_starting",
        started_at=started_at,
        publish_interval_s=publish_interval_s,
        webhook_interval_s=webhook_interval_s,
        metrics_interval_s=metrics_interval_s,
        ig_refresh_interval_s=ig_refresh_interval_s,
    )
    tasks = [
        asyncio.create_task(
            _publish_loop(db, publish_interval_s), name="nexoclip-publish-loop"
        ),
        asyncio.create_task(
            _webhook_loop(db, webhook_interval_s), name="nexoclip-webhook-loop"
        ),
        asyncio.create_task(
            _metrics_loop(db, metrics_interval_s), name="nexoclip-metrics-loop"
        ),
        asyncio.create_task(
            _instagram_refresh_loop(db, ig_refresh_interval_s),
            name="nexoclip-instagram-refresh-loop",
        ),
    ]
    try:
        yield
    finally:
        _log.info("lifespan_shutting_down")
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await task
