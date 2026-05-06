"""FastAPI app factory.

`create_app(db=...)` is the single entry point. The DB must be passed in
already opened; tests share a fixture-scoped DB and production wires one
up at boot before calling `create_app`.

A `pipeline_runner` callable is stored on `app.state` so tests can swap
the heavy `process_vod` for a stub. Default points at the real pipeline.

When `publisher_interval_s > 0`, the lifespan starts a background task
that wakes every `publisher_interval_s` seconds and drains pending
publish_jobs for every tenant. Tests pass `publisher_interval_s=0` so
the loop never fires.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

import structlog
from fastapi import FastAPI, Request

from nexoclip.db import Database

from ._pipeline import PipelineKickoff, PipelineRunner, default_pipeline_runner
from .auth import BearerAuthMiddleware
from .routers import clips as clips_router
from .routers import dashboard as dashboard_router
from .routers import llm_calls as llm_calls_router
from .routers import personas as personas_router
from .routers import streams as streams_router

__all__ = ["PipelineKickoff", "PipelineRunner", "create_app"]

_log = structlog.get_logger(__name__)
_DEFAULT_PUBLISHER_INTERVAL_S = 60.0


def create_app(
    *,
    db: Database,
    pipeline_runner: PipelineRunner | None = None,
    publisher_interval_s: float = 0.0,
) -> FastAPI:
    """Build a FastAPI app wired to `db`.

    Args:
        db: Pre-opened DB.
        pipeline_runner: Override `process_vod` for tests.
        publisher_interval_s: When > 0, the lifespan starts a background
            task that drains publish_jobs for every tenant every N seconds.
            Defaults to 0 so tests don't spin a loop. Production should
            pass 60 (or whatever cadence makes sense).
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        task: asyncio.Task[None] | None = None
        if publisher_interval_s > 0:
            task = asyncio.create_task(
                _publisher_loop(db, interval_s=publisher_interval_s)
            )
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with suppress(BaseException):
                    await task

    app = FastAPI(title="NexoClip API", version="0.1.0", lifespan=lifespan)
    app.state.db = db
    app.state.pipeline_runner = pipeline_runner or default_pipeline_runner

    app.add_middleware(BearerAuthMiddleware, db=db)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> dict[str, str]:
        active_db: Database = request.app.state.db
        await active_db.connect()
        return {"status": "ready"}

    app.include_router(streams_router.router)
    app.include_router(clips_router.router)
    app.include_router(personas_router.router)
    app.include_router(llm_calls_router.router)
    app.include_router(dashboard_router.router)

    return app


async def _publisher_loop(db: Database, *, interval_s: float) -> None:
    """Background drain: every `interval_s` seconds, run `run_publish_jobs`
    for every tenant. Errors are logged and swallowed so one bad tenant
    doesn't break the loop for the rest."""
    from nexoclip.db import TenantsRepo
    from nexoclip.publish import run_publish_jobs

    while True:
        try:
            tenants = await TenantsRepo(db).list_all()
            for tenant in tenants:
                try:
                    await run_publish_jobs(tenant.id, db)
                except Exception as e:  # per-tenant isolation
                    _log.warning(
                        "publisher_loop_error",
                        tenant_id=tenant.id,
                        error=str(e),
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # keep the loop alive
            _log.warning("publisher_loop_top_level_error", error=str(e))
        await asyncio.sleep(interval_s)
