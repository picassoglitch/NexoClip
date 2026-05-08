"""FastAPI app factory.

`create_app(db=...)` is the single entry point. The DB must be passed in
already opened; tests share a fixture-scoped DB and production wires one
up at boot before calling `create_app`.

A `pipeline_runner` callable is stored on `app.state` so tests can swap
the heavy `process_vod` for a stub. Default points at the real pipeline.

When `enable_background_drains=True` the lifespan starts three loops:
publish_jobs (60s), webhook dispatch (30s), and metrics ingest (1h).
Tests leave the flag off so loops never spin during test runs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from nexoclip.db import Database

from ._pipeline import PipelineKickoff, PipelineRunner, default_pipeline_runner
from .auth import BearerAuthMiddleware
from .lifespan import background_drains_lifespan
from .routers import clips as clips_router
from .routers import dashboard as dashboard_router
from .routers import llm_calls as llm_calls_router
from .routers import personas as personas_router
from .routers import streams as streams_router
from .routers import webhooks as webhooks_router

__all__ = ["PipelineKickoff", "PipelineRunner", "create_app"]


def create_app(
    *,
    db: Database,
    pipeline_runner: PipelineRunner | None = None,
    enable_background_drains: bool = False,
    publish_interval_s: float = 60.0,
    webhook_interval_s: float = 30.0,
    metrics_interval_s: float = 3600.0,
) -> FastAPI:
    """Build a FastAPI app wired to `db`.

    Args:
        db: Pre-opened DB.
        pipeline_runner: Override `process_vod` for tests.
        enable_background_drains: When True, the lifespan starts three
            drain loops (publish_jobs, webhook dispatch, metrics ingest)
            for every active tenant. Tests leave this off so the loops
            never spin during the test run.
        publish_interval_s / webhook_interval_s / metrics_interval_s:
            Per-loop cadences. Production typically leaves these at the
            documented defaults (60s / 30s / 1h).
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if enable_background_drains:
            async with background_drains_lifespan(
                _app,
                publish_interval_s=publish_interval_s,
                webhook_interval_s=webhook_interval_s,
                metrics_interval_s=metrics_interval_s,
            ):
                yield
        else:
            yield

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
    app.include_router(webhooks_router.router)

    return app
