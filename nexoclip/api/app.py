"""FastAPI app factory.

`create_app(db=...)` is the single entry point. The DB must be passed in
already opened; tests share a fixture-scoped DB and production wires one
up at boot before calling `create_app`.

A `pipeline_runner` callable is stored on `app.state` so tests can swap
the heavy `process_vod` for a stub. Default points at the real pipeline.
"""

from __future__ import annotations

from fastapi import FastAPI, Request

from nexoclip.db import Database

from ._pipeline import PipelineKickoff, PipelineRunner, default_pipeline_runner
from .auth import BearerAuthMiddleware
from .routers import clips as clips_router
from .routers import llm_calls as llm_calls_router
from .routers import personas as personas_router
from .routers import streams as streams_router

__all__ = ["PipelineKickoff", "PipelineRunner", "create_app"]


def create_app(
    *,
    db: Database,
    pipeline_runner: PipelineRunner | None = None,
) -> FastAPI:
    """Build a FastAPI app wired to `db`."""
    app = FastAPI(title="NexoClip API", version="0.1.0")
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

    return app
