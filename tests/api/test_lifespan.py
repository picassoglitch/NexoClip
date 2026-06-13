"""Lifespan auto-drains — boot starts the loops, shutdown cancels them.

We don't wait long enough for the loops to actually fire; their drain
functions are exercised by their own test suites. What we own here is
the lifespan plumbing: tasks created on boot, cancelled on shutdown,
and `enable_background_drains=False` keeps them off.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest_asyncio

from nexoclip.api import create_app
from nexoclip.db import Database, apply_migrations


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "lifespan.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


async def test_default_does_not_spin_background_loops(db: Database) -> None:
    """Default `enable_background_drains=False` keeps the lifespan a no-op.

    A lingering task at the end would fail this. We assert that hitting
    `/healthz` works and no NexoClip-named tasks survive.
    """
    app = create_app(db=db)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as c:
        r = await c.get("/healthz")
        assert r.status_code == 200

    # No drain tasks should be running after the lifespan exits.
    nexoclip_tasks = [
        t for t in asyncio.all_tasks() if (t.get_name() or "").startswith("nexoclip-")
    ]
    assert nexoclip_tasks == []


async def test_background_drains_lifespan_starts_three_named_loops(
    db: Database,
) -> None:
    """Direct test over `background_drains_lifespan` — three named tasks
    exist while the context is open and are cancelled on exit. We test
    the lifespan helper directly rather than through httpx.ASGITransport
    because the latter doesn't drive ASGI lifespan by default."""
    from fastapi import FastAPI

    from nexoclip.api.lifespan import background_drains_lifespan

    app = FastAPI()
    app.state.db = db
    async with background_drains_lifespan(
        app,
        publish_interval_s=3600.0,
        webhook_interval_s=3600.0,
        metrics_interval_s=3600.0,
    ):
        names = {t.get_name() for t in asyncio.all_tasks()}
        # The legacy publish_jobs drain loop was removed (Etapa A) —
        # publishing goes through Zernio now, not the per-platform worker.
        assert "nexoclip-publish-loop" not in names
        assert "nexoclip-webhook-loop" in names
        assert "nexoclip-metrics-loop" in names

    # Cancellation completes synchronously inside the lifespan's __aexit__.
    nexoclip_tasks = [
        t
        for t in asyncio.all_tasks()
        if (t.get_name() or "").startswith("nexoclip-") and not t.done()
    ]
    assert nexoclip_tasks == []
