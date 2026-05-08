"""GET /dashboard/streams/{id}/progress — HTMX fragment used to surface
live pipeline progress on the stream detail page."""

from __future__ import annotations

import datetime as _dt
import json

import httpx

from nexoclip.db import (
    Database,
    StreamsRepo,
)
from nexoclip.db.models import StreamRow
from nexoclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed_stream(db: Database, *, tenant_id: str, stream_id: str) -> None:
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id,
                tenant_id=tenant_id,
                vod_url=f"upload://{stream_id}.mp4",
                platform="upload",
                title="Test",
                channel=None,
                duration_s=120.0,
                source_video_path=f"/tmp/{stream_id}.mp4",
                source_audio_path=f"/tmp/{stream_id}.wav",
                status="ingested",
                created_at=_now(),
            )
        )


async def _emit_step_event(
    db: Database,
    *,
    tenant_id: str,
    stream_id: str,
    event_type: str,
    step: str,
    duration_s: float | None = None,
    error: str | None = None,
) -> None:
    """Insert a pipeline.step.* event directly via raw SQL (mirrors what the
    sync hook in `nexoclip.pipeline._record_step_event` does)."""
    payload: dict[str, object] = {"step": step, "stream_id": stream_id}
    if duration_s is not None:
        payload["duration_s"] = duration_s
    if error is not None:
        payload["error"] = error
    conn = await db.connect()
    await conn.execute(
        "INSERT INTO events (id, tenant_id, type, payload_json, ts) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            f"evt_{step}_{event_type}",
            tenant_id,
            event_type,
            json.dumps(payload),
            _now(),
        ),
    )
    await conn.commit()


async def test_progress_pending_when_no_events_yet(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """Fresh stream with no step events: every step renders as pending."""
    await _seed_stream(db, tenant_id=tenants["alice"]["id"], stream_id="str_p1")
    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/streams/str_p1/progress")
    assert r.status_code == 200
    body = r.text
    # All six steps appear by name.
    for step in ("ingest", "analyze_video", "transcribe", "detect", "cut", "variants"):
        assert step in body
    # No "running" / "done" badges yet.
    assert "(running…)" not in body


async def test_progress_running_step_polls_every_3s(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """A step.start without a matching step.done shows as running and the
    fragment includes the HTMX poll attribute so the page keeps refreshing."""
    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id=tenant_id, stream_id="str_p2")
    await _emit_step_event(
        db,
        tenant_id=tenant_id,
        stream_id="str_p2",
        event_type="pipeline.step.done",
        step="ingest",
        duration_s=2.0,
    )
    await _emit_step_event(
        db,
        tenant_id=tenant_id,
        stream_id="str_p2",
        event_type="pipeline.step.start",
        step="transcribe",
    )
    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/streams/str_p2/progress")
    assert r.status_code == 200
    body = r.text
    assert "(running…)" in body
    assert "transcribe" in body
    # Auto-refresh while running.
    assert 'hx-trigger="every 3s"' in body


async def test_progress_done_when_every_step_finished(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """Once every step has a `done` event, the fragment shows complete and
    drops the HTMX polling attribute (no more refresh)."""
    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id=tenant_id, stream_id="str_p3")
    for step in ("ingest", "analyze_video", "transcribe", "detect", "cut", "variants"):
        await _emit_step_event(
            db,
            tenant_id=tenant_id,
            stream_id="str_p3",
            event_type="pipeline.step.done",
            step=step,
            duration_s=1.5,
        )
    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/streams/str_p3/progress")
    assert r.status_code == 200
    body = r.text
    assert "Pipeline complete" in body
    assert 'hx-trigger="every 3s"' not in body


async def test_progress_failed_step_surfaces_error(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """A step.failed event shows the error text inline and stops polling."""
    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id=tenant_id, stream_id="str_p4")
    await _emit_step_event(
        db,
        tenant_id=tenant_id,
        stream_id="str_p4",
        event_type="pipeline.step.failed",
        step="transcribe",
        error="CUDA out of memory",
    )
    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/streams/str_p4/progress")
    assert r.status_code == 200
    body = r.text
    assert "Pipeline failed" in body
    assert "CUDA out of memory" in body
    assert 'hx-trigger="every 3s"' not in body


async def test_progress_isolates_events_per_stream(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    """Step events from another stream in the same tenant don't bleed in."""
    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id=tenant_id, stream_id="str_pA")
    await _seed_stream(db, tenant_id=tenant_id, stream_id="str_pB")
    # str_pA has finished transcribe; str_pB hasn't started anything.
    await _emit_step_event(
        db,
        tenant_id=tenant_id,
        stream_id="str_pA",
        event_type="pipeline.step.done",
        step="transcribe",
        duration_s=42.0,
    )
    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/streams/str_pB/progress")
    assert r.status_code == 200
    # str_pB's transcribe should still show as pending, not done.
    assert "Pipeline complete" not in r.text
