"""ZernioPublishesRepo — local, tenant-scoped publish history.

The whole reason this table exists: Zernio's GET /posts is scoped to
the company API key, so per-tenant history MUST come from local rows.
These tests pin the isolation (tenant A never sees tenant B's posts)
and the idempotent record() the duplicate-resolved publish path needs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nexoclip.db import Database, TenantsRepo, ZernioPublishesRepo
from nexoclip.db.migrations import apply_migrations
from nexoclip.tenancy import bound_tenant


@pytest.fixture
async def db(tmp_path: Path):
    db = Database(tmp_path / "zp.db")
    await db.connect()
    await apply_migrations(db)
    for tid in ("ten_A", "ten_B"):
        with bound_tenant(tid):
            await TenantsRepo(db).create(tenant_id=tid, name=tid)
    yield db
    await db.close()


@pytest.mark.asyncio
async def test_record_and_list_roundtrip(db: Database) -> None:
    repo = ZernioPublishesRepo(db)
    await repo.record(
        post_id="post_1",
        tenant_id="ten_A",
        clip_id="clp_1",
        platforms=["instagram", "tiktok"],
        content="Hola mundo",
    )
    with bound_tenant("ten_A"):
        rows = await repo.list_for_tenant()
    assert len(rows) == 1
    row = rows[0]
    assert row.post_id == "post_1"
    assert row.clip_id == "clp_1"
    assert row.platforms == "instagram,tiktok"
    assert row.content == "Hola mundo"
    assert row.created_at  # stamped


@pytest.mark.asyncio
async def test_history_is_tenant_isolated(db: Database) -> None:
    """Tenant A must NEVER see tenant B's publishes — the bug this
    table fixes (the raw Zernio feed was company-wide)."""
    repo = ZernioPublishesRepo(db)
    await repo.record(
        post_id="post_a", tenant_id="ten_A", clip_id="clp_a",
        platforms=["instagram"], content=None,
    )
    await repo.record(
        post_id="post_b", tenant_id="ten_B", clip_id="clp_b",
        platforms=["tiktok"], content=None,
    )
    with bound_tenant("ten_A"):
        rows_a = await repo.list_for_tenant()
    with bound_tenant("ten_B"):
        rows_b = await repo.list_for_tenant()
    assert [r.post_id for r in rows_a] == ["post_a"]
    assert [r.post_id for r in rows_b] == ["post_b"]


@pytest.mark.asyncio
async def test_record_is_idempotent_on_post_id(db: Database) -> None:
    """The duplicate-resolved publish path (Zernio 409 → existing post)
    records the same post id again — must not error or duplicate."""
    repo = ZernioPublishesRepo(db)
    for _ in range(3):
        await repo.record(
            post_id="post_dup", tenant_id="ten_A", clip_id="clp_1",
            platforms=["instagram"], content="x",
        )
    with bound_tenant("ten_A"):
        rows = await repo.list_for_tenant()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_exists_for_clip_ignores_cancelled_rows(db: Database) -> None:
    """A cancelled post must NOT block the clip from being re-scheduled —
    the tombstone used to black-hole the clip permanently (every
    scheduling surface skipped it, with no UI path back)."""
    repo = ZernioPublishesRepo(db)
    await repo.record(
        post_id="post_c1", tenant_id="ten_A", clip_id="clp_9",
        platforms=["tiktok"], content=None, status="scheduled",
    )
    assert await repo.exists_for_clip("ten_A", "clp_9") is True

    await repo.set_status("post_c1", status="cancelled")
    assert await repo.exists_for_clip("ten_A", "clp_9") is False


@pytest.mark.asyncio
async def test_count_by_platform_today_skips_cancelled_and_future(
    db: Database,
) -> None:
    """The daily-cap accounting: a cancelled post frees its slot (that's
    the point of cancelling to make room), and a post scheduled for a
    FUTURE day doesn't consume today's cap."""
    import datetime as _dt

    repo = ZernioPublishesRepo(db)
    now = _dt.datetime.now(_dt.UTC)
    today = now.isoformat()
    tomorrow = (now + _dt.timedelta(days=1)).isoformat()

    await repo.record(
        post_id="p_live", tenant_id="ten_A", clip_id="clp_a",
        platforms=["tiktok"], content=None, status="scheduled",
        scheduled_for=today,
    )
    await repo.record(
        post_id="p_tomorrow", tenant_id="ten_A", clip_id="clp_b",
        platforms=["tiktok"], content=None, status="scheduled",
        scheduled_for=tomorrow,
    )
    await repo.record(
        post_id="p_cancel", tenant_id="ten_A", clip_id="clp_c",
        platforms=["tiktok"], content=None, status="scheduled",
        scheduled_for=today,
    )
    await repo.set_status("p_cancel", status="cancelled")

    counts = await repo.count_by_platform_today("ten_A")
    assert counts.get("tiktok", 0) == 1  # only today's live post
