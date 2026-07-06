"""Per-platform abuse/rate-limit backoff: parsing + the cooldown repo."""

from __future__ import annotations

import datetime as _dt

import pytest

from nexoclip.db import Database, PlatformCooldownsRepo, apply_migrations
from nexoclip.integrations.zernio.errors import (
    cooldowns_from_failed_post,
    failure_anchor,
    parse_cooldown,
)


def test_parse_cooldown_minutes() -> None:
    assert parse_cooldown("Rate limit hit. Please wait 1438 minutes before posting again.") == _dt.timedelta(minutes=1438)


def test_parse_cooldown_hours_and_daily() -> None:
    assert parse_cooldown("try again in 2 hours") == _dt.timedelta(hours=2)
    assert parse_cooldown("daily upload limit reached. try again tomorrow.") == _dt.timedelta(hours=24)


def test_parse_cooldown_default_when_unparseable() -> None:
    assert parse_cooldown("something went wrong") == _dt.timedelta(hours=6)
    assert parse_cooldown(None) == _dt.timedelta(hours=6)


def test_cooldowns_from_failed_post_only_user_abuse() -> None:
    post = {
        "platforms": [
            {"platform": "youtube", "status": "failed",
             "errorCategory": "user_abuse",
             "errorMessage": "Rate limit hit. Please wait 60 minutes."},
            {"platform": "tiktok", "status": "failed",
             "errorCategory": "auth_expired", "errorMessage": "token expired"},
            {"platform": "instagram", "status": "published"},
        ]
    }
    cds = cooldowns_from_failed_post(post)
    assert set(cds) == {"youtube"}  # only the abuse one
    assert cds["youtube"] == _dt.timedelta(minutes=60)


def test_failure_anchor_parses_and_prefers_failure_time() -> None:
    """Cooldowns run from WHEN the post failed, not from when a sweep
    re-reads the (never-shrinking) failed list — anchoring at read time
    re-armed every cooldown forever. failedAt wins over createdAt; a Z
    suffix parses; the result is aware UTC."""
    post = {
        "createdAt": "2026-07-01T10:00:00Z",
        "failedAt": "2026-07-02T08:30:00Z",
    }
    anchor = failure_anchor(post)
    assert anchor == _dt.datetime(2026, 7, 2, 8, 30, tzinfo=_dt.UTC)

    # Falls back through updatedAt/createdAt when failedAt is absent.
    assert failure_anchor({"createdAt": "2026-07-01T10:00:00Z"}) == _dt.datetime(
        2026, 7, 1, 10, 0, tzinfo=_dt.UTC
    )


def test_failure_anchor_none_when_unparseable() -> None:
    """No timestamp → None. The caller must SKIP the post (guessing
    "now" recreates the eternal re-arm this fixes)."""
    assert failure_anchor({}) is None
    assert failure_anchor({"failedAt": "not-a-date", "updatedAt": ""}) is None


@pytest.mark.asyncio
async def test_cooldown_repo_active_and_expiry(tmp_path) -> None:
    db = Database(tmp_path / "x.db")
    await apply_migrations(db)
    conn = await db.connect()
    await conn.execute("INSERT INTO tenants (id, name, created_at) VALUES ('t1','T','2026-01-01')")
    await conn.commit()
    repo = PlatformCooldownsRepo(db)
    now = _dt.datetime.now(_dt.UTC)

    # Fresh cooldown (future) shows as active; canonicalizes x→twitter.
    await repo.set_cooldown("t1", "x", until=(now + _dt.timedelta(hours=2)).isoformat())
    await repo.set_cooldown("t1", "youtube", until=(now - _dt.timedelta(hours=1)).isoformat())
    active = await repo.active("t1")
    assert "twitter" in active  # x normalized + still active
    assert "youtube" not in active  # lapsed → pruned
    await db.close()
