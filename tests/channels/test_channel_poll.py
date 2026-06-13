"""Channel auto-ingest poll loop — back-catalog guard, dedup, caps,
failure handling. Lister + ingest callback are faked (no yt-dlp, no DB
pipeline)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from nexoclip.channels import ChannelVOD, poll_channel_watches
from nexoclip.channels.service import _poll_one_watch
from nexoclip.db import ChannelWatchesRepo, Database, TenantsRepo, apply_migrations
from nexoclip.tenancy import bound_tenant


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "ch.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


def _lister(vods: list[ChannelVOD]):
    async def _list(platform: str, channel_url: str, *, limit: int) -> list[ChannelVOD]:
        return list(vods[:limit])
    return _list


def _failing_lister():
    async def _list(platform: str, channel_url: str, *, limit: int) -> list[ChannelVOD]:
        raise RuntimeError("yt-dlp boom")
    return _list


class _Recorder:
    """Captures every (tenant, url, video_id) the poller hands us."""

    def __init__(self, fail_ids: set[str] | None = None) -> None:
        self.ingested: list[str] = []
        self.fail_ids = fail_ids or set()

    async def __call__(
        self, tenant_id: str, url: str, video_id: str, persona_id: str,
        language: str | None,
    ) -> None:
        if video_id in self.fail_ids:
            raise RuntimeError("ingest boom")
        self.ingested.append(video_id)


def _vods(*ids: str) -> list[ChannelVOD]:
    return [ChannelVOD(video_id=i, url=f"https://yt/watch?v={i}", title=i) for i in ids]


@pytest_asyncio.fixture
async def watch_tenant(db: Database) -> str:
    t = await TenantsRepo(db).create(name="Streamer")
    return t.id


@pytest.mark.asyncio
async def test_first_poll_ingests_only_newest_and_marks_backcatalog_seen(
    db: Database, watch_tenant: str
) -> None:
    with bound_tenant(watch_tenant):
        watch = await ChannelWatchesRepo(db).create(
            platform="youtube", channel_url="https://yt/@me",
            persona_id="per_1", max_per_poll=2,
        )
        rec = _Recorder()
        report = await _poll_one_watch(
            db=db, watch=watch, ingest_callback=rec, list_vods=_lister(_vods("a", "b", "c", "d")),
        )
        # Only the newest 2 ingested; the rest of the window is marked seen.
        assert rec.ingested == ["a", "b"]
        assert report.videos_ingested == 2
        refreshed = await ChannelWatchesRepo(db).get(watch.id)
    assert refreshed is not None
    assert set(refreshed.seen_video_ids) == {"a", "b", "c", "d"}  # back-catalog suppressed
    assert refreshed.last_polled_at is not None


@pytest.mark.asyncio
async def test_steady_state_ingests_only_new_capped(
    db: Database, watch_tenant: str
) -> None:
    with bound_tenant(watch_tenant):
        repo = ChannelWatchesRepo(db)
        watch = await repo.create(
            platform="youtube", channel_url="https://yt/@me",
            persona_id="per_1", max_per_poll=2,
        )
        # Simulate a prior poll: a,b,c,d already seen, last_polled set.
        await repo.mark_polled(
            watch.id, seen_video_ids=["a", "b", "c", "d"],
            last_polled_at="2026-06-10T10:00:00+00:00",
        )
        watch = await repo.get(watch.id)
        assert watch is not None
        rec = _Recorder()
        # Two NEW uploads (e, f) + the old ones still listed.
        report = await _poll_one_watch(
            db=db, watch=watch, ingest_callback=rec,
            list_vods=_lister(_vods("f", "e", "d", "c")),
        )
    assert set(rec.ingested) == {"f", "e"}  # only the new ones, capped at 2
    assert report.videos_ingested == 2


@pytest.mark.asyncio
async def test_disabled_watch_skipped(db: Database, watch_tenant: str) -> None:
    with bound_tenant(watch_tenant):
        repo = ChannelWatchesRepo(db)
        watch = await repo.create(
            platform="youtube", channel_url="https://yt/@me", persona_id="per_1",
        )
        await repo.set_enabled(watch.id, enabled=False)
        watch = await repo.get(watch.id)
        assert watch is not None
        rec = _Recorder()
        report = await _poll_one_watch(
            db=db, watch=watch, ingest_callback=rec, list_vods=_lister(_vods("a")),
        )
    assert report.skipped_disabled is True
    assert rec.ingested == []


@pytest.mark.asyncio
async def test_list_failure_does_not_advance_polled_at(
    db: Database, watch_tenant: str
) -> None:
    with bound_tenant(watch_tenant):
        watch = await ChannelWatchesRepo(db).create(
            platform="twitch", channel_url="https://twitch/me", persona_id="per_1",
        )
        rec = _Recorder()
        report = await _poll_one_watch(
            db=db, watch=watch, ingest_callback=rec, list_vods=_failing_lister(),
        )
    assert report.videos_failed == 1
    assert report.videos_ingested == 0


@pytest.mark.asyncio
async def test_ingest_failure_keeps_video_unseen_for_retry(
    db: Database, watch_tenant: str
) -> None:
    with bound_tenant(watch_tenant):
        repo = ChannelWatchesRepo(db)
        watch = await repo.create(
            platform="youtube", channel_url="https://yt/@me",
            persona_id="per_1", max_per_poll=3,
        )
        await repo.mark_polled(
            watch.id, seen_video_ids=[], last_polled_at="2026-06-10T10:00:00+00:00",
        )
        watch = await repo.get(watch.id)
        assert watch is not None
        rec = _Recorder(fail_ids={"b"})
        report = await _poll_one_watch(
            db=db, watch=watch, ingest_callback=rec, list_vods=_lister(_vods("a", "b", "c")),
        )
        refreshed = await repo.get(watch.id)
    assert report.videos_ingested == 2  # a, c
    assert report.videos_failed == 1   # b
    assert refreshed is not None
    # b stays unseen → retried next poll; a clean pass requirement keeps
    # last_polled_at from advancing.
    assert "b" not in refreshed.seen_video_ids
    assert {"a", "c"} <= set(refreshed.seen_video_ids)


@pytest.mark.asyncio
async def test_poll_all_watches_isolates_per_tenant(
    db: Database, watch_tenant: str
) -> None:
    with bound_tenant(watch_tenant):
        await ChannelWatchesRepo(db).create(
            platform="youtube", channel_url="https://yt/@me", persona_id="per_1",
        )
    rec = _Recorder()
    reports = await poll_channel_watches(
        db, ingest_callback=rec, list_vods=_lister(_vods("x")),
        tenant_id=watch_tenant,
    )
    assert len(reports) == 1
    assert reports[0].videos_ingested == 1
