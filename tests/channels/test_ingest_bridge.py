"""Channel ingest callback — VOD-level idempotency + ingest/pipeline
failure decoupling.

These guard the duplicate-ingest fix: a re-detected VOD must reuse its
existing stream id (no duplicate row, no re-download), and a pipeline
failure must NOT make the poller treat the VOD as un-ingested.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from nexoclip.channels import make_channel_ingest_callback
from nexoclip.db import Database, StreamsRepo, TenantsRepo, apply_migrations
from nexoclip.errors import IngestError
from nexoclip.ids import new_id
from nexoclip.ingest.models import Stream
from nexoclip.jobs import JobDispatcher, PipelineKickoff
from nexoclip.tenancy import bound_tenant


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "ib.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


@pytest_asyncio.fixture
async def tenant(db: Database) -> str:
    t = await TenantsRepo(db).create(name="Streamer")
    return t.id


class _FakeDispatcher(JobDispatcher):
    """Records kickoffs; optionally raises to simulate a pipeline failure."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.dispatched: list[PipelineKickoff] = []

    @property
    def name(self) -> str:
        return "fake"

    async def dispatch_pipeline(
        self, kickoff: PipelineKickoff, *, background_tasks: object | None = None
    ) -> None:
        if self.fail:
            raise RuntimeError("pipeline boom")
        self.dispatched.append(kickoff)


class _FakeIngest:
    """Stand-in for ingest_vod: honors stream_id (cache-hit) and records
    the id it was asked to use on each call."""

    def __init__(self, *, raises: bool = False) -> None:
        self.raises = raises
        self.stream_ids_seen: list[str | None] = []

    async def __call__(
        self,
        *,
        tenant_id: str,
        vod_url: str,
        output_dir: Path,
        stream_id: str | None = None,
        cookies_from_browser: str | None = None,
        cookies_file: str | None = None,
        db: object | None = None,
    ) -> Stream:
        if self.raises:
            raise IngestError("download boom")
        self.stream_ids_seen.append(stream_id)
        sid = stream_id or new_id("str")
        return Stream(
            id=sid,
            tenant_id=tenant_id,
            vod_url=vod_url,
            platform="kick",
            title="t",
            duration_s=10.0,
            source_video_path=output_dir / sid / "source" / "video.mp4",
            source_audio_path=output_dir / sid / "source" / "audio.wav",
        )


@pytest.mark.asyncio
async def test_reingest_reuses_stream_id_no_duplicate_row(
    db: Database, tenant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_ingest = _FakeIngest()
    monkeypatch.setattr("nexoclip.ingest.ingest_vod", fake_ingest)
    dispatcher = _FakeDispatcher()
    cb = make_channel_ingest_callback(db, dispatcher, output_dir=tmp_path)

    url = "https://kick.com/n3on/videos/abc"
    await cb(tenant, url, "vid1", "persona", "es")
    await cb(tenant, url, "vid1", "persona", "es")  # re-detected

    with bound_tenant(tenant):
        rows = await StreamsRepo(db).list_for_tenant()
    # One canonical row despite two ingest passes.
    assert len([r for r in rows if r.vod_url == url]) == 1
    # Second pass reused the first pass's id (cache-hit, no re-download).
    first_id = fake_ingest.stream_ids_seen[0]
    assert first_id is None  # first call minted a fresh id
    assert fake_ingest.stream_ids_seen[1] == rows[0].id


@pytest.mark.asyncio
async def test_pipeline_failure_does_not_propagate(
    db: Database, tenant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nexoclip.ingest.ingest_vod", _FakeIngest())
    dispatcher = _FakeDispatcher(fail=True)
    cb = make_channel_ingest_callback(db, dispatcher, output_dir=tmp_path)

    # Must NOT raise — a pipeline failure is not an ingest failure.
    await cb(tenant, "https://kick.com/n3on/videos/xyz", "vid2", "persona", "es")

    with bound_tenant(tenant):
        rows = await StreamsRepo(db).list_for_tenant()
    assert len(rows) == 1  # the stream row still landed


@pytest.mark.asyncio
async def test_download_failure_propagates(
    db: Database, tenant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nexoclip.ingest.ingest_vod", _FakeIngest(raises=True))
    dispatcher = _FakeDispatcher()
    cb = make_channel_ingest_callback(db, dispatcher, output_dir=tmp_path)

    # A genuine ingest/download failure must still raise so the poller
    # retries it (and does NOT mark the VOD seen).
    with pytest.raises(IngestError):
        await cb(tenant, "https://kick.com/n3on/videos/boom", "vid3", "persona", "es")
