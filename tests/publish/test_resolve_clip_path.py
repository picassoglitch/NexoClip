"""`_resolve_clip_path` prefers `clip_final.mp4` when present (slice F.7-E)."""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio

from nexoclip.db import (
    ClipsRepo,
    Database,
    StreamsRepo,
    TenantsRepo,
    apply_migrations,
)
from nexoclip.db.models import (
    CandidateRow,
    ClipRow,
    StreamRow,
)
from nexoclip.publish.service import _resolve_clip_path
from nexoclip.tenancy import bound_tenant


@pytest_asyncio.fixture
async def resolve_db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "resolve.db")
    try:
        await apply_migrations(d)
        yield d
    finally:
        await d.close()


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed_clip(
    db: Database, *, tenant_id: str, clip_path: Path
) -> None:
    from nexoclip.db import CandidatesRepo

    await TenantsRepo(db).create(tenant_id=tenant_id, name="A")
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_r", tenant_id=tenant_id, vod_url="x", platform="kick",
                title="t", channel="c", duration_s=60.0,
                source_video_path="/tmp/v", source_audio_path="/tmp/a",
                status="ingested", created_at=_now(),
            )
        )
        await CandidatesRepo(db).upsert_many([CandidateRow(
            id="cnd_r", stream_id="str_r", tenant_id=tenant_id,
            ts=10.0, score=0.9, reason="voice", evidence={},
            created_at=_now(),
        )])
        await ClipsRepo(db).upsert_many([ClipRow(
            id="clp_r", stream_id="str_r", tenant_id=tenant_id,
            candidate_id="cnd_r",
            start_s=0.0, end_s=10.0, duration_s=10.0,
            width=1080, height=1920,
            path=str(clip_path),
            status="approved", created_at=_now(),
        )])


async def test_resolve_returns_original_when_no_burned_file(
    resolve_db: Database, tmp_path: Path
) -> None:
    clip_dir = tmp_path / "clips" / "clp_r"
    clip_dir.mkdir(parents=True)
    clip = clip_dir / "clip.mp4"
    clip.write_bytes(b"fake")
    await _seed_clip(resolve_db, tenant_id="aldo", clip_path=clip)

    with bound_tenant("aldo"):
        out = await _resolve_clip_path(resolve_db, "clp_r")
    assert out == str(clip)


async def test_resolve_prefers_clip_final_when_present(
    resolve_db: Database, tmp_path: Path
) -> None:
    """The whole point of slice F.7-E: publishers upload the
    burned-in version when it exists."""
    clip_dir = tmp_path / "clips" / "clp_r"
    clip_dir.mkdir(parents=True)
    clip = clip_dir / "clip.mp4"
    clip.write_bytes(b"fake-original")
    final = clip_dir / "clip_final.mp4"
    final.write_bytes(b"fake-burned")

    await _seed_clip(resolve_db, tenant_id="aldo", clip_path=clip)
    with bound_tenant("aldo"):
        out = await _resolve_clip_path(resolve_db, "clp_r")
    assert out == str(final)


async def test_resolve_returns_empty_string_for_unknown_clip(
    resolve_db: Database,
) -> None:
    await TenantsRepo(resolve_db).create(tenant_id="aldo", name="A")
    with bound_tenant("aldo"):
        out = await _resolve_clip_path(resolve_db, "clp_nope")
    assert out == ""
