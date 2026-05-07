"""YouTube client + dispatcher integration."""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest_asyncio
import respx

from nexoclip.db import (
    CandidatesRepo,
    ClipsRepo,
    ConnectedAccountsRepo,
    Database,
    PersonasRepo,
    PublishJobsRepo,
    StreamsRepo,
    TenantsRepo,
    VariantsRepo,
    apply_migrations,
)
from nexoclip.db.models import (
    CandidateRow,
    ClipRow,
    ConnectedAccount,
    StreamRow,
    VariantRow,
)
from nexoclip.publish import YouTubeClient, YouTubeError, run_publish_jobs
from nexoclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "yt.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


@pytest_asyncio.fixture
async def seeded_yt(db: Database, tmp_path: Path) -> dict[str, str]:
    tenant = await TenantsRepo(db).create(name="Aldo")
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"video-data" * 256)
    with bound_tenant(tenant.id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_y",
                tenant_id=tenant.id,
                vod_url="x",
                platform="kick",
                title=None,
                channel=None,
                duration_s=60.0,
                source_video_path="/tmp/v",
                source_audio_path="/tmp/a",
                status="ingested",
                created_at=_now(),
            )
        )
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id="cnd_y",
                    stream_id="str_y",
                    tenant_id=tenant.id,
                    ts=10.0,
                    score=0.7,
                    reason="voice",
                    evidence={},
                    created_at=_now(),
                )
            ]
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id="clp_y",
                    stream_id="str_y",
                    tenant_id=tenant.id,
                    candidate_id="cnd_y",
                    start_s=0.0,
                    end_s=10.0,
                    duration_s=10.0,
                    width=1080,
                    height=1920,
                    path=str(clip_path),
                    status="approved",
                    created_at=_now(),
                )
            ]
        )
        await PersonasRepo(db).create(
            persona_id="aldo",
            name="A",
            primary_language="es",
            target_languages=["es"],
            voice_prompt="v",
        )
        await VariantsRepo(db).replace_for_clip_persona(
            "clp_y",
            "aldo",
            [
                VariantRow(
                    id="var_y",
                    clip_id="clp_y",
                    tenant_id=tenant.id,
                    persona_id="aldo",
                    language="es",
                    caption="great moment",
                    title_card_text="Wow",
                    hashtags=["shorts", "live"],
                    model="m",
                    created_at=_now(),
                )
            ],
        )
        far_future = (_dt.datetime.now(_dt.UTC) + _dt.timedelta(days=30)).isoformat()
        acct = await ConnectedAccountsRepo(db).create(
            platform="youtube",
            external_id="UC_abc",
            display_name="Aldo Channel",
            oauth_blob={"access_token": "yt_token"},
            refresh_token="yt_refresh",
            expires_at=far_future,
        )
        job = await PublishJobsRepo(db).enqueue(
            clip_id="clp_y",
            variant_id="var_y",
            account_id=acct.id,
            platform="youtube",
        )
    return {"tenant_id": tenant.id, "account_id": acct.id, "job_id": job.id}


# ---- YouTubeClient unit tests ----


@respx.mock
async def test_youtube_client_publish_happy_path(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"vid")

    respx.post(
        "https://www.googleapis.com/upload/youtube/v3/videos"
    ).mock(
        return_value=httpx.Response(
            200,
            headers={"location": "https://upload.example.com/yt-session-1"},
        )
    )
    respx.put("https://upload.example.com/yt-session-1").mock(
        return_value=httpx.Response(
            200, json={"id": "ytvid_123", "snippet": {"title": "Wow"}}
        )
    )

    variant = VariantRow(
        id="v",
        clip_id="c",
        tenant_id="t",
        persona_id="p",
        language="es",
        caption="caption",
        title_card_text="Wow",
        hashtags=["shorts"],
        model=None,
        created_at=_now(),
    )
    account = ConnectedAccount(
        id="a", tenant_id="t", platform="youtube",
        external_id="UC", oauth_blob={"access_token": "tok"}, created_at=_now(),
    )
    async with YouTubeClient(access_token="tok") as client:
        result = await client.publish(
            clip_path=str(clip_path), variant=variant, account=account
        )
    assert result.external_id == "ytvid_123"
    assert result.external_url == "https://www.youtube.com/watch?v=ytvid_123"


@respx.mock
async def test_youtube_treats_5xx_as_transient(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"v")

    respx.post(
        "https://www.googleapis.com/upload/youtube/v3/videos"
    ).mock(return_value=httpx.Response(502, text="bad gateway"))

    variant = VariantRow(
        id="v", clip_id="c", tenant_id="t", persona_id="p",
        language="es", caption="x", hashtags=[], model=None, created_at=_now(),
    )
    account = ConnectedAccount(
        id="a", tenant_id="t", platform="youtube",
        external_id="UC", oauth_blob={"access_token": "tok"}, created_at=_now(),
    )
    async with YouTubeClient(access_token="tok") as client:
        try:
            await client.publish(
                clip_path=str(clip_path), variant=variant, account=account
            )
        except YouTubeError as e:
            assert e.transient is True
        else:
            raise AssertionError("expected YouTubeError")


# ---- Dispatcher integration ----


@respx.mock
async def test_run_publish_jobs_routes_youtube_through_native_client(
    db: Database, seeded_yt: dict[str, str]
) -> None:
    respx.post("https://www.googleapis.com/upload/youtube/v3/videos").mock(
        return_value=httpx.Response(
            200, headers={"location": "https://upload.example.com/yt-x"}
        )
    )
    respx.put("https://upload.example.com/yt-x").mock(
        return_value=httpx.Response(
            200, json={"id": "ytvid_xyz", "snippet": {}}
        )
    )

    async def _no_sleep(_s: float) -> None:
        return None

    outcome = await run_publish_jobs(
        seeded_yt["tenant_id"], db, sleep=_no_sleep
    )
    assert outcome.sent == 1

    with bound_tenant(seeded_yt["tenant_id"]):
        jobs = await PublishJobsRepo(db).list_for_clip("clp_y")
    assert jobs[0].status == "sent"
    assert jobs[0].external_id == "ytvid_xyz"
    assert jobs[0].external_url == "https://www.youtube.com/watch?v=ytvid_xyz"
