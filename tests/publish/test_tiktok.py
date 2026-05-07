"""TikTok client + dispatcher integration."""

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
    StreamRow,
    VariantRow,
)
from nexoclip.publish import TikTokClient, TikTokError, run_publish_jobs
from nexoclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "tt.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


@pytest_asyncio.fixture
async def seeded_tiktok(db: Database, tmp_path: Path) -> dict[str, str]:
    """Tenant + clip + variant + tiktok account + 1 pending publish_job."""
    tenant = await TenantsRepo(db).create(name="Aldo")
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"fakempeg" * 1024)
    with bound_tenant(tenant.id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_t",
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
                    id="cnd_t",
                    stream_id="str_t",
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
                    id="clp_t",
                    stream_id="str_t",
                    tenant_id=tenant.id,
                    candidate_id="cnd_t",
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
            "clp_t",
            "aldo",
            [
                VariantRow(
                    id="var_t",
                    clip_id="clp_t",
                    tenant_id=tenant.id,
                    persona_id="aldo",
                    language="es",
                    caption="Hello world",
                    title_card_text="",
                    hashtags=["streamer", "viral"],
                    model="m",
                    created_at=_now(),
                )
            ],
        )
        # Far-future expiry so refresh isn't triggered.
        far_future = (_dt.datetime.now(_dt.UTC) + _dt.timedelta(days=30)).isoformat()
        acct = await ConnectedAccountsRepo(db).create(
            platform="tiktok",
            external_id="tt_user",
            display_name="Aldo TT",
            oauth_blob={"access_token": "tt_token_abc"},
            refresh_token="tt_refresh_abc",
            expires_at=far_future,
        )
        job = await PublishJobsRepo(db).enqueue(
            clip_id="clp_t",
            variant_id="var_t",
            account_id=acct.id,
            platform="tiktok",
        )
    return {"tenant_id": tenant.id, "account_id": acct.id, "job_id": job.id}


# ---- TikTokClient unit tests ----


@respx.mock
async def test_tiktok_client_publish_happy_path(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"video-bytes")

    respx.post("https://open.tiktokapis.com/v2/post/publish/inbox/video/init/").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "publish_id": "pub_abc",
                    "upload_url": "https://upload.example.com/x",
                }
            },
        )
    )
    respx.put("https://upload.example.com/x").mock(return_value=httpx.Response(200))

    variant = VariantRow(
        id="v",
        clip_id="c",
        tenant_id="t",
        persona_id="p",
        language="es",
        caption="caption",
        hashtags=["clip"],
        model=None,
        created_at=_now(),
    )
    from nexoclip.db.models import ConnectedAccount

    account = ConnectedAccount(
        id="acc_t",
        tenant_id="t",
        platform="tiktok",
        external_id="x",
        oauth_blob={"access_token": "tok"},
        created_at=_now(),
    )
    async with TikTokClient(access_token="tok") as client:
        result = await client.publish(
            clip_path=str(clip_path), variant=variant, account=account
        )
    assert result.external_id == "pub_abc"
    assert result.metadata["init"]["publish_id"] == "pub_abc"


@respx.mock
async def test_tiktok_client_treats_5xx_as_transient(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"v")

    respx.post(
        "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"
    ).mock(return_value=httpx.Response(503, text="overloaded"))

    variant = VariantRow(
        id="v",
        clip_id="c",
        tenant_id="t",
        persona_id="p",
        language="es",
        caption="caption",
        hashtags=[],
        model=None,
        created_at=_now(),
    )
    from nexoclip.db.models import ConnectedAccount

    account = ConnectedAccount(
        id="a", tenant_id="t", platform="tiktok",
        external_id="x", oauth_blob={"access_token": "tok"}, created_at=_now(),
    )
    async with TikTokClient(access_token="tok") as client:
        try:
            await client.publish(
                clip_path=str(clip_path), variant=variant, account=account
            )
        except TikTokError as e:
            assert e.transient is True
        else:
            raise AssertionError("expected TikTokError")


# ---- Dispatcher integration ----


@respx.mock
async def test_run_publish_jobs_routes_tiktok_through_native_client(
    db: Database, seeded_tiktok: dict[str, str]
) -> None:
    respx.post(
        "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "publish_id": "pub_xyz",
                    "upload_url": "https://upload.example.com/u",
                }
            },
        )
    )
    respx.put("https://upload.example.com/u").mock(return_value=httpx.Response(200))

    async def _no_sleep(_s: float) -> None:
        return None

    outcome = await run_publish_jobs(
        seeded_tiktok["tenant_id"], db, sleep=_no_sleep
    )
    assert outcome.sent == 1

    with bound_tenant(seeded_tiktok["tenant_id"]):
        jobs = await PublishJobsRepo(db).list_for_clip("clp_t")
    assert jobs[0].status == "sent"
    assert jobs[0].external_id == "pub_xyz"
