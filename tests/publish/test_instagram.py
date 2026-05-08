"""Instagram client + dispatcher integration."""

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
from nexoclip.publish import InstagramClient, InstagramError, run_publish_jobs
from nexoclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


_IG_BASE = "https://graph.facebook.com/v22.0"


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "ig.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


def _account(template: str | None = None) -> ConnectedAccount:
    blob: dict[str, object] = {"access_token": "tok"}
    if template is not None:
        blob["video_url_template"] = template
    return ConnectedAccount(
        id="acc",
        tenant_id="t",
        platform="instagram",
        external_id="17841400000000000",  # IG user id
        oauth_blob=blob,
        created_at=_now(),
    )


def _variant() -> VariantRow:
    return VariantRow(
        id="v",
        clip_id="c",
        tenant_id="t",
        persona_id="p",
        language="en",
        caption="check this out",
        title_card_text="",
        hashtags=["clip", "live"],
        model=None,
        created_at=_now(),
    )


# ---- InstagramClient happy path ----


@respx.mock
async def test_instagram_client_publish_two_step_flow(tmp_path: Path) -> None:
    clip_path = tmp_path / "myclip.mp4"
    clip_path.write_bytes(b"video-bytes")

    ig_user = "17841400000000000"
    container_route = respx.post(f"{_IG_BASE}/{ig_user}/media").mock(
        return_value=httpx.Response(200, json={"id": "container_123"})
    )
    publish_route = respx.post(f"{_IG_BASE}/{ig_user}/media_publish").mock(
        return_value=httpx.Response(200, json={"id": "media_xyz"})
    )

    account = _account(template="https://media.example/clips/{clip_id}.mp4")
    async with InstagramClient(access_token="tok") as client:
        result = await client.publish(
            clip_path=str(clip_path), variant=_variant(), account=account
        )

    assert result.external_id == "media_xyz"
    assert result.external_url == "https://www.instagram.com/reel/media_xyz/"
    # The container POST received the rendered video_url and the caption.
    assert container_route.called
    container_req = container_route.calls[0].request
    body = container_req.content.decode()
    assert "video_url=https" in body
    assert "myclip.mp4" in body
    assert "media_type=REELS" in body
    # The publish POST included the container id from step 1.
    publish_req = publish_route.calls[0].request
    assert "creation_id=container_123" in publish_req.content.decode()


@respx.mock
async def test_instagram_client_5xx_is_transient(tmp_path: Path) -> None:
    clip_path = tmp_path / "c.mp4"
    clip_path.write_bytes(b"v")
    respx.post(f"{_IG_BASE}/17841400000000000/media").mock(
        return_value=httpx.Response(503, text="overloaded")
    )

    account = _account(template="https://x.example/{clip_id}.mp4")
    async with InstagramClient(access_token="tok") as client:
        try:
            await client.publish(
                clip_path=str(clip_path), variant=_variant(), account=account
            )
        except InstagramError as e:
            assert e.transient is True
        else:
            raise AssertionError("expected InstagramError")


@respx.mock
async def test_instagram_client_4xx_is_fatal(tmp_path: Path) -> None:
    clip_path = tmp_path / "c.mp4"
    clip_path.write_bytes(b"v")
    respx.post(f"{_IG_BASE}/17841400000000000/media").mock(
        return_value=httpx.Response(400, text="bad request")
    )

    account = _account(template="https://x.example/{clip_id}.mp4")
    async with InstagramClient(access_token="tok") as client:
        try:
            await client.publish(
                clip_path=str(clip_path), variant=_variant(), account=account
            )
        except InstagramError as e:
            assert e.transient is False
        else:
            raise AssertionError("expected InstagramError")


async def test_instagram_client_missing_template_is_fatal(tmp_path: Path) -> None:
    """Phase 3 (no S3 yet) requires `oauth_blob.video_url_template`."""
    clip_path = tmp_path / "c.mp4"
    clip_path.write_bytes(b"v")
    account = _account(template=None)
    async with InstagramClient(access_token="tok") as client:
        try:
            await client.publish(
                clip_path=str(clip_path), variant=_variant(), account=account
            )
        except InstagramError as e:
            assert e.transient is False
            assert "video_url_template" in str(e)
        else:
            raise AssertionError("expected InstagramError")


async def test_instagram_client_missing_external_id_is_fatal(tmp_path: Path) -> None:
    clip_path = tmp_path / "c.mp4"
    clip_path.write_bytes(b"v")
    account = ConnectedAccount(
        id="acc",
        tenant_id="t",
        platform="instagram",
        external_id="",  # IG user id missing
        oauth_blob={"access_token": "tok", "video_url_template": "https://x/{clip_id}"},
        created_at=_now(),
    )
    async with InstagramClient(access_token="tok") as client:
        try:
            await client.publish(
                clip_path=str(clip_path), variant=_variant(), account=account
            )
        except InstagramError as e:
            assert e.transient is False
            assert "ig user id" in str(e).lower() or "external_id" in str(e)
        else:
            raise AssertionError("expected InstagramError")


# ---- Dispatcher integration ----


async def _seed_ig_job(db: Database, tmp_path: Path) -> dict[str, str]:
    tenant = await TenantsRepo(db).create(name="Aldo")
    clip_path = tmp_path / "ig_clip.mp4"
    clip_path.write_bytes(b"video-data" * 256)
    with bound_tenant(tenant.id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_ig",
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
                    id="cnd_ig",
                    stream_id="str_ig",
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
                    id="clp_ig",
                    stream_id="str_ig",
                    tenant_id=tenant.id,
                    candidate_id="cnd_ig",
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
            primary_language="en",
            target_languages=["en"],
            voice_prompt="v",
        )
        await VariantsRepo(db).replace_for_clip_persona(
            "clp_ig",
            "aldo",
            [
                VariantRow(
                    id="var_ig",
                    clip_id="clp_ig",
                    tenant_id=tenant.id,
                    persona_id="aldo",
                    language="en",
                    caption="great moment",
                    title_card_text="",
                    hashtags=["reels"],
                    model="m",
                    created_at=_now(),
                )
            ],
        )
        far_future = (_dt.datetime.now(_dt.UTC) + _dt.timedelta(days=30)).isoformat()
        acct = await ConnectedAccountsRepo(db).create(
            platform="instagram",
            external_id="17841400000000000",
            display_name="Aldo IG",
            oauth_blob={
                "access_token": "ig_token",
                "video_url_template": "https://media.example.com/clips/{clip_id}.mp4",
            },
            refresh_token="ig_token",  # Graph long-lived token doubles as the refresh credential
            expires_at=far_future,
        )
        job = await PublishJobsRepo(db).enqueue(
            clip_id="clp_ig",
            variant_id="var_ig",
            account_id=acct.id,
            platform="instagram",
        )
    return {"tenant_id": tenant.id, "account_id": acct.id, "job_id": job.id}


@respx.mock
async def test_run_publish_jobs_routes_instagram_through_native_client(
    db: Database, tmp_path: Path
) -> None:
    seeded = await _seed_ig_job(db, tmp_path)
    ig_user = "17841400000000000"
    respx.post(f"{_IG_BASE}/{ig_user}/media").mock(
        return_value=httpx.Response(200, json={"id": "container_alpha"})
    )
    respx.post(f"{_IG_BASE}/{ig_user}/media_publish").mock(
        return_value=httpx.Response(200, json={"id": "media_omega"})
    )

    async def _no_sleep(_s: float) -> None:
        return None

    outcome = await run_publish_jobs(seeded["tenant_id"], db, sleep=_no_sleep)
    assert outcome.sent == 1

    with bound_tenant(seeded["tenant_id"]):
        jobs = await PublishJobsRepo(db).list_for_clip("clp_ig")
    assert jobs[0].status == "sent"
    assert jobs[0].external_id == "media_omega"
    assert jobs[0].external_url == "https://www.instagram.com/reel/media_omega/"
