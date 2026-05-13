"""Inbox dashboard page — voice-markers spec slice E.3.

Covers:
  * Empty state when there are no clips.
  * VOD heading + clip count per stream.
  * Per-speaker grouping (uses candidate.evidence.speaker_label).
  * Undo strip surfaces scheduled jobs and hides past-window jobs.
  * Cross-tenant isolation: Alice's inbox doesn't show Bob's clips.
"""

from __future__ import annotations

import datetime as _dt

import httpx

from nexoclip.db import (
    BrandKitsRepo,
    CandidatesRepo,
    ClipsRepo,
    ConnectedAccountsRepo,
    Database,
    PublishJobsRepo,
    StreamsRepo,
)
from nexoclip.db.models import (
    CandidateRow,
    ClipRow,
    StreamRow,
)
from nexoclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def _future_iso(*, minutes_from_now: int) -> str:
    return (
        _dt.datetime.now(_dt.UTC) + _dt.timedelta(minutes=minutes_from_now)
    ).isoformat()


async def _seed_stream_with_clip(
    db: Database,
    *,
    tenant_id: str,
    stream_id: str = "str_x",
    clip_id: str = "clp_x",
    speaker_label: str | None = None,
    title: str = "Stream X",
) -> None:
    """Seed stream + candidate (carrying speaker_label) + clip."""
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id,
                tenant_id=tenant_id,
                vod_url="https://kick.com/x",
                platform="kick",
                title=title,
                channel="c",
                duration_s=60.0,
                source_video_path="/tmp/v",
                source_audio_path="/tmp/a",
                status="ingested",
                created_at=_now(),
            )
        )
        ev: dict[str, object] = {}
        if speaker_label is not None:
            ev["speaker_label"] = speaker_label
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id=f"cnd_{clip_id[4:]}",
                    stream_id=stream_id,
                    tenant_id=tenant_id,
                    ts=10.0,
                    score=0.9,
                    reason="voice",
                    evidence=ev,
                    created_at=_now(),
                )
            ]
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id=clip_id,
                    stream_id=stream_id,
                    tenant_id=tenant_id,
                    candidate_id=f"cnd_{clip_id[4:]}",
                    start_s=0.0,
                    end_s=10.0,
                    duration_s=10.0,
                    width=1080,
                    height=1920,
                    path="/tmp/c.mp4",
                    status="cut",
                    created_at=_now(),
                )
            ]
        )


async def _login(client: httpx.AsyncClient, token: str) -> None:
    await client.post("/dashboard/login", data={"token": token})


async def test_inbox_empty_state(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    await _login(client, tenants["alice"]["token"])
    r = await client.get("/dashboard/inbox")
    assert r.status_code == 200
    assert "Inbox" in r.text
    assert "No clips yet" in r.text


async def test_inbox_lists_streams_with_clips(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """A seeded stream + clip shows up under its title."""
    await _login(client, tenants["alice"]["token"])
    await _seed_stream_with_clip(
        db, tenant_id=tenants["alice"]["id"], title="Morning Stream"
    )
    r = await client.get("/dashboard/inbox")
    assert r.status_code == 200
    assert "Morning Stream" in r.text
    # 1 clip
    assert "1 clip" in r.text


async def test_inbox_groups_clips_by_speaker_label(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Two clips with different speaker_labels render as two groups."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_stream_with_clip(
        db,
        tenant_id=tid,
        clip_id="clp_one",
        speaker_label="SPEAKER_00",
        title="Multi",
    )
    # Second clip in the same stream, different speaker.
    with bound_tenant(tid):
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id="cnd_two",
                    stream_id="str_x",
                    tenant_id=tid,
                    ts=20.0,
                    score=0.8,
                    reason="voice",
                    evidence={"speaker_label": "SPEAKER_01"},
                    created_at=_now(),
                )
            ]
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id="clp_two",
                    stream_id="str_x",
                    tenant_id=tid,
                    candidate_id="cnd_two",
                    start_s=15.0,
                    end_s=25.0,
                    duration_s=10.0,
                    width=1080,
                    height=1920,
                    path="/tmp/two.mp4",
                    status="cut",
                    created_at=_now(),
                )
            ]
        )
    r = await client.get("/dashboard/inbox")
    assert r.status_code == 200
    assert "SPEAKER_00" in r.text
    assert "SPEAKER_01" in r.text


async def test_inbox_undo_strip_shows_scheduled_jobs(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """A pending job with scheduled_for in the future appears in the
    undo strip with a working Undo form."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_stream_with_clip(db, tenant_id=tid)
    with bound_tenant(tid):
        # Need a variant + connected account for the FK to land cleanly.
        from nexoclip.db import PersonasRepo, VariantsRepo
        from nexoclip.db.models import VariantRow

        await PersonasRepo(db).create(
            persona_id="p1",
            name="P",
            primary_language="es",
            target_languages=["es"],
            voice_prompt="v",
        )
        await VariantsRepo(db).replace_for_clip_persona(
            "clp_x",
            "p1",
            [
                VariantRow(
                    id="var_x",
                    clip_id="clp_x",
                    tenant_id=tid,
                    persona_id="p1",
                    language="es",
                    caption="c",
                    title_card_text="",
                    hashtags=[],
                    model=None,
                    created_at=_now(),
                )
            ],
        )
        acct = await ConnectedAccountsRepo(db).create(
            platform="tiktok", external_id="t1"
        )
        await PublishJobsRepo(db).enqueue(
            clip_id="clp_x",
            variant_id="var_x",
            account_id=acct.id,
            platform="tiktok",
            scheduled_for=_future_iso(minutes_from_now=45),
        )
    r = await client.get("/dashboard/inbox")
    assert r.status_code == 200
    assert "Undo" in r.text
    assert "tiktok" in r.text
    assert "/publish-jobs/" in r.text
    assert "scheduled to publish" in r.text


async def test_inbox_strips_jobs_past_undo_window(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Once scheduled_for has elapsed, the job leaves the undo strip
    (it's the worker's now)."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_stream_with_clip(db, tenant_id=tid)
    with bound_tenant(tid):
        from nexoclip.db import PersonasRepo, VariantsRepo
        from nexoclip.db.models import VariantRow

        await PersonasRepo(db).create(
            persona_id="p1",
            name="P",
            primary_language="es",
            target_languages=["es"],
            voice_prompt="v",
        )
        await VariantsRepo(db).replace_for_clip_persona(
            "clp_x",
            "p1",
            [
                VariantRow(
                    id="var_x",
                    clip_id="clp_x",
                    tenant_id=tid,
                    persona_id="p1",
                    language="es",
                    caption="c",
                    title_card_text="",
                    hashtags=[],
                    model=None,
                    created_at=_now(),
                )
            ],
        )
        acct = await ConnectedAccountsRepo(db).create(
            platform="tiktok", external_id="t1"
        )
        past = (
            _dt.datetime.now(_dt.UTC) - _dt.timedelta(minutes=5)
        ).isoformat()
        await PublishJobsRepo(db).enqueue(
            clip_id="clp_x",
            variant_id="var_x",
            account_id=acct.id,
            platform="tiktok",
            scheduled_for=past,
        )
    r = await client.get("/dashboard/inbox")
    assert r.status_code == 200
    # The undo strip section header should NOT appear when no jobs are
    # in-window.
    assert "scheduled to publish" not in r.text


async def test_inbox_isolated_per_tenant(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Bob's inbox doesn't show Alice's stream."""
    # Seed Alice
    await _seed_stream_with_clip(
        db,
        tenant_id=tenants["alice"]["id"],
        stream_id="str_alice",
        clip_id="clp_alice",
        title="Alice Secret",
    )
    # Bob has nothing.
    await _login(client, tenants["bob"]["token"])
    r = await client.get("/dashboard/inbox")
    assert r.status_code == 200
    assert "Alice Secret" not in r.text
    assert "No clips yet" in r.text


async def test_inbox_nav_link_renders(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    """The nav bar includes a link to /dashboard/inbox on every page."""
    await _login(client, tenants["alice"]["token"])
    r = await client.get("/dashboard/streams")
    assert r.status_code == 200
    assert '/dashboard/inbox"' in r.text or "/dashboard/inbox'" in r.text
    # Defensive — at minimum the link text 'Inbox' must be present.
    assert ">Inbox<" in r.text


# Used by test_inbox_undo_strip_shows_scheduled_jobs's BrandKit setup
# when the brand-kit table is queried indirectly via /dashboard/inbox.
# Keeping the import here makes it explicit that BrandKitsRepo isn't
# required for the inbox view itself — only for kit-driven auto-publish
# downstream.
_ = BrandKitsRepo
