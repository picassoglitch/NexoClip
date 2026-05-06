"""Clip detail, status updates, and publish-job enqueue."""

from __future__ import annotations

import datetime as _dt

import httpx

from nexoclip.db import (
    ClipsRepo,
    ConnectedAccountsRepo,
    Database,
    PublishJobsRepo,
    StreamsRepo,
    VariantsRepo,
)
from nexoclip.db.models import (
    CandidateRow,
    ClipRow,
    StreamRow,
    VariantRow,
)
from nexoclip.tenancy import bound_tenant

from .conftest import auth


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed(
    db: Database,
    *,
    tenant_id: str,
    stream_id: str,
    candidate_id: str,
    clip_id: str,
    persona_id: str,
    variant_id: str,
) -> None:
    """Seed the full chain so publish/clip endpoints have something to find."""
    from nexoclip.db import CandidatesRepo, PersonasRepo

    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id,
                tenant_id=tenant_id,
                vod_url="https://example.com/x",
                platform="kick",
                title="t",
                channel="c",
                duration_s=300.0,
                source_video_path="/tmp/x.mp4",
                source_audio_path="/tmp/x.wav",
                status="ingested",
                created_at=_now(),
            )
        )
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id=candidate_id,
                    stream_id=stream_id,
                    tenant_id=tenant_id,
                    ts=10.0,
                    score=0.9,
                    reason="voice",
                    evidence={},
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
                    candidate_id=candidate_id,
                    start_s=10.0,
                    end_s=55.0,
                    duration_s=45.0,
                    width=1080,
                    height=1920,
                    path=f"/tmp/{clip_id}.mp4",
                    status="cut",
                    created_at=_now(),
                )
            ]
        )
        await PersonasRepo(db).create(
            persona_id=persona_id,
            name="Aldo",
            primary_language="es",
            target_languages=["es", "en"],
            voice_prompt="direct",
        )
        await VariantsRepo(db).replace_for_clip_persona(
            clip_id,
            persona_id,
            [
                VariantRow(
                    id=variant_id,
                    clip_id=clip_id,
                    tenant_id=tenant_id,
                    persona_id=persona_id,
                    language="es",
                    caption="hello",
                    title_card_text="",
                    hashtags=["clip"],
                    model="claude-haiku-4-5-20251001",
                    created_at=_now(),
                )
            ],
        )


async def test_get_clip_returns_detail(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    await _seed(
        db,
        tenant_id=tenants["alice"]["id"],
        stream_id="str_a",
        candidate_id="cnd_a",
        clip_id="clp_a",
        persona_id="aldo",
        variant_id="var_a",
    )
    r = await client.get("/clips/clp_a", headers=auth(tenants["alice"]["token"]))
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "clp_a"
    assert body["status"] == "cut"


async def test_get_other_tenants_clip_returns_404(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    await _seed(
        db,
        tenant_id=tenants["alice"]["id"],
        stream_id="str_a",
        candidate_id="cnd_a",
        clip_id="clp_a",
        persona_id="aldo",
        variant_id="var_a",
    )
    r = await client.get("/clips/clp_a", headers=auth(tenants["bob"]["token"]))
    assert r.status_code == 404


async def test_patch_clip_transitions_status(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    await _seed(
        db,
        tenant_id=tenants["alice"]["id"],
        stream_id="str_a",
        candidate_id="cnd_a",
        clip_id="clp_a",
        persona_id="aldo",
        variant_id="var_a",
    )
    r = await client.patch(
        "/clips/clp_a",
        json={"status": "ready_for_review"},
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 200
    assert r.json()["status"] == "ready_for_review"

    # cut -> approved is NOT in the transition map -> 409.
    await _reset_status(db, tenants["alice"]["id"], "clp_a", "cut")
    r = await client.patch(
        "/clips/clp_a",
        json={"status": "approved"},
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 409


async def _reset_status(db: Database, tenant_id: str, clip_id: str, target: str) -> None:
    conn = await db.connect()
    await conn.execute(
        "UPDATE clips SET status = ? WHERE id = ? AND tenant_id = ?",
        (target, clip_id, tenant_id),
    )
    await conn.commit()


async def test_publish_clip_enqueues_one_job_per_account(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    tenant_id = tenants["alice"]["id"]
    await _seed(
        db,
        tenant_id=tenant_id,
        stream_id="str_a",
        candidate_id="cnd_a",
        clip_id="clp_a",
        persona_id="aldo",
        variant_id="var_a",
    )
    # Connect two Buffer accounts.
    with bound_tenant(tenant_id):
        await ConnectedAccountsRepo(db).create(
            platform="buffer", external_id="buf_one", display_name="Main"
        )
        await ConnectedAccountsRepo(db).create(
            platform="buffer", external_id="buf_two", display_name="Backup"
        )

    r = await client.post(
        "/clips/clp_a/publish",
        json={"variant_id": "var_a"},
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 202
    jobs = r.json()
    assert len(jobs) == 2
    assert {j["platform"] for j in jobs} == {"buffer"}
    assert all(j["status"] == "pending" for j in jobs)

    with bound_tenant(tenant_id):
        rows = await PublishJobsRepo(db).list_for_clip("clp_a")
    assert len(rows) == 2


async def test_publish_returns_409_when_no_accounts(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    await _seed(
        db,
        tenant_id=tenants["alice"]["id"],
        stream_id="str_a",
        candidate_id="cnd_a",
        clip_id="clp_a",
        persona_id="aldo",
        variant_id="var_a",
    )
    r = await client.post(
        "/clips/clp_a/publish",
        json={"variant_id": "var_a"},
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 409


async def test_publish_404_for_unknown_variant(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]]
) -> None:
    tenant_id = tenants["alice"]["id"]
    await _seed(
        db,
        tenant_id=tenant_id,
        stream_id="str_a",
        candidate_id="cnd_a",
        clip_id="clp_a",
        persona_id="aldo",
        variant_id="var_a",
    )
    with bound_tenant(tenant_id):
        await ConnectedAccountsRepo(db).create(
            platform="buffer", external_id="buf", display_name="m"
        )
    r = await client.post(
        "/clips/clp_a/publish",
        json={"variant_id": "var_does_not_exist"},
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 404
