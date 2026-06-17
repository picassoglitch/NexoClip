"""Auto-program + enrichment tests (Publish Center).

Covers the new whole-queue scheduler and the post-text composer:

  * `build_post` stitches viral hook (variant title card) + caption +
    AI hashtags + the tenant's fixed handle/hashtag suffix.
  * POST /compose/{clip_id} returns that composed title + caption.
  * POST /schedule/auto spreads every approved clip across best-time
    slots and ships the ENRICHED content to Zernio (asserted on the
    create_post payloads).
  * tag_suffix round-trips through /autopublish/save + /autopublish.json.

Zernio is respx-mocked; clip rendering is stubbed (the render path is
covered by the render tests, not here).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
import respx

import nexoclip.api._clip_render as _clip_render
from nexoclip.db import (
    AutopublishSettingsRepo,
    CandidatesRepo,
    ClipsRepo,
    Database,
    PersonasRepo,
    StreamsRepo,
    TenantsRepo,
    VariantsRepo,
)
from nexoclip.db.models import CandidateRow, ClipRow, StreamRow, VariantRow
from nexoclip.integrations.nexo_ai.service import sync_tenant_tier
from nexoclip.settings import get_settings
from nexoclip.tenancy import bound_tenant

from .conftest import auth

_ZBASE = "https://zernio.com/api/v1"


@pytest.fixture
def publish_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("NEXOCLIP_ZERNIO_API_KEY", "sk_test_ap2")
    monkeypatch.setenv("NEXOCLIP_INTERNAL_SIGNING_SECRET", "sign_me_ap2")
    get_settings.cache_clear()

    async def _rendered(**_kw: Any) -> None:
        return None

    monkeypatch.setattr(_clip_render, "ensure_clip_rendered", _rendered)
    try:
        yield
    finally:
        get_settings.cache_clear()


async def _seed_clip(
    db: Database, tid: str, *, clip_id: str, persona_id: str,
    caption: str, hook: str, hashtags: list[str], now: str,
) -> None:
    """Approved clip + a single variant carrying caption/hook/hashtags."""
    with bound_tenant(tid):
        await CandidatesRepo(db).upsert_many([
            CandidateRow(
                id=f"cnd_{clip_id}", stream_id="str_ap", tenant_id=tid,
                ts=10.0, score=0.7, reason="voice", evidence={}, created_at=now,
            )
        ])
        await ClipsRepo(db).upsert_many([
            ClipRow(
                id=clip_id, stream_id="str_ap", tenant_id=tid,
                candidate_id=f"cnd_{clip_id}", start_s=0.0, end_s=10.0,
                duration_s=10.0, width=1080, height=1920,
                path="/tmp/c.mp4", status="approved", created_at=now,
            )
        ])
        await VariantsRepo(db).replace_for_clip_persona(
            clip_id, persona_id,
            [VariantRow(
                id=f"var_{clip_id}", clip_id=clip_id, tenant_id=tid,
                persona_id=persona_id, language="es", caption=caption,
                title_card_text=hook, hashtags=hashtags, model="test",
                created_at=now,
            )],
        )


@pytest_asyncio.fixture
async def alice(
    db: Database, tenants: dict[str, dict[str, str]]
) -> dict[str, str]:
    """Alice: all-access, Zernio profile, persona, two approved clips with
    variants (clp_1, clp_2)."""
    tid = tenants["alice"]["id"]
    await sync_tenant_tier(db, tenant_id=tid, tier="all_access")
    await TenantsRepo(db).set_zernio_profile(
        tid, profile_id="prof_alice", profile_name="Alice",
    )
    now = _dt.datetime.now(_dt.UTC).isoformat()
    with bound_tenant(tid):
        await StreamsRepo(db).upsert(StreamRow(
            id="str_ap", tenant_id=tid, vod_url="x", platform="kick",
            title=None, channel=None, duration_s=60.0,
            source_video_path="/tmp/v", source_audio_path="/tmp/a",
            status="ingested", created_at=now,
        ))
        await PersonasRepo(db).create(
            persona_id="psn_ap", name="Voz", primary_language="es",
            target_languages=["es"], voice_prompt="energético",
        )
    await _seed_clip(
        db, tid, clip_id="clp_1", persona_id="psn_ap",
        caption="Cuerpo uno", hook="Hook uno", hashtags=["gaming", "clips"],
        now=now,
    )
    await _seed_clip(
        db, tid, clip_id="clp_2", persona_id="psn_ap",
        caption="Cuerpo dos", hook="Hook dos", hashtags=["stream"], now=now,
    )
    return tenants["alice"]


def _mock_accounts(mock: respx.Router, *platforms: str) -> None:
    mock.get(f"{_ZBASE}/accounts").mock(return_value=httpx.Response(
        200,
        json={"accounts": [
            {"platform": p, "_id": f"acct_{p}", "profileId": "prof_alice"}
            for p in platforms
        ]},
    ))


# ---- composer ----


@pytest.mark.asyncio
async def test_build_post_stitches_hook_body_hashtags_suffix(
    db: Database, alice: dict[str, str]
) -> None:
    from nexoclip.publish.compose import build_post

    with bound_tenant(alice["id"]):
        post = await build_post(
            db, "clp_1", handle_suffix="@minombre #marca",
        )
    assert post.hook == "Hook uno"
    assert post.title == "Hook uno"
    assert post.hashtags == ["#gaming", "#clips"]
    assert post.caption == "Hook uno\n\nCuerpo uno\n\n#gaming #clips\n\n@minombre #marca"


@pytest.mark.asyncio
async def test_build_post_no_variant_is_empty(
    db: Database, alice: dict[str, str]
) -> None:
    from nexoclip.publish.compose import build_post

    with bound_tenant(alice["id"]):
        post = await build_post(db, "clp_missing", handle_suffix="@x")
    # No variant → only the fixed suffix survives; no spurious title.
    assert post.hook == ""
    assert post.title is None
    assert post.caption == "@x"


# ---- /compose/{clip_id} ----


@pytest.mark.asyncio
async def test_compose_endpoint_returns_enriched_text(
    publish_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    await AutopublishSettingsRepo(db).upsert(
        alice["id"], enabled=False, mode="on_approve", targets="tiktok",
        post_mode="queue", daily_cap=10, score_threshold=0.6,
        tag_suffix="@minombre #marca",
    )
    resp = await client.post(
        "/dashboard/publish/zernio/compose/clp_1", headers=auth(alice["token"]),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["title"] == "Hook uno"
    assert body["hashtags"] == ["#gaming", "#clips"]
    assert "Hook uno" in body["caption"]
    assert "@minombre #marca" in body["caption"]


# ---- /schedule/auto ----


@pytest.mark.asyncio
async def test_schedule_auto_schedules_all_approved_with_enrichment(
    publish_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    await AutopublishSettingsRepo(db).upsert(
        alice["id"], enabled=True, mode="on_approve", targets="tiktok",
        post_mode="queue", daily_cap=10, score_threshold=0.6,
        tag_suffix="@minombre #marca",
    )
    posted: list[str] = []

    def _create(request: httpx.Request) -> httpx.Response:
        pid = f"post_auto_{len(posted)}"
        posted.append(pid)
        return httpx.Response(201, json={"success": True, "post": {"_id": pid}})

    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        # No analytics add-on → empty best-time → planner uses fallback hours.
        mock.get(f"{_ZBASE}/analytics/best-time").mock(
            return_value=httpx.Response(403, json={"error": "addon"})
        )
        post_route = mock.post(f"{_ZBASE}/posts").mock(side_effect=_create)
        resp = await client.post(
            "/dashboard/publish/zernio/schedule/auto", headers=auth(alice["token"]),
        )
        # The run now executes in the background; the POST returns immediately.
        assert resp.status_code == 200, resp.text
        assert resp.json()["state"] == "running"
        # Poll progress until the background task finishes — keep the respx
        # mock open so the task's POST /posts calls hit it.
        prog: dict[str, Any] = {}
        for _ in range(500):
            prog = (
                await client.get(
                    "/dashboard/publish/zernio/schedule/auto/progress",
                    headers=auth(alice["token"]),
                )
            ).json()
            if prog.get("state") in ("done", "error"):
                break
            await asyncio.sleep(0.02)

    assert prog.get("state") == "done", prog
    assert prog["scheduled"] == 2
    assert prog["failed"] == 0
    assert post_route.call_count == 2

    # Every scheduled post carries a future scheduledFor and the enriched
    # caption (hook + body + hashtags + fixed handle suffix).
    payloads = [json.loads(c.request.content.decode()) for c in post_route.calls]
    for p in payloads:
        assert p.get("scheduledFor")
        assert p["timezone"] == "UTC"
        assert "publishNow" not in p
    contents = {p["content"] for p in payloads}
    assert any("Hook uno" in c and "@minombre #marca" in c for c in contents)
    assert any("Hook dos" in c for c in contents)


@pytest.mark.asyncio
async def test_schedule_auto_no_clips_is_clean_ok(
    publish_env: None, client: httpx.AsyncClient, db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    tid = tenants["alice"]["id"]
    await sync_tenant_tier(db, tenant_id=tid, tier="all_access")
    await TenantsRepo(db).set_zernio_profile(tid, profile_id="prof_alice", profile_name="A")
    resp = await client.post(
        "/dashboard/publish/zernio/schedule/auto", headers=auth(tenants["alice"]["token"]),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["scheduled"] == 0


# ---- hands-free best-time spread ----


@pytest.mark.asyncio
async def test_handsfree_queue_spreads_and_enriches(
    publish_env: None, db: Database, alice: dict[str, str]
) -> None:
    """A finished VOD's clips, in hands_free + queue mode, schedule across
    best-time slots (scheduledFor set, not immediate) with enriched text."""
    from nexoclip.api.routers.zernio import autopublish_hands_free_sweep

    await AutopublishSettingsRepo(db).upsert(
        alice["id"], enabled=True, mode="hands_free", targets="tiktok",
        post_mode="queue", daily_cap=10, score_threshold=0.5,
        tag_suffix="@minombre #marca",
    )
    posted: list[str] = []

    def _create(request: httpx.Request) -> httpx.Response:
        pid = f"post_hf_{len(posted)}"
        posted.append(pid)
        return httpx.Response(201, json={"success": True, "post": {"_id": pid}})

    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        mock.get(f"{_ZBASE}/analytics/best-time").mock(
            return_value=httpx.Response(403, json={"error": "addon"})
        )
        post_route = mock.post(f"{_ZBASE}/posts").mock(side_effect=_create)
        n = await autopublish_hands_free_sweep(
            db=db, tenant_id=alice["id"], base_url="https://x.test",
            clip_scores=[("clp_1", 0.9), ("clp_2", 0.8)],
        )

    assert n == 2
    assert post_route.call_count == 2
    payloads = [json.loads(c.request.content.decode()) for c in post_route.calls]
    for p in payloads:
        assert p.get("scheduledFor")  # queue mode → spread, not immediate
        assert p.get("publishNow") in (None, False)
    assert any("@minombre #marca" in p["content"] for p in payloads)


@pytest.mark.asyncio
async def test_handsfree_now_mode_posts_immediately(
    publish_env: None, db: Database, alice: dict[str, str]
) -> None:
    """now mode keeps firing immediately — no scheduledFor."""
    from nexoclip.api.routers.zernio import autopublish_hands_free_sweep

    await AutopublishSettingsRepo(db).upsert(
        alice["id"], enabled=True, mode="hands_free", targets="tiktok",
        post_mode="now", daily_cap=10, score_threshold=0.5, tag_suffix="",
    )
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        post_route = mock.post(f"{_ZBASE}/posts").mock(
            side_effect=lambda r: httpx.Response(
                201, json={"success": True, "post": {"_id": "post_now"}}
            )
        )
        n = await autopublish_hands_free_sweep(
            db=db, tenant_id=alice["id"], base_url="https://x.test",
            clip_scores=[("clp_1", 0.9)],
        )
    assert n == 1
    payload = json.loads(post_route.calls.last.request.content.decode())
    assert payload["publishNow"] is True
    assert "scheduledFor" not in payload


@pytest.mark.asyncio
async def test_handsfree_empty_targets_falls_back_to_all_connected(
    publish_env: None, db: Database, alice: dict[str, str]
) -> None:
    """Regression: hands_free with NO target platforms picked must post to all
    connected accounts (it used to silently no-op on empty targets, so
    channel-auto clips never published)."""
    from nexoclip.api.routers.zernio import autopublish_hands_free_sweep

    await AutopublishSettingsRepo(db).upsert(
        alice["id"], enabled=True, mode="hands_free", targets="",  # EMPTY
        post_mode="now", daily_cap=10, score_threshold=0.5, tag_suffix="",
    )
    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok", "youtube")
        post_route = mock.post(f"{_ZBASE}/posts").mock(
            side_effect=lambda r: httpx.Response(
                201, json={"success": True, "post": {"_id": "post_ft"}}
            )
        )
        n = await autopublish_hands_free_sweep(
            db=db, tenant_id=alice["id"], base_url="https://x.test",
            clip_scores=[("clp_1", 0.9)],
        )
    # Old behavior: n == 0 (no_targets). Now it posts to the connected accounts.
    assert n == 1
    assert post_route.call_count == 1


# ---- tag_suffix round-trip ----


@pytest.mark.asyncio
async def test_tag_suffix_roundtrips_through_save_and_json(
    publish_env: None, client: httpx.AsyncClient, db: Database, alice: dict[str, str]
) -> None:
    resp = await client.post(
        "/dashboard/publish/zernio/autopublish/save",
        json={
            "enabled": True, "mode": "on_approve", "targets": ["tiktok"],
            "post_mode": "queue", "daily_cap": 7,
            "tag_suffix": "  @minombre #marca  ",
        },
        headers=auth(alice["token"]),
    )
    assert resp.status_code == 200
    s = await AutopublishSettingsRepo(db).get(alice["id"])
    assert s is not None
    assert s["tag_suffix"] == "@minombre #marca"  # trimmed

    with respx.mock() as mock:
        _mock_accounts(mock, "tiktok")
        got = await client.get(
            "/dashboard/publish/zernio/autopublish.json", headers=auth(alice["token"]),
        )
    assert got.status_code == 200
    assert got.json()["settings"]["tag_suffix"] == "@minombre #marca"
