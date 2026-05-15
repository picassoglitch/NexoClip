"""Per-clip overlay editor (slice F.6).

End-to-end through the dashboard:
  * GET /dashboard/clips/{id} renders the editor with the live-preview
    surface + control panels.
  * POST /dashboard/clips/{id}/overlay saves the overlay config but
    leaves status untouched.
  * POST /dashboard/clips/{id}/finalize saves AND walks the status to
    'approved' (the existing pre-publish standby).
  * Per-tenant isolation: Bob can't touch Alice's clip.
  * Idempotence: re-saving overwrites cleanly.
  * Status validation: a `published` clip can't be re-finalized.
"""

from __future__ import annotations

import datetime as _dt

import httpx

from nexoclip.db import (
    ClipsRepo,
    Database,
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


async def _seed_clip(
    db: Database,
    *,
    tenant_id: str,
    stream_id: str = "str_e",
    clip_id: str = "clp_e",
    status: str = "cut",
) -> None:
    from nexoclip.db import CandidatesRepo

    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id,
                tenant_id=tenant_id,
                vod_url="https://kick.com/x",
                platform="kick",
                title="t",
                channel="c",
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
                    id="cnd_e",
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
                    candidate_id="cnd_e",
                    start_s=0.0,
                    end_s=10.0,
                    duration_s=10.0,
                    width=1080,
                    height=1920,
                    path="/tmp/c.mp4",
                    status=status,
                    created_at=_now(),
                )
            ]
        )


async def _login(client: httpx.AsyncClient, token: str) -> None:
    await client.post("/dashboard/login", data={"token": token})


# ---- Page render -------------------------------------------------


async def test_clip_editor_page_renders_live_preview_surface(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """The editor page renders with the preview <video>, the four
    control panels, and the Complete + Save-draft buttons."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    r = await client.get("/dashboard/clips/clp_e")
    assert r.status_code == 200
    body = r.text

    # Preview surface + the four overlay layers.
    assert 'id="preview"' in body
    assert 'id="pv-title"' in body
    assert 'id="pv-banner"' in body
    assert 'id="pv-caption"' in body
    assert 'id="pv-rail"' in body

    # The four control panels.
    assert 'name="title_text"' in body
    assert 'name="banner_enabled"' in body
    assert 'name="banner_platform"' in body
    assert 'name="banner_url"' in body
    assert 'name="captions_preset"' in body
    assert 'name="captions_highlight_color"' in body
    assert 'name="comments_show"' in body

    # The two action buttons live on the same form via formaction.
    assert "Complete &amp; stage for publish" in body
    assert "Save draft" in body
    assert "/dashboard/clips/clp_e/overlay" in body
    assert "/dashboard/clips/clp_e/finalize" in body


async def test_clip_editor_pre_populates_from_saved_overlay(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """A previously-saved overlay config populates the form on next render."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    with bound_tenant(tid):
        await ClipsRepo(db).set_overlay_config(
            "clp_e",
            overlay_config={
                "title_text": "Adin Ross asked Clav",
                "banner": {
                    "enabled": True,
                    "platform": "kick",
                    "url": "kick.com/clavicular",
                    "color": "#53FC18",
                },
                "captions": {
                    "enabled": True,
                    "preset": "bold_block",
                    "highlight_color": "#FF00FF",
                },
                "comments": {"show_overlay": True, "fake_likes": 42},
            },
        )
    r = await client.get("/dashboard/clips/clp_e")
    assert r.status_code == 200
    body = r.text
    assert "Adin Ross asked Clav" in body
    assert "kick.com/clavicular" in body
    assert "#FF00FF" in body
    assert "42" in body  # fake_likes


# ---- Save (draft) ----------------------------------------------


async def test_overlay_save_persists_config_only(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """POST /clips/{id}/overlay writes the config but leaves status
    where it was."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid, status="cut")
    r = await client.post(
        "/dashboard/clips/clp_e/overlay",
        data={
            "title_text": "Hello world",
            "banner_enabled": "1",
            "banner_platform": "twitch",
            "banner_url": "twitch.tv/foo",
            "banner_color": "#9146FF",
            "captions_enabled": "1",
            "captions_preset": "karaoke_pop",
            "captions_highlight_color": "#FFD700",
            "comments_show": "1",
            "comments_fake_likes": "100",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard/clips/clp_e"
    with bound_tenant(tid):
        clip = await ClipsRepo(db).get("clp_e")
    assert clip is not None
    assert clip.status == "cut"  # unchanged
    cfg = clip.overlay_config
    assert cfg is not None
    assert cfg["title_text"] == "Hello world"
    banner = cfg["banner"]
    assert isinstance(banner, dict)
    assert banner["enabled"] is True
    assert banner["platform"] == "twitch"
    assert banner["url"] == "twitch.tv/foo"
    captions = cfg["captions"]
    assert isinstance(captions, dict)
    assert captions["preset"] == "karaoke_pop"
    comments = cfg["comments"]
    assert isinstance(comments, dict)
    assert comments["fake_likes"] == 100


async def test_overlay_save_blank_title_clears_to_none(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """A blank title_text form field collapses to None in storage —
    the editor is additive, not destructive."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid)
    r = await client.post(
        "/dashboard/clips/clp_e/overlay",
        data={
            "title_text": "",
            "banner_enabled": "",
            "banner_platform": "kick",
            "banner_url": "",
            "banner_color": "",
            "captions_enabled": "1",
            "captions_preset": "",
            "captions_highlight_color": "",
            "comments_show": "",
            "comments_fake_likes": "0",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    with bound_tenant(tid):
        clip = await ClipsRepo(db).get("clp_e")
    assert clip is not None
    cfg = clip.overlay_config
    assert cfg is not None
    assert cfg["title_text"] is None


async def test_overlay_save_404_for_unknown_clip(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
) -> None:
    await _login(client, tenants["alice"]["token"])
    r = await client.post(
        "/dashboard/clips/clp_nope/overlay",
        data={"title_text": "x"},
        follow_redirects=False,
    )
    assert r.status_code == 404


# ---- Finalize ---------------------------------------------------


async def test_finalize_walks_cut_to_approved(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """A `cut` clip → `ready_for_review` → `approved` in one POST."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid, status="cut")
    r = await client.post(
        "/dashboard/clips/clp_e/finalize",
        data={
            "title_text": "Final",
            "banner_enabled": "1",
            "banner_platform": "kick",
            "banner_url": "kick.com/me",
            "banner_color": "#53FC18",
            "captions_enabled": "1",
            "captions_preset": "",
            "captions_highlight_color": "#FFD700",
            "comments_show": "1",
            "comments_fake_likes": "8",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    with bound_tenant(tid):
        clip = await ClipsRepo(db).get("clp_e")
    assert clip is not None
    assert clip.status == "approved"
    assert clip.overlay_config is not None
    assert clip.overlay_config["title_text"] == "Final"


async def test_finalize_from_ready_for_review_goes_to_approved(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid, status="ready_for_review")
    r = await client.post(
        "/dashboard/clips/clp_e/finalize",
        data={
            "title_text": "x",
            "banner_enabled": "",
            "banner_platform": "kick",
            "banner_url": "",
            "banner_color": "",
            "captions_enabled": "1",
            "captions_preset": "",
            "captions_highlight_color": "#FFD700",
            "comments_show": "",
            "comments_fake_likes": "0",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    with bound_tenant(tid):
        clip = await ClipsRepo(db).get("clp_e")
    assert clip is not None
    assert clip.status == "approved"


async def test_finalize_idempotent_when_already_approved(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Re-finalizing an already-approved clip overwrites the overlay
    but leaves the status alone (no 409)."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid, status="approved")
    r = await client.post(
        "/dashboard/clips/clp_e/finalize",
        data={
            "title_text": "v2",
            "banner_enabled": "",
            "banner_platform": "kick",
            "banner_url": "",
            "banner_color": "",
            "captions_enabled": "1",
            "captions_preset": "",
            "captions_highlight_color": "#FFD700",
            "comments_show": "",
            "comments_fake_likes": "0",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    with bound_tenant(tid):
        clip = await ClipsRepo(db).get("clp_e")
    assert clip is not None
    assert clip.status == "approved"
    assert clip.overlay_config is not None
    assert clip.overlay_config["title_text"] == "v2"


async def test_finalize_409_on_published_clip(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """A `published` clip is terminal — can't be re-finalized."""
    tid = tenants["alice"]["id"]
    await _login(client, tenants["alice"]["token"])
    await _seed_clip(db, tenant_id=tid, status="published")
    r = await client.post(
        "/dashboard/clips/clp_e/finalize",
        data={
            "title_text": "x",
            "banner_enabled": "",
            "banner_platform": "kick",
            "banner_url": "",
            "banner_color": "",
            "captions_enabled": "1",
            "captions_preset": "",
            "captions_highlight_color": "#FFD700",
            "comments_show": "",
            "comments_fake_likes": "0",
        },
        follow_redirects=False,
    )
    assert r.status_code == 409


async def test_finalize_404_for_unknown_clip(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
) -> None:
    await _login(client, tenants["alice"]["token"])
    r = await client.post(
        "/dashboard/clips/clp_nope/finalize",
        data={"title_text": "x"},
        follow_redirects=False,
    )
    assert r.status_code == 404


# ---- Tenant isolation ------------------------------------------


async def test_clip_row_loader_tolerates_unknown_columns(
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Defense-in-depth regression: if a future migration adds a
    column to `clips` that the ClipRow Pydantic model doesn't yet
    declare, `_clip_from_row` must still load existing rows. Without
    this defense an out-of-sync deploy (DB ahead of code) crashes
    every dashboard page that reads a clip.

    We simulate this by injecting a fake column directly via SQL,
    then verifying the loader strips it without raising.
    """
    from nexoclip.db.repos import _clip_from_row

    tid = tenants["alice"]["id"]
    await _seed_clip(db, tenant_id=tid)
    conn = await db.connect()
    # Add a hypothetical future column. SQLite ALTER TABLE is cheap.
    await conn.execute("ALTER TABLE clips ADD COLUMN imaginary_v10_col TEXT")
    await conn.execute(
        "UPDATE clips SET imaginary_v10_col = ? WHERE id = ?",
        ("future-data", "clp_e"),
    )
    await conn.commit()
    cur = await conn.execute(
        "SELECT * FROM clips WHERE id = ?",
        ("clp_e",),
    )
    row = await cur.fetchone()
    assert row is not None
    # The unknown column is in the row dict, but the loader filters
    # it out before model_validate — no ValidationError raised.
    clip = _clip_from_row(row)
    assert clip.id == "clp_e"


async def test_overlay_endpoints_isolated_per_tenant(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Bob can't save / finalize Alice's clip — both endpoints 404."""
    await _seed_clip(db, tenant_id=tenants["alice"]["id"])
    await _login(client, tenants["bob"]["token"])
    for path in (
        "/dashboard/clips/clp_e/overlay",
        "/dashboard/clips/clp_e/finalize",
    ):
        r = await client.post(
            path,
            data={"title_text": "hijack"},
            follow_redirects=False,
        )
        assert r.status_code == 404, f"{path} returned {r.status_code}"
    # And Alice's clip is untouched.
    with bound_tenant(tenants["alice"]["id"]):
        clip = await ClipsRepo(db).get("clp_e")
    assert clip is not None
    assert clip.overlay_config is None
