"""Dashboard brand-kit CRUD pages (slice C.3).

Each test logs in via the cookie flow set by the `tenants` fixture and
hits the routes through httpx + ASGITransport. End-to-end assertion is
'POST does what it says on the tin' — the underlying repo is covered by
tests/branding/test_brand_kits.py.
"""

from __future__ import annotations

import httpx
import pytest

from nexoclip.db import BrandKitsRepo, Database
from nexoclip.tenancy import bound_tenant


async def _login(client: httpx.AsyncClient, token: str) -> None:
    await client.post("/dashboard/login", data={"token": token})


async def test_list_page_renders_empty_state(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    await _login(client, tenants["alice"]["token"])
    r = await client.get("/dashboard/brand-kits")
    assert r.status_code == 200
    assert "Brand kits" in r.text
    assert "No brand kits yet" in r.text


async def test_create_form_renders(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    await _login(client, tenants["alice"]["token"])
    r = await client.get("/dashboard/brand-kits/new")
    assert r.status_code == 200
    assert "New brand kit" in r.text
    assert 'name="primary_color"' in r.text
    assert 'name="forward_phrases"' in r.text


async def test_create_then_list_then_edit(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Three-step end-to-end: POST /brand-kits → row exists → list shows it
    → GET edit form renders prepopulated."""
    await _login(client, tenants["alice"]["token"])
    r = await client.post(
        "/dashboard/brand-kits",
        data={
            "name": "AARA",
            "primary_color": "#FF3366",
            "accent_color": "#FFD700",
            "text_color": "#FFFFFF",
            "font_family": "Inter",
            "font_weight": "800",
            "default_layout": "pip",
            "is_default": "1",
            "handle_tiktok": "@aara_art",
            "auto_publish_enabled": "",
            "auto_publish_platforms": "",
            "auto_publish_delay_min": "60",
            "forward_phrases": "córtalo\nclip this",
            "retroactive_phrases": "lo clipeaste",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    # Redirect goes to /dashboard/brand-kits/<id>
    assert r.headers["location"].startswith("/dashboard/brand-kits/brk_")
    kit_id = r.headers["location"].rsplit("/", 1)[1]

    # Repo round-trip: row exists for Alice, isolated from Bob.
    with bound_tenant(tenants["alice"]["id"]):
        kit = await BrandKitsRepo(db).get(kit_id)
    assert kit is not None
    assert kit.name == "AARA"
    assert kit.is_default is True
    assert kit.handle_tiktok == "@aara_art"
    assert kit.custom_trigger_phrases.forward == ["córtalo", "clip this"]
    assert kit.custom_trigger_phrases.retroactive == ["lo clipeaste"]

    # List page shows it (name + the trigger-phrase counter the list
    # template renders inline).
    r = await client.get("/dashboard/brand-kits")
    assert r.status_code == 200
    assert "AARA" in r.text
    assert "+2 fwd" in r.text   # 'córtalo' + 'clip this'
    assert "+1 retro" in r.text  # 'lo clipeaste'

    # Edit form renders prepopulated — checks both the name and one of
    # the custom phrases round-tripped into the textarea.
    r = await client.get(f"/dashboard/brand-kits/{kit_id}")
    assert r.status_code == 200
    assert "AARA" in r.text
    assert "@aara_art" in r.text   # handle field is on the edit form
    assert "córtalo" in r.text


async def test_update_via_form(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """POST /brand-kits/{id} updates the row."""
    await _login(client, tenants["alice"]["token"])
    with bound_tenant(tenants["alice"]["id"]):
        kit = await BrandKitsRepo(db).create(
            name="Original",
            primary_color="#000000",
            accent_color="#FFFFFF",
        )
    r = await client.post(
        f"/dashboard/brand-kits/{kit.id}",
        data={
            "name": "Renamed",
            "primary_color": "#FF0000",
            "accent_color": "#00FF00",
            "text_color": "#FFFFFF",
            "font_family": "Roboto",
            "font_weight": "700",
            "default_layout": "split_stack",
            "auto_publish_enabled": "1",
            "auto_publish_platforms": "tiktok, shorts",
            "auto_publish_delay_min": "30",
            "forward_phrases": "monchi eso",
            "retroactive_phrases": "",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    with bound_tenant(tenants["alice"]["id"]):
        updated = await BrandKitsRepo(db).get(kit.id)
    assert updated is not None
    assert updated.name == "Renamed"
    assert updated.primary_color == "#FF0000"
    assert updated.font_family == "Roboto"
    assert updated.default_layout == "split_stack"
    assert updated.auto_publish_enabled is True
    assert updated.auto_publish_platforms == ["tiktok", "shorts"]
    assert updated.auto_publish_delay_min == 30
    assert updated.custom_trigger_phrases.forward == ["monchi eso"]
    assert updated.custom_trigger_phrases.retroactive == []


async def test_set_default_demotes_existing(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """POST /brand-kits/{id}/default flips the default flag atomically."""
    await _login(client, tenants["alice"]["token"])
    with bound_tenant(tenants["alice"]["id"]):
        a = await BrandKitsRepo(db).create(
            name="A", primary_color="#000", accent_color="#FFF", is_default=True
        )
        b = await BrandKitsRepo(db).create(
            name="B", primary_color="#111", accent_color="#EEE"
        )
    r = await client.post(
        f"/dashboard/brand-kits/{b.id}/default", follow_redirects=False
    )
    assert r.status_code == 303
    with bound_tenant(tenants["alice"]["id"]):
        a_after = await BrandKitsRepo(db).get(a.id)
        b_after = await BrandKitsRepo(db).get(b.id)
    assert a_after is not None and a_after.is_default is False
    assert b_after is not None and b_after.is_default is True


async def test_delete_removes_kit(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    await _login(client, tenants["alice"]["token"])
    with bound_tenant(tenants["alice"]["id"]):
        kit = await BrandKitsRepo(db).create(
            name="Delete me", primary_color="#000", accent_color="#FFF"
        )
    r = await client.post(
        f"/dashboard/brand-kits/{kit.id}/delete", follow_redirects=False
    )
    assert r.status_code == 303
    with bound_tenant(tenants["alice"]["id"]):
        gone = await BrandKitsRepo(db).get(kit.id)
    assert gone is None


async def test_other_tenants_kit_404(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Bob can't edit Alice's brand kit even with a guessed id."""
    with bound_tenant(tenants["alice"]["id"]):
        kit = await BrandKitsRepo(db).create(
            name="Alice's secret",
            primary_color="#000",
            accent_color="#FFF",
        )
    await _login(client, tenants["bob"]["token"])
    r = await client.get(f"/dashboard/brand-kits/{kit.id}")
    assert r.status_code == 404


async def test_set_default_unknown_kit_404(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
) -> None:
    await _login(client, tenants["alice"]["token"])
    r = await client.post(
        "/dashboard/brand-kits/brk_nonexistent/default", follow_redirects=False
    )
    assert r.status_code == 404


async def test_caption_preset_persists_through_form(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Slice D.2: picking a preset in the form persists the full
    CaptionStyle dict to brand_kits.caption_style_json, so the renderer
    has every field it needs without re-resolving from the preset id."""
    await _login(client, tenants["alice"]["token"])
    with bound_tenant(tenants["alice"]["id"]):
        kit = await BrandKitsRepo(db).create(
            name="K", primary_color="#000", accent_color="#FFF"
        )
    r = await client.post(
        f"/dashboard/brand-kits/{kit.id}",
        data={
            "name": "K",
            "primary_color": "#000000",
            "accent_color": "#FFFFFF",
            "text_color": "#FFFFFF",
            "font_family": "Inter",
            "font_weight": "800",
            "default_layout": "pip",
            "caption_preset": "typewriter",
            "auto_publish_enabled": "",
            "auto_publish_platforms": "",
            "auto_publish_delay_min": "60",
            "forward_phrases": "",
            "retroactive_phrases": "",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    with bound_tenant(tenants["alice"]["id"]):
        updated = await BrandKitsRepo(db).get(kit.id)
    assert updated is not None
    assert updated.caption_style is not None
    assert updated.caption_style["preset"] == "typewriter"
    # Field that's distinctive to the typewriter preset:
    assert updated.caption_style["font_family"] == "JetBrains Mono"


async def test_create_form_uses_default_preset_when_unspecified(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """If the user doesn't pick a preset, the create endpoint falls back
    to karaoke_pop (the spec default)."""
    await _login(client, tenants["alice"]["token"])
    r = await client.post(
        "/dashboard/brand-kits",
        data={
            "name": "Defaults",
            "primary_color": "#FF0000",
            "accent_color": "#00FF00",
            "text_color": "#FFFFFF",
            "font_family": "Inter",
            "font_weight": "800",
            "default_layout": "pip",
            "auto_publish_enabled": "",
            "auto_publish_platforms": "",
            "auto_publish_delay_min": "60",
            "forward_phrases": "",
            "retroactive_phrases": "",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    kit_id = r.headers["location"].rsplit("/", 1)[1]
    with bound_tenant(tenants["alice"]["id"]):
        kit = await BrandKitsRepo(db).get(kit_id)
    assert kit is not None
    assert kit.caption_style is not None
    assert kit.caption_style["preset"] == "karaoke_pop"
