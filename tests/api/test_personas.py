"""Persona CRUD - list / create / patch."""

from __future__ import annotations

import httpx

from .conftest import auth


async def test_create_then_list_persona(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    r = await client.post(
        "/personas",
        json={
            "id": "aldo",
            "name": "Aldo Villanueva",
            "primary_language": "es",
            "target_languages": ["es", "en"],
            "voice_prompt": "Direct, confrontational",
            "routing_tags": ["mindset"],
        },
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 201
    body = r.json()
    assert body["id"] == "aldo"
    assert body["primary_language"] == "es"

    r = await client.get("/personas", headers=auth(tenants["alice"]["token"]))
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["id"] == "aldo"


async def test_persona_list_isolated_per_tenant(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    await client.post(
        "/personas",
        json={
            "id": "aldo",
            "name": "Aldo",
            "primary_language": "es",
            "voice_prompt": "voice",
        },
        headers=auth(tenants["alice"]["token"]),
    )
    r = await client.get("/personas", headers=auth(tenants["bob"]["token"]))
    assert r.status_code == 200
    assert r.json() == []


async def test_create_duplicate_persona_returns_409(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    payload = {
        "id": "aldo",
        "name": "Aldo",
        "primary_language": "es",
        "voice_prompt": "voice",
    }
    r1 = await client.post("/personas", json=payload, headers=auth(tenants["alice"]["token"]))
    assert r1.status_code == 201
    r2 = await client.post("/personas", json=payload, headers=auth(tenants["alice"]["token"]))
    assert r2.status_code == 409


async def test_patch_persona_updates_fields(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    await client.post(
        "/personas",
        json={
            "id": "aldo",
            "name": "Aldo",
            "primary_language": "es",
            "voice_prompt": "old",
        },
        headers=auth(tenants["alice"]["token"]),
    )
    r = await client.patch(
        "/personas/aldo",
        json={"voice_prompt": "new", "routing_tags": ["entrepreneur"]},
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["voice_prompt"] == "new"
    assert body["routing_tags"] == ["entrepreneur"]
    # Untouched fields preserved.
    assert body["primary_language"] == "es"
    assert body["name"] == "Aldo"


async def test_patch_unknown_persona_returns_404(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    r = await client.patch(
        "/personas/does_not_exist",
        json={"name": "x"},
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 404
