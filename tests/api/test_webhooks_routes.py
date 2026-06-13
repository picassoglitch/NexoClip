"""REST CRUD for /webhooks — create returns secret once, list omits it."""

from __future__ import annotations

import httpx

from .conftest import auth


async def test_create_webhook_returns_secret_once(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    r = await client.post(
        "/webhooks",
        json={"url": "https://example.com/x", "types": ["clip.*"]},
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 201
    body = r.json()
    assert body["url"] == "https://example.com/x"
    assert body["types"] == ["clip.*"]
    assert body["secret"]  # 64 hex chars
    assert len(body["secret"]) == 64
    assert body["status"] == "active"


async def test_list_webhooks_does_not_include_secret(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    r = await client.post(
        "/webhooks",
        json={"url": "https://example.com/x", "types": []},
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 201

    r = await client.get("/webhooks", headers=auth(tenants["alice"]["token"]))
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert "secret" not in rows[0]
    assert rows[0]["url"] == "https://example.com/x"


async def test_webhooks_isolated_per_tenant(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    await client.post(
        "/webhooks",
        json={"url": "https://example.com/a", "types": []},
        headers=auth(tenants["alice"]["token"]),
    )
    r = await client.get("/webhooks", headers=auth(tenants["bob"]["token"]))
    assert r.status_code == 200
    assert r.json() == []


async def test_delete_webhook_returns_204_then_404(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    r = await client.post(
        "/webhooks",
        json={"url": "https://example.com/x", "types": []},
        headers=auth(tenants["alice"]["token"]),
    )
    sub_id = r.json()["id"]
    r2 = await client.delete(
        f"/webhooks/{sub_id}", headers=auth(tenants["alice"]["token"])
    )
    assert r2.status_code == 204
    # Idempotent second call returns 404.
    r3 = await client.delete(
        f"/webhooks/{sub_id}", headers=auth(tenants["alice"]["token"])
    )
    assert r3.status_code == 404
