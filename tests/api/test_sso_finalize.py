"""GET /auth/sso — post-login redirect + external_user_id reconciliation.

Pins the funnel contract with nexo-ai:
  - a fresh SSO login lands on /dashboard/start (the "New clip" hero),
    matching the `next=/dashboard/start` nexo-ai sends on every launch URL
  - `next` is honored only as a same-origin relative path — /auth/sso must
    not be usable as an open redirect
  - the tenant's external_user_id is synced to the signed payload's user_id
    on EVERY login (backfill on NULL *and* overwrite on mismatch), because a
    stale link makes balance fetches read the wrong Nexo AI ledger.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from nexoclip.db import Database, TenantsRepo
from nexoclip.integrations.nexo_ai.sso import sign_sso_token
from nexoclip.settings import get_settings

_SECRET = "sso_shared_secret_value"


@pytest.fixture
def sso_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Strict-mode SSO: set the shared secret + bust the settings cache."""
    monkeypatch.setenv("NEXO_AI_SSO_SECRET", _SECRET)
    get_settings.cache_clear()
    try:
        yield _SECRET
    finally:
        get_settings.cache_clear()


def _token_for(tenant_id: str, *, user_id: str = "nexo_user_1") -> str:
    return sign_sso_token(
        user_id=user_id,
        email="alice@example.com",
        tenant_id=tenant_id,
        secret=_SECRET,
    )


async def test_sso_lands_on_dashboard_start_by_default(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
    sso_env: str,
) -> None:
    token = _token_for(tenants["alice"]["id"])
    r = await client.get(f"/auth/sso?token={token}", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard/start"
    assert "nexoclip_token" in r.headers.get("set-cookie", "")


async def test_sso_honors_relative_next(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
    sso_env: str,
) -> None:
    token = _token_for(tenants["alice"]["id"])
    r = await client.get(
        f"/auth/sso?token={token}&next=/dashboard/streams",
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard/streams"


@pytest.mark.parametrize(
    "evil_next",
    [
        "https://evil.example/phish",
        "//evil.example/phish",
        "/\\evil.example/phish",
        "",
    ],
)
async def test_sso_rejects_non_relative_next(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
    sso_env: str,
    evil_next: str,
) -> None:
    token = _token_for(tenants["alice"]["id"])
    r = await client.get(
        f"/auth/sso?token={token}&next={evil_next}",
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard/start"


async def test_sso_backfills_missing_external_user_id(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    sso_env: str,
) -> None:
    tenant_id = tenants["alice"]["id"]
    before = await TenantsRepo(db).get(tenant_id)
    assert before is not None and not before.external_user_id

    token = _token_for(tenant_id, user_id="nexo_user_fresh")
    r = await client.get(f"/auth/sso?token={token}", follow_redirects=False)
    assert r.status_code == 303

    after = await TenantsRepo(db).get(tenant_id)
    assert after is not None
    assert after.external_user_id == "nexo_user_fresh"


async def test_sso_overwrites_stale_external_user_id(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    sso_env: str,
) -> None:
    """A tenant linked to the WRONG platform user reads someone else's
    token ledger. The signed payload is nexo-ai's word on ownership, so a
    mismatch is corrected on login — not preserved."""
    tenant_id = tenants["alice"]["id"]
    await TenantsRepo(db).set_external_user_id(tenant_id, "nexo_user_stale")

    token = _token_for(tenant_id, user_id="nexo_user_current")
    r = await client.get(f"/auth/sso?token={token}", follow_redirects=False)
    assert r.status_code == 303

    after = await TenantsRepo(db).get(tenant_id)
    assert after is not None
    assert after.external_user_id == "nexo_user_current"
