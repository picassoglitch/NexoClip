"""Instagram 60-day proactive refresh scheduler tests.

The scheduler:
  - picks rows where expires_at < now+14d AND status=active AND
    platform=instagram AND token_type=long_lived
  - re-exchanges the stored long-lived USER token (decrypted from
    refresh_token_encrypted) for a new long-lived token
  - on success, writes the new token + extends expires_at
  - on failure, flips status='auth_failed' so the operator sees
    a red banner on Connect and re-OAuths

Tests pin:
  - happy path refreshes a near-expiry row
  - no-op when expires_at is far in the future
  - mark_status=auth_failed when Meta refuses the token
  - skip rows whose token_type isn't 'long_lived'
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import httpx
import pytest
import respx
from cryptography.fernet import Fernet

from nexoclip.db import (
    ConnectedAccountsRepo,
    Database,
    TenantsRepo,
)
from nexoclip.db.migrations import apply_migrations
from nexoclip.integrations.instagram.refresh import run_instagram_refresh
from nexoclip.integrations.oauth.encryption import (
    TokenEncryptor,
    reset_encryptor_for_tests,
)
from nexoclip.settings import get_settings
from nexoclip.tenancy import bound_tenant


def _iso(dt: _dt.datetime) -> str:
    return dt.isoformat()


@pytest.fixture
def fresh_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Pin a known Fernet key on settings + reset the encryptor
    singleton so each test gets a clean wrapper."""
    key = Fernet.generate_key().decode()
    monkeypatch.setattr(get_settings(), "nexoclip_creds_key", key, raising=False)
    monkeypatch.setattr(get_settings(), "meta_app_id", "appid", raising=False)
    monkeypatch.setattr(get_settings(), "meta_app_secret", "appsecret", raising=False)
    reset_encryptor_for_tests()
    yield key
    reset_encryptor_for_tests()


@pytest.fixture
async def seeded_db(tmp_path: Path, fresh_key: str):
    """Boot an in-memory-style DB with one tenant + a near-expiry
    IG row. Returns (db, tenant_id, account_id, enc)."""
    db = Database(tmp_path / "ig_refresh.db")
    await db.connect()
    await apply_migrations(db)

    enc = TokenEncryptor(fresh_key)

    tenant_id = "ten_test"
    with bound_tenant(tenant_id):
        await TenantsRepo(db).create(tenant_id=tenant_id, name="Test")

        repo = ConnectedAccountsRepo(db)
        # 5 days from now → INSIDE the 14-day refresh window.
        near_expiry = _dt.datetime.now(_dt.UTC) + _dt.timedelta(days=5)
        access_ct = enc.encrypt("page_access_token_v1")
        refresh_ct = enc.encrypt("long_lived_user_token_v1")
        account = await repo.upsert_oauth_connection(
            platform="instagram",
            platform_user_id="ig_999",
            platform_username="alice",
            platform_avatar_url=None,
            access_token_encrypted=access_ct,  # type: ignore[arg-type]
            refresh_token_encrypted=refresh_ct,  # type: ignore[arg-type]
            token_type="long_lived",
            expires_at=_iso(near_expiry),
            scopes=[],
            display_name="Alice",
            access_token_plaintext_mirror="page_access_token_v1",
        )

    yield db, tenant_id, account.id, enc
    await db.close()


@pytest.mark.asyncio
async def test_refresh_extends_near_expiry_row(seeded_db) -> None:
    db, tenant_id, account_id, enc = seeded_db
    body = {
        "access_token": "long_lived_user_token_v2",
        "token_type": "bearer",
        "expires_in": 5_184_000,  # ~60 days
    }
    with respx.mock(assert_all_called=True) as mock:
        mock.get("https://graph.facebook.com/v18.0/oauth/access_token").mock(
            return_value=httpx.Response(200, json=body)
        )
        n = await run_instagram_refresh(tenant_id, db)
    assert n == 1

    with bound_tenant(tenant_id):
        repo = ConnectedAccountsRepo(db)
        updated = await repo.get(account_id)
    assert updated is not None
    assert updated.status == "active"
    assert updated.refresh_token_encrypted is not None
    assert enc.decrypt(updated.refresh_token_encrypted) == "long_lived_user_token_v2"
    new_expires = _dt.datetime.fromisoformat(updated.expires_at)
    # New expiry is well beyond the 14-day window.
    assert new_expires > _dt.datetime.now(_dt.UTC) + _dt.timedelta(days=30)


@pytest.mark.asyncio
async def test_no_op_when_token_is_far_from_expiry(
    tmp_path: Path, fresh_key: str,
) -> None:
    """Row with expires_at > now+14d must NOT be refreshed.
    Wasteful + risks rate-limit if the loop hammers a stable token."""
    db = Database(tmp_path / "ig_far.db")
    await db.connect()
    await apply_migrations(db)

    enc = TokenEncryptor(fresh_key)
    tenant_id = "ten_far"
    far_future = _dt.datetime.now(_dt.UTC) + _dt.timedelta(days=45)
    with bound_tenant(tenant_id):
        await TenantsRepo(db).create(tenant_id=tenant_id, name="Test")
        repo = ConnectedAccountsRepo(db)
        await repo.upsert_oauth_connection(
            platform="instagram",
            platform_user_id="ig_999",
            platform_username="alice",
            platform_avatar_url=None,
            access_token_encrypted=enc.encrypt("pat"),  # type: ignore[arg-type]
            refresh_token_encrypted=enc.encrypt("lt"),  # type: ignore[arg-type]
            token_type="long_lived",
            expires_at=_iso(far_future),
            scopes=[],
            display_name="Alice",
            access_token_plaintext_mirror="pat",
        )

    # No respx mock — if the scheduler tried to hit Meta the test would
    # raise (no transport for graph.facebook.com), which is exactly what
    # we want to assert.
    n = await run_instagram_refresh(tenant_id, db)
    assert n == 0
    await db.close()


@pytest.mark.asyncio
async def test_meta_failure_flips_status_to_auth_failed(seeded_db) -> None:
    db, tenant_id, account_id, _enc = seeded_db
    body = {"error": {"message": "Token revoked by user", "code": 190}}
    with respx.mock() as mock:
        mock.get("https://graph.facebook.com/v18.0/oauth/access_token").mock(
            return_value=httpx.Response(400, json=body)
        )
        n = await run_instagram_refresh(tenant_id, db)
    assert n == 0

    with bound_tenant(tenant_id):
        updated = await ConnectedAccountsRepo(db).get(account_id)
    assert updated is not None
    assert updated.status == "auth_failed"


@pytest.mark.asyncio
async def test_skips_rows_with_wrong_token_type(
    tmp_path: Path, fresh_key: str,
) -> None:
    """A row tagged token_type='bearer' is the classic TikTok/Google
    model — the IG refresh strategy must NOT touch it."""
    db = Database(tmp_path / "ig_wrong_type.db")
    await db.connect()
    await apply_migrations(db)

    enc = TokenEncryptor(fresh_key)
    tenant_id = "ten_wt"
    near_expiry = _dt.datetime.now(_dt.UTC) + _dt.timedelta(days=5)
    with bound_tenant(tenant_id):
        await TenantsRepo(db).create(tenant_id=tenant_id, name="Test")
        repo = ConnectedAccountsRepo(db)
        await repo.upsert_oauth_connection(
            platform="instagram",
            platform_user_id="ig_999",
            platform_username="alice",
            platform_avatar_url=None,
            access_token_encrypted=enc.encrypt("at"),    # type: ignore[arg-type]
            refresh_token_encrypted=enc.encrypt("rt"),   # type: ignore[arg-type]
            token_type="bearer",   # <-- wrong for IG; refresh must skip
            expires_at=_iso(near_expiry),
            scopes=[],
            display_name="Alice",
            access_token_plaintext_mirror="at",
        )

    n = await run_instagram_refresh(tenant_id, db)
    assert n == 0
    await db.close()
