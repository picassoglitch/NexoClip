"""OAuth refresh — refresh-if-expiring + auth_failed flip on refresh failure."""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest_asyncio

from nexoclip.db import (
    ConnectedAccountsRepo,
    Database,
    EventsRepo,
    TenantsRepo,
    apply_migrations,
)
from nexoclip.publish.oauth import RefreshedToken, refresh_if_expiring
from nexoclip.tenancy import bound_tenant


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "oauth.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


async def _make_account(
    db: Database, *, tenant_id: str, expires_at: str | None
) -> str:
    with bound_tenant(tenant_id):
        acct = await ConnectedAccountsRepo(db).create(
            platform="tiktok",
            external_id="x",
            oauth_blob={"access_token": "old_at"},
            refresh_token="rt_xyz",
            expires_at=expires_at,
        )
    return acct.id


async def test_refresh_skipped_when_far_from_expiry(db: Database) -> None:
    tenant = await TenantsRepo(db).create(name="A")
    far_future = (
        _dt.datetime.now(_dt.UTC) + _dt.timedelta(hours=2)
    ).isoformat()
    acct_id = await _make_account(db, tenant_id=tenant.id, expires_at=far_future)
    with bound_tenant(tenant.id):
        acct = await ConnectedAccountsRepo(db).get(acct_id)
    assert acct is not None

    calls: list[str] = []

    async def fake_refresh(rt: str, http: httpx.AsyncClient) -> RefreshedToken:
        calls.append(rt)
        raise AssertionError("refresh should not have been called")

    async with httpx.AsyncClient() as http:
        with bound_tenant(tenant.id):
            updated = await refresh_if_expiring(
                acct,
                db=db,
                http=http,
                refresh_impl=fake_refresh,
                lead_s=300,
            )
    assert updated.refresh_token == "rt_xyz"
    assert updated.oauth_blob is not None
    assert updated.oauth_blob["access_token"] == "old_at"
    assert calls == []


async def test_refresh_runs_when_within_lead_window(db: Database) -> None:
    tenant = await TenantsRepo(db).create(name="A")
    soon = (
        _dt.datetime.now(_dt.UTC) + _dt.timedelta(seconds=60)
    ).isoformat()
    acct_id = await _make_account(db, tenant_id=tenant.id, expires_at=soon)
    with bound_tenant(tenant.id):
        acct = await ConnectedAccountsRepo(db).get(acct_id)
    assert acct is not None

    new_expiry = (
        _dt.datetime.now(_dt.UTC) + _dt.timedelta(hours=4)
    ).isoformat()

    async def fake_refresh(rt: str, http: httpx.AsyncClient) -> RefreshedToken:
        return RefreshedToken(
            access_token="new_at",
            refresh_token="rt_new",
            expires_at=new_expiry,
        )

    async with httpx.AsyncClient() as http:
        with bound_tenant(tenant.id):
            updated = await refresh_if_expiring(
                acct,
                db=db,
                http=http,
                refresh_impl=fake_refresh,
                lead_s=300,
            )
    assert updated.refresh_token == "rt_new"
    assert updated.oauth_blob is not None
    assert updated.oauth_blob["access_token"] == "new_at"
    assert updated.expires_at == new_expiry


async def test_refresh_failure_flips_auth_failed_and_emits_event(
    db: Database,
) -> None:
    tenant = await TenantsRepo(db).create(name="A")
    soon = (
        _dt.datetime.now(_dt.UTC) + _dt.timedelta(seconds=60)
    ).isoformat()
    acct_id = await _make_account(db, tenant_id=tenant.id, expires_at=soon)
    with bound_tenant(tenant.id):
        acct = await ConnectedAccountsRepo(db).get(acct_id)
    assert acct is not None

    async def boom(rt: str, http: httpx.AsyncClient) -> RefreshedToken:
        raise RuntimeError("refresh_token revoked")

    async with httpx.AsyncClient() as http:
        try:
            with bound_tenant(tenant.id):
                await refresh_if_expiring(
                    acct,
                    db=db,
                    http=http,
                    refresh_impl=boom,
                    lead_s=300,
                )
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")

    with bound_tenant(tenant.id):
        flipped = await ConnectedAccountsRepo(db).get(acct_id)
        events = await EventsRepo(db).list_for_tenant(
            type="connected_account.auth_failed"
        )
    assert flipped is not None
    assert flipped.status == "auth_failed"
    assert len(events) == 1
    assert events[0].payload.get("account_id") == acct_id


async def test_refresh_with_no_refresh_token_marks_auth_failed(
    db: Database,
) -> None:
    """An account that's expired AND has no refresh_token can't recover."""
    tenant = await TenantsRepo(db).create(name="A")
    soon = (
        _dt.datetime.now(_dt.UTC) + _dt.timedelta(seconds=60)
    ).isoformat()
    with bound_tenant(tenant.id):
        acct = await ConnectedAccountsRepo(db).create(
            platform="tiktok",
            external_id="x",
            oauth_blob={"access_token": "old"},
            refresh_token=None,  # missing
            expires_at=soon,
        )

    async def fake_refresh(rt: str, http: httpx.AsyncClient) -> RefreshedToken:
        raise AssertionError("should not be called")

    async with httpx.AsyncClient() as http:
        try:
            with bound_tenant(tenant.id):
                await refresh_if_expiring(
                    acct,
                    db=db,
                    http=http,
                    refresh_impl=fake_refresh,
                    lead_s=300,
                )
        except PermissionError:
            pass
        else:
            raise AssertionError("expected PermissionError")

    with bound_tenant(tenant.id):
        flipped = await ConnectedAccountsRepo(db).get(acct.id)
    assert flipped is not None
    assert flipped.status == "auth_failed"
