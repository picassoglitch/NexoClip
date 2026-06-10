"""ensure_profile_for_tenant tests (Zernio).

Pins the derive-or-reuse behavior: first call derives + persists the
profileId WITHOUT any network call (Zernio has no create-profile
endpoint — the profileId springs into existence on first use), and a
second call is a cache hit off the persisted value.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from nexoclip.db import Database, TenantsRepo
from nexoclip.db.migrations import apply_migrations
from nexoclip.integrations.zernio import (
    ZernioClient,
    ZernioError,
    ensure_profile_for_tenant,
)
from nexoclip.integrations.zernio.profiles import _derive_profile_id
from nexoclip.tenancy import bound_tenant

# ---- profileId derivation ----


def test_derive_profile_id_passes_ulid_through_lowercased() -> None:
    assert _derive_profile_id("ten_01KS0X2F34HBBJPMZBA45CQW77") == (
        "ten_01ks0x2f34hbbjpmzba45cqw77"
    )


def test_derive_profile_id_strips_unsafe_chars() -> None:
    assert _derive_profile_id("ten_alice@example.com") == "ten_alice-example-com"
    assert _derive_profile_id("ten/with spaces!!") == "ten-with-spaces"


def test_derive_profile_id_falls_back_when_empty() -> None:
    assert _derive_profile_id("!!!") == "tenant"


# ---- ensure_profile_for_tenant ----


@pytest.fixture
async def seeded_db(tmp_path: Path):
    db = Database(tmp_path / "zernio_profiles.db")
    await db.connect()
    await apply_migrations(db)
    tenant_id = "ten_01KS0TEST123"
    with bound_tenant(tenant_id):
        await TenantsRepo(db).create(tenant_id=tenant_id, name="Test")
    yield db, tenant_id
    await db.close()


@pytest.mark.asyncio
async def test_first_call_derives_and_persists_no_network(seeded_db) -> None:
    """First call must derive + persist the profileId WITHOUT any HTTP
    call — Zernio creates the profile implicitly. respx rejects any
    network attempt, so a stray call would fail the test."""
    db, tenant_id = seeded_db
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=False):
            client = ZernioClient(api_key="sk_x", http=http)
            with bound_tenant(tenant_id):
                profile_id = await ensure_profile_for_tenant(
                    db=db, tenant_id=tenant_id, client=client,
                )
    assert profile_id == "ten_01ks0test123"
    with bound_tenant(tenant_id):
        tenant = await TenantsRepo(db).get(tenant_id)
    assert tenant is not None
    assert tenant.zernio_profile_id == "ten_01ks0test123"


@pytest.mark.asyncio
async def test_second_call_is_cache_hit(seeded_db) -> None:
    db, tenant_id = seeded_db
    with bound_tenant(tenant_id):
        await TenantsRepo(db).set_zernio_profile_id(tenant_id, "ten_custom")
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=False):
            client = ZernioClient(api_key="sk_x", http=http)
            with bound_tenant(tenant_id):
                profile_id = await ensure_profile_for_tenant(
                    db=db, tenant_id=tenant_id, client=client,
                )
    assert profile_id == "ten_custom"


@pytest.mark.asyncio
async def test_missing_tenant_raises(seeded_db) -> None:
    db, _ = seeded_db
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=False):
            client = ZernioClient(api_key="sk_x", http=http)
            with (
                pytest.raises(ZernioError, match="tenant not found"),
                bound_tenant("ten_nope"),
            ):
                await ensure_profile_for_tenant(
                    db=db, tenant_id="ten_nope", client=client,
                )
