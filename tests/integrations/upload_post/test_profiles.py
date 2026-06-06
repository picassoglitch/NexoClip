"""ensure_profile_for_tenant tests.

Pins the get-or-create behavior: first call mints + persists, second
is a no-op cache hit, 409 from upload-post is treated as success
(profile already exists upstream, we just re-claim it locally),
403 propagates up (plan limit).
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from nexoclip.db import Database, TenantsRepo
from nexoclip.db.migrations import apply_migrations
from nexoclip.integrations.upload_post import (
    UploadPostClient,
    UploadPostError,
    ensure_profile_for_tenant,
)
from nexoclip.integrations.upload_post.profiles import _derive_username
from nexoclip.tenancy import bound_tenant


# ---- username derivation ----


def test_derive_username_passes_ulid_through_lowercased() -> None:
    """ULIDs are alphanumeric — the derive step lowercases and otherwise
    leaves them alone. Production tenant_ids are ULIDs like
    `ten_01KS0X2F34HBBJPMZBA45CQW77`."""
    assert _derive_username("ten_01KS0X2F34HBBJPMZBA45CQW77") == (
        "ten_01ks0x2f34hbbjpmzba45cqw77"
    )


def test_derive_username_strips_unsafe_chars() -> None:
    assert _derive_username("ten_alice@example.com") == "ten_alice-example-com"
    # Multiple unsafe runs collapse to single dashes via re.sub
    assert _derive_username("ten/with spaces!!") == "ten-with-spaces"


def test_derive_username_falls_back_when_empty() -> None:
    assert _derive_username("!!!") == "tenant"


# ---- ensure_profile_for_tenant ----


@pytest.fixture
async def seeded_db(tmp_path: Path):
    """In-memory-ish DB with one tenant."""
    db = Database(tmp_path / "up_profiles.db")
    await db.connect()
    await apply_migrations(db)
    tenant_id = "ten_01KS0TEST123"
    with bound_tenant(tenant_id):
        await TenantsRepo(db).create(tenant_id=tenant_id, name="Test")
    yield db, tenant_id
    await db.close()


@pytest.mark.asyncio
async def test_first_call_creates_and_persists(seeded_db) -> None:
    db, tenant_id = seeded_db
    body = {
        "success": True,
        "profile": {
            "username": "ten_01ks0test123",
            "created_at": "Fri, 02 May 2025 21:43:14 GMT",
            "social_accounts": {"tiktok": ""},
        },
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            mock.post("https://api.upload-post.com/api/uploadposts/users").mock(
                return_value=httpx.Response(201, json=body)
            )
            client = UploadPostClient(api_key="x", http=http)
            with bound_tenant(tenant_id):
                username = await ensure_profile_for_tenant(
                    db=db, tenant_id=tenant_id, client=client,
                )
    assert username == "ten_01ks0test123"

    # Persisted on the tenant row.
    with bound_tenant(tenant_id):
        tenant = await TenantsRepo(db).get(tenant_id)
    assert tenant is not None
    assert tenant.upload_post_profile_username == "ten_01ks0test123"


@pytest.mark.asyncio
async def test_second_call_is_cache_hit_no_network(seeded_db) -> None:
    """Pre-populated tenant row → ensure_profile must NOT hit the API.
    If it tries, respx will fail (no route registered)."""
    db, tenant_id = seeded_db
    with bound_tenant(tenant_id):
        await TenantsRepo(db).set_upload_post_profile_username(
            tenant_id, "ten_01ks0test123",
        )

    async with httpx.AsyncClient() as http:
        # No respx mock = any HTTP call will fail with httpx.ConnectError
        # under respx's default reject-all transport.
        with respx.mock(assert_all_called=False):
            client = UploadPostClient(api_key="x", http=http)
            with bound_tenant(tenant_id):
                username = await ensure_profile_for_tenant(
                    db=db, tenant_id=tenant_id, client=client,
                )
    assert username == "ten_01ks0test123"


@pytest.mark.asyncio
async def test_409_treated_as_success_and_persisted(seeded_db) -> None:
    """If upload-post says the profile already exists (409) but our
    local row didn't reflect that, ensure_profile re-claims it
    locally without surfacing an error to the caller."""
    db, tenant_id = seeded_db
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post("https://api.upload-post.com/api/uploadposts/users").mock(
                return_value=httpx.Response(
                    409, json={"success": False, "error": "Profile exists"}
                )
            )
            client = UploadPostClient(api_key="x", http=http)
            with bound_tenant(tenant_id):
                username = await ensure_profile_for_tenant(
                    db=db, tenant_id=tenant_id, client=client,
                )
    assert username == "ten_01ks0test123"
    # Persisted regardless.
    with bound_tenant(tenant_id):
        tenant = await TenantsRepo(db).get(tenant_id)
    assert tenant.upload_post_profile_username == "ten_01ks0test123"


@pytest.mark.asyncio
async def test_403_profile_limit_propagates(seeded_db) -> None:
    """Plan limit reached on upload-post → caller wants to surface
    the upgrade prompt, not silently swallow the error."""
    db, tenant_id = seeded_db
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post("https://api.upload-post.com/api/uploadposts/users").mock(
                return_value=httpx.Response(
                    403,
                    json={
                        "success": False,
                        "error_code": "PROFILE_LIMIT_REACHED",
                        "error": "Plan limit reached",
                    },
                )
            )
            client = UploadPostClient(api_key="x", http=http)
            with bound_tenant(tenant_id):
                with pytest.raises(UploadPostError) as ei:
                    await ensure_profile_for_tenant(
                        db=db, tenant_id=tenant_id, client=client,
                    )
    assert ei.value.status_code == 403

    # Tenant row stays unset.
    with bound_tenant(tenant_id):
        tenant = await TenantsRepo(db).get(tenant_id)
    assert tenant.upload_post_profile_username is None
