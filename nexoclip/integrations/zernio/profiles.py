"""Tenant ↔ Zernio profile creation + persistence.

`create_profile_for_tenant(...)` is the entry point used by the
dashboard's inline "Create profile" action. It calls Zernio's
`POST /profiles` (a real server-side create that returns a generated
`_id` like `prof_abc123`), then persists that id + the operator-chosen
name on the tenant row.

This replaces an earlier shortcut that fabricated the profileId from
the tenant id without ever calling Zernio — Zernio requires the
profile to exist before you can connect accounts or publish under it.

Single-profile model: each tenant has at most one Zernio profile
(`tenant.zernio_profile_id`). Creating again overwrites the binding
(use after /unlink to start fresh); connecting accounts to the new
profile is a separate step.
"""
from __future__ import annotations

import logging

from nexoclip.db import Database, TenantsRepo
from nexoclip.integrations.zernio.client import (
    ZernioClient,
    ZernioError,
    ZernioProfile,
)

_log = logging.getLogger("nexoclip.integrations.zernio.profiles")


async def create_profile_for_tenant(
    *,
    db: Database,
    tenant_id: str,
    client: ZernioClient,
    name: str,
    description: str | None = None,
) -> ZernioProfile:
    """Create a Zernio profile for this tenant and persist the binding.

    Raises `ZernioError` if the tenant row is missing or Zernio rejects
    the create. The returned profile's `profile_id` is what connect +
    publish use as `profileId`.
    """
    repo = TenantsRepo(db)
    tenant = await repo.get(tenant_id)
    if tenant is None:
        raise ZernioError(f"tenant not found: {tenant_id}")

    profile = await client.create_profile(name=name, description=description)
    await repo.set_zernio_profile(
        tenant_id,
        profile_id=profile.profile_id,
        profile_name=profile.name,
    )
    _log.info(
        "zernio.profile_created tenant=%s profile_id=%s name=%s",
        tenant_id, profile.profile_id, profile.name,
    )
    return profile


__all__ = ["create_profile_for_tenant"]
