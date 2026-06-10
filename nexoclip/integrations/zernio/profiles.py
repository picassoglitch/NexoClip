"""Tenant ↔ Zernio profile mapping.

`ensure_profile_for_tenant(db, tenant_id, client)` is the only entry
point. Idempotent: returns the existing profile id if one's already
on the tenant row; otherwise derives a fresh one, persists it
locally, and returns it. Subsequent calls fast-path off the
persisted value.

Unlike the upload-post integration this replaces, Zernio has no
create-profile endpoint: `profileId` is a free-form namespacing
string that springs into existence the first time we use it on a
`/connect` or `/posts` call. So this helper makes NO network call —
it derives + persists. The `client` argument is kept in the
signature for interface parity (and future use, e.g. validating
against `list_accounts`) but is currently unused.

The profileId we use is the tenant_id itself, sanitized so it's a
safe URL/query value. Real tenant_ids in NexoClip are ULIDs like
`ten_01KS0X2F34HBBJPMZBA45CQW77` which are already safe — we keep
them verbatim (lowercased). Defense-in-depth strips anything weird
if a future migration introduces non-ULID ids.
"""
from __future__ import annotations

import logging
import re
from typing import Final

from nexoclip.db import Database, TenantsRepo
from nexoclip.integrations.zernio.client import ZernioClient, ZernioError

_log = logging.getLogger("nexoclip.integrations.zernio.profiles")

# Lowercase alphanumeric + a few separators. ULIDs already match
# (after lowercase). Anything else gets the offending chars replaced
# with `-` so the profileId stays a clean URL/query value.
_SAFE_PROFILE_RE: Final = re.compile(r"[^a-z0-9_\-]")


def _derive_profile_id(tenant_id: str) -> str:
    """Stable Zernio profileId from a tenant_id.

    Deterministic — same tenant_id always yields the same profileId,
    so retries after a failed first-attempt land on the same value.
    """
    return _SAFE_PROFILE_RE.sub("-", tenant_id.lower()).strip("-") or "tenant"


async def ensure_profile_for_tenant(
    *,
    db: Database,
    tenant_id: str,
    client: ZernioClient,
) -> str:
    """Return the Zernio profileId for this tenant, deriving +
    persisting one on first use.

    Two branches:
      1. tenant.zernio_profile_id already set → return it.
      2. Not set → derive, persist, return. No network call — Zernio
         creates the profile implicitly on the first connect/post.

    Locking is intentionally not added: two concurrent first clicks
    would each derive the SAME deterministic id and persist it; the
    end state is identical.
    """
    repo = TenantsRepo(db)
    tenant = await repo.get(tenant_id)
    if tenant is None:
        raise ZernioError(f"tenant not found: {tenant_id}")

    if tenant.zernio_profile_id:
        return tenant.zernio_profile_id

    profile_id = _derive_profile_id(tenant_id)
    await repo.set_zernio_profile_id(tenant_id, profile_id)
    _log.info(
        "zernio.profile_persisted tenant=%s profile_id=%s",
        tenant_id, profile_id,
    )
    return profile_id


__all__ = ["ensure_profile_for_tenant"]
