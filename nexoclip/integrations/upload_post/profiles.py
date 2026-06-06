"""Tenant ↔ upload-post profile mapping.

`ensure_profile_for_tenant(db, tenant_id)` is the only entry point.
Idempotent: returns the existing profile username if one's already
on the tenant row; otherwise mints a fresh one, creates it on
upload-post's side, persists locally, and returns it. Subsequent
calls fast-path off the persisted value.

The username we send upload-post is the tenant_id itself, sanitized
so their `username` rule (alphanumeric + a few separators, typically)
doesn't reject it. Real tenant_ids in NexoClip are ULIDs like
`ten_01KS0X2F34HBBJPMZBA45CQW77` which are already safe — we keep
them verbatim. Defense-in-depth lowercases and strips anything
weird if a future migration introduces non-ULID ids.
"""
from __future__ import annotations

import logging
import re
from typing import Final

from nexoclip.db import Database, TenantsRepo
from nexoclip.integrations.upload_post.client import (
    UploadPostClient,
    UploadPostError,
)

_log = logging.getLogger("nexoclip.integrations.upload_post.profiles")

# Lowercase alphanumeric + a few separators. ULIDs already match
# (after lowercase). Anything else gets the offending chars
# replaced with `-` so upload-post's username validator doesn't
# 400 us at create time.
_SAFE_USERNAME_RE: Final = re.compile(r"[^a-z0-9_\-]")


def _derive_username(tenant_id: str) -> str:
    """Stable upload-post username from a tenant_id.

    Deterministic — same tenant_id always yields the same username,
    so retries after a failed first-attempt land on the same row.
    """
    return _SAFE_USERNAME_RE.sub("-", tenant_id.lower()).strip("-") or "tenant"


async def ensure_profile_for_tenant(
    *,
    db: Database,
    tenant_id: str,
    client: UploadPostClient,
) -> str:
    """Return the upload-post username for this tenant, creating the
    profile on upload-post's side if needed.

    Three branches:
      1. tenant.upload_post_profile_username already set → return it,
         no network call.
      2. Not set → derive username, call create_user_profile.
         - 200/201: persist + return.
         - 409 (already exists on upload-post but we lost the local
           row somehow): treat as success, persist the same username
           + return.
         - 403 (profile limit reached on the plan): raise
           UploadPostError so the caller can surface "upgrade plan".
         - other 4xx/5xx: raise.

    Locking is intentionally not added: two concurrent first-publish
    clicks would each hit the upload-post API, one would succeed,
    the other would get 409 and treat it as success. End state is
    the same.
    """
    repo = TenantsRepo(db)
    tenant = await repo.get(tenant_id)
    if tenant is None:
        raise UploadPostError(f"tenant not found: {tenant_id}")

    if tenant.upload_post_profile_username:
        return tenant.upload_post_profile_username

    username = _derive_username(tenant_id)
    try:
        await client.create_user_profile(username)
    except UploadPostError as e:
        if e.status_code == 409:
            # Profile already exists upstream — we lost our local
            # mapping somehow (db rollback, fresh dev env, etc.).
            # Re-claim it.
            _log.info(
                "upload_post.profile_already_exists tenant=%s username=%s",
                tenant_id, username,
            )
        else:
            raise

    await repo.set_upload_post_profile_username(tenant_id, username)
    _log.info(
        "upload_post.profile_persisted tenant=%s username=%s",
        tenant_id, username,
    )
    return username


__all__ = ["ensure_profile_for_tenant"]
