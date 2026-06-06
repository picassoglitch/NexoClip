"""Wave 2 Task 5 — proactive 60-day Instagram token refresh.

Meta long-lived tokens expire ~60 days after issue. Unlike TikTok /
Google where the refresh is on-demand right before a publish, Meta
gives no refresh_token — you re-exchange the long-lived token ITSELF
before expires_at to get a new long-lived token. If the long-lived
token lapses, you cannot mint a new one without a fresh OAuth round-
trip — which means an operator who connected once and didn't publish
for 60+ days silently loses their connection.

The fix is a scheduled refresh loop:

  * runs every `interval_s` seconds (production default 6h)
  * selects connected_accounts where:
      platform = 'instagram'
      AND status = 'active'
      AND token_type = 'long_lived'
      AND expires_at < (now + REFRESH_WINDOW_DAYS)
  * calls Meta's fb_exchange_token endpoint with the CURRENT
    long-lived token (stored as refresh_token_encrypted)
  * on success: writes the new long-lived USER token + updates
    expires_at. The Page Access Token (access_token_encrypted) is
    technically the same value Meta returned originally; per docs
    refreshing the long-lived user token implicitly extends the
    Page Access Token's lifetime too, but we re-store it for
    clarity / safety.
  * on failure: marks status='auth_failed' so the operator sees
    a red banner on the Connect tab and can re-OAuth.

REFRESH_WINDOW_DAYS picked generously: 14 days gives the loop ~3-4
opportunities to succeed before lapse, even with multi-day outages
(Railway dyno restart, Meta-side incident).
"""
from __future__ import annotations

import datetime as _dt
import logging
from typing import Final

from nexoclip.db import ConnectedAccountsRepo, Database
from nexoclip.errors import NexoClipError
from nexoclip.integrations.instagram.auth import (
    InstagramAuthError,
    refresh_long_lived_token,
)
from nexoclip.integrations.oauth.encryption import (
    DecryptionFailed,
    EncryptionKeyMissing,
    get_encryptor,
)
from nexoclip.settings import get_settings
from nexoclip.tenancy import bound_tenant

_log = logging.getLogger("nexoclip.integrations.instagram.refresh")

# Account is "near expiry" when expires_at is within this many days.
# Picked so the loop has multiple opportunities to retry before
# the token actually lapses.
_REFRESH_WINDOW_DAYS: Final = 14


async def run_instagram_refresh(tenant_id: str, db: Database) -> int:
    """Refresh near-expiry IG long-lived tokens for one tenant.

    Returns the number of tokens that were refreshed successfully.
    The lifespan loop calls this per-tenant inside its outer
    try/except so a single failing tenant doesn't crash the loop.
    """
    settings = get_settings()
    if not settings.meta_app_id or not settings.meta_app_secret:
        # Connect is disabled in production — refresh has nothing to do.
        return 0
    if not settings.nexoclip_creds_key:
        # Encryption is required to even read the stored token.
        _log.warning("instagram_refresh.encryption_unavailable tenant=%s", tenant_id)
        return 0

    try:
        enc = get_encryptor()
    except EncryptionKeyMissing:
        return 0

    refreshed = 0
    cutoff = _dt.datetime.now(_dt.UTC) + _dt.timedelta(days=_REFRESH_WINDOW_DAYS)

    with bound_tenant(tenant_id):
        repo = ConnectedAccountsRepo(db)
        for account in await repo.list_for_tenant():
            if account.platform != "instagram":
                continue
            if account.token_type != "long_lived":
                # Defensive: legacy / mis-tagged row. Skip rather than
                # call the wrong refresh strategy.
                continue
            if account.status != "active":
                continue
            if not account.expires_at:
                continue
            if account.refresh_token_encrypted is None:
                _log.warning(
                    "instagram_refresh.missing_refresh_credential "
                    "tenant=%s account=%s",
                    tenant_id, account.id,
                )
                continue

            try:
                expires_dt = _dt.datetime.fromisoformat(account.expires_at)
            except ValueError:
                _log.warning(
                    "instagram_refresh.bad_expires_at tenant=%s account=%s expires=%s",
                    tenant_id, account.id, account.expires_at,
                )
                continue
            if expires_dt > cutoff:
                # Not near expiry yet.
                continue

            try:
                long_lived_user_token = enc.decrypt(account.refresh_token_encrypted)
            except DecryptionFailed:
                _log.warning(
                    "instagram_refresh.decryption_failed tenant=%s account=%s",
                    tenant_id, account.id,
                )
                await repo.mark_status(account.id, "auth_failed")
                continue
            if long_lived_user_token is None:
                continue

            try:
                fresh = await refresh_long_lived_token(
                    app_id=settings.meta_app_id,
                    app_secret=settings.meta_app_secret,
                    current_long_lived_token=long_lived_user_token,
                )
            except InstagramAuthError as e:
                _log.warning(
                    "instagram_refresh.exchange_failed tenant=%s account=%s err=%s body=%s",
                    tenant_id, account.id, e, e.body,
                )
                # The token may have been revoked by the user, by
                # Meta (suspicious activity), or genuinely lapsed.
                # Either way, the operator has to re-OAuth.
                await repo.mark_status(account.id, "auth_failed")
                continue
            except NexoClipError as e:
                # Catch-all for our typed errors so a bug in the
                # auth module doesn't crash the loop. Don't flip
                # auth_failed for unknown errors — they may be
                # transient (network blip).
                _log.warning(
                    "instagram_refresh.unexpected_error tenant=%s account=%s err=%s",
                    tenant_id, account.id, e,
                )
                continue

            # Refresh succeeded. The fresh long-lived USER token
            # replaces the stored refresh credential. The Page
            # Access Token (access_token_encrypted) carries forward
            # — per Meta docs, refreshing the user token implicitly
            # extends the Page Access Token's lifetime to match.
            new_refresh_ct = enc.encrypt(fresh.access_token)
            await repo.update_oauth(
                account.id,
                refresh_token=fresh.access_token,
                expires_at=fresh.expires_at,
            )
            # update_oauth doesn't touch the encrypted column —
            # do that via a direct SQL update to avoid a new
            # method just for this case.
            conn = await db.connect()
            await conn.execute(
                "UPDATE connected_accounts "
                "SET refresh_token_encrypted = ?, expires_at = ? "
                "WHERE id = ? AND tenant_id = ?",
                (new_refresh_ct, fresh.expires_at, account.id, tenant_id),
            )
            await conn.commit()
            refreshed += 1
            _log.info(
                "instagram_refresh.success tenant=%s account=%s new_expires=%s",
                tenant_id, account.id, fresh.expires_at,
            )

    return refreshed


__all__ = ["run_instagram_refresh"]
