"""OAuth refresh flow shared between TikTok + YouTube clients.

`refresh_if_expiring(account, *, db, http)` is called once per drain pass
before posting. On a fresh refresh:
  * `oauth_blob.access_token` and `oauth_blob.refresh_token` updated
  * `expires_at` advanced
  * status stays `active`
On a refresh that fails (refresh_token revoked, network down, etc.):
  * status flips to `auth_failed`
  * `connected_account.auth_failed` event emitted
  * caller skips this account for the rest of the drain
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx
import structlog

from nexoclip.db import ConnectedAccountsRepo, Database, EventsRepo
from nexoclip.db.models import ConnectedAccount

_log = structlog.get_logger(__name__)

# Refresh when within this many seconds of expiry.
_DEFAULT_REFRESH_LEAD_S = 300


@dataclass(frozen=True)
class RefreshedToken:
    """What a successful refresh round-trip produced."""

    access_token: str
    refresh_token: str | None  # None = reuse the existing one
    expires_at: str  # ISO 8601


# A refresh implementation is a coroutine factory that takes the existing
# refresh_token + the platform's API base + an httpx client and returns the
# new credentials. Each platform plugs in its own.
RefreshImpl = Callable[
    [str, httpx.AsyncClient],
    Awaitable[RefreshedToken],
]


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


def _is_expiring(expires_at: str | None, *, lead_s: float, now: _dt.datetime) -> bool:
    if not expires_at:
        return False
    try:
        deadline = _dt.datetime.fromisoformat(expires_at)
    except ValueError:
        return False
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=_dt.UTC)
    return (deadline - now).total_seconds() <= lead_s


async def refresh_if_expiring(
    account: ConnectedAccount,
    *,
    db: Database,
    http: httpx.AsyncClient,
    refresh_impl: RefreshImpl,
    lead_s: float = _DEFAULT_REFRESH_LEAD_S,
    clock: Callable[[], _dt.datetime] | None = None,
) -> ConnectedAccount:
    """Refresh `account.oauth_blob.access_token` if it's near expiry.

    Returns the updated ConnectedAccount (refreshed or untouched). On
    refresh failure, marks the account `auth_failed`, emits the event,
    and *re-raises* the underlying exception so the caller can skip this
    account for the rest of the drain.
    """
    now = (clock or _utc_now)()
    if not _is_expiring(account.expires_at, lead_s=lead_s, now=now):
        return account
    if not account.refresh_token:
        # Nothing to refresh with - flip the account so the dashboard can
        # surface "reconnect required".
        await _mark_auth_failed(db, account, reason="no refresh_token")
        raise PermissionError(f"account {account.id} has no refresh_token")

    try:
        new_token = await refresh_impl(account.refresh_token, http)
    except Exception as e:
        await _mark_auth_failed(db, account, reason=f"refresh_failed: {e}")
        raise

    blob = dict(account.oauth_blob or {})
    blob["access_token"] = new_token.access_token
    return await ConnectedAccountsRepo(db).update_oauth(
        account.id,
        oauth_blob=blob,
        refresh_token=new_token.refresh_token or account.refresh_token,
        expires_at=new_token.expires_at,
    )


async def _mark_auth_failed(
    db: Database, account: ConnectedAccount, *, reason: str
) -> None:
    """Flip status + emit `connected_account.auth_failed`."""
    _log.warning(
        "connected_account_auth_failed",
        account_id=account.id,
        platform=account.platform,
        reason=reason,
    )
    await ConnectedAccountsRepo(db).mark_status(account.id, "auth_failed")
    await EventsRepo(db).emit(
        type="connected_account.auth_failed",
        payload={
            "account_id": account.id,
            "platform": account.platform,
            "reason": reason,
        },
    )
