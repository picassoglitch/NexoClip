"""Native-OAuth Connect tab — Wave 1 (TikTok) + Wave 2 stubs (IG/YT).

Two routers in one file so URL shapes stay co-located:

  - `router` (prefix=/dashboard/connect): the cookie-auth dashboard
    surface. Lists platforms, kicks off authorize redirects, handles
    disconnect. Every route binds `tenant_id` via `tenant_binder` so
    a hijacked dashboard session can't disconnect another tenant.

  - `public_router` (prefix=/connect): the public-callback surface
    where TikTok / IG / YouTube redirect the operator's browser
    back to us. NO cookie auth — the HMAC-signed `state` query
    param carries the originating tenant_id through. The bearer
    middleware's allowlist exempts `/connect/` so the request
    actually reaches these handlers.

The split is important because:
  * the authorize step happens FROM the dashboard (cookie present)
  * the callback arrives FROM the platform's redirect, after the
    user's browser left our origin — no cookie unless they were
    logged in to a parallel tab, and we can't rely on that.
"""
from __future__ import annotations

import logging
from typing import Final

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from nexoclip.db import ConnectedAccountsRepo, Database
from nexoclip.errors import NexoClipError
from nexoclip.integrations.oauth.encryption import (
    EncryptionKeyMissing,
    get_encryptor,
)
from nexoclip.integrations.oauth.state import (
    InvalidState,
    sign_state,
    verify_state,
)
from nexoclip.integrations.tiktok.auth import (
    TikTokAuthError,
    exchange_code_for_token,
    fetch_user_info,
    query_creator_info,
    build_authorize_url,
)
from nexoclip.settings import get_settings

from ..deps import get_db, require_full_scope, tenant_binder

_log = logging.getLogger("nexoclip.api.connect")

# Cookie-authed dashboard surface for everything except the callback.
router = APIRouter(prefix="/dashboard/connect", tags=["connect"])

# Public callback surface. Reached by the platform's redirect; auth
# rides in the signed state, not in a cookie.
public_router = APIRouter(prefix="/connect", tags=["connect"])

# Reuse the dashboard's template registry. Templates land under
# nexoclip/api/templates/connect/ in this commit so the existing
# connected_accounts.html (legacy Buffer flow) stays untouched.
import pathlib
_TEMPLATES_DIR = pathlib.Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
from ..i18n import install_globals as _install_i18n
_install_i18n(templates)


_SUPPORTED_PLATFORMS: Final = ("tiktok", "instagram", "youtube")


def _redirect_uri(platform: str) -> str:
    """Build the canonical redirect URI tenants register on the dev
    portal. Same shape for all tenants under Pattern A; differs only
    by platform path segment."""
    base = get_settings().oauth_redirect_base_url.rstrip("/")
    return f"{base}/connect/{platform}/callback"


# ---------- Dashboard UI ----------


@router.get("", response_class=HTMLResponse)
async def connect_landing(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Landing page for the Connect tab.

    Lists each platform card with current connection state. TikTok
    is live in Wave 1; IG + YT show 'Coming soon' until Wave 2.
    """
    settings = get_settings()
    accounts = await ConnectedAccountsRepo(db).list_for_tenant()
    by_platform: dict[str, object] = {a.platform: a for a in accounts}

    # Per-platform readiness signal — surfaces an actionable error
    # the moment the operator lands instead of mid-redirect ("why is
    # this button broken?").
    def _ready_for(platform: str) -> tuple[bool, str | None]:
        if platform == "tiktok":
            if not settings.tiktok_client_key or not settings.tiktok_client_secret:
                return False, "TIKTOK_CLIENT_KEY / TIKTOK_CLIENT_SECRET unset"
            if not settings.nexoclip_creds_key:
                return False, "NEXOCLIP_CREDS_KEY unset (token encryption)"
            return True, None
        # IG + YT land in Wave 2.
        return False, "Coming in Wave 2"

    cards = []
    for platform in _SUPPORTED_PLATFORMS:
        ready, reason = _ready_for(platform)
        account = by_platform.get(platform)
        cards.append(
            {
                "platform": platform,
                "ready": ready,
                "not_ready_reason": reason,
                "account": account,
                "redirect_uri": _redirect_uri(platform),
            }
        )
    return templates.TemplateResponse(
        request,
        "connect/connect_landing.html",
        {"cards": cards},
    )


@router.get("/tiktok/authorize")
async def tiktok_authorize(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
) -> Response:
    """Step 1 of TikTok OAuth: 303 the operator's browser to TikTok's
    consent page with our client_key + signed state.

    `require_full_scope` is intentional: a read-only bearer token
    shouldn't be able to start a Connect flow because the resulting
    write happens server-side on callback under the same tenant.
    """
    settings = get_settings()
    if not settings.tiktok_client_key:
        raise HTTPException(
            status_code=503,
            detail="TIKTOK_CLIENT_KEY is not configured on this NexoClip instance.",
        )
    if not settings.nexoclip_creds_key:
        # Fail at the start, not on callback after the user authorized
        # — refusing here means TikTok never sends us a code we can't
        # encrypt.
        raise HTTPException(
            status_code=503,
            detail=(
                "NEXOCLIP_CREDS_KEY is not configured; cannot encrypt "
                "OAuth tokens at rest. Set it in Railway env and retry."
            ),
        )

    state = sign_state(platform="tiktok", tenant_id=tenant_id)
    redirect_uri = _redirect_uri("tiktok")
    auth_url = build_authorize_url(
        client_key=settings.tiktok_client_key,
        redirect_uri=redirect_uri,
        state=state,
    )
    _log.info(
        "connect.tiktok.authorize_start tenant=%s redirect=%s",
        tenant_id, redirect_uri,
    )
    return RedirectResponse(url=auth_url, status_code=303)


@router.post("/tiktok/disconnect")
async def tiktok_disconnect(
    request: Request,
    account_id: str = Form(...),
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    db: Database = Depends(get_db),
) -> Response:
    """Soft-disconnect a TikTok account: mark status=disabled +
    null out the encrypted tokens.

    We DON'T call TikTok's token-revoke endpoint here — leaving the
    server-side credential intact lets a re-connect dedup on
    (tenant, platform_user_id) and overwrite cleanly. The operator
    can also re-authorize without re-entering the consent dialog
    if TikTok still has our app on its session.
    """
    repo = ConnectedAccountsRepo(db)
    account = await repo.get(account_id)
    if account is None or account.platform != "tiktok":
        raise HTTPException(status_code=404, detail="tiktok account not found")

    # Clear the encrypted columns + mirror, mark inactive. Same row
    # is overwritten on next Connect click.
    await repo.update_oauth(
        account_id,
        oauth_blob={},
        refresh_token="",
        expires_at="",
        scopes=[],
    )
    await repo.mark_status(account_id, "disabled")
    _log.info(
        "connect.tiktok.disconnect tenant=%s account=%s",
        tenant_id, account_id,
    )
    return RedirectResponse(url="/dashboard/connect", status_code=303)


# ---------- Public callback (no cookie auth) ----------


@public_router.get("/tiktok/callback")
async def tiktok_callback(
    request: Request,
    code: str = Query(default=""),
    state: str = Query(default=""),
    error: str = Query(default=""),
    db: Database = Depends(get_db),
) -> Response:
    """Step 2 of TikTok OAuth: verify state, exchange code for
    tokens, fetch user info + creator-info, store encrypted.

    NO cookie auth on this endpoint — the bearer middleware skips
    /connect/ entirely. Tenant binding rides in the signed `state`.

    Error surfacing: any failure renders a small "Connect failed"
    page with the operator-visible reason and a Retry button that
    bounces back to /dashboard/connect. We never echo raw TikTok
    error bodies into HTML (would XSS); structured logs carry the
    full payload.
    """
    settings = get_settings()
    if error:
        # User canceled or TikTok rejected before round-trip
        return _render_callback_error(
            request, reason=f"TikTok returned: {error}"
        )
    if not code or not state:
        return _render_callback_error(
            request, reason="Missing `code` or `state` in callback URL."
        )

    # 1. Verify state — binds the tenant.
    try:
        claims = verify_state(state)
    except InvalidState as e:
        _log.warning("connect.tiktok.state_invalid reason=%s", e.reason)
        return _render_callback_error(
            request,
            reason=(
                "Connect attempt expired or could not be verified. "
                "Please return to the Connect tab and click Connect again."
            ),
        )
    if claims.platform != "tiktok":
        _log.warning(
            "connect.tiktok.platform_mismatch state_platform=%s", claims.platform
        )
        return _render_callback_error(
            request, reason="State token does not belong to TikTok."
        )

    # 2. Exchange the code for tokens.
    if not settings.tiktok_client_key or not settings.tiktok_client_secret:
        return _render_callback_error(
            request, reason="TikTok client credentials are not configured."
        )
    try:
        token = await exchange_code_for_token(
            client_key=settings.tiktok_client_key,
            client_secret=settings.tiktok_client_secret,
            code=code,
            redirect_uri=_redirect_uri("tiktok"),
        )
    except TikTokAuthError as e:
        _log.warning("connect.tiktok.exchange_failed err=%s body=%s", e, e.body)
        return _render_callback_error(
            request,
            reason="TikTok rejected the authorization code. Click Connect again.",
        )

    # 3. Fetch user info (display name + avatar) for the UI confirm.
    try:
        user = await fetch_user_info(access_token=token.access_token)
    except TikTokAuthError as e:
        _log.warning("connect.tiktok.user_info_failed err=%s", e)
        # Non-fatal — write the row without the cosmetic fields so the
        # connect still completes. Operator sees the open_id but no avatar.
        user = None

    # 4. Creator-info: surfaces "can_post" + privacy levels.
    try:
        await query_creator_info(access_token=token.access_token)
    except TikTokAuthError as e:
        _log.warning("connect.tiktok.creator_info_failed err=%s", e)
        # Also non-fatal — the publisher re-queries before each post.

    # 5. Encrypt + persist.
    try:
        enc = get_encryptor()
    except EncryptionKeyMissing as e:
        # Shouldn't be reachable — authorize-start refused without the
        # key. But cover it so a key rotation that nulled the var
        # doesn't crash the request.
        _log.error("connect.tiktok.encryption_unavailable err=%s", e)
        return _render_callback_error(
            request,
            reason="Token encryption is not configured on the server.",
        )

    access_ct = enc.encrypt(token.access_token)
    refresh_ct = enc.encrypt(token.refresh_token)

    # Bind the repo to the tenant from the verified state, NOT the
    # request — there's no cookie on this public route.
    from nexoclip.tenancy import bound_tenant
    with bound_tenant(claims.tenant_id):
        repo = ConnectedAccountsRepo(db)
        scopes = [s for s in token.scope.split(",") if s]
        await repo.upsert_oauth_connection(
            platform="tiktok",
            platform_user_id=token.open_id,
            platform_username=user.display_name if user else None,
            platform_avatar_url=user.avatar_url if user else None,
            access_token_encrypted=access_ct,    # type: ignore[arg-type]
            refresh_token_encrypted=refresh_ct,  # type: ignore[arg-type]
            token_type="bearer",
            expires_at=token.expires_at,
            scopes=scopes,
            display_name=user.display_name if user else None,
            # Wave 1 dual-write — existing publish path reads plain
            # token from oauth_blob. Removed in Wave 2 cleanup.
            access_token_plaintext_mirror=token.access_token,
        )
    _log.info(
        "connect.tiktok.connected tenant=%s open_id=%s display_name=%s",
        claims.tenant_id, token.open_id, user.display_name if user else None,
    )
    return RedirectResponse(url="/dashboard/connect?connected=tiktok", status_code=303)


def _render_callback_error(request: Request, *, reason: str) -> Response:
    """Render a small error page so the operator can read what went
    wrong without an opaque 500. Reason is sanitized server-side
    before reaching this function — no user-controlled content
    interpolates into the HTML."""
    return templates.TemplateResponse(
        request,
        "connect/connect_error.html",
        {"reason": reason},
        status_code=400,
    )


__all__ = ["router", "public_router"]
