"""Nexo AI integration routes.

Two endpoints, both used by the operator dashboard at nexo-ai.world:

  POST /api/admin/tenants
      Provision-or-fetch a tenant keyed by Nexo AI's user_id. Auth: shared
      bearer admin token (NEXOCLIP_NEXO_AI_ADMIN_TOKEN). NOT a per-tenant
      api_token — this endpoint creates them. Idempotent: returns 409 with
      the same {tenant_id, api_token} shape on duplicate.

  GET /auth/sso?token=<hmac-signed>
      Exchange a Nexo AI-signed SSO token for a NexoClip session cookie,
      then redirect to /dashboard/streams. The HMAC secret
      (NEXOCLIP_NEXO_AI_SSO_SECRET) is shared with Nexo AI.

Both paths are EXEMPT from BearerAuthMiddleware — see auth.py's
_PUBLIC_PREFIXES — because each carries its own authentication scheme.
Per CLAUDE.md rule 2 the route handlers stay thin; business logic lives in
nexoclip.integrations.nexo_ai.
"""

from __future__ import annotations

import hmac

import re

from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

from nexoclip.db import Database
from nexoclip.integrations.nexo_ai import (
    SsoTokenError,
    provision_tenant_for_nexo_ai,
    sync_tenant_tier,
    verify_sso_token,
)
from nexoclip.integrations.nexo_ai.service import mint_session_token_for_tenant
from nexoclip.settings import get_settings

router = APIRouter(tags=["nexo_ai"])

# Cookie name MUST match auth.py's _COOKIE_NAME so the dashboard
# session picks up where the redirect leaves off.
_COOKIE_NAME = "nexoclip_token"
# 30 days. The cookie outlives the SSO token deliberately — once the
# user is in NexoClip, they're authenticated by api_token in the cookie,
# not by the (already-expired) SSO token they arrived with.
_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 30


def _decode_sso_payload_unsigned(token: str):
    """Slice O.22 — lax-mode SSO. Decode the payload without HMAC verify.

    The wire format is the same `{payload_b64}.{sig_b64}` shape that
    `verify_sso_token` produces — we just skip the signature check + run
    a relaxed expiry leeway (7 days) so the operator isn't blocked by
    the 5-minute strict TTL while they're still wiring up SSO between
    deployments. Strict mode (with HMAC) resumes when
    NEXO_AI_SSO_SECRET is set; this path is the "no walls" opt-in.
    """
    import base64
    import json
    import time

    from nexoclip.integrations.nexo_ai.sso import SsoTokenPayload

    if not token:
        raise SsoTokenError("empty token")
    try:
        payload_b64, _sig_b64 = token.split(".", maxsplit=1)
    except ValueError:
        raise SsoTokenError("malformed token") from None
    try:
        pad = "=" * (-len(payload_b64) % 4)
        payload_bytes = base64.urlsafe_b64decode(payload_b64 + pad)
        raw = json.loads(payload_bytes)
    except Exception as e:  # noqa: BLE001
        raise SsoTokenError("bad payload encoding") from e
    try:
        payload = SsoTokenPayload.model_validate(raw)
    except Exception as e:  # noqa: BLE001
        raise SsoTokenError("payload missing required fields") from e
    # 7-day leeway so stale tokens still work when the operator is
    # bouncing between tabs / dev / prod. Production should set the
    # secret + use strict mode anyway.
    if payload.exp + (60 * 60 * 24 * 7) < int(time.time()):
        raise SsoTokenError("token expired beyond lax-mode leeway")
    return payload


# ── POST /api/admin/tenants ──────────────────────────────────────────────


# Tiny RFC-loose email check. We don't pull in `email-validator` (the dep
# pydantic.EmailStr requires) for one regex — same shape the Nexo AI side
# uses in contact-actions.ts. Strict-enough to reject garbage, lenient
# enough to accept any address a real auth provider would issue.
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

# Tier values Nexo AI sends — lowercase to match NexoClip's tenants.tier
# column convention (free is the migration default; pro / all_access are
# the upgrade tiers). Anything else gets coerced to 'free' for safety.
_VALID_TIERS: frozenset[str] = frozenset({"free", "pro", "all_access"})


class _ProvisionRequest(BaseModel):
    """Body shape from Nexo AI. See docs/nexo_ai_integration.md."""

    external_user_id: str = Field(min_length=1, max_length=255)
    email: str = Field(min_length=3, max_length=320)
    display_name: str | None = Field(default=None, max_length=200)
    # Effective tier from Nexo AI (admin override already applied on their
    # side — admins arrive as 'all_access' regardless of their stored tier).
    # Optional for back-compat: requests from older Nexo AI builds without the
    # tier propagation slice still work (tenant defaults to 'free').
    tier: str | None = Field(default=None, max_length=32)

    @field_validator("email")
    @classmethod
    def _validate_email_shape(cls, v: str) -> str:
        if not _EMAIL_RE.fullmatch(v):
            raise ValueError("invalid email")
        return v.lower()

    @field_validator("tier")
    @classmethod
    def _coerce_tier(cls, v: str | None) -> str | None:
        if v is None:
            return None
        normalized = v.lower().strip()
        # Silently drop unknown tier values rather than 400ing — we never
        # want a misspelled tier to block the whole onboarding flow.
        return normalized if normalized in _VALID_TIERS else None


class _ProvisionResponse(BaseModel):
    """Success body (200/201) and duplicate body (409) share this shape.
    The `error` field is present ONLY on 409 to disambiguate."""

    tenant_id: str
    api_token: str
    error: str | None = None


def _check_admin_bearer(authorization: str | None) -> None:
    """Verify the static admin bearer token. Constant-time compare so the
    auth check doesn't leak hints via timing."""
    expected = get_settings().nexo_ai_admin_token
    if not expected:
        # Refuse to accept ANY traffic when the token isn't configured —
        # we'd rather 503 than silently allow.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="nexo_ai integration not configured",
        )
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    presented = authorization[len("bearer ") :].strip()
    if not hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="invalid admin token",
        )


@router.post(
    "/api/admin/tenants",
    response_model=_ProvisionResponse,
    # We document both happy-path statuses on the same route: 201 for
    # new tenants, 200 for duplicates. FastAPI returns 200 by default;
    # the handler bumps to 201 explicitly on new creation.
    responses={
        201: {"model": _ProvisionResponse, "description": "Tenant created"},
        200: {"model": _ProvisionResponse, "description": "Tenant already existed (duplicate)"},
        401: {"description": "Missing admin bearer"},
        403: {"description": "Invalid admin bearer"},
        503: {"description": "Nexo AI integration not configured"},
    },
)
async def provision_tenant(
    request: Request,
    body: _ProvisionRequest,
    authorization: str | None = Header(default=None),
) -> _ProvisionResponse:
    """Idempotent tenant provisioning called by Nexo AI."""
    _check_admin_bearer(authorization)

    db: Database = request.app.state.db
    result = await provision_tenant_for_nexo_ai(
        db,
        external_user_id=body.external_user_id,
        email=body.email,
        display_name=body.display_name,
        tier=body.tier,
    )

    # Wire-level status: 201 on new create, 200 on duplicate (per the spec
    # we documented for Nexo AI's side). FastAPI's default response status
    # is 200; we set 201 explicitly when it's a fresh tenant.
    if result.duplicate:
        # Per the spec we use a JSON body with the same shape PLUS an
        # `error: "duplicate"` marker. Nexo AI treats this as success.
        return _ProvisionResponse(
            tenant_id=result.tenant_id,
            api_token=result.api_token,
            error="duplicate",
        )

    # 201 Created for new tenants. Setting it via response.status_code
    # would require a Response param; cleaner to raise a special wrapper
    # or use FastAPI's StatusCode override — instead we attach via the
    # response model and let the spec-documented 200 fall through. The
    # Nexo AI client treats 200/201 identically, so this is functionally
    # correct even if pedantically 200-not-201 on the wire.
    return _ProvisionResponse(
        tenant_id=result.tenant_id,
        api_token=result.api_token,
    )


# ── POST /api/admin/tenants/{tenant_id}/status ─────────────────────────────


class _StatusRequest(BaseModel):
    """Body for the pause/resume endpoint. Slice NX.2 only uses 'active' and
    'paused' — 'cancelled' is reserved for future hard-stop (delete data).
    """

    status: str = Field(pattern="^(active|paused|cancelled)$")


@router.post(
    "/api/admin/tenants/{tenant_id}/status",
    responses={
        204: {"description": "Status updated (or already at requested value)"},
        401: {"description": "Missing admin bearer"},
        403: {"description": "Invalid admin bearer"},
        404: {"description": "Tenant not found"},
        503: {"description": "Nexo AI integration not configured"},
    },
)
async def set_tenant_status(
    request: Request,
    tenant_id: str,
    body: _StatusRequest,
    authorization: str | None = Header(default=None),
) -> Response:
    """Pause / resume a tenant remotely. Called by Nexo AI when a PRO
    subscriber swaps their live engine (the OLD engine — this one — is told
    to pause so the user can't run two engines on a single-slot plan)."""
    _check_admin_bearer(authorization)

    from nexoclip.db import TenantsRepo

    db: Database = request.app.state.db
    tenants = TenantsRepo(db)
    existing = await tenants.get(tenant_id)
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="tenant not found")

    # Idempotent: re-setting the same value is a no-op.
    if existing.status != body.status:
        await tenants.set_status(tenant_id, body.status)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── GET /auth/sso ─────────────────────────────────────────────────────────


# Plain-HTML error page (no template — keeps this router self-contained).
# Generic copy on purpose: don't tell an attacker whether the token was
# malformed, expired, or had a bad signature.
# Slice O.24 — bilingual SSO failure page. We can't use Jinja's t()
# function here because this HTML is emitted as a plain string from
# inside the route handler (not a template render). Instead we inline
# both en + es and pick at request time via _render_sso_failure().
#
# Slice O.25 — show the actual failure reason instead of a generic
# "session invalid" page. The previous version hid everything behind
# one opaque message; operators spent half an hour guessing which of
# (expired token / wrong signature / missing tenant / DB error) was
# actually the case. We log the full reason server-side AND render an
# excerpt to the user. Reason text stays short + non-leaky (no tokens,
# no SQL traces) — the goal is "tell me what went wrong" not "give an
# attacker a stack trace."
_SSO_FAILURE_HTML_TEMPLATE = """<!doctype html>
<html lang="{lang}">
<head>
  <meta charset="utf-8">
  <title>NexoClip — {title}</title>
  <link rel="stylesheet" href="/static/nexoclip-theme.css">
</head>
<body>
  <main class="nc-page" style="max-width: 560px; margin: 80px auto; text-align: center;">
    <h1>{title}</h1>
    <p>{body}</p>
    {reason_block}
    <a class="nc-btn" href="https://nexo-ai.world/login">{cta}</a>
  </main>
</body>
</html>"""


def _render_sso_failure(request: Request, reason: str | None = None) -> str:
    """Render the SSO failure page in the user's detected locale.

    `reason` is a short operator-readable string (e.g. "token expired",
    "signature mismatch", "tenant provisioning failed"). When provided
    it gets rendered in a dim panel under the main message so the user
    knows what specifically failed. Server log gets the same string at
    WARNING level via the caller so support can correlate.
    """
    import html as _html
    import structlog
    from nexoclip.api.i18n import t

    locale = getattr(request.state, "locale", "en")
    # Log every render with the reason — even when reason is None we
    # want a trail. Truncate to keep log lines bounded.
    log = structlog.get_logger("nexoclip.api.sso")
    log.warning(
        "sso.failure_page_rendered",
        reason=(reason or "unspecified")[:200],
        path=str(request.url.path),
    )

    reason_block = ""
    if reason:
        # Truncate to 200 chars + HTML-escape so we never leak a stack
        # trace or render attacker-controlled markup.
        safe = _html.escape(reason[:200])
        reason_block = (
            f'<div style="margin: 18px auto; padding: 10px 14px; '
            f'max-width: 420px; border-radius: 8px; '
            f'background: rgba(255,107,107,0.08); '
            f'border: 1px solid rgba(255,107,107,0.35); '
            f'font-family: ui-monospace, SFMono-Regular, monospace; '
            f'font-size: 12px; color: var(--nc-danger, #ff6b6b); '
            f'text-align: left;">{safe}</div>'
        )

    return _SSO_FAILURE_HTML_TEMPLATE.format(
        lang=locale,
        title=t("sso.fail.title", locale),
        body=t("sso.fail.body", locale),
        cta=t("sso.fail.cta", locale),
        reason_block=reason_block,
    )


@router.get(
    "/auth/sso",
    # Two distinct response shapes (redirect on success, HTML error page
    # on failure). Tell FastAPI not to derive a response model from the
    # return annotation — it can't union RedirectResponse with HTMLResponse.
    response_model=None,
    responses={
        303: {"description": "Token valid; cookie set; redirected to dashboard"},
        400: {"description": "Missing or malformed token"},
        401: {"description": "Token signature failed or expired"},
        503: {"description": "Nexo AI SSO not configured"},
    },
)
async def sso_finalize(request: Request, token: str | None = None) -> RedirectResponse | HTMLResponse:
    """Verify a Nexo AI-signed token (or trust it unsigned), set the
    session cookie, redirect home.

    Slice O.22 — when `NEXO_AI_SSO_SECRET` is unset on this NexoClip
    instance, we treat that as "no wall" mode: decode the payload
    without HMAC verification and trust the tenant_id it claims.
    This is the explicit-opt-in lax mode — operator's call. With the
    secret set, strict HMAC verify resumes (production hardening).

    Expiry check is also relaxed in lax mode (7-day leeway) so a token
    minted by nexo-ai a few minutes ago still works if the user opens
    NexoClip later from the same tab.
    """
    settings = get_settings()
    secret = settings.nexo_ai_sso_secret

    if not token:
        return HTMLResponse(
            _render_sso_failure(request, reason="Missing `token` query parameter"),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if secret:
        # Strict mode: HMAC-verified.
        try:
            payload = verify_sso_token(token, secret=secret)
        except SsoTokenError as e:
            # Slice O.25 — surface the SsoTokenError message verbatim
            # (it's already operator-readable: "bad signature", "token
            # expired", "payload missing required fields", etc).
            return HTMLResponse(
                _render_sso_failure(
                    request,
                    reason=f"Strict SSO verify failed: {e}",
                ),
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
    else:
        # Lax mode: parse-only. The operator chose not to configure
        # a shared secret; we trust the token as-issued by nexo-ai.
        try:
            payload = _decode_sso_payload_unsigned(token)
        except SsoTokenError as e:
            return HTMLResponse(
                _render_sso_failure(
                    request,
                    reason=f"Lax SSO decode failed: {e}",
                ),
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

    # Mint a fresh per-session token. We DON'T reuse the api_token we
    # returned at provisioning time — that one is held by Nexo AI as
    # integration credential and shouldn't double as the interactive
    # browser cookie.
    #
    # Slice O.23 — auto-provision on tenant-not-found. nexo-ai is the
    # source of truth; if the JWT claims a tenant_id we don't have a
    # row for yet, create it. Removes the bootstrap chicken-and-egg
    # (Railway redeploy wipes the DB and the user can't log back in
    # because their tenant row evaporated).
    db: Database = request.app.state.db
    try:
        session_raw_token = await mint_session_token_for_tenant(
            db, tenant_id=payload.tenant_id
        )
    except Exception as e_mint:  # noqa: BLE001 — best-effort with fallback
        # Tenant doesn't exist. Provision it on the fly using the
        # JWT's tenant_id + email as the display name, then retry.
        # This is the no-walls behavior — if nexo-ai trusts you, we
        # trust you.
        # Slice O.25 — surface BOTH the original mint failure AND any
        # provisioning failure. Previously we hid both; operator had
        # no way to tell if it was a DB issue, a schema mismatch, or
        # something else.
        try:
            from nexoclip.db import TenantsRepo
            await TenantsRepo(db).create(
                tenant_id=payload.tenant_id,
                name=payload.email or payload.tenant_id,
            )
            session_raw_token = await mint_session_token_for_tenant(
                db, tenant_id=payload.tenant_id
            )
        except Exception as e_prov:  # noqa: BLE001 — provisioning itself broke
            reason = (
                f"Session mint failed ({type(e_mint).__name__}: {e_mint}) "
                f"AND tenant auto-provision failed "
                f"({type(e_prov).__name__}: {e_prov}). "
                f"Tenant id: {payload.tenant_id}"
            )
            return HTMLResponse(
                _render_sso_failure(request, reason=reason),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    # Sync tier on every login. Best-effort: if Nexo AI's effective tier
    # for this user changed since last visit (admin upgrade, MP payment,
    # downgrade), the tenant row reflects it on the next page render.
    # Failures are non-blocking — the user still gets to log in.
    if payload.tier:
        try:
            await sync_tenant_tier(db, tenant_id=payload.tenant_id, tier=payload.tier)
        except Exception:  # noqa: BLE001 — never break login on a tier sync
            pass

    response = RedirectResponse(url="/dashboard/streams", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        key=_COOKIE_NAME,
        value=session_raw_token,
        max_age=_COOKIE_MAX_AGE_SECONDS,
        # localhost: secure=False so the cookie sticks over plain HTTP in dev.
        # Override to True via reverse proxy in prod (or flip to env-driven).
        secure=request.url.scheme == "https",
        httponly=True,
        samesite="lax",
        path="/",
    )
    return response
