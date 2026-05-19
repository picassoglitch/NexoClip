"""Bearer-token auth middleware.

Reads `Authorization: Bearer <raw>`, hashes the raw token, looks up
`api_tokens.hash`, and stashes `(tenant_id, scope)` on `request.state`.
The dependency `tenant_binder` in `deps.py` binds the contextvar for the
duration of the handler so repos pick it up.

Routes whose path matches `_PUBLIC_PATHS` skip auth - reserved for
liveness/readiness checks that monitoring can hit unauthenticated.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from nexoclip.db import ApiTokensRepo, Database
from nexoclip.tenancy import hash_token

# These paths are exempt from auth. Keep tight - everything else MUST carry
# a valid bearer token before any handler logic runs.
# Anything matching `_PUBLIC_PREFIXES` (path startswith) skips auth.
# `/dashboard/login` is public so the login form can render and POST without
# a token; everything under `/dashboard` past login still needs auth via cookie.
_PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/",
        "/healthz",
        "/readyz",
        "/openapi.json",
        "/docs",
        "/redoc",
        # AI / search-bot discovery surface — always public so crawlers
        # can find NexoClip without auth (dashboard.html landing page is
        # also public via the "/" entry above).
        "/llms.txt",
        "/robots.txt",
        "/sitemap.xml",
    }
)
_PUBLIC_PREFIXES: tuple[str, ...] = (
    "/dashboard/login",
    "/static/",
    # Slice M.4 — OBS Browser Source overlays. Configuration travels
    # via the query string (channelId / handle / scale / etc), not
    # via cookies — OBS can't carry the dashboard's session token,
    # so requiring auth here would make the feature unusable. The
    # overlay routes themselves do NO database writes and only read
    # the parameters they were called with.
    "/overlay/",
    # Slice NX.1 — Nexo AI integration surfaces. Both carry their own
    # auth (admin bearer secret and HMAC SSO token respectively); the
    # handler enforces the right scheme. We exempt them from the
    # tenant-token middleware here so the handler runs at all.
    "/api/admin/",
    "/auth/sso",
    # Slice O.44 — internal endpoints (e.g. signed-URL audio fetch the
    # Modal Whisper provider pulls from). Each route in /api/internal/
    # carries its own HMAC signature check, so the bearer-cookie path
    # would only get in the way (Modal can't carry our cookie).
    "/api/internal/",
)
_COOKIE_NAME = "nexoclip_token"


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Authenticate every non-public request with a bearer API token.

    Reads the token from `Authorization: Bearer ...` first; falls through
    to the `nexoclip_token` cookie so the HTMX dashboard works after a
    `POST /dashboard/login` sets the cookie.
    """

    def __init__(self, app: object, *, db: Database) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._db = db

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Slice O.24 — detect locale FIRST (even for public paths), so
        # the landing page + login form render in the user's language.
        # Cheap: parses a single header string, no I/O.
        try:
            from nexoclip.api.i18n import detect_locale
            request.state.locale = detect_locale(request)
        except Exception:  # noqa: BLE001 — i18n is non-essential
            request.state.locale = "en"

        path = request.url.path
        if path in _PUBLIC_PATHS or any(path.startswith(p) for p in _PUBLIC_PREFIXES):
            return await call_next(request)

        raw = _extract_token(request)
        if not raw:
            return _unauthorized(request, "missing bearer token")

        try:
            token_hash = hash_token(raw)
        except Exception:
            return _unauthorized(request, "invalid token")

        token_row = await ApiTokensRepo(self._db).lookup_by_hash(token_hash)
        if token_row is None:
            return _unauthorized(request, "unknown token")

        request.state.tenant_id = token_row.tenant_id
        request.state.token_scope = token_row.scope

        # Slice O.1 — also resolve the tenant's subscription tier here
        # so dashboard templates can render the tier chip via
        # `{{ request.state.tenant_tier }}` without every handler having
        # to pass it in TemplateResponse context. Best-effort: any
        # lookup failure falls back to "free" (the default migration
        # value, also the conservative choice from a watermark-policy
        # standpoint).
        #
        # Slice NX.2 — also stash tenant.status so handlers that DO actual
        # work (pipeline kickoff, publish enqueue, LLM calls) can gate on
        # 'active'. Base template reads `request.state.tenant_status` to
        # render a "paused by Nexo AI" banner.
        try:
            from nexoclip.db import TenantsRepo
            tenant = await TenantsRepo(self._db).get(token_row.tenant_id)
            request.state.tenant_tier = (tenant.tier if tenant else "free") or "free"
            request.state.tenant_status = (
                (tenant.status if tenant else "active") or "active"
            )
            # Slice NX.4 — token balance cache. Populated by the usage reporter
            # after each LLM call. None until the first call lands (the chip
            # shows a "—" placeholder in that case).
            if tenant is not None and tenant.cached_balance_at:
                request.state.token_balance = {
                    "remaining": tenant.cached_balance_remaining or 0,
                    "unlimited": bool(tenant.cached_balance_unlimited),
                    "monthly_used": tenant.cached_balance_monthly_used or 0,
                    "at": tenant.cached_balance_at,
                }
            else:
                request.state.token_balance = None
        except Exception:  # noqa: BLE001 — best-effort
            request.state.tenant_tier = "free"
            request.state.tenant_status = "active"
            request.state.token_balance = None

        # Slice O.9 — admin gating. Templates read `request.state.is_admin`
        # to decide whether to render operator-only nav items (LLM spend,
        # LLM settings). Driven by the `NEXOCLIP_ADMIN_TENANT_IDS` env
        # var (comma-separated). Default empty → no admins (creator UX
        # for everyone, including local dev). Set it to your own tenant
        # id to flip those nav items back on.
        try:
            from nexoclip.settings import get_settings
            raw = (get_settings().admin_tenant_ids or "").strip()
            admin_ids = {p.strip() for p in raw.split(",") if p.strip()}
            request.state.is_admin = token_row.tenant_id in admin_ids
        except Exception:  # noqa: BLE001 — best-effort
            request.state.is_admin = False

        return await call_next(request)


def _extract_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        token = header[len("bearer ") :].strip()
        if token:
            return token
    return request.cookies.get(_COOKIE_NAME, "").strip()


def _unauthorized(request: Request, detail: str) -> Response:
    """Dashboard pages get an HTML redirect to nexo-ai; API JSON otherwise.

    Slice O.23 — login removed. nexo-ai is the only gatekeeper. Any
    /dashboard/* hit without a valid session cookie bounces straight
    to nexo-ai's login. There is no in-house fallback page anymore.
    """
    if request.url.path.startswith("/dashboard"):
        from starlette.responses import RedirectResponse

        try:
            from nexoclip.settings import get_settings
            login_url = (get_settings().nexo_ai_login_url or "").strip()
        except Exception:  # noqa: BLE001 — settings unreachable, hardcode default
            login_url = ""
        target = login_url or "https://nexo-ai.world/login"
        return RedirectResponse(url=target, status_code=status.HTTP_303_SEE_OTHER)
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": detail},
        headers={"WWW-Authenticate": "Bearer"},
    )
