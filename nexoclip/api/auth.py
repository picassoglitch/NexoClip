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
        return await call_next(request)


def _extract_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        token = header[len("bearer ") :].strip()
        if token:
            return token
    return request.cookies.get(_COOKIE_NAME, "").strip()


def _unauthorized(request: Request, detail: str) -> Response:
    """Dashboard pages get an HTML redirect to /dashboard/login; API JSON otherwise."""
    if request.url.path.startswith("/dashboard"):
        from starlette.responses import RedirectResponse

        return RedirectResponse(url="/dashboard/login", status_code=status.HTTP_303_SEE_OTHER)
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": detail},
        headers={"WWW-Authenticate": "Bearer"},
    )
