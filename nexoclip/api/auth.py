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
_PUBLIC_PATHS: frozenset[str] = frozenset({"/healthz", "/readyz", "/openapi.json", "/docs", "/redoc"})


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Authenticate every non-public request with a bearer API token."""

    def __init__(self, app: object, *, db: Database) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._db = db

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            return _unauthorized("missing bearer token")
        raw = header[len("bearer ") :].strip()
        if not raw:
            return _unauthorized("empty bearer token")

        try:
            token_hash = hash_token(raw)
        except Exception:
            return _unauthorized("invalid token")

        token_row = await ApiTokensRepo(self._db).lookup_by_hash(token_hash)
        if token_row is None:
            return _unauthorized("unknown token")

        request.state.tenant_id = token_row.tenant_id
        request.state.token_scope = token_row.scope
        return await call_next(request)


def _unauthorized(detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": detail},
        headers={"WWW-Authenticate": "Bearer"},
    )
