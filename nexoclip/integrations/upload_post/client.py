"""Async httpx client for the upload-post REST API.

One thin method per endpoint we use. Each returns a typed dataclass
or a plain dict for read-many responses; methods raise
`UploadPostError` on any non-2xx or schema-violating response.

Auth model is dead simple: every call carries the same company-wide
API key in the `Authorization: Apikey <key>` header. There's no per-
tenant auth on our side — multi-tenancy is just the `user` field on
the upload payload (the upload-post profile username, which we
manage in `profiles.py`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import httpx

from nexoclip.errors import NexoClipError


class UploadPostError(NexoClipError):
    """Raised on any failure in an upload-post API call.

    Caller surfaces as a user-visible error on the publish UI and
    logs the wrapped body for diagnosis. `status_code` is preserved
    so callers can branch (e.g. 409 = profile exists, retry as get).
    """

    def __init__(
        self, message: str, *, status_code: int | None = None, body: object = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


_DEFAULT_BASE_URL: Final = "https://api.upload-post.com"


@dataclass(frozen=True, slots=True)
class UploadPostProfile:
    """Subset of the GET /api/uploadposts/users/{username} response.

    `social_accounts` is a dict from platform name → either a string
    (legacy / unconnected) or an object with display_name + avatar.
    Caller is expected to handle both shapes.
    """

    username: str
    created_at: str
    social_accounts: dict[str, Any]


@dataclass(frozen=True, slots=True)
class UploadJwtLink:
    """Result of POST /api/uploadposts/users/generate-jwt.

    `access_url` is the magic link the tenant's browser visits to
    connect their social accounts. Valid for `duration` (their docs
    say 48h; we surface whatever they return).
    """

    access_url: str
    duration: str


@dataclass(frozen=True, slots=True)
class UploadResult:
    """Result of POST /api/upload (or upload_photos / upload_text).

    `is_async` is True when upload-post took the job for background
    processing; in that case caller polls
    `get_status(request_id=...)`. `platforms` is the per-platform
    result dict — only present on the synchronous path; absent on
    async + scheduled.
    """

    success: bool
    request_id: str
    job_id: str | None
    is_async: bool
    platforms: dict[str, Any] | None
    message: str | None


@dataclass(frozen=True, slots=True)
class UploadStatus:
    """Result of GET /api/uploadposts/status?request_id=...

    `status` follows upload-post's vocab: PENDING / PROCESSING /
    FINISHED / ERROR. `result.platforms` is populated once FINISHED.
    """

    request_id: str
    status: str
    is_async: bool
    result: dict[str, Any] | None


class UploadPostClient:
    """Thin async wrapper. Construct once per request handler or
    long-lived task; the underlying httpx client is re-used across
    calls when you pass `http` in, otherwise each method opens +
    closes its own short-lived client.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        http: httpx.AsyncClient | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        if not api_key:
            raise UploadPostError("UPLOAD_POST_API_KEY is not configured")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http = http
        self._timeout_s = timeout_s

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Apikey {self._api_key}"}

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        """Single round-trip. Returns parsed JSON body on 2xx; raises
        UploadPostError on anything else. Reuses the injected
        AsyncClient when present, else spawns a one-shot."""
        url = f"{self._base_url}{path}"
        headers = dict(self._headers)
        if json is not None:
            headers["Content-Type"] = "application/json"
        client = self._http or httpx.AsyncClient(
            timeout=timeout_s or self._timeout_s,
        )
        try:
            resp = await client.request(
                method, url, headers=headers, json=json, params=params,
            )
        finally:
            if self._http is None:
                await client.aclose()

        if resp.status_code >= 400:
            body: object
            try:
                body = resp.json()
            except ValueError:
                body = resp.text[:500]
            raise UploadPostError(
                f"upload-post {method} {path} returned HTTP {resp.status_code}",
                status_code=resp.status_code,
                body=body,
            )
        try:
            return resp.json()
        except ValueError as e:
            raise UploadPostError(
                f"upload-post {method} {path} returned non-JSON",
                status_code=resp.status_code,
                body=resp.text[:500],
            ) from e

    # ---- Profile management ----

    async def create_user_profile(self, username: str) -> UploadPostProfile:
        """POST /api/uploadposts/users — create a profile keyed by
        our tenant_id. 409 means it already exists; caller treats
        that as success and follows up with get_user_profile()."""
        body = await self._request(
            "POST",
            "/api/uploadposts/users",
            json={"username": username},
        )
        profile = body.get("profile") if isinstance(body, dict) else None
        if not isinstance(profile, dict):
            raise UploadPostError(
                "create_user_profile response missing profile",
                body=body,
            )
        return UploadPostProfile(
            username=str(profile.get("username") or username),
            created_at=str(profile.get("created_at") or ""),
            social_accounts=(
                profile.get("social_accounts") or {}
                if isinstance(profile.get("social_accounts"), dict) else {}
            ),
        )

    async def get_user_profile(self, username: str) -> UploadPostProfile:
        """GET /api/uploadposts/users/{username} — read a profile's
        social_accounts. Used by the dashboard to show which platforms
        the tenant has connected."""
        body = await self._request(
            "GET", f"/api/uploadposts/users/{username}",
        )
        profile = body.get("profile") if isinstance(body, dict) else None
        if not isinstance(profile, dict):
            raise UploadPostError(
                "get_user_profile response missing profile", body=body,
            )
        return UploadPostProfile(
            username=str(profile.get("username") or username),
            created_at=str(profile.get("created_at") or ""),
            social_accounts=(
                profile.get("social_accounts") or {}
                if isinstance(profile.get("social_accounts"), dict) else {}
            ),
        )

    async def generate_connect_jwt(
        self,
        username: str,
        *,
        redirect_url: str | None = None,
        platforms: list[str] | None = None,
        logo_image: str | None = None,
        connect_title: str | None = None,
    ) -> UploadJwtLink:
        """POST /api/uploadposts/users/generate-jwt — mint the 48h
        magic link the tenant visits to connect their social
        accounts on upload-post's hosted UI."""
        payload: dict[str, Any] = {"username": username}
        if redirect_url:
            payload["redirect_url"] = redirect_url
        if platforms:
            payload["platforms"] = platforms
        if logo_image:
            payload["logo_image"] = logo_image
        if connect_title:
            payload["connect_title"] = connect_title
        body = await self._request(
            "POST",
            "/api/uploadposts/users/generate-jwt",
            json=payload,
        )
        access_url = body.get("access_url") if isinstance(body, dict) else None
        if not isinstance(access_url, str) or not access_url:
            raise UploadPostError(
                "generate_connect_jwt response missing access_url", body=body,
            )
        return UploadJwtLink(
            access_url=access_url,
            duration=str(body.get("duration") or "48h"),
        )

    # ---- Publishing ----

    async def upload_video_from_url(
        self,
        *,
        username: str,
        video_url: str,
        platforms: list[str],
        title: str | None = None,
        description: str | None = None,
        request_id: str | None = None,
        async_upload: bool = True,
        scheduled_date: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> UploadResult:
        """POST /api/upload — publish a video by URL.

        We default `async_upload=True` because the upload-post side
        may take many seconds (download our signed URL → re-host →
        per-platform publish). Async returns 202 with the request_id
        and we poll status from there.

        `extra` carries any platform-specific overrides (e.g.
        `tiktok_disable_duet`, `youtube_visibility`). The full
        catalog is in upload-post's docs; we don't enumerate them
        here, just pass through.
        """
        payload: dict[str, Any] = {
            "user": username,
            "platform[]": platforms,  # upload-post expects array form
            "video": video_url,
        }
        if title:
            payload["title"] = title
        if description:
            payload["description"] = description
        if request_id:
            payload["request_id"] = request_id
        if async_upload:
            payload["async_upload"] = True
        if scheduled_date:
            payload["scheduled_date"] = scheduled_date
        if extra:
            payload.update(extra)

        body = await self._request(
            "POST", "/api/upload", json=payload,
            # Async path returns fast; sync can take minutes for
            # multi-platform publish so allow a much longer ceiling.
            timeout_s=300.0,
        )
        if not isinstance(body, dict):
            raise UploadPostError("upload response is not an object", body=body)
        return UploadResult(
            success=bool(body.get("success")),
            request_id=str(body.get("request_id") or request_id or ""),
            job_id=(
                str(body["job_id"]) if isinstance(body.get("job_id"), str) else None
            ),
            is_async=bool(async_upload or body.get("message", "").lower().count("queued")),
            platforms=(
                body.get("platforms") if isinstance(body.get("platforms"), dict) else None
            ),
            message=(
                str(body["message"]) if isinstance(body.get("message"), str) else None
            ),
        )

    # ---- Status + history ----

    async def get_status(
        self,
        *,
        request_id: str | None = None,
        job_id: str | None = None,
    ) -> UploadStatus:
        """GET /api/uploadposts/status — poll an async upload (via
        request_id) or a scheduled post (via job_id). Exactly one
        of the two must be set."""
        if not (request_id or job_id):
            raise UploadPostError("get_status needs request_id or job_id")
        params: dict[str, Any] = {}
        if request_id:
            params["request_id"] = request_id
        if job_id:
            params["job_id"] = job_id
        body = await self._request(
            "GET", "/api/uploadposts/status", params=params,
        )
        if not isinstance(body, dict):
            raise UploadPostError("status response is not an object", body=body)
        return UploadStatus(
            request_id=str(body.get("request_id") or request_id or ""),
            status=str(body.get("status") or "PENDING"),
            is_async=bool(body.get("is_async")),
            result=(
                body.get("result") if isinstance(body.get("result"), dict) else None
            ),
        )

    async def get_history(
        self, *, page: int = 1, limit: int = 20,
    ) -> dict[str, Any]:
        """GET /api/uploadposts/history — feed of past uploads. Returns
        the raw dict (history list + total + page + limit) so the UI
        can render whatever the backend gives back."""
        body = await self._request(
            "GET", "/api/uploadposts/history",
            params={"page": page, "limit": limit},
        )
        if not isinstance(body, dict):
            raise UploadPostError("history response is not an object", body=body)
        return body

    async def get_scheduled(self) -> list[dict[str, Any]]:
        """GET /api/uploadposts/schedule — list scheduled (queued)
        posts not yet published. Returns the raw array."""
        body = await self._request("GET", "/api/uploadposts/schedule")
        if isinstance(body, list):
            return body
        if isinstance(body, dict) and isinstance(body.get("scheduled"), list):
            return body["scheduled"]
        raise UploadPostError(
            "scheduled response shape unexpected", body=body,
        )


__all__ = [
    "UploadPostClient",
    "UploadPostError",
    "UploadPostProfile",
    "UploadJwtLink",
    "UploadResult",
    "UploadStatus",
]
