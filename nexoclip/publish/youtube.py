"""YouTube Data API v3 client for the Shorts publisher.

Phase 2 implements the resumable upload protocol:
  1. POST /upload/youtube/v3/videos?uploadType=resumable
     with the metadata body. The server returns an upload session URL via
     Location header.
  2. PUT to that URL with the raw bytes.
  3. The final response carries the new video resource (id, snippet, ...).

For a 60s Short the bytes fit comfortably in one chunk - we don't bother
with byte-range resume here. Phase 3 can add that for longer clips.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import structlog

from nexoclip.db.models import ConnectedAccount, VariantRow

from .oauth import RefreshedToken
from .protocol import PublisherError

_log = structlog.get_logger(__name__)

_DEFAULT_BASE_URL = "https://www.googleapis.com"
_DEFAULT_UPLOAD_BASE = "https://www.googleapis.com/upload"


class YouTubeError(PublisherError):
    """YouTube Data API error (transient or fatal)."""


@dataclass(frozen=True)
class YouTubePublishResult:
    external_id: str
    external_url: str | None
    metadata: dict[str, Any]


class YouTubeClient:
    """Async YouTube Data API client for the dispatcher."""

    def __init__(
        self,
        *,
        access_token: str,
        upload_base_url: str = _DEFAULT_UPLOAD_BASE,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._access_token = access_token
        self._upload_base = upload_base_url.rstrip("/")
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def __aenter__(self) -> YouTubeClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_client:
            await self._client.aclose()

    @property
    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token}"}

    async def publish(
        self,
        *,
        clip_path: str,
        variant: VariantRow,
        account: ConnectedAccount,
    ) -> YouTubePublishResult:
        """Init resumable upload + push bytes. Returns external_id + URL."""
        video_bytes = Path(clip_path).read_bytes()
        metadata_body = {
            "snippet": {
                "title": _compose_title(variant),
                "description": variant.caption,
                "tags": list(variant.hashtags or []),
                "categoryId": "22",  # 22 = People & Blogs
            },
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": False,
            },
        }
        upload_url = await self._init_resumable_upload(
            metadata_body=metadata_body, size_bytes=len(video_bytes)
        )
        result = await self._upload_video(upload_url, video_bytes)
        video_id = result.get("id")
        if not isinstance(video_id, str):
            raise YouTubeError(
                f"youtube response missing video id: {result!r}", transient=False
            )
        return YouTubePublishResult(
            external_id=video_id,
            external_url=f"https://www.youtube.com/watch?v={video_id}",
            metadata=result,
        )

    async def _init_resumable_upload(
        self, *, metadata_body: dict[str, Any], size_bytes: int
    ) -> str:
        try:
            resp = await self._client.post(
                f"{self._upload_base}/youtube/v3/videos"
                "?uploadType=resumable&part=snippet,status",
                headers={
                    **self.auth_headers,
                    "Content-Type": "application/json; charset=UTF-8",
                    "X-Upload-Content-Length": str(size_bytes),
                    "X-Upload-Content-Type": "video/mp4",
                },
                content=json.dumps(metadata_body),
            )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise YouTubeError(f"youtube init network: {e}", transient=True) from e
        if resp.status_code >= 500 or resp.status_code in {408, 409, 429}:
            raise YouTubeError(
                f"youtube init {resp.status_code}: {resp.text[:200]}",
                transient=True,
            )
        if resp.status_code >= 400:
            raise YouTubeError(
                f"youtube init {resp.status_code}: {resp.text[:200]}",
                transient=False,
            )
        location = resp.headers.get("location")
        if not location:
            raise YouTubeError(
                "youtube init returned 200 but no Location header", transient=False
            )
        return str(location)

    async def _upload_video(self, upload_url: str, body: bytes) -> dict[str, Any]:
        try:
            resp = await self._client.put(
                upload_url,
                content=body,
                headers={"Content-Type": "video/mp4", **self.auth_headers},
            )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise YouTubeError(
                f"youtube upload network: {e}", transient=True
            ) from e
        if resp.status_code >= 500 or resp.status_code in {408, 409, 429}:
            raise YouTubeError(
                f"youtube upload {resp.status_code}: {resp.text[:200]}",
                transient=True,
            )
        if resp.status_code >= 400:
            raise YouTubeError(
                f"youtube upload {resp.status_code}: {resp.text[:200]}",
                transient=False,
            )
        try:
            body_json = resp.json()
        except ValueError as e:
            raise YouTubeError(
                f"youtube upload non-json: {e}", transient=False
            ) from e
        if not isinstance(body_json, dict):
            raise YouTubeError(
                f"youtube upload non-object body: {body_json!r}", transient=False
            )
        return body_json


def _compose_title(variant: VariantRow) -> str:
    title_card = (variant.title_card_text or "").strip()
    if title_card:
        return title_card[:100]
    cap = variant.caption.strip()
    return cap[:100] if cap else "Clip"


# ---- OAuth refresh implementation ----


_GOOGLE_REFRESH_URL = "https://oauth2.googleapis.com/token"


async def refresh_youtube_token(
    refresh_token: str,
    http: httpx.AsyncClient,
    *,
    client_id: str = "",
    client_secret: str = "",
) -> RefreshedToken:
    """POST to Google's OAuth token endpoint with grant_type=refresh_token."""
    if not client_id or not client_secret:
        raise PublisherError(
            "youtube refresh requires client_id + client_secret", transient=False
        )
    resp = await http.post(
        _GOOGLE_REFRESH_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if resp.status_code >= 500:
        raise YouTubeError(f"youtube refresh {resp.status_code}", transient=True)
    if resp.status_code >= 400:
        raise YouTubeError(
            f"youtube refresh {resp.status_code}: {resp.text[:200]}",
            transient=False,
        )
    body = resp.json()
    if not isinstance(body, dict):
        raise YouTubeError("youtube refresh non-object body", transient=False)
    access = body.get("access_token")
    expires_in = body.get("expires_in", 0)
    if not isinstance(access, str) or not isinstance(expires_in, int | float):
        raise YouTubeError(f"youtube refresh malformed: {body!r}", transient=False)
    deadline = _dt.datetime.now(_dt.UTC) + _dt.timedelta(seconds=int(expires_in))
    # Google's refresh response usually doesn't include a new refresh_token;
    # we'll keep the existing one.
    return RefreshedToken(
        access_token=access,
        refresh_token=None,
        expires_at=deadline.isoformat(),
    )
