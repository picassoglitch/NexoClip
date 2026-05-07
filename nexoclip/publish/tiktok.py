"""TikTok Content Posting API client.

Phase 2 implements the `direct_post` flow:
  1. POST /v2/post/publish/inbox/video/init/  -> upload_url + publish_id
  2. PUT  upload_url   (raw video bytes)      -> 200 OK
  3. (optional) POST .../publish_status/      -> poll status if needed

For the v0.1 dashboard we just need (1) + (2) and we stash the publish_id
as our `external_id` while the URL stays None until the user opens TikTok.
The metadata blob keeps the raw API responses so the publisher dashboard
can debug failures.

Errors split:
  * 4xx (except 408/409/429) -> fatal (raise PublisherError(transient=False))
  * 408/409/429 + 5xx + network/timeouts -> transient
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import structlog

from nexoclip.db.models import ConnectedAccount, VariantRow

from .oauth import RefreshedToken
from .protocol import PublisherError

_log = structlog.get_logger(__name__)

_DEFAULT_BASE_URL = "https://open.tiktokapis.com"
_REFRESH_PATH = "/v2/oauth/token/"


class TikTokError(PublisherError):
    """TikTok API error (transient or fatal)."""


@dataclass(frozen=True)
class TikTokPublishResult:
    external_id: str
    external_url: str | None
    metadata: dict[str, Any]


class TikTokClient:
    """Async TikTok client for the dispatcher."""

    def __init__(
        self,
        *,
        access_token: str,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._access_token = access_token
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def __aenter__(self) -> TikTokClient:
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
    ) -> TikTokPublishResult:
        """Init upload + push bytes. Returns external_id + raw metadata."""
        video_bytes = Path(clip_path).read_bytes()
        init = await self._init_upload(
            caption=_compose_caption(variant), size_bytes=len(video_bytes)
        )
        upload_url = init.get("upload_url")
        publish_id = init.get("publish_id")
        if not isinstance(upload_url, str) or not isinstance(publish_id, str):
            raise TikTokError(
                f"tiktok init missing upload_url or publish_id: {init!r}",
                transient=False,
            )
        await self._upload_video(upload_url, video_bytes)
        return TikTokPublishResult(
            external_id=publish_id,
            external_url=None,  # tiktok returns the URL via webhook later
            metadata={"init": init},
        )

    async def _init_upload(
        self, *, caption: str, size_bytes: int
    ) -> dict[str, Any]:
        try:
            resp = await self._client.post(
                f"{self._base_url}/v2/post/publish/inbox/video/init/",
                headers=self.auth_headers,
                json={
                    "post_info": {"caption": caption[:2200]},
                    "source_info": {"source": "FILE_UPLOAD", "video_size": size_bytes},
                },
            )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise TikTokError(f"tiktok network: {e}", transient=True) from e
        return _parse(resp)

    async def _upload_video(self, upload_url: str, body: bytes) -> None:
        try:
            resp = await self._client.put(
                upload_url,
                content=body,
                headers={"Content-Type": "video/mp4"},
            )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise TikTokError(f"tiktok upload network: {e}", transient=True) from e
        if resp.status_code >= 500 or resp.status_code in {408, 409, 429}:
            raise TikTokError(
                f"tiktok upload {resp.status_code}", transient=True
            )
        if resp.status_code >= 400:
            raise TikTokError(
                f"tiktok upload {resp.status_code}: {resp.text[:200]}",
                transient=False,
            )


def _compose_caption(variant: VariantRow) -> str:
    parts = [variant.caption.strip()]
    if variant.hashtags:
        parts.append(" ".join(f"#{h.lstrip('#')}" for h in variant.hashtags))
    return " ".join(p for p in parts if p)


def _parse(resp: httpx.Response) -> dict[str, Any]:
    if resp.status_code >= 500 or resp.status_code in {408, 409, 429}:
        raise TikTokError(f"tiktok {resp.status_code}: {resp.text[:200]}", transient=True)
    if resp.status_code >= 400:
        raise TikTokError(f"tiktok {resp.status_code}: {resp.text[:200]}", transient=False)
    try:
        body = resp.json()
    except ValueError as e:
        raise TikTokError(f"tiktok non-json response: {e}", transient=False) from e
    if not isinstance(body, dict):
        raise TikTokError(f"tiktok non-object body: {body!r}", transient=False)
    data = body.get("data")
    return data if isinstance(data, dict) else body


# ---- OAuth refresh implementation ----


async def refresh_tiktok_token(
    refresh_token: str,
    http: httpx.AsyncClient,
    *,
    client_key: str = "",
    client_secret: str = "",
    base_url: str = _DEFAULT_BASE_URL,
) -> RefreshedToken:
    """POST to /v2/oauth/token/ with grant_type=refresh_token.

    `client_key` / `client_secret` come from the dashboard's OAuth setup
    (Phase 3 wires them through; Phase 2's defaults make this fail
    explicitly until the operator configures them)."""
    if not client_key or not client_secret:
        raise PublisherError(
            "tiktok refresh requires client_key + client_secret", transient=False
        )
    resp = await http.post(
        f"{base_url}{_REFRESH_PATH}",
        data={
            "client_key": client_key,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if resp.status_code >= 500:
        raise TikTokError(f"tiktok refresh {resp.status_code}", transient=True)
    if resp.status_code >= 400:
        raise TikTokError(
            f"tiktok refresh {resp.status_code}: {resp.text[:200]}", transient=False
        )
    body = resp.json()
    if not isinstance(body, dict):
        raise TikTokError("tiktok refresh non-object body", transient=False)
    access = body.get("access_token")
    refresh = body.get("refresh_token")
    expires_in = body.get("expires_in", 0)
    if not isinstance(access, str) or not isinstance(expires_in, int | float):
        raise TikTokError(f"tiktok refresh malformed: {body!r}", transient=False)
    import datetime as _dt

    deadline = _dt.datetime.now(_dt.UTC) + _dt.timedelta(seconds=int(expires_in))
    return RefreshedToken(
        access_token=access,
        refresh_token=refresh if isinstance(refresh, str) else None,
        expires_at=deadline.isoformat(),
    )
