"""Instagram Reels publisher (Graph API).

The Reels endpoint takes a *public video URL* rather than multipart bytes,
so the operator must store a `video_url_template` (e.g.
"https://media.example.com/clips/{clip_id}.mp4") in the connected
account's `oauth_blob` for now. Phase 3's cloud-lift swaps this for
auto-provisioned S3 URLs and the template requirement goes away.

Two-step publish:
  1. POST /{ig-user-id}/media       create the container (media_type=REELS)
  2. POST /{ig-user-id}/media_publish  with creation_id=container_id

`account.external_id` is the IG user/business account id.

Errors split:
  * 408 / 409 / 429 / 5xx / network / timeout -> transient (retry)
  * other 4xx -> fatal (give up)
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

_DEFAULT_GRAPH_BASE_URL = "https://graph.facebook.com/v22.0"
_DEFAULT_OAUTH_BASE_URL = "https://graph.facebook.com/v22.0"


class InstagramError(PublisherError):
    """Instagram Graph API error (transient or fatal)."""


@dataclass(frozen=True)
class InstagramPublishResult:
    external_id: str
    external_url: str | None
    metadata: dict[str, Any]


class InstagramClient:
    """Async Instagram Reels client for the dispatcher."""

    def __init__(
        self,
        *,
        access_token: str,
        base_url: str = _DEFAULT_GRAPH_BASE_URL,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._access_token = access_token
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def __aenter__(self) -> InstagramClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def publish(
        self,
        *,
        clip_path: str,
        variant: VariantRow,
        account: ConnectedAccount,
    ) -> InstagramPublishResult:
        """Two-step IG Reels publish (create container, then publish)."""
        video_url = _resolve_video_url(account, clip_path=clip_path)
        ig_user_id = account.external_id
        if not ig_user_id:
            raise InstagramError(
                f"instagram account {account.id} has no external_id (ig user id)",
                transient=False,
            )
        caption = _compose_caption(variant)

        container = await self._create_container(
            ig_user_id=ig_user_id, video_url=video_url, caption=caption
        )
        container_id = container.get("id")
        if not isinstance(container_id, str):
            raise InstagramError(
                f"instagram /media missing container id: {container!r}",
                transient=False,
            )

        published = await self._publish_container(
            ig_user_id=ig_user_id, creation_id=container_id
        )
        media_id = published.get("id")
        if not isinstance(media_id, str):
            raise InstagramError(
                f"instagram /media_publish missing media id: {published!r}",
                transient=False,
            )
        return InstagramPublishResult(
            external_id=media_id,
            external_url=f"https://www.instagram.com/reel/{media_id}/",
            metadata={"container": container, "publish": published},
        )

    async def _create_container(
        self, *, ig_user_id: str, video_url: str, caption: str
    ) -> dict[str, Any]:
        try:
            resp = await self._client.post(
                f"{self._base_url}/{ig_user_id}/media",
                params={"access_token": self._access_token},
                data={
                    "media_type": "REELS",
                    "video_url": video_url,
                    "caption": caption[:2200],
                    "share_to_feed": "true",
                },
            )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise InstagramError(f"instagram network: {e}", transient=True) from e
        return _parse(resp, op="container")

    async def _publish_container(
        self, *, ig_user_id: str, creation_id: str
    ) -> dict[str, Any]:
        try:
            resp = await self._client.post(
                f"{self._base_url}/{ig_user_id}/media_publish",
                params={"access_token": self._access_token},
                data={"creation_id": creation_id},
            )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise InstagramError(
                f"instagram publish network: {e}", transient=True
            ) from e
        return _parse(resp, op="publish")


def _resolve_video_url(account: ConnectedAccount, *, clip_path: str) -> str:
    """Render `oauth_blob.video_url_template` with the clip's stem.

    Until Phase 3's cloud-lift provisions S3 URLs automatically, the
    operator must configure a public URL pattern per account. Missing
    template = fatal config error.
    """
    blob = account.oauth_blob or {}
    template = blob.get("video_url_template")
    if not isinstance(template, str) or not template:
        raise InstagramError(
            f"instagram account {account.id} oauth_blob is missing "
            "'video_url_template' — Phase 3 cloud-lift will provision S3 "
            "URLs automatically; until then, configure a public URL "
            "template like 'https://media.example.com/clips/{clip_id}.mp4'",
            transient=False,
        )
    clip_id = Path(clip_path).stem
    try:
        return template.format(clip_id=clip_id)
    except (KeyError, IndexError) as e:
        raise InstagramError(
            f"instagram video_url_template render failed: {e}", transient=False
        ) from e


def _compose_caption(variant: VariantRow) -> str:
    parts = [variant.caption.strip()]
    if variant.hashtags:
        parts.append(" ".join(f"#{h.lstrip('#')}" for h in variant.hashtags))
    return " ".join(p for p in parts if p)


def _parse(resp: httpx.Response, *, op: str) -> dict[str, Any]:
    if resp.status_code >= 500 or resp.status_code in {408, 409, 429}:
        raise InstagramError(
            f"instagram {op} {resp.status_code}: {resp.text[:200]}", transient=True
        )
    if resp.status_code >= 400:
        raise InstagramError(
            f"instagram {op} {resp.status_code}: {resp.text[:200]}",
            transient=False,
        )
    try:
        body = resp.json()
    except ValueError as e:
        raise InstagramError(
            f"instagram {op} non-json: {e}", transient=False
        ) from e
    if not isinstance(body, dict):
        raise InstagramError(
            f"instagram {op} non-object body: {body!r}", transient=False
        )
    return body


# ---------------------------------------------------------------------------
# OAuth refresh
# ---------------------------------------------------------------------------


async def refresh_instagram_token(
    refresh_token: str,
    http: httpx.AsyncClient,
    *,
    app_id: str = "",
    app_secret: str = "",
    base_url: str = _DEFAULT_OAUTH_BASE_URL,
) -> RefreshedToken:
    """Extend a long-lived Instagram/Facebook token via fb_exchange_token.

    IG Business tokens are long-lived (~60 days). The Graph API "refresh"
    is really an extension - swap the current long-lived token for a new
    one with a fresh 60-day window. The old `refresh_token` argument here
    is the existing long-lived access token, not a separate refresh
    credential (Graph API doesn't issue those).

    Caller must configure `app_id` / `app_secret` on the connected
    account's oauth_blob; absence raises a fatal PublisherError so the
    refresh path doesn't silently no-op.
    """
    if not app_id or not app_secret:
        raise PublisherError(
            "instagram refresh requires app_id + app_secret on oauth_blob",
            transient=False,
        )
    resp = await http.get(
        f"{base_url}/oauth/access_token",
        params={
            "grant_type": "fb_exchange_token",
            "client_id": app_id,
            "client_secret": app_secret,
            "fb_exchange_token": refresh_token,
        },
    )
    if resp.status_code >= 500:
        raise InstagramError(f"instagram refresh {resp.status_code}", transient=True)
    if resp.status_code >= 400:
        raise InstagramError(
            f"instagram refresh {resp.status_code}: {resp.text[:200]}",
            transient=False,
        )
    body = resp.json()
    if not isinstance(body, dict):
        raise InstagramError(
            "instagram refresh non-object body", transient=False
        )
    access = body.get("access_token")
    expires_in = body.get("expires_in", 0)
    if not isinstance(access, str) or not isinstance(expires_in, int | float):
        raise InstagramError(
            f"instagram refresh malformed: {body!r}", transient=False
        )
    import datetime as _dt

    deadline = _dt.datetime.now(_dt.UTC) + _dt.timedelta(seconds=int(expires_in))
    # Graph "refresh" returns a new access_token but no new refresh credential
    # — the new long-lived token IS the refresh credential next time.
    return RefreshedToken(
        access_token=access,
        refresh_token=access,
        expires_at=deadline.isoformat(),
    )
