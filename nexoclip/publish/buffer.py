"""Thin Buffer API client.

Buffer's public API is a REST surface; we only exercise the one endpoint
we need: `POST /1/updates/create.json`. Auth is `access_token` (Phase 3
swaps in OAuth refresh; Phase 1 stores the access token raw inside
`connected_accounts.oauth_blob_json`).

The client distinguishes transient (5xx, 408, 429, network/timeout)
from fatal (4xx other than 429) failures. The orchestrator (`service.py`)
retries transient ones with exponential backoff and gives up after a
configured cap; fatal ones go straight to `failed`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from .protocol import PublisherError

if TYPE_CHECKING:
    from nexoclip.db.models import ConnectedAccount, VariantRow

# Buffer Classic API. Phase 3 will switch to the new "Publishing API" once
# their migration is GA; the client surface here doesn't change.
DEFAULT_BASE_URL = "https://api.bufferapp.com"

_TRANSIENT_STATUSES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})


class BufferError(PublisherError):
    """Buffer API call failed.

    Inherits the `transient` flag from `PublisherError`; the orchestrator
    retries transient errors with backoff and gives up after `max_attempts`.
    """

    def __init__(
        self,
        message: str,
        *,
        transient: bool = False,
        status_code: int | None = None,
    ):
        super().__init__(message, transient=transient)
        self.status_code = status_code


class BufferClient:
    """Async Buffer API client.

    Phase 1 only implements the create-update flow. The client owns its
    httpx.AsyncClient so the worker can reuse a single connection across
    a drain pass.
    """

    def __init__(
        self,
        access_token: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 15.0,
    ):
        if not access_token:
            raise BufferError("BufferClient requires a non-empty access_token")
        self._access_token = access_token
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> BufferClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def create_update(
        self,
        *,
        profile_external_id: str,
        text: str,
        media_url: str | None = None,
    ) -> dict[str, Any]:
        """Queue a single update on a Buffer profile.

        Returns the JSON Buffer responds with - includes the external id
        we record on the publish_job row.
        """
        url = f"{self._base_url}/1/updates/create.json"
        data: dict[str, Any] = {
            "access_token": self._access_token,
            "profile_ids[]": profile_external_id,
            "text": text,
        }
        if media_url:
            data["media[link]"] = media_url

        try:
            response = await self._client.post(url, data=data)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
            raise BufferError(f"buffer transient: {e}", transient=True) from e
        except httpx.HTTPError as e:
            raise BufferError(f"buffer http error: {e}", transient=False) from e

        if response.status_code in _TRANSIENT_STATUSES:
            raise BufferError(
                f"buffer {response.status_code}: {response.text[:200]}",
                transient=True,
                status_code=response.status_code,
            )
        if response.status_code >= 400:
            raise BufferError(
                f"buffer {response.status_code}: {response.text[:200]}",
                transient=False,
                status_code=response.status_code,
            )

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as e:
            raise BufferError(f"buffer returned non-JSON: {e}", transient=False) from e
        return payload


@dataclass(frozen=True)
class BufferPublishResult:
    external_id: str
    external_url: str | None
    metadata: dict[str, Any]


class BufferPublisher:
    """Adapter so BufferClient fits the unified `Publisher` protocol.

    The dispatcher's loop calls `publisher.publish(clip_path, variant,
    account)` regardless of platform; the adapter turns that into Buffer's
    `create_update(profile_external_id, text)`.
    """

    def __init__(self, client: BufferClient) -> None:
        self._client = client

    async def __aenter__(self) -> BufferPublisher:
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.__aexit__(*exc)

    async def publish(
        self,
        *,
        clip_path: str,
        variant: VariantRow,
        account: ConnectedAccount,
    ) -> BufferPublishResult:
        text = _compose_caption(variant)
        resp = await self._client.create_update(
            profile_external_id=account.external_id, text=text
        )
        return BufferPublishResult(
            external_id=_extract_external_id(resp),
            external_url=None,
            metadata=resp,
        )


def _compose_caption(variant: VariantRow) -> str:
    parts = [variant.caption.strip()]
    if variant.hashtags:
        parts.append(" ".join(f"#{h.lstrip('#')}" for h in variant.hashtags))
    return " ".join(p for p in parts if p)


def _extract_external_id(payload: dict[str, Any]) -> str:
    updates = payload.get("updates")
    if isinstance(updates, list) and updates:
        first = updates[0]
        if isinstance(first, dict):
            first_id = first.get("id")
            if isinstance(first_id, str):
                return first_id
    top_id = payload.get("id")
    if isinstance(top_id, str):
        return top_id
    return ""
