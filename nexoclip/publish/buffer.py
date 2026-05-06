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

from typing import Any

import httpx

# Buffer Classic API. Phase 3 will switch to the new "Publishing API" once
# their migration is GA; the client surface here doesn't change.
DEFAULT_BASE_URL = "https://api.bufferapp.com"

_TRANSIENT_STATUSES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})


class BufferError(Exception):
    """Buffer API call failed.

    `transient=True` means the orchestrator should retry; `transient=False`
    means give up immediately.
    """

    def __init__(
        self,
        message: str,
        *,
        transient: bool = False,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.transient = transient
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
