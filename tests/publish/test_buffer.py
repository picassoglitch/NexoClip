"""BufferClient unit tests using respx to mock the Buffer API."""

from __future__ import annotations

import httpx
import pytest
import respx

from nexoclip.publish import BufferClient, BufferError
from nexoclip.publish.buffer import DEFAULT_BASE_URL


@respx.mock
async def test_create_update_returns_payload() -> None:
    route = respx.post(f"{DEFAULT_BASE_URL}/1/updates/create.json").mock(
        return_value=httpx.Response(200, json={"updates": [{"id": "buf_update_1"}]})
    )
    async with BufferClient("btok_abc") as c:
        out = await c.create_update(profile_external_id="prof_1", text="hi")
    assert out == {"updates": [{"id": "buf_update_1"}]}
    assert route.called


@respx.mock
async def test_5xx_raises_transient() -> None:
    respx.post(f"{DEFAULT_BASE_URL}/1/updates/create.json").mock(
        return_value=httpx.Response(503, text="upstream busy")
    )
    async with BufferClient("btok_abc") as c:
        with pytest.raises(BufferError) as exc_info:
            await c.create_update(profile_external_id="p", text="t")
    assert exc_info.value.transient is True
    assert exc_info.value.status_code == 503


@respx.mock
async def test_4xx_raises_fatal() -> None:
    respx.post(f"{DEFAULT_BASE_URL}/1/updates/create.json").mock(
        return_value=httpx.Response(401, text="bad token")
    )
    async with BufferClient("btok_abc") as c:
        with pytest.raises(BufferError) as exc_info:
            await c.create_update(profile_external_id="p", text="t")
    assert exc_info.value.transient is False
    assert exc_info.value.status_code == 401


@respx.mock
async def test_429_is_transient() -> None:
    respx.post(f"{DEFAULT_BASE_URL}/1/updates/create.json").mock(
        return_value=httpx.Response(429, text="slow down")
    )
    async with BufferClient("btok_abc") as c:
        with pytest.raises(BufferError) as exc_info:
            await c.create_update(profile_external_id="p", text="t")
    assert exc_info.value.transient is True


def test_empty_token_rejected() -> None:
    with pytest.raises(BufferError, match="non-empty"):
        BufferClient("")
