"""TikTok OAuth + user/creator-info wiring tests.

We mock httpx via `respx` (already used elsewhere in this repo for
Buffer + LLM tests). Each test pins a specific platform contract:
  - authorize URL shape (params, encoding, scopes joined by comma)
  - token exchange happy path + error body handling
  - user_info parses display_name + avatar_url + open_id from
    the nested data.user object
  - creator_info parses privacy_level_options and reflects can_post
    as the empty-options sentinel
  - refresh round-trip swaps refresh_token for the new one
"""
from __future__ import annotations

import httpx
import pytest
import respx

from nexoclip.integrations.tiktok.auth import (
    TikTokAuthError,
    build_authorize_url,
    exchange_code_for_token,
    fetch_user_info,
    query_creator_info,
    refresh_access_token,
)


# ---- authorize URL ----


def test_authorize_url_has_required_params() -> None:
    url = build_authorize_url(
        client_key="awxyz",
        redirect_uri="https://nexoclip.nexo-ai.world/connect/tiktok/callback",
        state="signed-state-token",
    )
    assert url.startswith("https://www.tiktok.com/v2/auth/authorize/?")
    # All four required params present.
    assert "client_key=awxyz" in url
    assert "response_type=code" in url
    assert "state=signed-state-token" in url
    # redirect_uri gets percent-encoded.
    assert (
        "redirect_uri=https%3A%2F%2Fnexoclip.nexo-ai.world%2Fconnect%2Ftiktok%2Fcallback"
        in url
    )
    # Scopes are comma-joined per TikTok contract (NOT %20 / +).
    assert "scope=user.info.basic%2Cvideo.upload%2Cvideo.publish" in url


def test_authorize_url_refuses_empty_client_key() -> None:
    with pytest.raises(TikTokAuthError, match="TIKTOK_CLIENT_KEY"):
        build_authorize_url(
            client_key="",
            redirect_uri="https://x.test/cb",
            state="s",
        )


# ---- token exchange ----


@pytest.mark.asyncio
async def test_exchange_code_parses_happy_response() -> None:
    body = {
        "access_token": "at_abc",
        "refresh_token": "rt_def",
        "open_id": "openid_alice",
        "scope": "user.info.basic,video.upload",
        "expires_in": 86400,
        "refresh_expires_in": 31_536_000,
        "token_type": "Bearer",
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            mock.post("https://open.tiktokapis.com/v2/oauth/token/").mock(
                return_value=httpx.Response(200, json=body)
            )
            tok = await exchange_code_for_token(
                client_key="ck",
                client_secret="cs",
                code="the-code",
                redirect_uri="https://x.test/cb",
                http=http,
            )
    assert tok.access_token == "at_abc"
    assert tok.refresh_token == "rt_def"
    assert tok.open_id == "openid_alice"
    assert tok.scope == "user.info.basic,video.upload"
    # ISO timestamps, not raw seconds.
    assert "T" in tok.expires_at
    assert "T" in tok.refresh_expires_at


@pytest.mark.asyncio
async def test_exchange_code_raises_on_tiktok_error_body() -> None:
    body = {
        "error": "invalid_grant",
        "error_description": "Authorization code has expired",
    }
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post("https://open.tiktokapis.com/v2/oauth/token/").mock(
                return_value=httpx.Response(200, json=body)
            )
            with pytest.raises(TikTokAuthError, match="invalid_grant"):
                await exchange_code_for_token(
                    client_key="ck",
                    client_secret="cs",
                    code="bad",
                    redirect_uri="https://x.test/cb",
                    http=http,
                )


@pytest.mark.asyncio
async def test_exchange_code_raises_on_5xx() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post("https://open.tiktokapis.com/v2/oauth/token/").mock(
                return_value=httpx.Response(500, text="oops")
            )
            with pytest.raises(TikTokAuthError, match="HTTP 500"):
                await exchange_code_for_token(
                    client_key="ck",
                    client_secret="cs",
                    code="x",
                    redirect_uri="https://x.test/cb",
                    http=http,
                )


@pytest.mark.asyncio
async def test_exchange_code_refuses_missing_open_id() -> None:
    """If TikTok somehow omits open_id we can't dedup the row; raise
    rather than silently storing a partial record."""
    body = {
        "access_token": "at",
        "refresh_token": "rt",
        # open_id intentionally missing
        "scope": "",
        "expires_in": 86400,
        "refresh_expires_in": 31_536_000,
    }
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post("https://open.tiktokapis.com/v2/oauth/token/").mock(
                return_value=httpx.Response(200, json=body)
            )
            with pytest.raises(TikTokAuthError, match="open_id"):
                await exchange_code_for_token(
                    client_key="ck",
                    client_secret="cs",
                    code="x",
                    redirect_uri="https://x.test/cb",
                    http=http,
                )


# ---- user info ----


@pytest.mark.asyncio
async def test_fetch_user_info_parses_nested_data_user() -> None:
    body = {
        "data": {
            "user": {
                "open_id": "openid_alice",
                "union_id": "unionid_alice",
                "avatar_url": "https://p16.tiktokcdn.com/avatar.jpg",
                "display_name": "Alice the Streamer",
            }
        }
    }
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get(
                "https://open.tiktokapis.com/v2/user/info/",
            ).mock(return_value=httpx.Response(200, json=body))
            u = await fetch_user_info(access_token="at", http=http)
    assert u.open_id == "openid_alice"
    assert u.union_id == "unionid_alice"
    assert u.display_name == "Alice the Streamer"
    assert u.avatar_url == "https://p16.tiktokcdn.com/avatar.jpg"


# ---- creator info ----


@pytest.mark.asyncio
async def test_creator_info_reports_can_post_when_options_present() -> None:
    body = {
        "data": {
            "privacy_level_options": [
                "PUBLIC_TO_EVERYONE",
                "MUTUAL_FOLLOW_FRIENDS",
                "SELF_ONLY",
            ],
            "duet_disabled": False,
            "comment_disabled": False,
            "stitch_disabled": False,
            "max_video_post_duration_sec": 600,
        }
    }
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post(
                "https://open.tiktokapis.com/v2/post/publish/creator_info/query/",
            ).mock(return_value=httpx.Response(200, json=body))
            info = await query_creator_info(access_token="at", http=http)
    assert info.can_post is True
    assert "PUBLIC_TO_EVERYONE" in info.privacy_level_options
    assert info.max_video_post_duration_sec == 600


@pytest.mark.asyncio
async def test_creator_info_empty_options_is_can_post_false() -> None:
    """Sandbox apps + restricted accounts come back with no privacy
    options. UI treats that as 'cannot post'."""
    body = {
        "data": {
            "privacy_level_options": [],
        }
    }
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post(
                "https://open.tiktokapis.com/v2/post/publish/creator_info/query/",
            ).mock(return_value=httpx.Response(200, json=body))
            info = await query_creator_info(access_token="at", http=http)
    assert info.can_post is False
    assert info.privacy_level_options == []


# ---- refresh ----


@pytest.mark.asyncio
async def test_refresh_swaps_refresh_token() -> None:
    """TikTok rotates the refresh token on every refresh. The new
    one MUST replace the stored one or the next refresh 401s."""
    body = {
        "access_token": "at_v2",
        "refresh_token": "rt_v2",
        "open_id": "openid_alice",
        "scope": "user.info.basic,video.upload",
        "expires_in": 86400,
        "refresh_expires_in": 31_536_000,
    }
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post("https://open.tiktokapis.com/v2/oauth/token/").mock(
                return_value=httpx.Response(200, json=body)
            )
            tok = await refresh_access_token(
                client_key="ck",
                client_secret="cs",
                refresh_token="rt_v1",
                http=http,
            )
    assert tok.access_token == "at_v2"
    assert tok.refresh_token == "rt_v2"  # rotated, NOT the old one
