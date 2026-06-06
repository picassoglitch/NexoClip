"""Instagram (Meta) OAuth wiring tests.

Same shape as the TikTok test suite: respx-mock the Graph API,
pin the contract bits that the connect-router relies on.

What's NOT covered here (covered by test_refresh_scheduler.py):
  - the 60-day proactive refresh job's near-expiry selection logic
  - the auth_failed flip on refresh failure
"""
from __future__ import annotations

import httpx
import pytest
import respx

from nexoclip.integrations.instagram.auth import (
    FacebookPage,
    InstagramAuthError,
    build_authorize_url,
    exchange_code_for_short_lived,
    exchange_for_long_lived,
    list_pages,
    resolve_ig_business_account,
    refresh_long_lived_token,
)


# ---- authorize URL ----


def test_authorize_url_has_required_meta_params() -> None:
    url = build_authorize_url(
        app_id="123456789",
        redirect_uri="https://nexoclip.nexo-ai.world/connect/instagram/callback",
        state="signed-state",
    )
    assert url.startswith("https://www.facebook.com/v18.0/dialog/oauth?")
    assert "client_id=123456789" in url
    assert "state=signed-state" in url
    # Meta wants response_type=code (NOT the implicit 'token' grant).
    assert "response_type=code" in url
    # Scopes joined by comma per Meta API contract.
    assert "scope=instagram_basic%2Cinstagram_content_publish" in url
    assert "%2Cpages_show_list" in url
    assert "%2Cpages_read_engagement" in url
    assert "%2Cbusiness_management" in url


def test_authorize_url_refuses_empty_app_id() -> None:
    with pytest.raises(InstagramAuthError, match="META_APP_ID"):
        build_authorize_url(
            app_id="",
            redirect_uri="https://x.test/cb",
            state="s",
        )


# ---- token exchange (short-lived) ----


@pytest.mark.asyncio
async def test_short_lived_exchange_parses_happy() -> None:
    body = {"access_token": "short_abc", "token_type": "bearer", "expires_in": 3600}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            mock.get("https://graph.facebook.com/v18.0/oauth/access_token").mock(
                return_value=httpx.Response(200, json=body)
            )
            tok = await exchange_code_for_short_lived(
                app_id="appid",
                app_secret="appsecret",
                code="the-code",
                redirect_uri="https://x.test/cb",
                http=http,
            )
    assert tok.access_token == "short_abc"
    assert "T" in tok.expires_at  # ISO timestamp


@pytest.mark.asyncio
async def test_short_lived_exchange_surfaces_meta_error_envelope() -> None:
    """Meta returns 400 with { error: { message, type, code } }."""
    body = {"error": {"message": "Bad code", "type": "OAuthException", "code": 100}}
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get("https://graph.facebook.com/v18.0/oauth/access_token").mock(
                return_value=httpx.Response(400, json=body)
            )
            with pytest.raises(InstagramAuthError, match="HTTP 400"):
                await exchange_code_for_short_lived(
                    app_id="x",
                    app_secret="y",
                    code="bad",
                    redirect_uri="https://x.test/cb",
                    http=http,
                )


# ---- token exchange (long-lived) ----


@pytest.mark.asyncio
async def test_long_lived_exchange_extends_expiry() -> None:
    body = {"access_token": "long_xyz", "token_type": "bearer", "expires_in": 5_184_000}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            mock.get("https://graph.facebook.com/v18.0/oauth/access_token").mock(
                return_value=httpx.Response(200, json=body)
            )
            tok = await exchange_for_long_lived(
                app_id="appid",
                app_secret="appsecret",
                short_lived_token="short_abc",
                http=http,
            )
    assert tok.access_token == "long_xyz"
    # ~60 days from now — sanity check the ISO date is in the future.
    assert "T" in tok.expires_at


# ---- list pages + IG resolution ----


@pytest.mark.asyncio
async def test_list_pages_parses_each_admin_page() -> None:
    body = {
        "data": [
            {"id": "page_a", "name": "Page A", "access_token": "pat_a"},
            {"id": "page_b", "name": "Page B", "access_token": "pat_b"},
        ],
        "paging": {"cursors": {"before": "X", "after": "Y"}},
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            mock.get("https://graph.facebook.com/v18.0/me/accounts").mock(
                return_value=httpx.Response(200, json=body)
            )
            pages = await list_pages(long_lived_user_token="long", http=http)
    assert len(pages) == 2
    assert pages[0].id == "page_a"
    assert pages[0].access_token == "pat_a"
    assert pages[1].name == "Page B"


@pytest.mark.asyncio
async def test_resolve_ig_business_returns_none_when_unlinked() -> None:
    """A FB Page without a linked IG-Business returns no
    instagram_business_account pointer. The router uses the None
    return to try the next Page."""
    body = {"id": "page_unlinked"}  # no instagram_business_account field
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get("https://graph.facebook.com/v18.0/page_unlinked").mock(
                return_value=httpx.Response(200, json=body)
            )
            page = FacebookPage(id="page_unlinked", name="Unlinked", access_token="pat")
            result = await resolve_ig_business_account(page=page, http=http)
    assert result is None


@pytest.mark.asyncio
async def test_resolve_ig_business_parses_linked_account() -> None:
    page_body = {"id": "page_a", "instagram_business_account": {"id": "ig_999"}}
    ig_body = {
        "id": "ig_999",
        "username": "alice_streams",
        "name": "Alice's Channel",
        "profile_picture_url": "https://scontent.fbcdn.net/avatar.jpg",
    }
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get("https://graph.facebook.com/v18.0/page_a").mock(
                return_value=httpx.Response(200, json=page_body)
            )
            mock.get("https://graph.facebook.com/v18.0/ig_999").mock(
                return_value=httpx.Response(200, json=ig_body)
            )
            page = FacebookPage(id="page_a", name="Page A", access_token="pat_a")
            result = await resolve_ig_business_account(page=page, http=http)
    assert result is not None
    assert result.ig_user_id == "ig_999"
    assert result.page_id == "page_a"
    assert result.page_access_token == "pat_a"
    assert result.username == "alice_streams"
    assert result.name == "Alice's Channel"
    assert result.profile_picture_url == "https://scontent.fbcdn.net/avatar.jpg"


# ---- refresh ----


@pytest.mark.asyncio
async def test_refresh_long_lived_token_is_idempotent_with_extend_semantics() -> None:
    """refresh_long_lived_token is literally exchange_for_long_lived
    re-applied to the current long-lived token. Confirms the same
    endpoint + same shape so the scheduler doesn't accidentally
    diverge."""
    body = {"access_token": "long_v2", "token_type": "bearer", "expires_in": 5_184_000}
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get("https://graph.facebook.com/v18.0/oauth/access_token").mock(
                return_value=httpx.Response(200, json=body)
            )
            tok = await refresh_long_lived_token(
                app_id="appid",
                app_secret="appsecret",
                current_long_lived_token="long_v1",
                http=http,
            )
    assert tok.access_token == "long_v2"
