"""Zernio API client tests.

respx-mocks every endpoint we care about and pins the auth header,
URL shape, body shape, and response parsing. These are the contract
tests — when Zernio changes their API, these break first.
"""
from __future__ import annotations

import json

import httpx
import pytest
import respx

from nexoclip.integrations.zernio.client import ZernioClient, ZernioError

_BASE = "https://zernio.com/api/v1"


def _client(http: httpx.AsyncClient) -> ZernioClient:
    return ZernioClient(api_key="sk_test_abc", http=http)


# ---- auth + ctor ----


def test_constructor_refuses_empty_api_key() -> None:
    with pytest.raises(ZernioError, match="ZERNIO_API_KEY"):
        ZernioClient(api_key="")


# ---- create_profile ----


@pytest.mark.asyncio
async def test_create_profile_sends_name_and_parses_id() -> None:
    body = {"profile": {"_id": "prof_abc123", "name": "My Brand"}}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{_BASE}/profiles").mock(
                return_value=httpx.Response(201, json=body)
            )
            profile = await _client(http).create_profile(
                name="My Brand", description="Testing the Zernio API",
            )
    assert profile.profile_id == "prof_abc123"
    assert profile.name == "My Brand"
    sent = route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer sk_test_abc"
    payload = json.loads(sent.content.decode())
    assert payload["name"] == "My Brand"
    assert payload["description"] == "Testing the Zernio API"


@pytest.mark.asyncio
async def test_create_profile_missing_id_raises() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post(f"{_BASE}/profiles").mock(
                return_value=httpx.Response(201, json={"profile": {"name": "X"}})
            )
            with pytest.raises(ZernioError, match="missing _id"):
                await _client(http).create_profile(name="X")


# ---- connect_url ----


@pytest.mark.asyncio
async def test_connect_url_returns_authurl_with_profile_id() -> None:
    body = {"authUrl": "https://zernio.com/oauth/tiktok?state=xyz"}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(f"{_BASE}/connect/tiktok").mock(
                return_value=httpx.Response(200, json=body)
            )
            link = await _client(http).connect_url(
                "tiktok",
                profile_id="ten_alice",
                redirect_url="https://nexoclip.test/dashboard/publish/zernio/connected",
            )
    assert link.auth_url.startswith("https://zernio.com/oauth/tiktok")
    sent = route.calls.last.request
    # Bearer auth (NOT Apikey).
    assert sent.headers["Authorization"] == "Bearer sk_test_abc"
    # profileId scopes the connection to the tenant.
    assert sent.url.params.get("profileId") == "ten_alice"
    # redirect_url sends the post-OAuth popup back to OUR page instead
    # of Zernio's dashboard (white-label flow).
    assert sent.url.params.get("redirect_url") == (
        "https://nexoclip.test/dashboard/publish/zernio/connected"
    )


@pytest.mark.asyncio
async def test_connect_url_missing_authurl_raises() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get(f"{_BASE}/connect/tiktok").mock(
                return_value=httpx.Response(200, json={"oops": True})
            )
            with pytest.raises(ZernioError, match="authUrl"):
                await _client(http).connect_url("tiktok", profile_id="ten_alice")


@pytest.mark.asyncio
async def test_connect_url_headless_sets_param() -> None:
    """headless=true makes Zernio's redirect carry the selection state
    (tempToken & co) so WE render the page/board picker."""
    body = {"authUrl": "https://facebook.com/oauth?x=1"}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(f"{_BASE}/connect/facebook").mock(
                return_value=httpx.Response(200, json=body)
            )
            await _client(http).connect_url(
                "facebook",
                profile_id="ten_alice",
                redirect_url="https://nexoclip.test/dashboard/publish/zernio/connected?platform=facebook",
                headless=True,
            )
    sent = route.calls.last.request
    assert sent.url.params.get("headless") == "true"
    # The platform rides on the redirect_url query string, percent-
    # encoding intact, so /connected knows which chip just connected.
    assert sent.url.params.get("redirect_url") == (
        "https://nexoclip.test/dashboard/publish/zernio/connected?platform=facebook"
    )


@pytest.mark.asyncio
async def test_connect_url_default_omits_headless() -> None:
    body = {"authUrl": "https://tiktok.com/oauth"}
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            route = mock.get(f"{_BASE}/connect/tiktok").mock(
                return_value=httpx.Response(200, json=body)
            )
            await _client(http).connect_url("tiktok", profile_id="ten_alice")
    assert "headless" not in route.calls.last.request.url.params


# ---- list_accounts ----


@pytest.mark.asyncio
async def test_list_accounts_parses_rows() -> None:
    body = {
        "accounts": [
            {"platform": "tiktok", "_id": "acct_tt_1"},
            {"platform": "youtube", "_id": "acct_yt_2"},
            {"platform": "x", "no_id": True},  # dropped (no _id)
        ]
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(f"{_BASE}/accounts").mock(
                return_value=httpx.Response(200, json=body)
            )
            accts = await _client(http).list_accounts(profile_id="ten_alice")
    assert {(a.platform, a.account_id) for a in accts} == {
        ("tiktok", "acct_tt_1"),
        ("youtube", "acct_yt_2"),
    }
    assert route.calls.last.request.url.params.get("profileId") == "ten_alice"


@pytest.mark.asyncio
async def test_list_accounts_filters_by_profile_id_when_present() -> None:
    body = {
        "accounts": [
            {"platform": "tiktok", "_id": "a1", "profileId": "ten_alice"},
            {"platform": "youtube", "_id": "a2", "profileId": "ten_bob"},
        ]
    }
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get(f"{_BASE}/accounts").mock(
                return_value=httpx.Response(200, json=body)
            )
            accts = await _client(http).list_accounts(profile_id="ten_alice")
    assert [a.account_id for a in accts] == ["a1"]


@pytest.mark.asyncio
async def test_list_accounts_keeps_row_when_profileid_is_object() -> None:
    """Regression: a profileId returned as a nested object (or absent)
    must NOT be dropped — that bug made connected accounts vanish."""
    body = {
        "accounts": [
            {"platform": "tiktok", "_id": "a1", "profileId": {"_id": "ten_alice"}},
            {"platform": "instagram", "_id": "a2"},  # no profileId field at all
        ]
    }
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get(f"{_BASE}/accounts").mock(
                return_value=httpx.Response(200, json=body)
            )
            accts = await _client(http).list_accounts(profile_id="ten_alice")
    assert {(a.platform, a.account_id) for a in accts} == {
        ("tiktok", "a1"),
        ("instagram", "a2"),
    }


# ---- transport errors wrap into ZernioError ----


@pytest.mark.asyncio
async def test_transport_error_wraps_into_zernio_error() -> None:
    """Timeouts / connect failures must surface as ZernioError so
    callers' `except ZernioError` degrades gracefully (a raw httpx
    exception 500ed the dashboard when Zernio was slow)."""
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get(f"{_BASE}/accounts").mock(
                side_effect=httpx.ReadTimeout("read timed out")
            )
            with pytest.raises(ZernioError, match="request failed"):
                await _client(http).list_accounts(profile_id="ten_alice")


# ---- disconnect_account ----


@pytest.mark.asyncio
async def test_disconnect_account_tolerates_empty_204() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.delete(f"{_BASE}/accounts/acc_1").mock(
                return_value=httpx.Response(204)
            )
            # Must NOT raise on an empty 204 body.
            await _client(http).disconnect_account("acc_1")
    assert route.calls.last.request.headers["Authorization"] == "Bearer sk_test_abc"


@pytest.mark.asyncio
async def test_disconnect_account_raises_on_error() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.delete(f"{_BASE}/accounts/acc_x").mock(
                return_value=httpx.Response(404, json={"error": "not found"})
            )
            with pytest.raises(ZernioError) as ei:
                await _client(http).disconnect_account("acc_x")
    assert ei.value.status_code == 404


# ---- headless Facebook page selection ----


@pytest.mark.asyncio
async def test_list_facebook_pages_parses_rows_and_never_keeps_tokens() -> None:
    body = {
        "pages": [
            {
                "id": "123", "name": "My Brand Page", "username": "mybrand",
                "access_token": "EAAxxxxx", "category": "Brand",
                "tasks": ["MANAGE"],
            },
            {"id": "456", "name": "Side Page"},
            {"name": "no id — dropped"},
            "not-a-dict",
        ]
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(f"{_BASE}/connect/facebook/select-page").mock(
                return_value=httpx.Response(200, json=body)
            )
            pages = await _client(http).list_facebook_pages(
                profile_id="ten_alice", temp_token="EAAtmp",
            )
    sent = route.calls.last.request
    assert sent.url.params.get("profileId") == "ten_alice"
    assert sent.url.params.get("tempToken") == "EAAtmp"
    assert [(p.page_id, p.name, p.username, p.category) for p in pages] == [
        ("123", "My Brand Page", "mybrand", "Brand"),
        ("456", "Side Page", None, None),
    ]
    # The page-scoped access_token must not survive parsing — we never
    # store or log platform tokens.
    assert not any(hasattr(p, "access_token") for p in pages)


@pytest.mark.asyncio
async def test_list_facebook_pages_missing_pages_raises() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get(f"{_BASE}/connect/facebook/select-page").mock(
                return_value=httpx.Response(200, json={"oops": True})
            )
            with pytest.raises(ZernioError, match="missing pages"):
                await _client(http).list_facebook_pages(
                    profile_id="ten_alice", temp_token="EAAtmp",
                )


@pytest.mark.asyncio
async def test_select_facebook_page_posts_selection_and_parses_account() -> None:
    body = {
        "message": "Facebook page connected successfully",
        "account": {
            "accountId": "acct_fb_1", "platform": "facebook",
            "username": "mybrand", "displayName": "My Brand Page",
        },
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{_BASE}/connect/facebook/select-page").mock(
                return_value=httpx.Response(200, json=body)
            )
            acct = await _client(http).select_facebook_page(
                profile_id="ten_alice",
                page_id="123",
                temp_token="EAAtmp",
                user_profile={"id": "987", "name": "Alice"},
            )
    assert acct.platform == "facebook"
    assert acct.account_id == "acct_fb_1"
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload == {
        "profileId": "ten_alice",
        "pageId": "123",
        "tempToken": "EAAtmp",
        "userProfile": {"id": "987", "name": "Alice"},
    }


@pytest.mark.asyncio
async def test_select_facebook_page_missing_account_raises() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post(f"{_BASE}/connect/facebook/select-page").mock(
                return_value=httpx.Response(200, json={"message": "ok"})
            )
            with pytest.raises(ZernioError, match="missing account"):
                await _client(http).select_facebook_page(
                    profile_id="ten_alice", page_id="123",
                    temp_token="EAAtmp", user_profile={"id": "987"},
                )


@pytest.mark.asyncio
async def test_select_facebook_page_404_unknown_page_raises_with_status() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post(f"{_BASE}/connect/facebook/select-page").mock(
                return_value=httpx.Response(
                    404, json={"error": "Selected page not found"}
                )
            )
            with pytest.raises(ZernioError) as ei:
                await _client(http).select_facebook_page(
                    profile_id="ten_alice", page_id="999",
                    temp_token="EAAtmp", user_profile={"id": "987"},
                )
    assert ei.value.status_code == 404


# ---- create_post ----


@pytest.mark.asyncio
async def test_create_post_sends_media_url_accounts_and_tiktok_consent() -> None:
    body = {"success": True, "post": {"_id": "post_123"}}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{_BASE}/posts").mock(
                return_value=httpx.Response(201, json=body)
            )
            result = await _client(http).create_post(
                profile_id="ten_alice",
                content="My caption",
                media_url="https://nexoclip.test/api/internal/clip/clp_x?sig=...",
                platforms=[("tiktok", "acct_tt_1"), ("youtube", "acct_yt_2")],
                publish_now=True,
                tiktok_settings={
                    "content_preview_confirmed": True,
                    "express_consent_given": True,
                },
            )
    assert result.post_id == "post_123"

    sent = route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer sk_test_abc"
    assert sent.headers["Content-Type"].startswith("application/json")
    payload = json.loads(sent.content.decode())
    # Media referenced BY URL (no presigned upload in the common path).
    assert payload["mediaItems"] == [
        {"type": "video", "url": "https://nexoclip.test/api/internal/clip/clp_x?sig=..."}
    ]
    # Per-platform accountId is required on every platforms[] entry.
    assert payload["platforms"] == [
        {"platform": "tiktok", "accountId": "acct_tt_1"},
        {"platform": "youtube", "accountId": "acct_yt_2"},
    ]
    assert payload["publishNow"] is True
    assert "scheduledFor" not in payload
    # TikTok legal consent flags must be present.
    assert payload["tiktokSettings"]["content_preview_confirmed"] is True
    assert payload["tiktokSettings"]["express_consent_given"] is True


@pytest.mark.asyncio
async def test_create_post_scheduled_sets_scheduledfor_not_publishnow() -> None:
    body = {"success": True, "post": {"_id": "post_sched"}}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(f"{_BASE}/posts").mock(
                return_value=httpx.Response(201, json=body)
            )
            await _client(http).create_post(
                profile_id="ten_alice",
                content="Later",
                media_url="https://x.test/v.mp4",
                platforms=[("tiktok", "acct_tt_1")],
                scheduled_for="2026-12-31T23:00:00Z",
                timezone="America/New_York",
            )
    payload = json.loads(route.calls.last.request.content.decode())
    assert payload["scheduledFor"] == "2026-12-31T23:00:00Z"
    assert payload["timezone"] == "America/New_York"
    assert "publishNow" not in payload


@pytest.mark.asyncio
async def test_create_post_5xx_raises_with_status() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post(f"{_BASE}/posts").mock(
                return_value=httpx.Response(500, text="oops")
            )
            with pytest.raises(ZernioError) as ei:
                await _client(http).create_post(
                    profile_id="ten_alice",
                    content="x",
                    media_url="https://x.test/v.mp4",
                    platforms=[("tiktok", "acct_tt_1")],
                )
    assert ei.value.status_code == 500


# ---- get_post ----


@pytest.mark.asyncio
async def test_get_post_parses_status_and_platforms() -> None:
    body = {
        "post": {
            "_id": "post_123",
            "status": "published",
            "platforms": [
                {"platform": "tiktok", "status": "published", "url": "https://tiktok.com/x"}
            ],
        }
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            mock.get(f"{_BASE}/posts/post_123").mock(
                return_value=httpx.Response(200, json=body)
            )
            s = await _client(http).get_post("post_123")
    assert s.post_id == "post_123"
    assert s.status == "published"
    assert isinstance(s.platforms, list)
    assert s.platforms[0]["url"].startswith("https://tiktok.com/")


# ---- list_posts ----


@pytest.mark.asyncio
async def test_list_posts_passes_pagination() -> None:
    body = {"posts": [{"_id": "p1"}], "total": 1}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(f"{_BASE}/posts").mock(
                return_value=httpx.Response(200, json=body)
            )
            out = await _client(http).list_posts(page=2, limit=50)
    assert out["posts"][0]["_id"] == "p1"
    sent = route.calls.last.request
    assert sent.url.params.get("page") == "2"
    assert sent.url.params.get("limit") == "50"
