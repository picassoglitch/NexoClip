"""upload-post API client tests.

respx-mocks every endpoint we care about and pins the auth header,
URL shape, body shape, and response parsing. These are the contract
tests — when upload-post changes their API, these break first.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from nexoclip.integrations.upload_post.client import (
    UploadPostClient,
    UploadPostError,
)


def _client(http: httpx.AsyncClient) -> UploadPostClient:
    return UploadPostClient(api_key="test_apikey_abc", http=http)


# ---- auth + ctor ----


def test_constructor_refuses_empty_api_key() -> None:
    with pytest.raises(UploadPostError, match="UPLOAD_POST_API_KEY"):
        UploadPostClient(api_key="")


# ---- create_user_profile ----


@pytest.mark.asyncio
async def test_create_user_profile_happy_path() -> None:
    body = {
        "success": True,
        "profile": {
            "username": "ten_alice",
            "created_at": "Fri, 02 May 2025 21:43:14 GMT",
            "social_accounts": {"tiktok": ""},
        },
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post("https://api.upload-post.com/api/uploadposts/users").mock(
                return_value=httpx.Response(201, json=body)
            )
            profile = await _client(http).create_user_profile("ten_alice")
    assert profile.username == "ten_alice"
    assert profile.social_accounts == {"tiktok": ""}
    # Auth header must be "Apikey <key>" (not Bearer, not raw).
    sent = route.calls.last.request
    assert sent.headers["Authorization"] == "Apikey test_apikey_abc"
    assert sent.headers["Content-Type"].startswith("application/json")


@pytest.mark.asyncio
async def test_create_user_profile_409_raises_with_status() -> None:
    """Profile-already-exists is 409. Callers branch on status_code
    to treat it as success + go fetch the existing profile."""
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post("https://api.upload-post.com/api/uploadposts/users").mock(
                return_value=httpx.Response(
                    409, json={"success": False, "error": "Profile exists"}
                )
            )
            with pytest.raises(UploadPostError) as ei:
                await _client(http).create_user_profile("ten_alice")
    assert ei.value.status_code == 409


@pytest.mark.asyncio
async def test_create_user_profile_403_profile_limit_reached() -> None:
    """Plan limit reached. Caller wants to surface the upgrade pitch."""
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post("https://api.upload-post.com/api/uploadposts/users").mock(
                return_value=httpx.Response(
                    403,
                    json={
                        "success": False,
                        "error_code": "PROFILE_LIMIT_REACHED",
                        "error": "Plan limit hit",
                    },
                )
            )
            with pytest.raises(UploadPostError) as ei:
                await _client(http).create_user_profile("ten_bob")
    assert ei.value.status_code == 403


# ---- get_user_profile ----


@pytest.mark.asyncio
async def test_get_user_profile_parses_social_accounts() -> None:
    body = {
        "success": True,
        "profile": {
            "username": "ten_alice",
            "created_at": "2023-10-27T10:00:00Z",
            "social_accounts": {
                "tiktok": {
                    "username": "alice_streams",
                    "display_name": "Alice",
                    "social_images": "https://x.test/avatar.jpg",
                },
                "instagram": None,
            },
        },
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            mock.get(
                "https://api.upload-post.com/api/uploadposts/users/ten_alice"
            ).mock(return_value=httpx.Response(200, json=body))
            p = await _client(http).get_user_profile("ten_alice")
    assert p.username == "ten_alice"
    assert isinstance(p.social_accounts["tiktok"], dict)
    assert p.social_accounts["instagram"] is None


# ---- generate_connect_jwt ----


@pytest.mark.asyncio
async def test_generate_connect_jwt_returns_access_url() -> None:
    body = {
        "success": True,
        "access_url": "https://app.upload-post.com/connect?token=GENERATED_JWT",
        "duration": "48h",
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post(
                "https://api.upload-post.com/api/uploadposts/users/generate-jwt"
            ).mock(return_value=httpx.Response(200, json=body))
            link = await _client(http).generate_connect_jwt(
                "ten_alice",
                redirect_url="https://nexoclip.test/dashboard/publish/upload-post",
                redirect_button_text="Back to NexoClip",
                connect_title="Connect your accounts",
                connect_description="Link your socials here.",
                language="en",
            )
    assert link.access_url.startswith("https://app.upload-post.com/connect?token=")
    assert link.duration == "48h"

    # Body must carry the optional params we passed in.
    sent_body = route.calls.last.request.content.decode()
    assert "ten_alice" in sent_body
    assert "redirect_url" in sent_body
    assert "Connect your accounts" in sent_body
    # The button text override is the critical one — without it
    # the upload-post UI shows "Logout connection" as the back-
    # to-app button, which is confusing.
    assert "redirect_button_text" in sent_body
    assert "Back to NexoClip" in sent_body
    assert "language" in sent_body


# ---- upload_video_from_url ----


@pytest.mark.asyncio
async def test_upload_video_from_url_sends_platform_array_and_async_flag() -> None:
    body = {
        "success": True,
        "request_id": "req_abc",
        "message": "Upload queued for processing",
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.post("https://api.upload-post.com/api/upload").mock(
                return_value=httpx.Response(202, json=body)
            )
            result = await _client(http).upload_video_from_url(
                username="ten_alice",
                video_url="https://nexoclip.test/api/internal/clip/clp_xxx?...sig",
                platforms=["tiktok", "instagram", "youtube"],
                title="My clip",
                description="Description here",
                async_upload=True,
            )
    assert result.request_id == "req_abc"
    assert result.is_async is True
    # Verify the JSON body contained the platform list (upload-post
    # expects `platform[]` style in the keys).
    sent = route.calls.last.request.content.decode()
    assert "ten_alice" in sent
    assert "tiktok" in sent
    assert "instagram" in sent
    assert "youtube" in sent
    assert "My clip" in sent
    assert "async_upload" in sent


@pytest.mark.asyncio
async def test_upload_video_5xx_raises() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post("https://api.upload-post.com/api/upload").mock(
                return_value=httpx.Response(500, text="oops")
            )
            with pytest.raises(UploadPostError) as ei:
                await _client(http).upload_video_from_url(
                    username="ten_alice",
                    video_url="https://x.test/v.mp4",
                    platforms=["tiktok"],
                )
    assert ei.value.status_code == 500


# ---- get_status ----


@pytest.mark.asyncio
async def test_get_status_pending() -> None:
    body = {
        "success": True,
        "request_id": "req_abc",
        "is_async": True,
        "status": "PENDING",
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            mock.get(
                "https://api.upload-post.com/api/uploadposts/status"
            ).mock(return_value=httpx.Response(200, json=body))
            s = await _client(http).get_status(request_id="req_abc")
    assert s.status == "PENDING"
    assert s.result is None


@pytest.mark.asyncio
async def test_get_status_finished_with_result() -> None:
    body = {
        "success": True,
        "request_id": "req_abc",
        "is_async": True,
        "status": "FINISHED",
        "result": {
            "platforms": {
                "tiktok": {
                    "success": True,
                    "platform_post_id": "tt_123",
                    "post_url": "https://tiktok.com/@alice/video/123",
                }
            }
        },
    }
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.get(
                "https://api.upload-post.com/api/uploadposts/status"
            ).mock(return_value=httpx.Response(200, json=body))
            s = await _client(http).get_status(request_id="req_abc")
    assert s.status == "FINISHED"
    assert s.result is not None
    assert s.result["platforms"]["tiktok"]["post_url"].startswith("https://tiktok.com/")


@pytest.mark.asyncio
async def test_get_status_requires_id() -> None:
    async with httpx.AsyncClient() as http:
        with pytest.raises(UploadPostError, match="request_id or job_id"):
            await _client(http).get_status()


# ---- get_history ----


@pytest.mark.asyncio
async def test_get_history_passes_pagination() -> None:
    body = {"history": [{"platform": "tiktok"}], "total": 1, "page": 2, "limit": 50}
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            route = mock.get(
                "https://api.upload-post.com/api/uploadposts/history"
            ).mock(return_value=httpx.Response(200, json=body))
            out = await _client(http).get_history(page=2, limit=50)
    assert out["page"] == 2
    assert out["limit"] == 50
    sent = route.calls.last.request
    assert sent.url.params.get("page") == "2"
    assert sent.url.params.get("limit") == "50"


# ---- get_scheduled ----


@pytest.mark.asyncio
async def test_get_scheduled_unwraps_list() -> None:
    body = [
        {"job_id": "j_1", "scheduled_date": "2025-12-31T23:00:00Z", "post_type": "video", "title": "X"},
        {"job_id": "j_2", "scheduled_date": "2026-01-01T08:00:00Z", "post_type": "photo", "title": "Y"},
    ]
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            mock.get(
                "https://api.upload-post.com/api/uploadposts/schedule"
            ).mock(return_value=httpx.Response(200, json=body))
            out = await _client(http).get_scheduled()
    assert len(out) == 2
    assert out[0]["job_id"] == "j_1"
