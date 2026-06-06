"""YouTube (Google OAuth) wiring tests.

The single most important pin: the authorize URL MUST carry
access_type=offline AND prompt=consent, otherwise Google sometimes
omits the refresh_token and we cannot ever refresh access tokens —
upload silently breaks 60 minutes after every connect.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from nexoclip.integrations.youtube.auth import (
    YouTubeAuthError,
    build_authorize_url,
    exchange_code_for_token,
    refresh_access_token,
)


# ---- authorize URL ----


def test_authorize_url_has_offline_and_consent() -> None:
    """If either of these gets dropped, refresh_tokens stop coming
    back from Google on re-connects. The test pins both."""
    url = build_authorize_url(
        client_id="goog_client",
        redirect_uri="https://nexoclip.nexo-ai.world/connect/youtube/callback",
        state="signed-state",
    )
    assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert "client_id=goog_client" in url
    assert "response_type=code" in url
    assert "state=signed-state" in url
    # The two non-defaults — MUST be present.
    assert "access_type=offline" in url
    assert "prompt=consent" in url


def test_authorize_url_uses_upload_only_scope() -> None:
    """We deliberately ship youtube.upload alone (no readonly) to
    keep Google's OAuth verification surface minimal."""
    url = build_authorize_url(
        client_id="goog_client",
        redirect_uri="https://x.test/cb",
        state="s",
    )
    # Single scope — space-separated would mean multiple.
    assert "scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fyoutube.upload" in url
    assert "youtube.readonly" not in url
    assert "youtube.force-ssl" not in url


def test_authorize_url_refuses_empty_client_id() -> None:
    with pytest.raises(YouTubeAuthError, match="GOOGLE_CLIENT_ID"):
        build_authorize_url(
            client_id="",
            redirect_uri="https://x.test/cb",
            state="s",
        )


# ---- code exchange ----


@pytest.mark.asyncio
async def test_exchange_code_parses_happy_with_refresh() -> None:
    body = {
        "access_token": "at_abc",
        "refresh_token": "rt_def",
        "expires_in": 3599,
        "scope": "https://www.googleapis.com/auth/youtube.upload",
        "token_type": "Bearer",
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            mock.post("https://oauth2.googleapis.com/token").mock(
                return_value=httpx.Response(200, json=body)
            )
            tok = await exchange_code_for_token(
                client_id="cid",
                client_secret="csec",
                code="code",
                redirect_uri="https://x.test/cb",
                http=http,
            )
    assert tok.access_token == "at_abc"
    assert tok.refresh_token == "rt_def"
    assert tok.token_type == "Bearer"
    assert "T" in tok.expires_at


@pytest.mark.asyncio
async def test_exchange_code_raises_when_google_omits_refresh_token() -> None:
    """The contract guard — if Google ever returns a token response
    without refresh_token (means our authorize URL silently lost
    prompt=consent), fail loud at connect time, not later when
    upload breaks."""
    body = {
        "access_token": "at_abc",
        # refresh_token intentionally missing
        "expires_in": 3599,
        "scope": "https://www.googleapis.com/auth/youtube.upload",
        "token_type": "Bearer",
    }
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post("https://oauth2.googleapis.com/token").mock(
                return_value=httpx.Response(200, json=body)
            )
            with pytest.raises(YouTubeAuthError, match="refresh_token"):
                await exchange_code_for_token(
                    client_id="cid",
                    client_secret="csec",
                    code="code",
                    redirect_uri="https://x.test/cb",
                    http=http,
                )


@pytest.mark.asyncio
async def test_exchange_code_raises_on_google_error_envelope() -> None:
    body = {"error": "invalid_grant", "error_description": "Code expired"}
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.post("https://oauth2.googleapis.com/token").mock(
                return_value=httpx.Response(200, json=body)
            )
            with pytest.raises(YouTubeAuthError, match="invalid_grant"):
                await exchange_code_for_token(
                    client_id="cid",
                    client_secret="csec",
                    code="bad",
                    redirect_uri="https://x.test/cb",
                    http=http,
                )


# ---- refresh ----


@pytest.mark.asyncio
async def test_refresh_does_not_require_refresh_token_in_response() -> None:
    """Google does NOT re-issue refresh_token on a refresh call.
    refresh() must accept a refresh-less response without raising —
    that's the SUCCESS path. Caller preserves the stored refresh_token."""
    body = {
        "access_token": "at_v2",
        "expires_in": 3599,
        "scope": "https://www.googleapis.com/auth/youtube.upload",
        "token_type": "Bearer",
    }
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            mock.post("https://oauth2.googleapis.com/token").mock(
                return_value=httpx.Response(200, json=body)
            )
            tok = await refresh_access_token(
                client_id="cid",
                client_secret="csec",
                refresh_token="rt_v1",
                http=http,
            )
    assert tok.access_token == "at_v2"
    # Refresh path: refresh_token field is None on the result so the
    # caller knows NOT to overwrite the stored refresh_token.
    assert tok.refresh_token is None
