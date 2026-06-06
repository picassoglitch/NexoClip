"""YouTube (Google OAuth 2.0) — connect flow.

Pattern A: NexoClip owns ONE OAuth 2.0 web client on
console.cloud.google.com. Required:

  * youtube.upload scope only. We deliberately skip youtube.readonly
    (which is what channels.list?mine=true requires) to keep the
    Google OAuth verification surface minimal — Google asks more
    questions when read-scopes are involved. The channelId is
    captured from the first videos.insert response in the publish
    adapter and written back to platform_user_id.

  * access_type=offline AND prompt=consent on the authorize URL.
    Without prompt=consent, Google may decide not to issue a
    refresh_token on subsequent connects (the user "already
    granted" the scope), which leaves us with a 1-hour access
    token and no way to refresh — exactly the kind of "works in
    dev, breaks in prod" bug we want to slam shut at build time.

  * Service accounts DO NOT WORK with YouTube Data API — they
    return NoLinkedYouTubeAccount. Must be the OAuth user flow.

Token model — classic access + refresh:

  * Access token: ~1 hour. Mint on demand before each upload.
  * Refresh token: long-lived. Issued once at connect time when
    consent is granted; survives until the user revokes our app.
    No scheduled refresh job needed (unlike Meta's 60-day exchange);
    we just refresh-on-demand right before an upload.

Endpoints:

  * https://accounts.google.com/o/oauth2/v2/auth     (browser redirect)
  * https://oauth2.googleapis.com/token              (code + refresh both)
  * (publish-time, in nexoclip.publish.youtube) videos.insert with
    uploadType=resumable — see publish adapter; this module doesn't
    touch upload.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlencode

import httpx

from nexoclip.errors import NexoClipError


class YouTubeAuthError(NexoClipError):
    """Raised on any failure in the Google OAuth round-trip."""

    def __init__(self, message: str, *, body: object = None) -> None:
        super().__init__(message)
        self.body = body


_AUTHORIZE_URL: Final = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL: Final = "https://oauth2.googleapis.com/token"

# Single scope: upload only. channels.list (needs youtube.readonly)
# would require a separate Google verification questionnaire — not
# worth it. channelId comes from the first videos.insert response.
_DEFAULT_SCOPES: Final = ("https://www.googleapis.com/auth/youtube.upload",)


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


def _expires_iso(seconds_from_now: int | float | None) -> str:
    s = int(seconds_from_now or 0)
    return (_now() + _dt.timedelta(seconds=s)).isoformat()


def build_authorize_url(
    *,
    client_id: str,
    redirect_uri: str,
    state: str,
    scopes: tuple[str, ...] = _DEFAULT_SCOPES,
) -> str:
    """Construct the Google OAuth consent URL.

    `access_type=offline` + `prompt=consent` are NOT defaults — both
    are required for Google to reliably return a refresh_token.
    Removing either is a silent break on subsequent re-connects.
    """
    if not client_id:
        raise YouTubeAuthError("GOOGLE_CLIENT_ID is not configured")
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        "state": state,
        # The two non-defaults that guarantee a refresh_token comes back.
        "access_type": "offline",
        "prompt": "consent",
        # Google includes_granted_scopes lets us add upload later if
        # we ever expand without forcing a re-consent on existing
        # users. Cheap to set; safe default.
        "include_granted_scopes": "true",
    }
    return f"{_AUTHORIZE_URL}?{urlencode(params)}"


@dataclass(frozen=True, slots=True)
class GoogleToken:
    """Result of a successful Google token exchange or refresh.

    `refresh_token` may be None on a REFRESH response — Google does
    NOT re-issue a refresh_token when you refresh the access token,
    so the stored one stays valid. Callers must NOT overwrite the
    stored refresh_token with None.
    """

    access_token: str
    refresh_token: str | None
    expires_at: str
    scope: str
    token_type: str  # "Bearer"


async def exchange_code_for_token(
    *,
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
    http: httpx.AsyncClient | None = None,
) -> GoogleToken:
    """First-connect exchange. Yields BOTH access_token and
    refresh_token (the latter ONLY because we forced prompt=consent
    at authorize time — otherwise Google sometimes omits it)."""
    if not client_secret:
        raise YouTubeAuthError("GOOGLE_CLIENT_SECRET is not configured")
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }
    tok = await _post_token(payload, http=http)
    if tok.refresh_token is None:
        # If we ever see this in prod it means our authorize URL
        # silently lost the prompt=consent / access_type=offline
        # pair. Fail loud at connect time, not later when the access
        # token expires and we have no way to refresh.
        raise YouTubeAuthError(
            "Google omitted refresh_token. Confirm the authorize URL "
            "carries access_type=offline AND prompt=consent."
        )
    return tok


async def refresh_access_token(
    *,
    client_id: str,
    client_secret: str,
    refresh_token: str,
    http: httpx.AsyncClient | None = None,
) -> GoogleToken:
    """Mint a fresh access_token from a stored refresh_token.

    The response does NOT carry a refresh_token (Google never re-
    issues it on refresh). The returned `GoogleToken` will have
    `refresh_token=None` — callers must preserve the stored one.
    """
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    return await _post_token(payload, http=http)


async def _post_token(
    payload: dict[str, str],
    *,
    http: httpx.AsyncClient | None,
) -> GoogleToken:
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    client = http or httpx.AsyncClient(timeout=20.0)
    try:
        resp = await client.post(_TOKEN_URL, data=payload, headers=headers)
    finally:
        if http is None:
            await client.aclose()

    if resp.status_code != 200:
        raise YouTubeAuthError(
            f"Google /token returned HTTP {resp.status_code}",
            body=resp.text[:500],
        )
    try:
        body = resp.json()
    except ValueError as e:
        raise YouTubeAuthError("Google /token returned non-JSON", body=resp.text[:500]) from e
    if not isinstance(body, dict):
        raise YouTubeAuthError("Google /token response is not an object", body=body)

    err = body.get("error")
    if err:
        raise YouTubeAuthError(
            f"Google refused token: {err} - {body.get('error_description')}",
            body=body,
        )

    access = body.get("access_token")
    refresh = body.get("refresh_token")
    expires_in = body.get("expires_in")
    scope = body.get("scope") or ""
    token_type = body.get("token_type") or "Bearer"

    if not isinstance(access, str) or not access:
        raise YouTubeAuthError("Google response missing access_token", body=body)
    if refresh is not None and not isinstance(refresh, str):
        raise YouTubeAuthError("Google response refresh_token is not a string", body=body)

    return GoogleToken(
        access_token=access,
        refresh_token=refresh if isinstance(refresh, str) and refresh else None,
        expires_at=_expires_iso(expires_in),
        scope=scope,
        token_type=token_type,
    )


__all__ = [
    "YouTubeAuthError",
    "GoogleToken",
    "build_authorize_url",
    "exchange_code_for_token",
    "refresh_access_token",
]
