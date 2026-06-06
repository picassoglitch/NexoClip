"""TikTok Login Kit + Content Posting API — OAuth connect flow.

Pattern A: NexoClip owns ONE TikTok app registered on
developer.tiktok.com with `TIKTOK_CLIENT_KEY` / `TIKTOK_CLIENT_SECRET`
in process env. End users click "Connect with TikTok" and TikTok asks
their TikTok account whether to grant our app access; we never see
their TikTok password and they never see our client_secret.

Endpoints we hit:

  * GET  https://www.tiktok.com/v2/auth/authorize/      (browser redirect)
  * POST https://open.tiktokapis.com/v2/oauth/token/    (code → tokens)
  * GET  https://open.tiktokapis.com/v2/user/info/      (handle + avatar)
  * POST https://open.tiktokapis.com/v2/post/publish/creator_info/query/
    (allowed privacy levels + post-eligibility check)
  * POST https://open.tiktokapis.com/v2/oauth/token/    (refresh)

Scopes:

  * user.info.basic    — handle + avatar for the post-connect display
  * video.upload       — INIT a video upload
  * video.publish      — finalize + publish a video

The 25 videos/day cap and 6 req/min token rate are platform-side;
this module doesn't enforce them — that's the publisher's job. The
`query_creator_info` call here surfaces TikTok's "can_post" verdict
which is what the operator sees in the post-connect UI.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlencode

import httpx

from nexoclip.errors import NexoClipError


class TikTokAuthError(NexoClipError):
    """Raised on any failure in the connect / refresh round-trip.

    Caller surfaces as "Connect attempt failed — TikTok said: <msg>"
    and offers a retry. The wrapped HTTP body (where present) goes
    into the structured log so we can diagnose without spamming
    the user with TikTok's full JSON payload.
    """

    def __init__(self, message: str, *, body: object = None) -> None:
        super().__init__(message)
        self.body = body


_AUTHORIZE_URL: Final = "https://www.tiktok.com/v2/auth/authorize/"
_TOKEN_URL: Final = "https://open.tiktokapis.com/v2/oauth/token/"
_USER_INFO_URL: Final = "https://open.tiktokapis.com/v2/user/info/"
_CREATOR_INFO_URL: Final = (
    "https://open.tiktokapis.com/v2/post/publish/creator_info/query/"
)

# Scopes requested at authorize time. Order matters less than presence;
# we ship them comma-joined because that's TikTok's API contract.
_DEFAULT_SCOPES: Final = ("user.info.basic", "video.upload", "video.publish")


def build_authorize_url(
    *,
    client_key: str,
    redirect_uri: str,
    state: str,
    scopes: tuple[str, ...] = _DEFAULT_SCOPES,
) -> str:
    """Construct the URL the operator's browser is bounced to.

    `state` MUST be the HMAC-signed token from
    `nexoclip.integrations.oauth.state.sign_state`. TikTok echoes
    it back on the callback verbatim.
    """
    if not client_key:
        raise TikTokAuthError("TIKTOK_CLIENT_KEY is not configured")
    params = {
        "client_key": client_key,
        "scope": ",".join(scopes),
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "state": state,
    }
    return f"{_AUTHORIZE_URL}?{urlencode(params)}"


@dataclass(frozen=True, slots=True)
class ExchangedToken:
    """Result of a successful code → token exchange."""

    access_token: str
    refresh_token: str
    open_id: str
    scope: str
    expires_at: str             # ISO 8601 UTC
    refresh_expires_at: str     # ISO 8601 UTC — TikTok refresh tokens are 365d


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


def _expires_iso(seconds_from_now: int | float | None) -> str:
    """Convert TikTok's expires_in seconds into an ISO timestamp.

    TikTok returns expires_in / refresh_expires_in for the token
    lifetimes. We persist absolute ISO timestamps because they're
    easier to reason about in the dashboard than a relative seconds
    count.
    """
    s = int(seconds_from_now or 0)
    return (_now() + _dt.timedelta(seconds=s)).isoformat()


async def exchange_code_for_token(
    *,
    client_key: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
    http: httpx.AsyncClient | None = None,
) -> ExchangedToken:
    """Exchange an OAuth auth code for an access + refresh token pair.

    The `redirect_uri` MUST match the one passed to
    `build_authorize_url` exactly (TikTok validates the binding).
    `http` is injectable for tests; production calls construct a
    short-lived client.
    """
    if not client_secret:
        raise TikTokAuthError("TIKTOK_CLIENT_SECRET is not configured")

    payload = {
        "client_key": client_key,
        "client_secret": client_secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }
    return await _post_token(payload, http=http)


async def refresh_access_token(
    *,
    client_key: str,
    client_secret: str,
    refresh_token: str,
    http: httpx.AsyncClient | None = None,
) -> ExchangedToken:
    """Mint a fresh access_token from a stored refresh_token.

    TikTok rotates refresh_token on each refresh — the response
    carries a NEW refresh_token that must replace the stored one.
    The returned `ExchangedToken` always carries the latest pair.
    """
    payload = {
        "client_key": client_key,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    return await _post_token(payload, http=http)


async def _post_token(
    payload: dict[str, str],
    *,
    http: httpx.AsyncClient | None,
) -> ExchangedToken:
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Cache-Control": "no-cache",
    }
    client = http or httpx.AsyncClient(timeout=20.0)
    try:
        resp = await client.post(_TOKEN_URL, data=payload, headers=headers)
    finally:
        if http is None:
            await client.aclose()

    if resp.status_code != 200:
        # TikTok returns 200 even on logical failures (body carries
        # error_code) — but cover network 4xx / 5xx too.
        raise TikTokAuthError(
            f"TikTok /v2/oauth/token returned HTTP {resp.status_code}",
            body=resp.text[:500],
        )

    body = resp.json()
    if not isinstance(body, dict):
        raise TikTokAuthError("TikTok token response is not an object", body=body)

    # Per docs the success payload is FLAT — no `data` wrapper on the
    # v2 token endpoint. Refresh-token responses share the shape.
    err = body.get("error")
    if err and err != "":
        raise TikTokAuthError(
            f"TikTok refused token exchange: {err} - {body.get('error_description')}",
            body=body,
        )

    access = body.get("access_token")
    refresh = body.get("refresh_token")
    open_id = body.get("open_id")
    scope = body.get("scope") or ""
    expires_in = body.get("expires_in")
    refresh_expires_in = body.get("refresh_expires_in")

    if not isinstance(access, str) or not access:
        raise TikTokAuthError("TikTok response missing access_token", body=body)
    if not isinstance(refresh, str) or not refresh:
        raise TikTokAuthError("TikTok response missing refresh_token", body=body)
    if not isinstance(open_id, str) or not open_id:
        raise TikTokAuthError("TikTok response missing open_id", body=body)

    return ExchangedToken(
        access_token=access,
        refresh_token=refresh,
        open_id=open_id,
        scope=scope,
        expires_at=_expires_iso(expires_in),
        refresh_expires_at=_expires_iso(refresh_expires_in),
    )


@dataclass(frozen=True, slots=True)
class TikTokUserInfo:
    """Subset of /v2/user/info/ we surface to the operator."""

    open_id: str
    union_id: str | None
    display_name: str | None
    avatar_url: str | None


async def fetch_user_info(
    *,
    access_token: str,
    http: httpx.AsyncClient | None = None,
) -> TikTokUserInfo:
    """Pull display_name + avatar_url so the post-connect UI can
    confirm to the operator "you're connected as @<handle>".

    The fields param is required; TikTok rejects requests that ask
    for everything. We pick the minimum sufficient set.
    """
    fields = "open_id,union_id,avatar_url,display_name"
    url = f"{_USER_INFO_URL}?fields={fields}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Cache-Control": "no-cache",
    }
    client = http or httpx.AsyncClient(timeout=15.0)
    try:
        resp = await client.get(url, headers=headers)
    finally:
        if http is None:
            await client.aclose()

    if resp.status_code != 200:
        raise TikTokAuthError(
            f"TikTok /v2/user/info returned HTTP {resp.status_code}",
            body=resp.text[:500],
        )
    body = resp.json()
    if not isinstance(body, dict):
        raise TikTokAuthError("TikTok user_info response is not an object", body=body)

    data = body.get("data")
    if not isinstance(data, dict):
        raise TikTokAuthError("TikTok user_info missing data", body=body)
    user = data.get("user")
    if not isinstance(user, dict):
        raise TikTokAuthError("TikTok user_info missing data.user", body=body)

    return TikTokUserInfo(
        open_id=str(user.get("open_id") or ""),
        union_id=user.get("union_id") if isinstance(user.get("union_id"), str) else None,
        display_name=(
            user.get("display_name") if isinstance(user.get("display_name"), str) else None
        ),
        avatar_url=(
            user.get("avatar_url") if isinstance(user.get("avatar_url"), str) else None
        ),
    )


@dataclass(frozen=True, slots=True)
class CreatorInfo:
    """Subset of /v2/post/publish/creator_info/query/ for the UI.

    `privacy_level_options` is the set TikTok says THIS account is
    allowed to post at (audited apps see all three; unaudited see a
    restricted set). The publisher must pick one and pass it along
    in the publish payload, so it's surfaced at connect time so the
    operator knows which knob they're working with.

    `can_post` is the explicit "this account can currently publish"
    flag. Comes back False during sandbox-mode rollouts, account
    deactivations, etc. Surfaced on the read-only display.
    """

    can_post: bool
    privacy_level_options: list[str]
    duet_disabled: bool | None
    comment_disabled: bool | None
    stitch_disabled: bool | None
    max_video_post_duration_sec: int | None


async def query_creator_info(
    *,
    access_token: str,
    http: httpx.AsyncClient | None = None,
) -> CreatorInfo:
    """Run TikTok's "can this account post right now?" check.

    Required BEFORE every publish call per platform docs. We also
    call it once at connect time so the post-connect UI can warn
    early if e.g. duets are disabled or the account is shadow-
    restricted.
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
    }
    client = http or httpx.AsyncClient(timeout=15.0)
    try:
        resp = await client.post(_CREATOR_INFO_URL, headers=headers, json={})
    finally:
        if http is None:
            await client.aclose()

    if resp.status_code != 200:
        raise TikTokAuthError(
            f"TikTok creator_info returned HTTP {resp.status_code}",
            body=resp.text[:500],
        )
    body = resp.json()
    if not isinstance(body, dict):
        raise TikTokAuthError("creator_info response is not an object", body=body)
    data = body.get("data")
    if not isinstance(data, dict):
        # Some sandbox apps return data: null with an error block —
        # treat as can-post:False so the UI doesn't claim everything's
        # fine.
        return CreatorInfo(
            can_post=False,
            privacy_level_options=[],
            duet_disabled=None,
            comment_disabled=None,
            stitch_disabled=None,
            max_video_post_duration_sec=None,
        )

    privacy_opts = data.get("privacy_level_options") or []
    if not isinstance(privacy_opts, list):
        privacy_opts = []

    return CreatorInfo(
        can_post=bool(privacy_opts),  # empty options ⇒ effectively no-post
        privacy_level_options=[str(x) for x in privacy_opts if isinstance(x, str)],
        duet_disabled=(
            bool(data["duet_disabled"]) if "duet_disabled" in data else None
        ),
        comment_disabled=(
            bool(data["comment_disabled"]) if "comment_disabled" in data else None
        ),
        stitch_disabled=(
            bool(data["stitch_disabled"]) if "stitch_disabled" in data else None
        ),
        max_video_post_duration_sec=(
            int(data["max_video_post_duration_sec"])
            if isinstance(data.get("max_video_post_duration_sec"), int | float)
            else None
        ),
    )


__all__ = [
    "TikTokAuthError",
    "ExchangedToken",
    "TikTokUserInfo",
    "CreatorInfo",
    "build_authorize_url",
    "exchange_code_for_token",
    "refresh_access_token",
    "fetch_user_info",
    "query_creator_info",
]
