"""Instagram (via Facebook Login) — OAuth connect flow.

Pattern A: NexoClip owns ONE Meta app on developers.facebook.com,
with Facebook Login + Instagram Graph API + Business Verification.
Tenants never see app_secret; they OAuth their personal Facebook
account, which has the IG Business / Creator account linked via a
Facebook Page they admin. We resolve Page → linked IG-Business id,
store the IG-Business id as platform_user_id (canonical for publish),
and the long-lived Page Access Token as the credential the publisher
will use.

Token model — different from TikTok/Google:

  * NO classic refresh_token. Meta gives you a SHORT-lived user token
    (~1 hour) which you exchange for a LONG-lived user token (~60d).
    To "refresh", you re-exchange the long-lived token itself for a
    new long-lived token BEFORE expires_at. The scheduler does this
    proactively (Wave 2 Task 5).

  * The Page Access Token (what you actually use to call IG endpoints)
    inherits the user token's lifetime — refreshing the user token
    isn't enough; you also re-fetch the Page Access Token. We store
    the Page Access Token as the access_token_encrypted, the long-
    lived USER token as the refresh_token_encrypted slot (overloading
    the column — it's the only "refresh credential" Meta exposes),
    and tag token_type='long_lived'.

Endpoints:

  * https://www.facebook.com/v18.0/dialog/oauth   (browser redirect)
  * https://graph.facebook.com/v18.0/oauth/access_token
        — short → long-lived USER token exchange + the refresh re-
          exchange (same endpoint, fb_exchange_token grant)
  * https://graph.facebook.com/v18.0/me/accounts  — list Pages
  * https://graph.facebook.com/v18.0/{page-id}?fields=instagram_business_account,access_token
  * https://graph.facebook.com/v18.0/{ig-user-id}?fields=username,name,profile_picture_url
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlencode

import httpx

from nexoclip.errors import NexoClipError


class InstagramAuthError(NexoClipError):
    """Raised on any failure in the Meta OAuth round-trip."""

    def __init__(self, message: str, *, body: object = None) -> None:
        super().__init__(message)
        self.body = body


_FB_GRAPH_VER: Final = "v18.0"
_AUTHORIZE_URL: Final = (
    f"https://www.facebook.com/{_FB_GRAPH_VER}/dialog/oauth"
)
_OAUTH_TOKEN_URL: Final = (
    f"https://graph.facebook.com/{_FB_GRAPH_VER}/oauth/access_token"
)
_GRAPH_BASE: Final = f"https://graph.facebook.com/{_FB_GRAPH_VER}"

# Required permissions for IG content publishing. Each requires Meta
# App Review before non-test users can grant — see SHIP.md.
_DEFAULT_SCOPES: Final = (
    "instagram_basic",
    "instagram_content_publish",
    "pages_show_list",
    "pages_read_engagement",
    "business_management",
)


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


def _expires_iso(seconds_from_now: int | float | None) -> str:
    s = int(seconds_from_now or 0)
    return (_now() + _dt.timedelta(seconds=s)).isoformat()


def build_authorize_url(
    *,
    app_id: str,
    redirect_uri: str,
    state: str,
    scopes: tuple[str, ...] = _DEFAULT_SCOPES,
) -> str:
    """Construct the Facebook Login dialog URL.

    `state` MUST be the HMAC-signed token from
    `nexoclip.integrations.oauth.state.sign_state`. Meta echoes it
    back on the callback verbatim.
    """
    if not app_id:
        raise InstagramAuthError("META_APP_ID is not configured")
    params = {
        "client_id": app_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "scope": ",".join(scopes),
        "response_type": "code",
    }
    return f"{_AUTHORIZE_URL}?{urlencode(params)}"


@dataclass(frozen=True, slots=True)
class ShortLivedToken:
    """Intermediate result — exchanged for a long-lived token next."""

    access_token: str
    expires_at: str


@dataclass(frozen=True, slots=True)
class LongLivedToken:
    """The ~60-day USER token. Drives the Page Access Token fetch."""

    access_token: str
    expires_at: str


@dataclass(frozen=True, slots=True)
class FacebookPage:
    """A Page the connected user administers — IG-Business comes from here."""

    id: str
    name: str
    access_token: str  # Page Access Token; what the publisher uses


@dataclass(frozen=True, slots=True)
class InstagramBusinessAccount:
    """What ends up in connected_accounts: the IG-Business account
    linked to a FB Page, with the Page Access Token (long-lived,
    inherits the user token's expiry)."""

    ig_user_id: str
    page_id: str
    page_access_token: str
    username: str | None
    name: str | None
    profile_picture_url: str | None


async def exchange_code_for_short_lived(
    *,
    app_id: str,
    app_secret: str,
    code: str,
    redirect_uri: str,
    http: httpx.AsyncClient | None = None,
) -> ShortLivedToken:
    """Step 1: code → short-lived user token (~1h)."""
    if not app_secret:
        raise InstagramAuthError("META_APP_SECRET is not configured")

    params = {
        "client_id": app_id,
        "client_secret": app_secret,
        "code": code,
        "redirect_uri": redirect_uri,
    }
    client = http or httpx.AsyncClient(timeout=20.0)
    try:
        resp = await client.get(_OAUTH_TOKEN_URL, params=params)
    finally:
        if http is None:
            await client.aclose()

    body = _parse_token_response(resp, step="short-lived exchange")
    access = body.get("access_token")
    expires_in = body.get("expires_in")
    if not isinstance(access, str) or not access:
        raise InstagramAuthError(
            "Meta short-lived response missing access_token", body=body
        )
    return ShortLivedToken(
        access_token=access,
        expires_at=_expires_iso(expires_in),
    )


async def exchange_for_long_lived(
    *,
    app_id: str,
    app_secret: str,
    short_lived_token: str,
    http: httpx.AsyncClient | None = None,
) -> LongLivedToken:
    """Step 2: short-lived user token → long-lived user token (~60d).

    The same endpoint is used for the refresh job (Wave 2 Task 5) —
    just call this again with the CURRENT long-lived token as the
    `short_lived_token` arg before expires_at. Meta extends the
    expiry from the call time, not from the original issue.
    """
    params = {
        "grant_type": "fb_exchange_token",
        "client_id": app_id,
        "client_secret": app_secret,
        "fb_exchange_token": short_lived_token,
    }
    client = http or httpx.AsyncClient(timeout=20.0)
    try:
        resp = await client.get(_OAUTH_TOKEN_URL, params=params)
    finally:
        if http is None:
            await client.aclose()

    body = _parse_token_response(resp, step="long-lived exchange")
    access = body.get("access_token")
    expires_in = body.get("expires_in")
    if not isinstance(access, str) or not access:
        raise InstagramAuthError(
            "Meta long-lived response missing access_token", body=body
        )
    return LongLivedToken(
        access_token=access,
        expires_at=_expires_iso(expires_in),
    )


async def list_pages(
    *,
    long_lived_user_token: str,
    http: httpx.AsyncClient | None = None,
) -> list[FacebookPage]:
    """Step 3: enumerate the FB Pages this user administers.

    Returns a list because a user may admin multiple pages; the
    callback handler picks the first one whose linked IG-Business
    account exists, OR surfaces an "you have N pages, pick one"
    UI in a follow-up (Wave 2 V1 picks the first).
    """
    url = f"{_GRAPH_BASE}/me/accounts"
    client = http or httpx.AsyncClient(timeout=15.0)
    try:
        resp = await client.get(
            url,
            params={
                "access_token": long_lived_user_token,
                "fields": "id,name,access_token",
            },
        )
    finally:
        if http is None:
            await client.aclose()

    body = _parse_token_response(resp, step="me/accounts")
    data = body.get("data")
    if not isinstance(data, list):
        raise InstagramAuthError("me/accounts response missing data array", body=body)

    pages: list[FacebookPage] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        pid = raw.get("id")
        pname = raw.get("name")
        ptoken = raw.get("access_token")
        if not (isinstance(pid, str) and isinstance(ptoken, str)):
            continue
        pages.append(
            FacebookPage(
                id=pid,
                name=pname if isinstance(pname, str) else pid,
                access_token=ptoken,
            )
        )
    return pages


async def resolve_ig_business_account(
    *,
    page: FacebookPage,
    http: httpx.AsyncClient | None = None,
) -> InstagramBusinessAccount | None:
    """Step 4: pull the IG-Business account linked to a FB Page.

    Returns None if the Page has no IG-Business account linked —
    caller iterates the next Page from `list_pages`. Common case
    on first connect: user admins multiple Pages but only one has
    IG linked, so we want to try each.
    """
    page_info_url = f"{_GRAPH_BASE}/{page.id}"
    client = http or httpx.AsyncClient(timeout=15.0)
    try:
        # Pull the IG-Business pointer off the Page.
        page_resp = await client.get(
            page_info_url,
            params={
                "access_token": page.access_token,
                "fields": "instagram_business_account",
            },
        )
    finally:
        if http is None:
            await client.aclose()

    page_body = _parse_token_response(page_resp, step=f"page/{page.id} info")
    ig_pointer = page_body.get("instagram_business_account")
    if not isinstance(ig_pointer, dict):
        return None
    ig_id = ig_pointer.get("id")
    if not isinstance(ig_id, str):
        return None

    # Now pull the IG account's display fields. Re-use the SAME http
    # client since the caller owns its lifecycle when one was passed.
    ig_client = http or httpx.AsyncClient(timeout=15.0)
    try:
        ig_resp = await ig_client.get(
            f"{_GRAPH_BASE}/{ig_id}",
            params={
                "access_token": page.access_token,
                "fields": "username,name,profile_picture_url",
            },
        )
    finally:
        if http is None:
            await ig_client.aclose()

    ig_body = _parse_token_response(ig_resp, step=f"ig/{ig_id} info")
    return InstagramBusinessAccount(
        ig_user_id=ig_id,
        page_id=page.id,
        page_access_token=page.access_token,
        username=ig_body.get("username") if isinstance(ig_body.get("username"), str) else None,
        name=ig_body.get("name") if isinstance(ig_body.get("name"), str) else None,
        profile_picture_url=(
            ig_body.get("profile_picture_url")
            if isinstance(ig_body.get("profile_picture_url"), str) else None
        ),
    )


async def refresh_long_lived_token(
    *,
    app_id: str,
    app_secret: str,
    current_long_lived_token: str,
    http: httpx.AsyncClient | None = None,
) -> LongLivedToken:
    """Wave 2 Task 5 entry point — re-exchange the long-lived token
    before expires_at. Output expires_at is fresh (~60 days from
    this call), not from the original issue.

    Functionally identical to `exchange_for_long_lived`; named
    separately so the scheduler's call site reads as intent rather
    than 'we're re-doing step 2 for some reason'.
    """
    return await exchange_for_long_lived(
        app_id=app_id,
        app_secret=app_secret,
        short_lived_token=current_long_lived_token,
        http=http,
    )


def _parse_token_response(resp: httpx.Response, *, step: str) -> dict:
    """Validate + parse a Meta JSON response. Centralized so each
    endpoint's error handling reads the same."""
    if resp.status_code != 200:
        raise InstagramAuthError(
            f"Meta {step} returned HTTP {resp.status_code}",
            body=resp.text[:500],
        )
    try:
        body = resp.json()
    except ValueError as e:
        raise InstagramAuthError(
            f"Meta {step} returned non-JSON",
            body=resp.text[:500],
        ) from e
    if not isinstance(body, dict):
        raise InstagramAuthError(
            f"Meta {step} response is not an object", body=body,
        )
    # Meta's error envelope: { "error": { "message": "...", ... } }
    err = body.get("error")
    if isinstance(err, dict) and err.get("message"):
        raise InstagramAuthError(
            f"Meta {step}: {err.get('message')}",
            body=err,
        )
    return body


__all__ = [
    "InstagramAuthError",
    "ShortLivedToken",
    "LongLivedToken",
    "FacebookPage",
    "InstagramBusinessAccount",
    "build_authorize_url",
    "exchange_code_for_short_lived",
    "exchange_for_long_lived",
    "list_pages",
    "resolve_ig_business_account",
    "refresh_long_lived_token",
]
