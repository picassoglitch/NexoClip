"""Async httpx client for the Zernio REST API.

One thin method per endpoint we use. Each returns a typed dataclass
or a plain dict for read-many responses; methods raise `ZernioError`
on any non-2xx or schema-violating response.

Auth model is dead simple: every call carries the same company-wide
bearer key in the `Authorization: Bearer <key>` header (a 67-char
`sk_...` secret). There's no per-tenant auth on our side —
multi-tenancy is the `profileId` we scope connect + accounts by
(the Zernio profile id we manage in `profiles.py`).

Differences from the old upload-post client this replaces:
  - Bearer auth (not `Apikey`), JSON bodies everywhere (upload-post's
    /api/upload wanted form-urlencoded `platform[]` repeat keys).
  - Connect is per-platform (`GET /connect/{platform}`) returning an
    OAuth `authUrl`, not one multi-platform JWT magic link.
  - Posts reference media BY URL (`mediaItems[].url`), so we pass our
    existing signed clip URL straight through — same mechanism
    upload-post used for `video`.
  - Publishing needs the per-platform `accountId` (from GET /accounts),
    where upload-post keyed everything off the profile username alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import httpx

from nexoclip.errors import NexoClipError


class ZernioError(NexoClipError):
    """Raised on any failure in a Zernio API call.

    Caller surfaces as a user-visible error on the publish UI and
    logs the wrapped body for diagnosis. `status_code` is preserved
    so callers can branch (e.g. 404 = profile/account not found).
    """

    def __init__(
        self, message: str, *, status_code: int | None = None, body: object = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


_DEFAULT_BASE_URL: Final = "https://zernio.com/api/v1"


@dataclass(frozen=True, slots=True)
class ZernioProfile:
    """Result of POST /profiles.

    A profile groups a tenant's connected social accounts. `profile_id`
    is Zernio's server-generated `_id` (e.g. `prof_abc123`) — the value
    we pass as `profileId` to connect + accounts calls.
    """

    profile_id: str
    name: str


@dataclass(frozen=True, slots=True)
class ZernioConnectLink:
    """Result of GET /connect/{platform}.

    `auth_url` is the hosted OAuth URL the tenant's browser visits to
    authorize ONE platform. Unlike upload-post's single JWT magic
    link, Zernio mints a separate URL per platform.
    """

    auth_url: str


@dataclass(frozen=True, slots=True)
class ZernioAccount:
    """One connected account from GET /accounts.

    `account_id` is Zernio's `_id` — the handle we MUST pass as
    `accountId` in each `platforms[]` entry when creating a post.
    """

    platform: str
    account_id: str


@dataclass(frozen=True, slots=True)
class ZernioFacebookPage:
    """One selectable page from GET /connect/facebook/select-page.

    Facebook's headless connect flow lands the operator back on our
    /connected page with a short-lived `tempToken`; this is one of the
    Pages that token can manage. `page_id` is what we POST back as
    `pageId` to finish the connection. The page-scoped `access_token`
    Zernio returns is deliberately NOT kept here — we never store or
    log platform tokens, and select-page only needs the pageId.
    """

    page_id: str
    name: str
    username: str | None = None
    category: str | None = None


@dataclass(frozen=True, slots=True)
class ZernioPostResult:
    """Result of POST /posts.

    `post_id` is Zernio's `post._id` — the job handle we poll status
    from (analogous to upload-post's `request_id`).
    """

    success: bool
    post_id: str
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ZernioPostStatus:
    """Result of GET /posts/{id}.

    `status` follows Zernio's vocab (e.g. scheduled / publishing /
    published / failed). `platforms` is the per-platform result
    list/dict once Zernio starts dispatching.
    """

    post_id: str
    status: str
    platforms: Any = field(default=None)


def _unwrap_post(body: dict[str, Any]) -> dict[str, Any]:
    """Return the post object from a response — either nested under
    `post` or the body itself when Zernio returns it flat."""
    nested = body.get("post")
    return nested if isinstance(nested, dict) else body


class ZernioClient:
    """Thin async wrapper. Construct once per request handler or
    long-lived task; the underlying httpx client is re-used across
    calls when you pass `http` in, otherwise each method opens +
    closes its own short-lived client.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        http: httpx.AsyncClient | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        # Strip surrounding whitespace — a trailing newline pasted into
        # the env var would otherwise ride along in the `Bearer <key>`
        # header and Zernio rejects it with 401 Unauthorized.
        api_key = (api_key or "").strip()
        if not api_key:
            raise ZernioError("ZERNIO_API_KEY is not configured")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http = http
        self._timeout_s = timeout_s

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        parse_json: bool = True,
    ) -> Any:
        """Single round-trip. Returns parsed JSON body on 2xx; raises
        ZernioError on anything else. Reuses the injected AsyncClient
        when present, else spawns a one-shot.

        Everything is application/json — Zernio takes JSON bodies on
        every write endpoint (no form-urlencoded special-casing the
        way upload-post needed for its `platform[]` array).
        """
        url = f"{self._base_url}{path}"
        headers = dict(self._headers)
        request_kwargs: dict[str, Any] = {"params": params}
        if json is not None:
            headers["Content-Type"] = "application/json"
            request_kwargs["json"] = json
        client = self._http or httpx.AsyncClient(
            timeout=timeout_s or self._timeout_s,
        )
        try:
            resp = await client.request(
                method, url, headers=headers, **request_kwargs,
            )
        except httpx.HTTPError as e:
            # Timeouts / connect errors surface as ZernioError so every
            # caller's `except ZernioError` degrades gracefully instead
            # of a raw httpx exception 500ing the page.
            raise ZernioError(
                f"zernio {method} {path} request failed: "
                f"{type(e).__name__}: {e}",
            ) from e
        finally:
            if self._http is None:
                await client.aclose()

        if resp.status_code >= 400:
            body: object
            try:
                body = resp.json()
            except ValueError:
                body = resp.text[:500]
            raise ZernioError(
                f"zernio {method} {path} returned HTTP {resp.status_code}",
                status_code=resp.status_code,
                body=body,
            )
        if not parse_json:
            # Endpoints like DELETE may return 204 / empty body on success.
            return None
        try:
            return resp.json()
        except ValueError as e:
            raise ZernioError(
                f"zernio {method} {path} returned non-JSON",
                status_code=resp.status_code,
                body=resp.text[:500],
            ) from e

    # ---- Profiles ----

    async def create_profile(
        self,
        *,
        name: str,
        description: str | None = None,
    ) -> ZernioProfile:
        """POST /profiles — create a profile that groups social accounts.

        Zernio returns a server-generated `_id` (e.g. `prof_abc123`)
        which is what every later connect + accounts call references as
        `profileId`. The tenant names the profile from the dashboard.
        """
        payload: dict[str, Any] = {"name": name}
        if description:
            payload["description"] = description
        body = await self._request("POST", "/profiles", json=payload)
        if not isinstance(body, dict):
            raise ZernioError("create_profile response is not an object", body=body)
        profile = body.get("profile")
        if not isinstance(profile, dict):
            profile = body
        profile_id = profile.get("_id") or profile.get("id")
        if not isinstance(profile_id, str) or not profile_id:
            raise ZernioError("create_profile response missing _id", body=body)
        return ZernioProfile(
            profile_id=profile_id,
            name=str(profile.get("name") or name),
        )

    # ---- Account connection ----

    async def connect_url(
        self,
        platform: str,
        *,
        profile_id: str,
        redirect_url: str | None = None,
        headless: bool = False,
    ) -> ZernioConnectLink:
        """GET /connect/{platform}?profileId=... — mint the hosted
        OAuth URL the tenant visits to authorize one platform.

        `profileId` namespaces the connection to this tenant so the
        resulting account shows up scoped to them on GET /accounts.
        `redirect_url` is where Zernio sends the browser AFTER the
        OAuth callback completes — point it at our own /connected page
        so the operator lands back on NexoClip instead of Zernio's
        dashboard (their white-label flow).
        `headless=True` skips Zernio's own post-OAuth selection UI for
        platforms that need one (Facebook page, Pinterest board, …):
        the redirect carries the selection state (`tempToken` + co) and
        WE render the selector, finishing via the platform-specific
        select endpoint (e.g. `select_facebook_page`).
        """
        params: dict[str, Any] = {"profileId": profile_id}
        if redirect_url:
            params["redirect_url"] = redirect_url
        if headless:
            params["headless"] = "true"
        body = await self._request(
            "GET",
            f"/connect/{platform}",
            params=params,
        )
        auth_url = body.get("authUrl") if isinstance(body, dict) else None
        if not isinstance(auth_url, str) or not auth_url:
            raise ZernioError(
                "connect_url response missing authUrl", body=body,
            )
        return ZernioConnectLink(auth_url=auth_url)

    async def list_accounts(
        self,
        *,
        profile_id: str | None = None,
    ) -> list[ZernioAccount]:
        """GET /accounts — list connected social accounts.

        We pass `profileId` so Zernio scopes the result server-side
        (the primary isolation). As defense-in-depth against the server
        ignoring the param, we ALSO drop rows whose profileId clearly
        mismatches — but only on a clear mismatch: the profileId field
        may come back as a plain string OR an object ({_id: ...}), and
        when we can't compare it we KEEP the row (dropping a connected
        account is worse than a rare cross-profile show in the single-
        profile-per-tenant model).
        """
        params = {"profileId": profile_id} if profile_id else None
        body = await self._request("GET", "/accounts", params=params)
        rows = body.get("accounts") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            raise ZernioError("accounts response missing accounts", body=body)
        out: list[ZernioAccount] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            platform = row.get("platform")
            account_id = row.get("_id") or row.get("accountId") or row.get("id")
            if not (isinstance(platform, str) and isinstance(account_id, str)):
                continue
            if profile_id:
                row_pid = row.get("profileId")
                # Tolerate profileId being a nested object.
                if isinstance(row_pid, dict):
                    row_pid = row_pid.get("_id") or row_pid.get("id")
                # Drop ONLY on a clear string mismatch; keep when absent
                # or unrecognizable.
                if isinstance(row_pid, str) and row_pid and row_pid != profile_id:
                    continue
            out.append(ZernioAccount(platform=platform, account_id=account_id))
        return out

    async def disconnect_account(self, account_id: str) -> None:
        """DELETE /accounts/{id} — disconnect a connected social account.

        Tolerates an empty / 204 success body. Raises ZernioError on a
        non-2xx so the caller can surface it."""
        await self._request(
            "DELETE", f"/accounts/{account_id}", parse_json=False,
        )

    # ---- Webhook settings ----
    # Company-key-wide (NOT per profile): ONE webhook config receives
    # every tenant's events; the receiver resolves the tenant per event
    # from the payload's profileId / post id.

    async def list_webhooks(self) -> list[dict[str, Any]]:
        """GET /webhooks/settings — all configured webhooks (max 10)."""
        body = await self._request("GET", "/webhooks/settings")
        rows = body.get("webhooks") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            raise ZernioError("webhooks response missing webhooks", body=body)
        return [r for r in rows if isinstance(r, dict)]

    async def create_webhook(
        self,
        *,
        name: str,
        url: str,
        secret: str,
        events: list[str],
    ) -> dict[str, Any]:
        """POST /webhooks/settings — create one webhook subscription.

        `secret` keys Zernio's HMAC-SHA256 X-Zernio-Signature on every
        delivery. Zernio auto-disables a webhook after 10 consecutive
        delivery failures — re-running registration re-activates it."""
        body = await self._request(
            "POST",
            "/webhooks/settings",
            json={
                "name": name,
                "url": url,
                "secret": secret,
                "events": events,
                "isActive": True,
            },
        )
        webhook = body.get("webhook") if isinstance(body, dict) else None
        if not isinstance(webhook, dict):
            raise ZernioError("create_webhook response missing webhook", body=body)
        return webhook

    async def update_webhook(
        self,
        webhook_id: str,
        *,
        url: str | None = None,
        secret: str | None = None,
        events: list[str] | None = None,
        is_active: bool | None = None,
    ) -> dict[str, Any]:
        """PUT /webhooks/settings — update one webhook by _id. Only the
        provided fields change."""
        payload: dict[str, Any] = {"_id": webhook_id}
        if url is not None:
            payload["url"] = url
        if secret is not None:
            payload["secret"] = secret
        if events is not None:
            payload["events"] = events
        if is_active is not None:
            payload["isActive"] = is_active
        body = await self._request("PUT", "/webhooks/settings", json=payload)
        webhook = body.get("webhook") if isinstance(body, dict) else None
        return webhook if isinstance(webhook, dict) else {"_id": webhook_id}

    # ---- Headless connect: Facebook page selection ----
    # Facebook OAuth grants a user token that can manage N pages; ONE
    # page must be picked to finish the connection. With headless=true
    # on connect_url, Zernio's redirect carries the selection state
    # (profileId + tempToken + userProfile) instead of showing its own
    # picker — these two calls let us render the picker in NexoClip.

    async def list_facebook_pages(
        self,
        *,
        profile_id: str,
        temp_token: str,
    ) -> list[ZernioFacebookPage]:
        """GET /connect/facebook/select-page — pages the OAuth'd user
        can manage. `temp_token` is the short-lived Facebook token from
        the headless redirect; never log it."""
        body = await self._request(
            "GET",
            "/connect/facebook/select-page",
            params={"profileId": profile_id, "tempToken": temp_token},
        )
        rows = body.get("pages") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            raise ZernioError("select-page response missing pages", body=body)
        out: list[ZernioFacebookPage] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            page_id = row.get("id")
            name = row.get("name")
            if not (isinstance(page_id, str) and page_id):
                continue
            username = row.get("username")
            category = row.get("category")
            out.append(
                ZernioFacebookPage(
                    page_id=page_id,
                    name=str(name) if name else page_id,
                    username=username if isinstance(username, str) else None,
                    category=category if isinstance(category, str) else None,
                )
            )
        return out

    async def select_facebook_page(
        self,
        *,
        profile_id: str,
        page_id: str,
        temp_token: str,
        user_profile: dict[str, Any],
    ) -> ZernioAccount:
        """POST /connect/facebook/select-page — finish the headless
        flow by saving the picked page. `user_profile` is the decoded
        userProfile object from the OAuth redirect, passed through
        verbatim (Zernio requires id/name/profilePicture)."""
        body = await self._request(
            "POST",
            "/connect/facebook/select-page",
            json={
                "profileId": profile_id,
                "pageId": page_id,
                "tempToken": temp_token,
                "userProfile": user_profile,
            },
        )
        account = body.get("account") if isinstance(body, dict) else None
        if not isinstance(account, dict):
            raise ZernioError("select-page response missing account", body=body)
        account_id = account.get("accountId") or account.get("_id") or account.get("id")
        if not isinstance(account_id, str) or not account_id:
            raise ZernioError("select-page response missing accountId", body=body)
        return ZernioAccount(
            platform=str(account.get("platform") or "facebook"),
            account_id=account_id,
        )

    # ---- Publishing ----

    async def create_post(
        self,
        *,
        profile_id: str,
        content: str,
        media_url: str,
        platforms: list[tuple[str, str]],
        publish_now: bool = True,
        scheduled_for: str | None = None,
        timezone: str | None = None,
        tiktok_settings: dict[str, Any] | None = None,
        youtube_settings: dict[str, Any] | None = None,
        instagram_settings: dict[str, Any] | None = None,
        title: str | None = None,
        is_draft: bool = False,
        queued_from_profile: str | None = None,
        custom_content: dict[str, str] | None = None,
        platform_specific_data: dict[str, dict[str, Any]] | None = None,
    ) -> ZernioPostResult:
        """POST /posts — publish (or schedule) a video to one or more
        platforms.

        `media_url` is referenced verbatim as `mediaItems[].url` — we
        pass our own signed clip URL and Zernio downloads from it (the
        same contract upload-post had for `video`). `platforms` is a
        list of (platform, account_id) pairs; the account_id comes
        from `list_accounts` and is REQUIRED per platform.

        Modes (mutually exclusive, in precedence order):
          - `is_draft=True` → saved unpublished (no platforms required
            by Zernio, but we still send the targets we have)
          - `queued_from_profile=<profileId>` → Zernio assigns the next
            available queue slot itself (do NOT pre-compute next-slot;
            that bypasses Zernio's queue locking)
          - `scheduled_for` (ISO 8601) → scheduled, with `timezone`
          - otherwise `publishNow=True` fires immediately

        Per-platform extras: `custom_content[platform]` overrides the
        caption for that target; `platform_specific_data[platform]`
        flows verbatim into that target's `platformSpecificData` (e.g.
        `{"youtube": {"title": ..., "firstComment": ...}}`). Root-level
        settings (tiktokSettings etc.) still apply to all targets of
        that platform, with per-target data taking precedence (Zernio
        merges them).

        For TikTok the caller MUST include `content_preview_confirmed`
        and `express_consent_given` (Zernio rejects the post otherwise —
        a TikTok legal requirement).
        """
        platform_entries: list[dict[str, Any]] = []
        for p, account_id in platforms:
            entry: dict[str, Any] = {"platform": p, "accountId": account_id}
            if custom_content and p in custom_content:
                entry["customContent"] = custom_content[p]
            if platform_specific_data and p in platform_specific_data:
                entry["platformSpecificData"] = platform_specific_data[p]
            platform_entries.append(entry)
        payload: dict[str, Any] = {
            "content": content,
            "mediaItems": [{"type": "video", "url": media_url}],
            "platforms": platform_entries,
        }
        if title:
            payload["title"] = title
        if is_draft:
            payload["isDraft"] = True
        elif queued_from_profile:
            payload["queuedFromProfile"] = queued_from_profile
        elif scheduled_for:
            payload["scheduledFor"] = scheduled_for
            if timezone:
                payload["timezone"] = timezone
        else:
            payload["publishNow"] = publish_now
        if tiktok_settings:
            payload["tiktokSettings"] = tiktok_settings
        if youtube_settings:
            payload["youtubeSettings"] = youtube_settings
        if instagram_settings:
            payload["instagramSettings"] = instagram_settings

        body = await self._request(
            "POST", "/posts", json=payload,
            # Zernio re-hosts our media + dispatches per platform;
            # allow a generous ceiling for the synchronous ack.
            timeout_s=120.0,
        )
        if not isinstance(body, dict):
            raise ZernioError("post response is not an object", body=body)
        post = _unwrap_post(body)
        post_id = post.get("_id") or post.get("id") or body.get("_id")
        if not isinstance(post_id, str) or not post_id:
            raise ZernioError("post response missing _id", body=body)
        return ZernioPostResult(
            success=bool(body.get("success", True)),
            post_id=post_id,
            message=(
                str(body["message"]) if isinstance(body.get("message"), str) else None
            ),
        )

    # ---- Analytics ----

    async def best_time_slots(
        self,
        *,
        profile_id: str,
        platform: str | None = None,
    ) -> list[dict[str, Any]]:
        """GET /analytics/best-time — engagement-ranked posting slots.

        Returns the raw slot list ({day_of_week 0=Mon, hour UTC,
        avg_engagement, post_count}); empty for accounts without
        enough history (callers fall back to a sane default). 403
        (Analytics add-on required) is surfaced as ZernioError — the
        caller treats it like "no data"."""
        params: dict[str, Any] = {"profileId": profile_id}
        if platform:
            params["platform"] = platform
        body = await self._request("GET", "/analytics/best-time", params=params)
        slots = body.get("slots") if isinstance(body, dict) else None
        return [s for s in slots if isinstance(s, dict)] if isinstance(slots, list) else []

    # ---- Queue (recurring schedule slots) ----
    # NOTE: a queue SLOT's dayOfWeek is 0=Sunday..6=Saturday — NOT the
    # best-time convention (0=Monday). Don't mix the two.

    async def list_queues(self, *, profile_id: str) -> list[dict[str, Any]]:
        """GET /queue/slots?all=true — every recurring queue for the
        profile. Returns the raw QueueSchedule dicts (each with `_id`,
        `name`, `timezone`, `slots:[{dayOfWeek,time}]`, `active`,
        `isDefault`). Empty list when the profile has no queues yet."""
        body = await self._request(
            "GET", "/queue/slots",
            params={"profileId": profile_id, "all": "true"},
        )
        rows = body.get("queues") if isinstance(body, dict) else None
        return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []

    async def upsert_default_queue(
        self,
        *,
        profile_id: str,
        slots: list[dict[str, Any]],
        timezone: str = "UTC",
        name: str = "NexoClip Queue",
        reshuffle_existing: bool = False,
    ) -> dict[str, Any]:
        """PUT /queue/slots — create-or-update the profile's DEFAULT
        queue (no queueId → default). `slots` is a list of
        {dayOfWeek 0=Sun..6=Sat, time "HH:mm"}. Returns the saved
        QueueSchedule."""
        body = await self._request(
            "PUT", "/queue/slots",
            json={
                "profileId": profile_id,
                "name": name,
                "timezone": timezone,
                "slots": slots,
                "active": True,
                "reshuffleExisting": reshuffle_existing,
            },
        )
        sched = body.get("schedule") if isinstance(body, dict) else None
        return sched if isinstance(sched, dict) else (body if isinstance(body, dict) else {})

    async def delete_queue(self, *, profile_id: str, queue_id: str) -> None:
        """DELETE /queue/slots?profileId=&queueId= — remove one queue.
        Deleting the default promotes another queue to default."""
        await self._request(
            "DELETE", "/queue/slots",
            params={"profileId": profile_id, "queueId": queue_id},
            parse_json=False,
        )

    # ---- Status + history ----

    async def get_post(self, post_id: str) -> ZernioPostStatus:
        """GET /posts/{id} — read one post's status + per-platform
        results. Used by the job detail page and the status poll."""
        body = await self._request("GET", f"/posts/{post_id}")
        if not isinstance(body, dict):
            raise ZernioError("post status response is not an object", body=body)
        post = _unwrap_post(body)
        return ZernioPostStatus(
            post_id=str(post.get("_id") or post.get("id") or post_id),
            status=str(post.get("status") or "unknown"),
            platforms=post.get("platforms"),
        )

    async def delete_post(self, post_id: str) -> None:
        """DELETE /posts/{id} — delete a draft or scheduled post.

        Published posts can't be deleted this way (Zernio 400s; their
        Unpublish endpoint is the path for those). Upload quota is
        refunded by Zernio automatically."""
        await self._request("DELETE", f"/posts/{post_id}", parse_json=False)

    async def list_posts(
        self,
        *,
        page: int = 1,
        limit: int = 25,
        status: str | None = None,
        profile_id: str | None = None,
        sort_by: str | None = None,
    ) -> dict[str, Any]:
        """GET /posts — feed of past + scheduled posts. Returns the
        raw dict so the UI can render whatever the backend gives
        back (mirrors the old history feed).

        `status` (draft|scheduled|published|failed), `profile_id`, and
        `sort_by` (e.g. scheduled-asc) narrow the feed — used by the
        Upcoming list (scheduled, soonest-first) and the drafts panel."""
        params: dict[str, Any] = {"page": page, "limit": limit}
        if status:
            params["status"] = status
        if profile_id:
            params["profileId"] = profile_id
        if sort_by:
            params["sortBy"] = sort_by
        body = await self._request("GET", "/posts", params=params)
        if not isinstance(body, dict):
            raise ZernioError("posts list response is not an object", body=body)
        return body

    async def list_failed(
        self, *, profile_id: str | None = None, limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Posts that failed to publish, for the retry affordance.

        There is NO /posts/failed endpoint — the spec lists failed
        posts via GET /posts?status=failed (confirmed against the
        OpenAPI; the old path 404'd). Scoped by profileId so a tenant
        only sees their own failures. Returns the raw array."""
        params: dict[str, Any] = {"status": "failed", "limit": limit}
        if profile_id:
            params["profileId"] = profile_id
        body = await self._request("GET", "/posts", params=params)
        if isinstance(body, dict):
            posts = body.get("posts")
            if isinstance(posts, list):
                return [p for p in posts if isinstance(p, dict)]
        if isinstance(body, list):
            return [p for p in body if isinstance(p, dict)]
        raise ZernioError("failed-posts response shape unexpected", body=body)

    async def retry_post(self, post_id: str) -> ZernioPostResult:
        """POST /posts/{id}/retry — re-attempt a failed post."""
        body = await self._request("POST", f"/posts/{post_id}/retry")
        if not isinstance(body, dict):
            raise ZernioError("retry response is not an object", body=body)
        post = _unwrap_post(body)
        return ZernioPostResult(
            success=bool(body.get("success", True)),
            post_id=str(post.get("_id") or post.get("id") or post_id),
            message=(
                str(body["message"]) if isinstance(body.get("message"), str) else None
            ),
        )


__all__ = [
    "ZernioAccount",
    "ZernioClient",
    "ZernioConnectLink",
    "ZernioError",
    "ZernioPostResult",
    "ZernioPostStatus",
    "ZernioProfile",
]
