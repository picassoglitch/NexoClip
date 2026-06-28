"""Zernio dashboard + publish actions.

Hangs off /dashboard/publish/zernio. The operator clicks through to
this surface to:

  - Connect their socials (one hosted-OAuth button per platform; we
    303 the browser to Zernio's authUrl for that platform)
  - Publish a clip to one or more connected platforms
  - See history of past + scheduled posts

Endpoints
  GET  /dashboard/publish/zernio                  — the dashboard page
  POST /dashboard/publish/zernio/connect          — mint authUrl, 303 away
  POST /dashboard/publish/zernio/accounts/claim   — bind an existing profileId
  POST /dashboard/publish/zernio/unlink           — forget the binding
  POST /dashboard/publish/zernio/post/{clip_id}   — publish one clip
  POST /dashboard/publish/zernio/bulk-post        — publish many clips
  GET  /dashboard/publish/zernio/feed.json        — HTMX-polled feed
  GET  /dashboard/publish/zernio/job/{post_id}    — per-post detail page
  GET  /dashboard/publish/zernio/status/{post_id}.json — single-post poll

Replaces the upload-post.com surface (routers/upload_post.py). The
signed-clip-URL mechanism (mint_signed_clip_url) is reused verbatim:
Zernio downloads the clip MP4 from `mediaItems[].url` exactly as
upload-post downloaded `video`.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from nexoclip.db import (
    ClipsRepo,
    Database,
    EventsRepo,
    TenantsRepo,
    ZernioBroadcastLogRepo,
    ZernioCalendarRepo,
    ZernioInboxRepo,
    ZernioPublishesRepo,
)
from nexoclip.integrations.zernio import (
    ZernioAccount,
    ZernioClient,
    ZernioError,
    create_profile_for_tenant,
)
from nexoclip.publish.hub import PublishOptions, build_per_platform_payload
from nexoclip.settings import get_settings, resolve_db_target

from ..deps import get_db, require_full_scope, tenant_binder
from ..status_gate import require_paid_tier
from .clips import _VALID_STATUS_TRANSITIONS
from .internal import mint_signed_clip_url, signed_clip_ttl_for_schedule

_log = logging.getLogger("nexoclip.api.zernio")

router = APIRouter(prefix="/dashboard/publish/zernio", tags=["zernio"])

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
from ..i18n import install_globals as _install_i18n  # noqa: E402

_install_i18n(templates)


def _build_client() -> ZernioClient:
    """Build a ZernioClient from settings. Raises 503 on the request
    path when the API key isn't configured so the operator sees a
    clear error instead of an opaque 500."""
    settings = get_settings()
    if not settings.zernio_api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "ZERNIO_API_KEY is not configured on this NexoClip "
                "instance. Add NEXOCLIP_ZERNIO_API_KEY to Railway env "
                "and redeploy."
            ),
        )
    return ZernioClient(
        api_key=settings.zernio_api_key,
        base_url=settings.zernio_base_url,
    )


def _public_base_url(request: Request) -> str:
    """Return the externally-reachable origin so signed clip URLs
    point at the right hostname. Prefers the public_url setting when
    set (production custom domain), falls back to the request's own
    scheme+host (works for localhost dev)."""
    settings = get_settings()
    if settings.public_url and not settings.public_url.startswith(
        ("http://localhost", "http://127.0.0.1"),
    ):
        return settings.public_url.rstrip("/")
    scheme = request.headers.get("x-forwarded-proto") or request.url.scheme or "https"
    host = request.headers.get("host") or request.url.netloc
    return f"{scheme}://{host}"


# ---------- Dashboard page ----------


_SUPPORTED_PLATFORMS = [
    # (id used by the Zernio API,  human label,        tabler icon class)
    # NOTE: Zernio uses `twitter`, NOT upload-post's `x`.
    ("tiktok",    "TikTok",    "ti-brand-tiktok"),
    ("instagram", "Instagram", "ti-brand-instagram"),
    ("youtube",   "YouTube",   "ti-brand-youtube"),
    ("twitter",   "X",         "ti-brand-x"),
    ("linkedin",  "LinkedIn",  "ti-brand-linkedin"),
    ("facebook",  "Facebook",  "ti-brand-facebook"),
    ("threads",   "Threads",   "ti-brand-threads"),
    ("pinterest", "Pinterest", "ti-brand-pinterest"),
    ("bluesky",   "Bluesky",   "ti-brand-bluesky"),
]

# Platforms whose api id Zernio publishes to. Used to validate the
# checkbox group on the publish path.
_SUPPORTED_PLATFORM_IDS = frozenset(p[0] for p in _SUPPORTED_PLATFORMS)

# Minimum minutes between consecutive hands-free "now"-mode posts. Firing a
# whole batch of eligible clips at once trips TikTok's "wait 10 minutes"
# rate limit (the burst failures seen in prod), so the first clip goes out
# immediately and the rest are staggered at least this far apart. Sits just
# above TikTok's 10-min cooldown.
_NOW_MODE_BURST_INTERVAL_MIN = 12

# Community channels — connectable like any platform, but NOT clip
# targets (phase 11): they get Connect buttons + a notification toggle,
# never publish chips.
_COMMUNITY_PLATFORMS = [
    ("discord",  "Discord",  "ti-brand-discord"),
    ("telegram", "Telegram", "ti-brand-telegram"),
]
_COMMUNITY_PLATFORM_IDS = frozenset(p[0] for p in _COMMUNITY_PLATFORMS)
_CONNECTABLE_PLATFORM_IDS = _SUPPORTED_PLATFORM_IDS | _COMMUNITY_PLATFORM_IDS

# Platforms whose OAuth needs a post-selection step (pick ONE page /
# board) that we render ourselves via Zernio's headless connect mode.
# Facebook is implemented; Pinterest boards follow the same pattern
# when we add `/connect/pinterest/select-board` client methods.
_HEADLESS_SELECTION_PLATFORMS = frozenset({"facebook"})


def _clip_display_title(clip: Any) -> str:
    """Operator-readable title for a clip card.

    Prefers the overlay_config.title_text the operator typed in the
    editor; falls back to duration + a truncated id so cards never
    render as just hash IDs."""
    ov = getattr(clip, "overlay_config", None) or {}
    title = ov.get("title_text") if isinstance(ov, dict) else None
    if isinstance(title, str) and title.strip():
        return title.strip()
    duration_s = float(getattr(clip, "duration_s", 0) or 0)
    return f"Clip {duration_s:.0f}s · {getattr(clip, 'id', '')[:14]}…"


def _account_map(accounts: list[ZernioAccount]) -> dict[str, str]:
    """platform → account_id, lowercased keys. Zernio requires the
    per-platform `accountId` on every post, so we resolve it from the
    connected-accounts list. Last write wins if a tenant connected the
    same platform twice (rare); the most recent connection is fine."""
    return {a.platform.lower(): a.account_id for a in accounts}


def _connected_platforms(accounts: list[ZernioAccount]) -> set[str]:
    """Platform keys ready to ship to — every connected account's
    platform, lowercased."""
    return {a.platform.lower() for a in accounts}


def _order_publish_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order the Publicados feed so the operator sees what is about to go
    out, not just what was most recently queued.

    Upcoming scheduled posts come first, soonest-to-publish at the top
    (ordered by `scheduled_for`); already-published / immediate history
    follows, newest first. A scheduled post's effective time is its
    publish time; everything else uses `created_at`.

    All upcoming posts are kept — never truncated, which was the "últimos
    25 nada más" complaint where a backlog of scheduled clips fell off the
    bottom. Only the published tail is capped (at 25)."""

    def _is_upcoming(h: dict[str, Any]) -> bool:
        return h.get("status") == "scheduled" and bool(h.get("scheduled_for"))

    def _when(h: dict[str, Any]) -> str:
        if _is_upcoming(h):
            return str(h.get("scheduled_for") or "")
        return str(h.get("created_at") or "")

    upcoming = sorted((h for h in history if _is_upcoming(h)), key=_when)
    published = sorted(
        (h for h in history if not _is_upcoming(h)), key=_when, reverse=True
    )
    return upcoming + published[:25]


def _tiktok_settings() -> dict[str, Any]:
    """Default TikTok publishing settings.

    `content_preview_confirmed` + `express_consent_given` are MANDATORY
    (a TikTok legal requirement Zernio enforces) — Zernio rejects the
    post without them. The rest are sensible public-video defaults;
    a future UI can let the operator override privacy/duet/stitch.
    """
    return {
        "privacy_level": "PUBLIC_TO_EVERYONE",
        "allow_comment": True,
        "allow_duet": True,
        "allow_stitch": True,
        "content_preview_confirmed": True,
        "express_consent_given": True,
    }


def _account_limit(request: Request) -> int | None:
    """Connected-account cap for the requesting tenant's tier.

    pro = 1, all_access (VIP) = None (unlimited). The tier was
    normalized by the auth middleware; tiers.zernio_account_limit holds
    the per-tier numbers (single source of truth)."""
    from nexoclip.tiers import zernio_account_limit

    tier = getattr(request.state, "tenant_tier", None)
    return zernio_account_limit(tier)


def _account_limit_message(limit: int) -> str:
    """Operator-facing copy for hitting the per-tier account cap."""
    return (
        f"Your plan allows {limit} connected social account"
        f"{'' if limit == 1 else 's'}. Disconnect one first, or upgrade "
        f"to All-Access for unlimited accounts."
    )


def _existing_post_id_from_conflict(e: ZernioError) -> str | None:
    """Pull `details.existingPostId` out of a Zernio 409 duplicate-content
    error, or None when the conflict is something else.

    Zernio's shape: {"error": "This exact content is already scheduled,
    publishing, or was posted ...", "details": {"accountId": ...,
    "platform": ..., "existingPostId": ...}}.
    """
    if e.status_code != 409 or not isinstance(e.body, dict):
        return None
    details = e.body.get("details")
    if not isinstance(details, dict):
        return None
    existing = details.get("existingPostId")
    return existing if isinstance(existing, str) and existing else None


async def _read_json(request: Request) -> Any:
    """Return the parsed JSON body, or None on an empty/invalid body.

    FastAPI's `await request.json()` raises (→ 500) on an empty or
    non-JSON body; this swallows that so handlers can return a clean
    4xx instead."""
    try:
        return await request.json()
    except Exception:
        # Any malformed/empty body → treat as absent; caller returns 4xx.
        return None


async def _require_profile(db: Database, tenant_id: str) -> str:
    """Return the tenant's Zernio profile_id, or raise 409 telling the
    operator to create a profile first.

    Connecting accounts + publishing both require a profile to exist on
    Zernio (created via POST /profiles from the dashboard). We no longer
    fabricate one — Zernio needs a real `prof_...` id."""
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    if not profile_id:
        raise HTTPException(
            status_code=409,
            detail=(
                "No Zernio profile yet. Create a profile on the Publish "
                "Center first, then connect accounts and publish."
            ),
        )
    return profile_id


async def _publish_clip(
    *,
    client: ZernioClient,
    db: Database,
    request: Request | None,
    tenant_id: str,
    profile_id: str,
    account_map: dict[str, str],
    clip_id: str,
    platforms: list[str],
    content: str,
    title: str | None = None,
    mode: str = "now",
    scheduled_for: str | None = None,
    options: PublishOptions | None = None,
    tiktok_privacy: str | None = None,
    # Background callers (e.g. the auto-program worker) have no live request,
    # so they pass these three request-derived values explicitly. When
    # `request` is provided they're ignored and derived from it instead.
    tenant_tier: str | None = None,
    base_url: str | None = None,
    session_cookie: str | None = None,
) -> str:
    """Shared publish core for the single + bulk paths.

    Resolves each selected platform to its connected accountId, mints a
    signed clip URL, and fires one POST /posts. Returns the Zernio
    post_id. Raises HTTPException with an operator-readable message on
    every failure mode (unknown clip, unconnected platform, missing
    signing secret, Zernio error).

    Phase-4 power-ups: `mode` is now|draft|schedule (draft → isDraft,
    schedule → scheduledFor); `options` carries per-platform caption
    overrides + first comment (built into customContent /
    platformSpecificData via the SAME helper the internal API uses);
    `tiktok_privacy` must be a creator-info value when set.
    """
    repo = ClipsRepo(db)
    clip = await repo.get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")

    # Auto-approve on publish. Choosing to publish a clip implies
    # approving it — and `approved` is the state whose saved overlay
    # config the render burns in, so this also guarantees the rendered
    # MP4 we ship below matches the editor preview. Walk the transition
    # graph (cut -> ready_for_review -> approved) one or two steps.
    if clip.status not in ("approved", "published"):
        allowed = _VALID_STATUS_TRANSITIONS.get(clip.status, set())
        if "approved" in allowed:
            clip = await repo.update_status(clip_id, status="approved")
        elif "ready_for_review" in allowed:
            await repo.update_status(clip_id, status="ready_for_review")
            clip = await repo.update_status(clip_id, status="approved")
        # else: an unexpected status with no path to approved — leave it
        # as-is; the render below still produces the edited MP4.

    # Map selected platforms → (platform, accountId). A platform the
    # operator checked but hasn't connected on Zernio is a clear 409,
    # not a raw Zernio 400 later.
    targets: list[tuple[str, str]] = []
    missing: list[str] = []
    for p in platforms:
        account_id = account_map.get(p.lower())
        if account_id:
            targets.append((p, account_id))
        else:
            missing.append(p)
    if missing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Not connected on Zernio: {', '.join(missing)}. "
                f"Connect these accounts first, then publish."
            ),
        )

    # Per-tier cap also applies at publish time — covers a tenant who
    # connected several accounts on a higher tier and then downgraded.
    if request is not None:
        limit = _account_limit(request)
    else:
        from nexoclip.tiers import zernio_account_limit
        limit = zernio_account_limit(tenant_tier)
    if limit is not None and len(targets) > limit:
        raise HTTPException(
            status_code=402,
            detail=(
                f"Your plan publishes with {limit} connected account"
                f"{'' if limit == 1 else 's'} — deselect the extra "
                f"platform(s) or upgrade to All-Access."
            ),
        )

    base = _public_base_url(request) if request is not None else (base_url or "")

    # Make sure the edited MP4 (overlays + captions burned in) exists on
    # disk BEFORE we hand Zernio the URL — this is what closes the
    # "published clip missing hooks/subs" bug. It's the same file the
    # download path serves, so publish == download. Usually a cache hit
    # because approve pre-renders; renders inline here only when cold.
    # The /render page is auth-gated, so pass the operator's session
    # cookie through to the headless recorder.
    settings = get_settings()
    if request is not None:
        cookie_val = request.cookies.get("nexoclip_token", "") or None
    else:
        cookie_val = session_cookie
    try:
        from nexoclip.api._clip_render import ensure_clip_rendered
        await ensure_clip_rendered(
            db=db,
            clip=clip,
            tenant_id=tenant_id,
            base_url=base,
            auth_cookie_value=cookie_val,
            db_path=resolve_db_target(settings),
        )
    except RuntimeError as e:
        _log.error(
            "zernio.publish.render_failed tenant=%s clip=%s err=%s",
            tenant_id, clip_id, e,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Couldn't render the edited clip (hooks + captions) "
                f"before publishing: {e}. Open the clip and click "
                "Download to surface the render error, then re-publish."
            ),
        ) from e

    try:
        # A scheduled post is fetched at posting time, not now — so the
        # signed URL must live until then. Size its TTL to the scheduled
        # time (+ margin) instead of a flat 1h, which would expire before
        # a post hours/days out is ever downloaded.
        ttl_seconds = signed_clip_ttl_for_schedule(
            scheduled_for if mode == "schedule" else None
        )
        media_url = mint_signed_clip_url(
            clip_id=clip_id,
            tenant_id=tenant_id,
            base_url=base,
            ttl_seconds=ttl_seconds,
        )
    except RuntimeError as e:
        _log.error(
            "zernio.publish.signing_secret_missing tenant=%s clip=%s err=%s",
            tenant_id, clip_id, e,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "NEXOCLIP_INTERNAL_SIGNING_SECRET is not configured on "
                "this NexoClip instance. Add it to Railway env (same "
                "secret used for the Modal Whisper audio fetch) and "
                "redeploy. Zernio needs a signed URL to download your "
                "clip MP4."
            ),
        ) from e

    tiktok = _tiktok_settings() if any(p.lower() == "tiktok" for p, _ in targets) else None
    platform_keys = [p.lower() for p, _ in targets]
    opts = options or PublishOptions()
    custom_content, platform_data = build_per_platform_payload(
        opts, platform_keys, title=title, tiktok_privacy=tiktok_privacy,
    )
    try:
        result = await client.create_post(
            profile_id=profile_id,
            content=content,
            media_url=media_url,
            platforms=targets,
            publish_now=(mode == "now"),
            title=title,
            is_draft=(mode == "draft"),
            scheduled_for=scheduled_for if mode == "schedule" else None,
            timezone="UTC" if (mode == "schedule" and scheduled_for) else None,
            tiktok_settings=tiktok,
            custom_content=custom_content,
            platform_specific_data=platform_data,
        )
    except ZernioError as e:
        # Zernio 409s when the EXACT same content already posted (or is
        # scheduled) to this account within 24h, and tells us the
        # existing post id. Treat a duplicate publish as idempotent:
        # resolve to the existing post instead of surfacing an error —
        # double-clicks and retries land on the same post.
        existing = _existing_post_id_from_conflict(e)
        if existing:
            _log.info(
                "zernio.publish.duplicate_resolved tenant=%s clip=%s post_id=%s",
                tenant_id, clip_id, existing,
            )
            post_id = existing
        else:
            raise
    else:
        post_id = result.post_id

    # Record the publish locally — this table IS the tenant's publish
    # history (Zernio's GET /posts is company-key-wide; migration 030).
    # Idempotent on post_id, so the duplicate-resolved path is safe.
    # Drafts seed status='draft' (the Borradores panel reads it) and
    # snapshot the per-platform extras so re-publish rebuilds the
    # exact payload after the signed URL expires.
    import json as _json

    options_json = None
    if opts.per_platform_captions or opts.first_comment or tiktok_privacy or title:
        options_json = _json.dumps(
            {
                "title": title,
                "per_platform_captions": opts.per_platform_captions,
                "first_comment": opts.first_comment,
                "tiktok_privacy": tiktok_privacy,
            }
        )
    await ZernioPublishesRepo(db).record(
        post_id=post_id,
        tenant_id=tenant_id,
        clip_id=clip_id,
        platforms=[p for p, _ in targets],
        content=content,
        status="draft" if mode == "draft" else (
            "scheduled" if mode == "schedule" else None
        ),
        options_json=options_json,
        scheduled_for=scheduled_for if mode == "schedule" else None,
    )
    # Flip the clip out of the "ready to publish" grid. Best-effort —
    # the publish already happened; a status-write hiccup must not fail
    # the request. Drafts stay approved: the clip isn't out the door
    # yet, and the Borradores panel is its new home.
    if clip.status == "approved" and mode != "draft":
        try:
            await repo.update_status(clip_id, status="published")
        except Exception:
            _log.warning(
                "zernio.publish.status_flip_failed tenant=%s clip=%s",
                tenant_id, clip_id,
            )
    return post_id


@router.get("", response_class=HTMLResponse)
async def zernio_dashboard(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Publish Center — single-page surface with two tabs:
      - Single Publish: thumbnail grid + selected-clip publish panel
      - Bulk Publish: per-clip rows with inline platform chip toggles

    Sections (top to bottom):
      1. Compact "Connected accounts" row (one Connect button per platform)
      2. Tab switcher
      3. Active tab content
      4. History feed at the bottom
    """
    settings = get_settings()
    configured = bool(settings.zernio_api_key)

    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None

    # Self-heal: a tenant with no Zernio profile sees ZERO tabs (every
    # surface gates on profile_id). Auto-provision one on first open of
    # the Publish Center so the tabs just appear — no manual "Create
    # profile" step. Idempotent (find-or-create by deterministic name);
    # best-effort, so a Zernio hiccup falls back to the manual form.
    if configured and tenant is not None and not profile_id:
        from nexoclip.integrations.zernio import ensure_zernio_profile_for_tenant
        with contextlib.suppress(Exception):
            profile_id = await ensure_zernio_profile_for_tenant(
                db=db, tenant_id=tenant_id, client=_build_client(),
            )
    profile_name = tenant.zernio_profile_name if tenant else None

    # Per-tenant publish history comes from OUR table (migration 030) —
    # Zernio's GET /posts is company-key-wide and must never be rendered
    # unfiltered (every tenant would see every other tenant's posts).
    # limit=100 (not 25) so the membership set below doesn't miss older
    # rows and wrongly re-list their clips as "marked manually".
    publishes = await ZernioPublishesRepo(db).list_for_tenant(limit=100)

    accounts: list[ZernioAccount] = []
    status_by_post: dict[str, dict[str, Any]] = {}
    connect_fetch_failed = False

    if configured and profile_id:
        client = _build_client()
        try:
            accounts = await client.list_accounts(profile_id=profile_id)
        except ZernioError as e:
            _log.warning(
                "zernio.dashboard.accounts_fetch_failed tenant=%s err=%s",
                tenant_id, e,
            )
            connect_fetch_failed = True
        # Live status enrichment: ONE feed call, matched against the
        # tenant's own post ids — failures are silent (history still
        # renders from the local rows, statuses just read "queued").
        if publishes:
            try:
                feed = await client.list_posts(page=1, limit=100)
                raw = feed.get("posts")
                if isinstance(raw, list):
                    for p in raw:
                        if isinstance(p, dict):
                            pid = p.get("_id") or p.get("id")
                            if isinstance(pid, str):
                                status_by_post[pid] = p
            except ZernioError as e:
                _log.warning(
                    "zernio.dashboard.feed_fetch_failed tenant=%s err=%s",
                    tenant_id, e,
                )

    connected_set = _connected_platforms(accounts)

    # Published clips — for clip titles on history rows, and to surface
    # clips that are status='published' but have NO publish record
    # (flipped via Mark-published before rows were written for it, or
    # published before migration 030 existed). Those still belong on
    # the Published tab.
    published_clips: list[Any] = []
    with contextlib.suppress(Exception):  # display is best-effort
        published_clips = await ClipsRepo(db).list_for_tenant_with_status(
            ["published"], limit=100,
        )
    title_by_clip = {c.id: _clip_display_title(c) for c in published_clips}

    # Join local rows with live status (only the tenant's own posts).
    # Two kinds: "zernio" (a real post — View opens the job page) and
    # "manual" (marked published by the operator; post_id is synthetic
    # `manual_<clip_id>` — View opens the clip page).
    history: list[dict[str, Any]] = []
    for row in publishes:
        # Drafts live in their own Borradores panel; deleted rows are
        # tombstones (replaced/removed drafts) — neither is history.
        if row.status in ("draft", "deleted"):
            continue
        manual = row.post_id.startswith("manual_")
        live = status_by_post.get(row.post_id, {})
        # Status precedence: live company-key feed (freshest) → our local
        # row.status (fed by post.* webhooks) → "queued". The webhook
        # fallback is what keeps a post truthful when Zernio's feed misses
        # it — deleted on Zernio (the GET /posts/{id} 404), beyond page 1,
        # or created from a different workspace. Without it, an already
        # published/failed post would read "queued" forever.
        status = (
            "published" if manual
            else str(live.get("status") or row.status or "queued")
        )
        history.append(
            {
                "post_id": row.post_id,
                "kind": "manual" if manual else "zernio",
                "clip_id": row.clip_id,
                "title": title_by_clip.get(row.clip_id),
                "platforms": [p for p in row.platforms.split(",") if p],
                "content": row.content,
                "created_at": row.created_at,
                "scheduled_for": row.scheduled_for,
                "status": status,
            }
        )
    # Fallback entries: published clips with no record at all (clips
    # marked before mark-published wrote rows). Timestamp is the clip's
    # creation time — the actual publish moment was never recorded.
    recorded_clip_ids = {row.clip_id for row in publishes}
    for c in published_clips:
        if c.id in recorded_clip_ids:
            continue
        history.append(
            {
                "post_id": f"manual_{c.id}",
                "kind": "manual",
                "clip_id": c.id,
                "title": title_by_clip.get(c.id),
                "platforms": [],
                "content": None,
                "created_at": c.created_at,
                "status": "published",
            }
        )
    history = _order_publish_history(history)

    # Publishable clips for both tabs — APPROVED only: a clip flips to
    # 'published' on its first successful publish and leaves this grid
    # (it stays visible on the Published tab). Decorate each with
    # display_title + thumbnail URL so the template stays dumb.
    raw_clips: list[Any] = []
    with contextlib.suppress(Exception):  # display is best-effort
        raw_clips = await ClipsRepo(db).list_for_tenant_with_status(
            ["approved"], limit=60,
        )
    publishable = [
        {
            "id": c.id,
            "stream_id": c.stream_id,
            "duration_s": c.duration_s,
            "start_s": c.start_s,
            "end_s": c.end_s,
            "title": _clip_display_title(c),
            "thumbnail_url": f"/dashboard/clips/{c.id}/thumbnail",
            "status": c.status,
        }
        for c in raw_clips
    ]

    # Stats strip — single pass over the tenant's (capped) history.
    def _s(h: dict[str, Any]) -> str:
        return str(h.get("status", "")).lower()

    stats = {
        "published": sum(
            1 for h in history if _s(h) in {"published", "finished", "success"}
        ),
        "failed": sum(1 for h in history if _s(h) in {"failed", "error"}),
        "scheduled": sum(
            1 for h in history
            if _s(h) in {"scheduled", "pending", "queued", "publishing", "processing"}
        ),
        "total": len(history),
    }

    return templates.TemplateResponse(
        request,
        "publish/zernio_dashboard.html",
        {
            "configured": configured,
            "profile_id": profile_id,
            "profile_name": profile_name,
            "accounts": accounts,
            "connected_set": connected_set,
            "account_by_platform": _account_map(accounts),
            "account_limit": _account_limit(request),
            "account_count": len(accounts),
            "connect_fetch_failed": connect_fetch_failed,
            "history": history,
            # Clips still in `published` status — drives the "Republicar"
            # action on the Published tab (reopen -> approved -> re-publish).
            "published_clip_ids": sorted(c.id for c in published_clips),
            "publishable_clips": publishable,
            "supported_platforms": _SUPPORTED_PLATFORMS,
            "stats": stats,
            # Borradores — local rows seeded status='draft' at save time
            # (publishes already tenant-filtered; limit=100 above).
            "drafts": [p for p in publishes if p.status == "draft"],
            # Phase 10 growth layer is Pro-gated — the tab shows the
            # panels for paid tiers, an upsell card otherwise.
            "is_pro": _is_paid_tier(request),
            # Phase 11 — Discord/Telegram connect chips (community
            # channels, not clip targets).
            "community_platforms": _COMMUNITY_PLATFORMS,
            "community_connected": sorted(
                connected_set & _COMMUNITY_PLATFORM_IDS
            ),
            # Phase 12 — feature flags (default OFF) gate the ads +
            # whatsapp UI; the controls don't render until turned on.
            "feature_ads": get_settings().feature_ads,
            "feature_whatsapp": get_settings().feature_whatsapp,
        },
    )


def _is_paid_tier(request: Request) -> bool:
    """True when the requesting tenant is on a paid tier (Pro / All-
    Access). Gates the growth tab's content (upsell otherwise)."""
    from nexoclip.tiers import PAID_TIERS

    tier = getattr(request.state, "tenant_tier", None) or "free"
    return tier in PAID_TIERS


# ---------- Profile ----------


@router.post("/profile")
async def zernio_create_profile(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Create the tenant's Zernio profile (inline, AJAX from the
    dashboard). JSON body: {"name": "...", "description": "..."?}.

    Calls Zernio's POST /profiles, persists the returned profile id +
    name on the tenant row, and returns JSON so the dashboard can update
    in place without a navigation. Connecting social accounts to the new
    profile is the next step (the per-platform Connect buttons).
    """
    body = await _read_json(request)
    if not isinstance(body, dict):
        return JSONResponse(
            {"ok": False, "error": "Body must be a JSON object."}, status_code=400,
        )
    name = (body.get("name") or "").strip()
    description = (body.get("description") or "").strip() or None
    if not name:
        return JSONResponse(
            {"ok": False, "error": "Profile name is required."}, status_code=400,
        )

    client = _build_client()
    try:
        profile = await create_profile_for_tenant(
            db=db,
            tenant_id=tenant_id,
            client=client,
            name=name,
            description=description,
        )
    except ZernioError as e:
        _log.warning(
            "zernio.profile.create_failed tenant=%s err=%s status=%s body=%s",
            tenant_id, e, e.status_code, e.body,
        )
        return JSONResponse(
            {"ok": False, "error": f"Zernio profile create failed: {e}"},
            status_code=502,
        )
    _log.info(
        "zernio.profile.created tenant=%s profile_id=%s",
        tenant_id, profile.profile_id,
    )
    return JSONResponse(
        {"ok": True, "profile_id": profile.profile_id, "name": profile.name},
    )


# ---------- Connect ----------


@router.post("/connect")
async def zernio_connect(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Mint a hosted-OAuth authUrl for ONE platform and return it as
    JSON so the dashboard opens it in a NEW TAB.

    Why a new tab instead of a 303: Zernio's connect flow has no
    redirect-back parameter — after OAuth it lands the browser on
    Zernio's own dashboard. Navigating our whole tab away would strand
    the operator on Zernio. Opening Zernio in a separate tab keeps the
    NexoClip tab alive; when the operator returns to it, the dashboard
    refreshes and shows the newly connected account.

    JSON body: {"platform": "tiktok"}. Requires the tenant's Zernio
    profile to already exist (created via POST /profile); accounts
    attach to that profile.
    """
    data = await _read_json(request)
    if isinstance(data, dict):
        platform = str(data.get("platform") or "")
    else:
        # Tolerate a form-encoded POST from a cached/old dashboard page.
        form = await request.form()
        platform = str(form.get("platform") or "")
    platform = platform.strip().lower()
    if platform not in _CONNECTABLE_PLATFORM_IDS:
        return JSONResponse(
            {"ok": False, "error": f"Unsupported platform: {platform!r}"},
            status_code=400,
        )

    profile_id = await _require_profile(db, tenant_id)
    client = _build_client()

    # Per-tier connected-account cap: pro = 1, all_access = unlimited.
    # Counted against Zernio (the source of truth for connections), and
    # checked BEFORE minting the OAuth URL so the operator gets a clear
    # paywall message instead of a dead OAuth round-trip.
    limit = _account_limit(request)
    if limit is not None:
        try:
            current = await client.list_accounts(profile_id=profile_id)
        except ZernioError as e:
            _log.warning(
                "zernio.connect.limit_check_failed tenant=%s err=%s",
                tenant_id, e,
            )
            return JSONResponse(
                {"ok": False, "error": f"Couldn't verify connected accounts: {e}"},
                status_code=502,
            )
        if len(current) >= limit:
            return JSONResponse(
                {
                    "ok": False,
                    "reason": "account_limit",
                    "error": _account_limit_message(limit),
                },
                status_code=402,
            )

    # After the OAuth callback, Zernio redirects the popup HERE instead
    # of its own dashboard — our /connected page closes the popup and
    # notifies the main tab. The operator never sees Zernio's UI. The
    # platform rides on the query string so /connected can tell the
    # opener WHICH chip just connected.
    redirect_back = (
        f"{_public_base_url(request)}/dashboard/publish/zernio/connected"
        f"?platform={platform}"
    )
    # Facebook needs a post-OAuth page selection. headless=true makes
    # Zernio's redirect carry the selection state (tempToken & co)
    # instead of showing Zernio's own picker, and /connected renders
    # OUR picker — the operator never leaves NexoClip's UI.
    headless = platform in _HEADLESS_SELECTION_PLATFORMS
    try:
        link = await client.connect_url(
            platform,
            profile_id=profile_id,
            redirect_url=redirect_back,
            headless=headless,
        )
    except ZernioError as e:
        _log.warning(
            "zernio.connect.failed tenant=%s platform=%s err=%s status=%s body=%s",
            tenant_id, platform, e, e.status_code, e.body,
        )
        if e.status_code == 402:
            msg = (
                "Zernio plan limit reached — your Zernio plan's connected-account "
                "limit is full (the free plan allows 2). Disconnect an account "
                "below, or upgrade your Zernio plan, then try again."
            )
        else:
            msg = f"Zernio connect failed: {e}"
        return JSONResponse(
            {"ok": False, "error": msg, "status": e.status_code},
            status_code=502,
        )

    _log.info(
        "zernio.connect.authurl_minted tenant=%s platform=%s profile_id=%s",
        tenant_id, platform, profile_id,
    )
    return JSONResponse({"ok": True, "auth_url": link.auth_url})


_CONNECTED_CLOSE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Cuenta conectada</title></head>
<body style="font-family:system-ui,sans-serif;padding:24px;color:#333">
<p>Cuenta conectada ✓ — volviendo a NexoClip…</p>
<script>
  var znPlatform = new URLSearchParams(window.location.search).get("platform")
    || new URLSearchParams(window.location.search).get("connected") || "";
  if (window.opener && !window.opener.closed) {
    try {
      window.opener.postMessage(
        { type: "zernio:connected", platform: znPlatform },
        window.location.origin
      );
    } catch (e) { /* ignore */ }
    window.close();
    // If the browser refused to close us, give the operator a way home.
    setTimeout(function () {
      window.location.href = "/dashboard/publish/zernio";
    }, 1500);
  } else {
    window.location.href = "/dashboard/publish/zernio";
  }
</script>
</body></html>"""


# Facebook page picker, rendered inside the connect popup on the
# headless redirect. NO server-side interpolation of query params —
# the JS reads location.search itself, so tempToken/userProfile never
# pass through templating (no escaping pitfalls, nothing logged).
_FB_SELECT_PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Elige tu página de Facebook</title>
<style>
  body { font-family: system-ui, sans-serif; padding: 24px; color: #e8e8e8;
         background: #101014; max-width: 460px; margin: 0 auto; }
  h1 { font-size: 18px; }
  .pg { display: flex; align-items: center; gap: 10px; padding: 10px 12px;
        border: 1px solid #333; border-radius: 8px; margin: 8px 0;
        cursor: pointer; }
  .pg:hover { border-color: #c5f82a; }
  .pg input { accent-color: #c5f82a; }
  .pg small { color: #9a9aa2; }
  button { margin-top: 14px; padding: 10px 18px; border-radius: 8px;
           border: none; background: #c5f82a; color: #101014;
           font-weight: 600; cursor: pointer; width: 100%; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  .err { color: #ff2d95; margin-top: 10px; }
</style></head>
<body>
<h1>Elige la página de Facebook a conectar</h1>
<p style="color:#9a9aa2">Tu cuenta administra varias páginas; los clips se publicarán en la que elijas.</p>
<form id="fb-form"><div id="fb-pages">Cargando páginas…</div>
<button type="submit" id="fb-submit" disabled>Conectar página</button>
<div class="err" id="fb-err"></div></form>
<script>
(function () {
  var qs = new URLSearchParams(window.location.search);
  var tempToken = qs.get("tempToken") || "";
  var userProfileRaw = qs.get("userProfile") || "";
  var listEl = document.getElementById("fb-pages");
  var errEl = document.getElementById("fb-err");
  var submitBtn = document.getElementById("fb-submit");
  function fail(msg) { errEl.textContent = msg; }
  fetch("/dashboard/publish/zernio/fb-pages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ temp_token: tempToken }),
  })
    .then(function (r) { return r.json(); })
    .then(function (body) {
      if (!body.ok) { throw new Error(body.error || "No se pudieron listar las páginas"); }
      if (!body.pages.length) {
        listEl.textContent = "Tu cuenta de Facebook no administra ninguna página.";
        return;
      }
      listEl.innerHTML = "";
      body.pages.forEach(function (p, i) {
        var label = document.createElement("label");
        label.className = "pg";
        var input = document.createElement("input");
        input.type = "radio"; input.name = "page"; input.value = p.page_id;
        if (i === 0) { input.checked = true; }
        var span = document.createElement("span");
        span.textContent = p.name + (p.category ? " " : "");
        var small = document.createElement("small");
        small.textContent = p.category || "";
        label.appendChild(input); label.appendChild(span); label.appendChild(small);
        listEl.appendChild(label);
      });
      submitBtn.disabled = false;
    })
    .catch(function (e) { listEl.textContent = ""; fail(String(e.message || e)); });
  document.getElementById("fb-form").addEventListener("submit", function (ev) {
    ev.preventDefault();
    var picked = document.querySelector('input[name="page"]:checked');
    if (!picked) { return; }
    submitBtn.disabled = true;
    fetch("/dashboard/publish/zernio/fb-page/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        page_id: picked.value,
        temp_token: tempToken,
        user_profile_raw: userProfileRaw,
      }),
    })
      .then(function (r) { return r.json(); })
      .then(function (body) {
        if (!body.ok) { throw new Error(body.error || "No se pudo conectar la página"); }
        if (window.opener && !window.opener.closed) {
          try {
            window.opener.postMessage(
              { type: "zernio:connected", platform: "facebook" },
              window.location.origin
            );
          } catch (e) { /* ignore */ }
          window.close();
          setTimeout(function () {
            window.location.href = "/dashboard/publish/zernio";
          }, 1500);
        } else {
          window.location.href = "/dashboard/publish/zernio";
        }
      })
      .catch(function (e) { submitBtn.disabled = false; fail(String(e.message || e)); });
  });
})();
</script>
</body></html>"""


@router.get("/connected", response_class=HTMLResponse)
async def zernio_connected_landing(request: Request) -> Response:
    """Post-OAuth landing page — where Zernio's `redirect_url` sends the
    popup after the account connects.

    Same-origin, so `window.close()` works on the script-opened popup:
    notify the opener (main NexoClip tab) via postMessage (with the
    platform, read client-side from the query string), then close. If
    the page was somehow opened as a full navigation (no opener), fall
    back to the dashboard. The operator never sees Zernio's UI.

    Headless branch: when the redirect carries selection state
    (`tempToken` without an `accountId`) for a platform we connect in
    headless mode, the account is NOT created yet — render our own
    picker (Facebook page selection) instead of closing.
    """
    qp = request.query_params
    platform = (qp.get("platform") or qp.get("connected") or "").strip().lower()
    needs_selection = bool(qp.get("tempToken")) and not qp.get("accountId")
    if needs_selection and platform == "facebook":
        return HTMLResponse(_FB_SELECT_PAGE_HTML)
    return HTMLResponse(_CONNECTED_CLOSE_HTML)


@router.post("/fb-pages")
async def zernio_facebook_pages(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """List the Facebook Pages the just-OAuth'd user can manage.

    POST (not GET) so the short-lived `temp_token` travels in the body,
    never in a URL that access logs would capture. The popup shares the
    dashboard's session cookie, so the normal tenant gates apply and
    the profileId is resolved server-side — the client never picks it.
    """
    data = await _read_json(request)
    temp_token = str(data.get("temp_token") or "") if isinstance(data, dict) else ""
    if not temp_token:
        return JSONResponse(
            {"ok": False, "error": "Missing temp_token"}, status_code=400,
        )
    profile_id = await _require_profile(db, tenant_id)
    client = _build_client()
    try:
        pages = await client.list_facebook_pages(
            profile_id=profile_id, temp_token=temp_token,
        )
    except ZernioError as e:
        _log.warning(
            "zernio.fb_pages.failed tenant=%s status=%s", tenant_id, e.status_code,
        )
        return JSONResponse(
            {"ok": False, "error": f"Couldn't list Facebook pages: {e}"},
            status_code=502,
        )
    return JSONResponse(
        {
            "ok": True,
            "pages": [
                {
                    "page_id": p.page_id,
                    "name": p.name,
                    "username": p.username,
                    "category": p.category,
                }
                for p in pages
            ],
        }
    )


def _decode_user_profile(raw: str) -> dict[str, Any]:
    """Decode the `userProfile` query param Zernio appends on the
    headless redirect. Documented as the 'decoded user profile object
    from the OAuth callback'; observed encodings are JSON and
    base64(JSON), so try both and fall back to {} (Zernio then rejects
    the selection with a clear 400 rather than us 500ing)."""
    import base64
    import binascii
    import json as _json

    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        decoded = _json.loads(raw)
        if isinstance(decoded, dict):
            return decoded
    except ValueError:
        pass
    try:
        decoded = _json.loads(base64.b64decode(raw, validate=True))
        if isinstance(decoded, dict):
            return decoded
    except (ValueError, binascii.Error):
        pass
    return {}


@router.post("/fb-page/select")
async def zernio_facebook_select_page(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Finish the headless Facebook connect: save the picked page.

    Body: {page_id, temp_token, user_profile_raw}. Same body-not-URL
    rule as /fb-pages — the temp token is an OAuth credential.
    """
    data = await _read_json(request)
    if not isinstance(data, dict):
        return JSONResponse({"ok": False, "error": "Missing body"}, status_code=400)
    page_id = str(data.get("page_id") or "")
    temp_token = str(data.get("temp_token") or "")
    if not page_id or not temp_token:
        return JSONResponse(
            {"ok": False, "error": "Missing page_id or temp_token"},
            status_code=400,
        )
    user_profile = _decode_user_profile(str(data.get("user_profile_raw") or ""))
    profile_id = await _require_profile(db, tenant_id)
    client = _build_client()
    try:
        account = await client.select_facebook_page(
            profile_id=profile_id,
            page_id=page_id,
            temp_token=temp_token,
            user_profile=user_profile,
        )
    except ZernioError as e:
        _log.warning(
            "zernio.fb_select.failed tenant=%s page_id=%s status=%s",
            tenant_id, page_id, e.status_code,
        )
        if e.status_code == 402:
            msg = (
                "Zernio plan limit reached — your Zernio plan's connected-account "
                "limit is full (the free plan allows 2). Disconnect an account, "
                "or upgrade your Zernio plan, then try again."
            )
        else:
            msg = f"Couldn't connect the Facebook page: {e}"
        return JSONResponse(
            {"ok": False, "error": msg, "status": e.status_code}, status_code=502,
        )
    _log.info(
        "zernio.fb_select.connected tenant=%s account_id=%s",
        tenant_id, account.account_id,
    )
    return JSONResponse({"ok": True, "account_id": account.account_id})


@router.get("/accounts-panel", response_class=HTMLResponse)
async def zernio_accounts_panel(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Server-rendered connected-accounts row (the same partial the
    dashboard includes). The connect popup's postMessage triggers a
    fetch of this and swaps it in — chips update without a reload,
    keeping the limit/disconnect logic in ONE Jinja template."""
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    settings = get_settings()
    if not profile_id or not settings.zernio_api_key:
        return HTMLResponse("")
    client = _build_client()
    try:
        accounts = await client.list_accounts(profile_id=profile_id)
    except ZernioError as e:
        _log.warning("zernio.accounts_panel.failed tenant=%s err=%s", tenant_id, e)
        # 502 → the popup JS falls back to a full reload.
        return HTMLResponse("", status_code=502)
    return templates.TemplateResponse(
        request,
        "publish/_zernio_accounts.html",
        {
            "profile_id": profile_id,
            "connected_set": _connected_platforms(accounts),
            "account_by_platform": _account_map(accounts),
            "account_limit": _account_limit(request),
            "account_count": len(accounts),
            "supported_platforms": _SUPPORTED_PLATFORMS,
        },
    )


# ---------- Programación (recurring queue slots + upcoming + best time) ----------
# Queue SLOT dayOfWeek is 0=Sunday..6=Saturday (Zernio's convention,
# different from best-time's 0=Monday). The UI sends weekday+time pairs
# straight through; we never translate.


@router.get("/schedule.json")
async def zernio_schedule_json(
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Programación payload: recurring queue slots + the upcoming
    (scheduled/queued) posts list. Read-only; tenant-scoped via the
    profile. Empty + ok when no profile/key so the panel renders a
    clean empty state instead of erroring."""
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    settings = get_settings()
    if not profile_id or not settings.zernio_api_key:
        return JSONResponse({"ok": True, "queues": [], "upcoming": []})
    client = _build_client()
    queues: list[dict[str, Any]] = []
    upcoming: list[dict[str, Any]] = []
    try:
        queues = await client.list_queues(profile_id=profile_id)
    except ZernioError as e:
        _log.warning("zernio.schedule.queues_failed tenant=%s err=%s", tenant_id, e)
    try:
        feed = await client.list_posts(
            status="scheduled", profile_id=profile_id,
            sort_by="scheduled-asc", limit=50,
        )
        raw = feed.get("posts")
        if isinstance(raw, list):
            for p in raw:
                if not isinstance(p, dict):
                    continue
                pid = p.get("_id") or p.get("id")
                upcoming.append(
                    {
                        "post_id": pid,
                        "content": (p.get("content") or "")[:80],
                        "scheduled_for": p.get("scheduledFor"),
                        "status": p.get("status"),
                        "platforms": [
                            pl.get("platform")
                            for pl in (p.get("platforms") or [])
                            if isinstance(pl, dict)
                        ],
                    }
                )
    except ZernioError as e:
        _log.warning("zernio.schedule.upcoming_failed tenant=%s err=%s", tenant_id, e)
    return JSONResponse({"ok": True, "queues": queues, "upcoming": upcoming})


@router.post("/schedule/slots")
async def zernio_save_slots(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Replace the default queue's recurring slots.

    JSON body: {"timezone": "America/Mexico_City", "slots": [
      {"dayOfWeek": 1, "time": "09:00"}, ...]}. dayOfWeek is
    0=Sunday..6=Saturday, time "HH:mm". Validated before the Zernio
    call so a bad weekday/time is a clean 400, not a Zernio 400."""
    data = await _read_json(request)
    if not isinstance(data, dict):
        return JSONResponse({"ok": False, "error": "Body must be JSON"}, status_code=400)
    raw_slots = data.get("slots")
    if not isinstance(raw_slots, list) or not raw_slots:
        return JSONResponse(
            {"ok": False, "error": "slots must be a non-empty list"},
            status_code=400,
        )
    import re as _re

    slots: list[dict[str, Any]] = []
    for s in raw_slots:
        if not isinstance(s, dict):
            return JSONResponse(
                {"ok": False, "error": "each slot must be an object"},
                status_code=400,
            )
        dow = s.get("dayOfWeek")
        tm = s.get("time")
        if not isinstance(dow, int) or not (0 <= dow <= 6):
            return JSONResponse(
                {"ok": False, "error": f"dayOfWeek must be 0-6, got {dow!r}"},
                status_code=400,
            )
        if not isinstance(tm, str) or not _re.fullmatch(
            r"([01][0-9]|2[0-3]):[0-5][0-9]", tm
        ):
            return JSONResponse(
                {"ok": False, "error": f"time must be HH:mm, got {tm!r}"},
                status_code=400,
            )
        slots.append({"dayOfWeek": dow, "time": tm})
    timezone = str(data.get("timezone") or "UTC")

    profile_id = await _require_profile(db, tenant_id)
    client = _build_client()
    try:
        sched = await client.upsert_default_queue(
            profile_id=profile_id, slots=slots, timezone=timezone,
            reshuffle_existing=bool(data.get("reshuffle_existing")),
        )
    except ZernioError as e:
        _log.warning("zernio.schedule.save_failed tenant=%s err=%s", tenant_id, e)
        return JSONResponse(
            {"ok": False, "error": f"Couldn't save the schedule: {e}"},
            status_code=502,
        )
    _log.info(
        "zernio.schedule.saved tenant=%s slots=%d tz=%s",
        tenant_id, len(slots), timezone,
    )
    return JSONResponse({"ok": True, "queue_id": sched.get("_id")})


@router.post("/schedule/queue/{queue_id}/delete")
async def zernio_delete_queue(
    queue_id: str,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Delete one recurring queue."""
    profile_id = await _require_profile(db, tenant_id)
    client = _build_client()
    try:
        await client.delete_queue(profile_id=profile_id, queue_id=queue_id)
    except ZernioError as e:
        return JSONResponse(
            {"ok": False, "error": f"Couldn't delete the queue: {e}"},
            status_code=502,
        )
    return JSONResponse({"ok": True})


async def _tenant_account_ids(db: Database, tenant_id: str) -> list[str]:
    """The Zernio social-account ids the tenant owns — the isolation
    boundary for the tenant-free inbox/calendar stores. Empty on no
    profile / no key / Zernio error (caller renders an empty state)."""
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    if not profile_id or not get_settings().zernio_api_key:
        return []
    try:
        accounts = await _build_client().list_accounts(profile_id=profile_id)
    except ZernioError:
        return []
    return [a.account_id for a in accounts]


# ---------- Feature-flagged extras: Ads + WhatsApp (phase 12) ----------
# Both default OFF (extra cost/complexity). Routes 404 when the flag is
# off so the surface is invisible; the UI hides the controls too.


def _require_feature_ads() -> None:
    if not get_settings().feature_ads:
        raise HTTPException(status_code=404, detail="ads feature is disabled")


def _require_feature_whatsapp() -> None:
    if not get_settings().feature_whatsapp:
        raise HTTPException(status_code=404, detail="whatsapp feature is disabled")


@router.get("/ads/campaigns.json")
async def zernio_ads_campaigns_json(
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Read-only ad campaigns (404 when FEATURE_ADS is off). Shown next
    to organic metrics in Rendimiento."""
    _require_feature_ads()
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    if not profile_id or not get_settings().zernio_api_key:
        return JSONResponse({"ok": True, "campaigns": []})
    try:
        campaigns = await _build_client().list_ad_campaigns(profile_id=profile_id)
    except ZernioError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)
    return JSONResponse({"ok": True, "campaigns": campaigns})


@router.post("/ads/boost")
async def zernio_ads_boost(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Boost a published clip into a paid ad (404 when FEATURE_ADS off).
    Behind a confirmation. Body: {account_id, ad_account_id, name, goal,
    budget_amount, budget_type, post_id?/platform_post_id?, confirm}."""
    _require_feature_ads()
    data = await _read_json(request)
    if not isinstance(data, dict):
        return JSONResponse({"ok": False, "error": "Body must be JSON"}, status_code=400)
    if not data.get("confirm"):
        return JSONResponse(
            {"ok": False, "error": "Confirmación requerida (esto gasta dinero)."},
            status_code=400,
        )
    account_id = str(data.get("account_id") or "")
    ad_account_id = str(data.get("ad_account_id") or "")
    name = str(data.get("name") or "").strip()
    goal = str(data.get("goal") or "").strip()
    try:
        budget_amount = float(data.get("budget_amount") or 0)
    except (TypeError, ValueError):
        budget_amount = 0.0
    budget_type = str(data.get("budget_type") or "daily")
    if not (account_id and ad_account_id and name and goal and budget_amount > 0):
        return JSONResponse(
            {"ok": False, "error": "Faltan campos (cuenta, ad account, nombre, goal, presupuesto)."},
            status_code=400,
        )
    if account_id not in await _tenant_account_ids(db, tenant_id):
        raise HTTPException(status_code=403, detail="account not owned by tenant")
    try:
        result = await _build_client().boost_post(
            account_id=account_id, ad_account_id=ad_account_id, name=name,
            goal=goal, budget_amount=budget_amount, budget_type=budget_type,
            post_id=str(data.get("post_id") or "") or None,
            platform_post_id=str(data.get("platform_post_id") or "") or None,
        )
    except ZernioError as e:
        return JSONResponse(
            {"ok": False, "error": f"No se pudo crear el anuncio: {e}"},
            status_code=502,
        )
    _log.info("zernio.ads.boost tenant=%s account=%s", tenant_id, account_id)
    return JSONResponse({"ok": True, "result": result})


@router.get("/whatsapp/status.json")
async def zernio_whatsapp_status_json(
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """WhatsApp number provisioning status (404 when FEATURE_WHATSAPP
    off). Fed by whatsapp.number.* webhooks."""
    _require_feature_whatsapp()
    from nexoclip.db import ZernioWhatsappNumbersRepo

    account_ids = await _tenant_account_ids(db, tenant_id)
    rows = await ZernioWhatsappNumbersRepo(db).list_for_accounts(account_ids)
    return JSONResponse({"ok": True, "numbers": rows})


# ---------- Comunidad: Discord/Telegram notifications (phase 11) ----------


@router.get("/community/settings.json")
async def zernio_community_settings_json(
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Current community-notification settings + the connected
    Discord/Telegram accounts the operator can pick as the channel."""
    from nexoclip.db import ZernioCommunityRepo

    settings_row = await ZernioCommunityRepo(db).get_settings(tenant_id) or {
        "enabled": False, "discord_account_id": None, "telegram_account_id": None,
        "brand_name": None, "brand_avatar_url": None, "weekly_digest": False,
    }
    # Offer the tenant's connected discord/telegram accounts.
    channels: list[dict[str, str]] = []
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    if profile_id and get_settings().zernio_api_key:
        try:
            for a in await _build_client().list_accounts(profile_id=profile_id):
                if a.platform.lower() in ("discord", "telegram"):
                    channels.append(
                        {"platform": a.platform.lower(), "account_id": a.account_id}
                    )
        except ZernioError:
            pass
    return JSONResponse({"ok": True, "settings": settings_row, "channels": channels})


@router.post("/community/settings")
async def zernio_community_save_settings(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Save the "Avisar a mi comunidad" toggle + channel + brand
    identity + weekly-digest toggle. Body: {enabled, discord_account_id?,
    telegram_account_id?, brand_name?, brand_avatar_url?, weekly_digest?}."""
    from nexoclip.db import ZernioCommunityRepo

    data = await _read_json(request)
    if not isinstance(data, dict):
        return JSONResponse({"ok": False, "error": "Body must be JSON"}, status_code=400)
    # Validate the chosen channels are ones the tenant actually owns.
    owned = set(await _tenant_account_ids(db, tenant_id))
    discord_id = str(data.get("discord_account_id") or "") or None
    telegram_id = str(data.get("telegram_account_id") or "") or None
    for chosen in (discord_id, telegram_id):
        if chosen and chosen not in owned:
            raise HTTPException(status_code=403, detail="account not owned by tenant")
    await ZernioCommunityRepo(db).upsert_settings(
        tenant_id,
        enabled=bool(data.get("enabled")),
        discord_account_id=discord_id,
        telegram_account_id=telegram_id,
        brand_name=str(data.get("brand_name") or "") or None,
        brand_avatar_url=str(data.get("brand_avatar_url") or "") or None,
        weekly_digest=bool(data.get("weekly_digest")),
    )
    return JSONResponse({"ok": True})


# ---------- Auto-publish: "Piloto automático" (Publish Center tab) ----------
# Per-tenant. When enabled, approved clips auto-enqueue to Zernio with their
# burned-in render (hooks + captions already composited) — no manual platform
# picking. mode=on_approve fires from the editor's Ship/approve; hands_free
# (every generated clip, no review) is phase 2. post_mode queue|now is the
# Zernio publish mode (queue → publish_now=False → the profile's recurring
# slots). daily_cap is an anti-spam ceiling per UTC day. Routed through
# `_publish_clip` (renders first) — the legacy publish_jobs worker is gone.

_AUTOPUBLISH_MODES = ("on_approve", "hands_free")
_AUTOPUBLISH_POST_MODES = ("queue", "now")
_AUTOPUBLISH_CONTENT_STRATEGIES = ("unique", "variations")


@router.get("/autopublish.json")
async def zernio_autopublish_json(
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Auto-publish settings + the connected platforms the operator can
    target + today's post count (for the daily-cap meter)."""
    from nexoclip.db import AutopublishSettingsRepo

    s = await AutopublishSettingsRepo(db).get(tenant_id) or {
        "enabled": False, "mode": "on_approve", "targets": None,
        "post_mode": "queue", "daily_cap": 10, "score_threshold": 0.6,
        "tag_suffix": "", "content_strategy": "unique",
    }
    platforms: list[str] = []
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    if profile_id and get_settings().zernio_api_key:
        with contextlib.suppress(ZernioError):
            platforms = sorted(
                _connected_platforms(
                    await _build_client().list_accounts(profile_id=profile_id)
                )
            )
    return JSONResponse({
        "ok": True,
        "settings": s,
        "platforms": platforms,
        "used_today": await _autopublish_count_today(db, tenant_id),
    })


@router.post("/autopublish/save")
async def zernio_autopublish_save(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    db: Database = Depends(get_db),
) -> Response:
    """Save auto-publish settings. Body: {enabled, mode, targets:[...],
    post_mode, daily_cap}. `hands_free` is accepted but does not fire yet
    (phase 2 — the pipeline hook isn't wired)."""
    from nexoclip.db import AutopublishSettingsRepo

    data = await _read_json(request)
    if not isinstance(data, dict):
        return JSONResponse({"ok": False, "error": "Body must be JSON"}, status_code=400)
    mode = str(data.get("mode") or "on_approve")
    if mode not in _AUTOPUBLISH_MODES:
        mode = "on_approve"
    post_mode = str(data.get("post_mode") or "queue")
    if post_mode not in _AUTOPUBLISH_POST_MODES:
        post_mode = "queue"
    content_strategy = str(data.get("content_strategy") or "unique")
    if content_strategy not in _AUTOPUBLISH_CONTENT_STRATEGIES:
        content_strategy = "unique"
    raw_targets = data.get("targets")
    targets = (
        ",".join(str(t).strip().lower() for t in raw_targets if str(t).strip())
        if isinstance(raw_targets, list)
        else None
    )
    try:
        daily_cap = max(0, int(data.get("daily_cap", 10)))
    except (TypeError, ValueError):
        daily_cap = 10
    try:
        score_threshold = min(1.0, max(0.0, float(data.get("score_threshold", 0.6))))
    except (TypeError, ValueError):
        score_threshold = 0.6
    # Fixed @handles + brand hashtags appended to every auto-published /
    # auto-programmed caption. Capped so a runaway paste can't bloat posts.
    tag_suffix = str(data.get("tag_suffix") or "").strip()[:500]
    await AutopublishSettingsRepo(db).upsert(
        tenant_id,
        enabled=bool(data.get("enabled")),
        mode=mode,
        targets=targets,
        post_mode=post_mode,
        daily_cap=daily_cap,
        score_threshold=score_threshold,
        tag_suffix=tag_suffix,
        content_strategy=content_strategy,
    )
    return JSONResponse({"ok": True})


async def _autopublish_count_today(db: Database, tenant_id: str) -> int:
    """Posts this tenant has published so far in the current UTC day — the
    anti-spam daily-cap counter. Reads the local `zernio_publishes` ledger."""
    import datetime as _dt

    day = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d")
    conn = await db.connect()
    cur = await conn.execute(
        "SELECT COUNT(*) FROM zernio_publishes "
        "WHERE tenant_id = ? AND substr(created_at, 1, 10) = ?",
        (tenant_id, day),
    )
    row = await cur.fetchone()
    return int(row[0]) if row else 0


async def maybe_autopublish_on_approve(
    *, request: Request, db: Database, tenant_id: str, clip_id: str
) -> str | None:
    """Best-effort auto-publish of a just-approved clip. Returns the Zernio
    post_id on success, or None when it didn't fire (disabled, wrong mode,
    daily cap hit, no connected targets, no profile). NEVER raises — it must
    not block the approve flow.

    Goes through `_publish_clip`, which renders the edited clip (hooks +
    captions burned in) before posting, so what auto-publishes is exactly
    what the operator would have downloaded."""
    import structlog

    log = structlog.get_logger("nexoclip.api.zernio")
    try:
        from nexoclip.db import AutopublishSettingsRepo
        from nexoclip.publish.compose import build_post

        s = await AutopublishSettingsRepo(db).get(tenant_id)
        if not s or not s["enabled"] or s["mode"] != "on_approve":
            return None
        cap = int(s.get("daily_cap") or 0)
        if cap > 0 and await _autopublish_count_today(db, tenant_id) >= cap:
            log.info("autopublish.cap_reached", tenant_id=tenant_id, cap=cap)
            return None
        if not get_settings().zernio_api_key:
            return None
        tenant = await TenantsRepo(db).get(tenant_id)
        profile_id = tenant.zernio_profile_id if tenant else None
        if not profile_id:
            return None
        client = _build_client()
        accounts = await client.list_accounts(profile_id=profile_id)
        connected = _connected_platforms(accounts)
        want = [t for t in (s["targets"] or "").split(",") if t.strip()]
        # Empty targets → all connected platforms (matches the manual
        # Auto-programar scheduler). Without this, a tenant who enabled
        # auto-publish but never picked target platforms gets a silent no-op.
        targets = [t for t in want if t in connected] or sorted(connected)
        if not targets:
            log.info(
                "autopublish.no_connected_targets",
                tenant_id=tenant_id, want=want, connected=sorted(connected),
            )
            return None
        # Enrich: viral hook (variant title card) + caption + AI hashtags +
        # the tenant's fixed @handles/hashtags suffix — so auto-publish ships
        # a complete, swipe-stopping post, not a bare caption.
        composed = await build_post(
            db, clip_id, handle_suffix=str(s.get("tag_suffix") or ""),
        )
        post_id = await _publish_clip(
            client=client,
            db=db,
            request=request,
            tenant_id=tenant_id,
            profile_id=profile_id,
            account_map=_account_map(accounts),
            clip_id=clip_id,
            platforms=targets,
            content=composed.caption,
            title=composed.title,
            mode=s["post_mode"],
        )
        log.info(
            "autopublish.posted",
            tenant_id=tenant_id, clip_id=clip_id, post_id=post_id,
            mode=s["post_mode"], targets=targets,
        )
        return post_id
    except Exception as e:  # auto-publish must never block the approve flow
        log.warning(
            "autopublish.failed", tenant_id=tenant_id, clip_id=clip_id, error=str(e)
        )
        return None


async def autopublish_hands_free_sweep(
    *,
    db: Database,
    tenant_id: str,
    base_url: str,
    clip_scores: list[tuple[str, float]],
) -> int:
    """Hands-free auto-publish: post every clip whose score >= the tenant's
    threshold, when auto-publish is enabled in `hands_free` mode. Renders each
    clip via a signed URL (no cookie — runs in the background pipeline) and
    posts through Zernio. Returns how many were published. NEVER raises — the
    pipeline must not fail because publishing did. Idempotent per clip (skips
    clips already in zernio_publishes). Queue mode drips one post every
    30/60 min (content strategy); the daily cap only bounds `now` mode."""
    import structlog

    log = structlog.get_logger("nexoclip.api.zernio")
    published = 0
    try:
        from nexoclip.api._clip_render import ensure_clip_rendered
        from nexoclip.api.routers.internal import (
            mint_signed_clip_url,
            sign_render_query,
            signed_clip_ttl_for_schedule,
        )
        from nexoclip.db import AutopublishSettingsRepo, ClipsRepo
        from nexoclip.publish.compose import build_post
        from nexoclip.tenancy import bound_tenant

        s = await AutopublishSettingsRepo(db).get(tenant_id)
        if not s or not s["enabled"] or s["mode"] != "hands_free":
            return 0
        settings = get_settings()
        if not settings.zernio_api_key or not base_url:
            return 0
        tenant = await TenantsRepo(db).get(tenant_id)
        profile_id = tenant.zernio_profile_id if tenant else None
        if not profile_id:
            return 0

        client = _build_client()
        accounts = await client.list_accounts(profile_id=profile_id)
        account_map = _account_map(accounts)
        connected = _connected_platforms(accounts)
        want = [t for t in (s["targets"] or "").split(",") if t.strip()]
        # Empty targets → all connected platforms (matches the manual
        # Auto-programar scheduler). Without this, hands_free with no target
        # platforms picked silently posts nothing.
        chosen = [p for p in want if p in connected] or sorted(connected)
        post_targets = [
            (p, account_map[p]) for p in chosen if p in account_map
        ]
        if not post_targets:
            log.info("autopublish.handsfree.no_targets", tenant_id=tenant_id, want=want)
            return 0

        threshold = float(s.get("score_threshold") or 0.6)
        cap = int(s.get("daily_cap") or 0)
        post_mode = s["post_mode"]
        used = await _autopublish_count_today(db, tenant_id)
        pubs = ZernioPublishesRepo(db)

        # Eligible clips: the PUBLISHABILITY verdict (slice I.3 — "is THIS
        # render safe to ship?") gates auto-publish, NOT the raw detector
        # score. A low-signal detection can still be fully publish-ready: a
        # YouTube VOD has no chat heat, so its candidate score is ~0.1, but
        # the rendered clip scores 55+/100. Gating on the detector score
        # silently dropped every such clip. `score_threshold` (0-1) is the
        # operator's floor on the publishability score; a `reject` verdict is
        # always excluded; a not-yet-scored clip is allowed through rather
        # than silently dropped. Not already posted; daily cap only bounds the
        # `now` path (the queue drip is interval-spaced, not capped).
        eligible: list[str] = []
        clips_by_id: dict[str, Any] = {}
        skipped_unpublishable = 0
        for clip_id, _score in clip_scores:
            if await pubs.exists_for_clip(tenant_id, clip_id):
                continue
            with bound_tenant(tenant_id):
                clip = await ClipsRepo(db).get(clip_id)
            if clip is None:
                continue
            pub = clip.publishability_score
            if clip.publishability_status == "reject" or (
                pub is not None and pub / 100.0 < threshold
            ):
                skipped_unpublishable += 1
                continue
            clips_by_id[clip_id] = clip
            eligible.append(clip_id)
        if cap and post_mode == "now":
            eligible = eligible[: max(0, cap - used)]
        if skipped_unpublishable:
            log.info(
                "autopublish.handsfree.skipped_unpublishable",
                tenant_id=tenant_id, threshold=threshold,
                skipped=skipped_unpublishable, eligible=len(eligible),
            )
        if not eligible:
            log.info(
                "autopublish.handsfree.nothing_eligible",
                tenant_id=tenant_id, considered=len(clip_scores),
                threshold=threshold,
            )
            return 0

        # Spacing rules:
        #   Queue mode → drip one post every 30 min (unique) / 60 min
        #     (same-video variations) from the tenant's content strategy. No
        #     best-time slots, no daily cap — the interval is the only spacing.
        #   Now mode   → publish the FIRST clip immediately, then stagger the
        #     rest at a rate-limit-safe gap. Firing the whole batch at once
        #     trips TikTok's "wait 10 minutes" limit (the burst failures in
        #     prod), so only clip 0 is truly "now"; the tail is spaced.
        schedule_times: list[Any] = []
        if post_mode != "now" or len(eligible) > 1:
            import datetime as _dt

            from nexoclip.publish.hub import drip_interval_minutes, plan_drip_times

            interval = drip_interval_minutes(s.get("content_strategy"))
            if post_mode == "now":
                interval = max(interval, _NOW_MODE_BURST_INTERVAL_MIN)
            schedule_times = plan_drip_times(
                len(eligible),
                now=_dt.datetime.now(_dt.UTC),
                interval_minutes=interval,
            )

        for i, clip_id in enumerate(eligible):
            try:
                clip = clips_by_id[clip_id]
                with bound_tenant(tenant_id):
                    composed = await build_post(
                        db, clip_id, handle_suffix=str(s.get("tag_suffix") or ""),
                    )
                content = composed.caption
                await ensure_clip_rendered(
                    db=db, clip=clip, tenant_id=tenant_id, base_url=base_url,
                    auth_cookie_value=None, db_path=resolve_db_target(settings),
                    auth_query=sign_render_query(clip_id=clip_id, tenant_id=tenant_id),
                )
                # Now mode: clip 0 fires immediately (when=None); the rest are
                # staggered (i > 0). Queue mode: every clip is dripped.
                first_now = post_mode == "now" and i == 0
                when = (
                    None
                    if first_now or i >= len(schedule_times)
                    else schedule_times[i].isoformat()
                )
                # Size the signed URL's TTL to the scheduled time — a
                # best-time slot can be days out, and the platform fetches
                # the media then, not now. A flat 1h TTL would expire first.
                media_url = mint_signed_clip_url(
                    clip_id=clip_id, tenant_id=tenant_id, base_url=base_url,
                    ttl_seconds=signed_clip_ttl_for_schedule(when),
                )
                result = await client.create_post(
                    profile_id=profile_id, content=content, media_url=media_url,
                    platforms=post_targets, publish_now=(when is None),
                    title=composed.title,
                    scheduled_for=when,
                    timezone="UTC" if when else None,
                )
                with bound_tenant(tenant_id):
                    await pubs.record(
                        post_id=result.post_id, tenant_id=tenant_id, clip_id=clip_id,
                        platforms=[p for p, _ in post_targets], content=content,
                        status="scheduled" if when else None,
                    )
                    await ClipsRepo(db).update_status(clip_id, status="published")
                published += 1
                log.info(
                    "autopublish.handsfree.posted",
                    tenant_id=tenant_id, clip_id=clip_id, post_id=result.post_id,
                    mode=post_mode, scheduled_for=when,
                )
            except Exception as e:  # one clip's failure must not stop the sweep
                log.warning(
                    "autopublish.handsfree.clip_failed",
                    tenant_id=tenant_id, clip_id=clip_id, error=str(e),
                )
                continue
    except Exception as e:  # the sweep must never break the pipeline
        log.warning("autopublish.handsfree.failed", tenant_id=tenant_id, error=str(e))
    return published


# ---------- Crecimiento: funnel machine (phase 10, Pro-gated) ----------
# Automations (IG/FB only), contacts, sequences, broadcasts. All
# mutating routes carry require_paid_tier (non-Pro → 402, the JS shows
# an upsell). Broadcasts carry a per-tenant daily cap (irreversible
# mass-DM guardrail).

# The "Bienvenida Nexo" sequence template (3 steps: welcome / tips /
# CTA). delayMinutes: immediate, +1 day, +3 days.
_BIENVENIDA_NEXO_STEPS: list[dict[str, Any]] = [
    {"order": 1, "delayMinutes": 0,
     "message": {"text": "¡Bienvenido! Gracias por escribir 🙌"}},
    {"order": 2, "delayMinutes": 1440,
     "message": {"text": "Tip: mira mis últimos clips para no perderte nada 🎬"}},
    {"order": 3, "delayMinutes": 4320,
     "message": {"text": "¿Te sumas a la comunidad? Aquí el link 👉"}},
]

# Comment-to-DM automations are Instagram/Facebook only (Zernio
# enforces it; we gate first for a clean error).
_AUTOMATION_PLATFORMS = frozenset({"instagram", "facebook"})


async def _account_platform(db: Database, tenant_id: str, account_id: str) -> str | None:
    """Resolve one of the tenant's connected accounts to its platform,
    or None if the account isn't theirs."""
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    if not profile_id or not get_settings().zernio_api_key:
        return None
    try:
        accounts = await _build_client().list_accounts(profile_id=profile_id)
    except ZernioError:
        return None
    for a in accounts:
        if a.account_id == account_id:
            return a.platform.lower()
    return None


@router.get("/growth/contacts.json")
async def zernio_growth_contacts_json(
    tag: str = "",
    q: str = "",
    tenant_id: str = Depends(tenant_binder),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Contacts seeded from comments/DMs (phase 9), with optional tag
    filter + free-text search over name/username."""
    account_ids = await _tenant_account_ids(db, tenant_id)
    rows = await ZernioInboxRepo(db).list_contacts(account_ids, tag=(tag or None))
    needle = (q or "").strip().lower()
    if needle:
        rows = [
            r for r in rows
            if needle in (r.get("name") or "").lower()
            or needle in (r.get("username") or "").lower()
        ]
    return JSONResponse({"ok": True, "contacts": rows})


@router.get("/growth/automations.json")
async def zernio_growth_automations_json(
    tenant_id: str = Depends(tenant_binder),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """List the tenant's comment-to-DM automations with stats."""
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    if not profile_id or not get_settings().zernio_api_key:
        return JSONResponse({"ok": True, "automations": []})
    try:
        autos = await _build_client().list_comment_automations(profile_id=profile_id)
    except ZernioError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)
    return JSONResponse({"ok": True, "automations": autos})


@router.post("/growth/automations")
async def zernio_growth_create_automation(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Create a comment-to-DM automation. IG/FB only (enforced). Body:
    {account_id, name, dm_message, keywords?, platform_post_id?,
    post_id?, comment_reply?}."""
    data = await _read_json(request)
    if not isinstance(data, dict):
        return JSONResponse({"ok": False, "error": "Body must be JSON"}, status_code=400)
    account_id = str(data.get("account_id") or "")
    name = str(data.get("name") or "").strip()
    dm_message = str(data.get("dm_message") or "").strip()
    if not (account_id and name and dm_message):
        return JSONResponse(
            {"ok": False, "error": "account_id, name y dm_message obligatorios"},
            status_code=400,
        )
    platform = await _account_platform(db, tenant_id, account_id)
    if platform is None:
        raise HTTPException(status_code=403, detail="account not owned by tenant")
    if platform not in _AUTOMATION_PLATFORMS:
        return JSONResponse(
            {
                "ok": False,
                "error": "Las automatizaciones comentario→DM solo funcionan en "
                         "Instagram y Facebook.",
            },
            status_code=409,
        )
    profile_id = await _require_profile(db, tenant_id)
    keywords = data.get("keywords")
    keywords_list = (
        [str(k).strip() for k in keywords if str(k).strip()]
        if isinstance(keywords, list) else None
    )
    try:
        result = await _build_client().create_comment_automation(
            profile_id=profile_id,
            account_id=account_id,
            name=name,
            dm_message=dm_message,
            keywords=keywords_list,
            platform_post_id=str(data.get("platform_post_id") or "") or None,
            post_id=str(data.get("post_id") or "") or None,
            comment_reply=str(data.get("comment_reply") or "") or None,
        )
    except ZernioError as e:
        return JSONResponse(
            {"ok": False, "error": f"No se pudo crear la automatización: {e}"},
            status_code=502,
        )
    return JSONResponse({"ok": True, "automation": result})


@router.post("/growth/automations/{automation_id}/toggle")
async def zernio_growth_toggle_automation(
    automation_id: str,
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Activate/pause an automation. Body: {active: bool}."""
    data = await _read_json(request)
    active = bool(data.get("active")) if isinstance(data, dict) else False
    try:
        await _build_client().set_comment_automation_active(
            automation_id, is_active=active,
        )
    except ZernioError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)
    return JSONResponse({"ok": True})


@router.get("/growth/sequences.json")
async def zernio_growth_sequences_json(
    tenant_id: str = Depends(tenant_binder),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """List the tenant's drip sequences."""
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    if not profile_id or not get_settings().zernio_api_key:
        return JSONResponse({"ok": True, "sequences": []})
    try:
        seqs = await _build_client().list_sequences(profile_id=profile_id)
    except ZernioError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)
    return JSONResponse({"ok": True, "sequences": seqs})


def _validate_sequence_steps(raw: Any) -> list[dict[str, Any]] | None:
    """Coerce + validate the steps editor payload. Returns None on a
    bad shape (caller → 400). Each step needs order (int), delayMinutes
    (int ≥ 0), and message text."""
    if not isinstance(raw, list) or not raw:
        return None
    steps: list[dict[str, Any]] = []
    for i, s in enumerate(raw):
        if not isinstance(s, dict):
            return None
        delay = s.get("delayMinutes", s.get("delay_minutes"))
        text = ""
        msg = s.get("message")
        if isinstance(msg, dict):
            text = str(msg.get("text") or "")
        elif isinstance(s.get("text"), str):
            text = s["text"]
        if not isinstance(delay, int) or delay < 0 or not text.strip():
            return None
        steps.append(
            {"order": i + 1, "delayMinutes": delay, "message": {"text": text.strip()}}
        )
    return steps


@router.post("/growth/sequences")
async def zernio_growth_create_sequence(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Create a drip sequence. Body: {account_id, name, steps:[...],
    description?, template?}. template='bienvenida' uses the built-in
    "Bienvenida Nexo" 3-step template."""
    data = await _read_json(request)
    if not isinstance(data, dict):
        return JSONResponse({"ok": False, "error": "Body must be JSON"}, status_code=400)
    account_id = str(data.get("account_id") or "")
    name = str(data.get("name") or "").strip()
    if not (account_id and name):
        return JSONResponse(
            {"ok": False, "error": "account_id y name obligatorios"}, status_code=400,
        )
    platform = await _account_platform(db, tenant_id, account_id)
    if platform is None:
        raise HTTPException(status_code=403, detail="account not owned by tenant")
    if data.get("template") == "bienvenida":
        steps: list[dict[str, Any]] | None = list(_BIENVENIDA_NEXO_STEPS)
    else:
        steps = _validate_sequence_steps(data.get("steps"))
    if steps is None:
        return JSONResponse(
            {"ok": False, "error": "Cada paso necesita delayMinutes (≥0) y texto."},
            status_code=400,
        )
    profile_id = await _require_profile(db, tenant_id)
    try:
        result = await _build_client().create_sequence(
            profile_id=profile_id, account_id=account_id, platform=platform,
            name=name, steps=steps,
            description=str(data.get("description") or "") or None,
        )
    except ZernioError as e:
        return JSONResponse(
            {"ok": False, "error": f"No se pudo crear la secuencia: {e}"},
            status_code=502,
        )
    return JSONResponse({"ok": True, "sequence": result})


@router.post("/growth/sequences/{sequence_id}/toggle")
async def zernio_growth_toggle_sequence(
    sequence_id: str,
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    data = await _read_json(request)
    active = bool(data.get("active")) if isinstance(data, dict) else False
    try:
        await _build_client().set_sequence_active(sequence_id, active=active)
    except ZernioError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)
    return JSONResponse({"ok": True})


@router.post("/growth/sequences/{sequence_id}/enroll")
async def zernio_growth_enroll_sequence(
    sequence_id: str,
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Enroll contacts. Body: {contact_ids:[...]} OR {tag: "..."} to
    enroll all contacts carrying that tag."""
    data = await _read_json(request)
    if not isinstance(data, dict):
        return JSONResponse({"ok": False, "error": "Body must be JSON"}, status_code=400)
    contact_ids = data.get("contact_ids")
    ids = (
        [str(c) for c in contact_ids if c]
        if isinstance(contact_ids, list) else []
    )
    tag = str(data.get("tag") or "").strip()
    if tag and not ids:
        # "enroll all with tag X" — resolve to the contacts' Zernio ids.
        account_ids = await _tenant_account_ids(db, tenant_id)
        rows = await ZernioInboxRepo(db).list_contacts(account_ids, tag=tag)
        ids = [
            str(r["zernio_contact_id"]) for r in rows if r.get("zernio_contact_id")
        ]
    if not ids:
        return JSONResponse(
            {"ok": False, "error": "Sin contactos para inscribir (faltan ids de Zernio)."},
            status_code=400,
        )
    try:
        result = await _build_client().enroll_in_sequence(
            sequence_id, contact_ids=ids,
        )
    except ZernioError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)
    return JSONResponse({"ok": True, "result": result})


@router.post("/growth/broadcasts/send")
async def zernio_growth_send_broadcast(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Create + populate + SEND a broadcast in one call (the UI's
    confirm modal gates the click). Body: {account_id, name, message,
    contact_ids?, tag?, confirm: true}.

    Guardrail: the per-tenant daily cap is checked BEFORE the send —
    a broadcast is an irreversible mass DM. `confirm` must be true."""
    import datetime as _dt

    data = await _read_json(request)
    if not isinstance(data, dict):
        return JSONResponse({"ok": False, "error": "Body must be JSON"}, status_code=400)
    if not data.get("confirm"):
        return JSONResponse(
            {"ok": False, "error": "Confirmación requerida."}, status_code=400,
        )
    account_id = str(data.get("account_id") or "")
    name = str(data.get("name") or "").strip()
    message = str(data.get("message") or "").strip()
    if not (account_id and name and message):
        return JSONResponse(
            {"ok": False, "error": "account_id, name y message obligatorios"},
            status_code=400,
        )
    platform = await _account_platform(db, tenant_id, account_id)
    if platform is None:
        raise HTTPException(status_code=403, detail="account not owned by tenant")

    # --- daily cap (irreversible mass-DM guardrail) ---
    day = _dt.datetime.now(_dt.UTC).date().isoformat()
    cap = get_settings().hub_max_broadcasts_per_day
    log = ZernioBroadcastLogRepo(db)
    if await log.count_for_day(tenant_id, day=day) >= cap:
        return JSONResponse(
            {
                "ok": False,
                "reason": "daily_cap",
                "error": f"Límite diario de broadcasts alcanzado ({cap}/día). "
                         "Inténtalo mañana.",
            },
            status_code=429,
        )

    profile_id = await _require_profile(db, tenant_id)
    client = _build_client()
    contact_ids = data.get("contact_ids")
    ids = [str(c) for c in contact_ids if c] if isinstance(contact_ids, list) else []
    tag = str(data.get("tag") or "").strip()
    if tag and not ids:
        account_ids = await _tenant_account_ids(db, tenant_id)
        rows = await ZernioInboxRepo(db).list_contacts(account_ids, tag=tag)
        ids = [str(r["zernio_contact_id"]) for r in rows if r.get("zernio_contact_id")]
    try:
        created = await client.create_broadcast(
            profile_id=profile_id, account_id=account_id, platform=platform,
            name=name, message_text=message,
        )
        broadcast_id = str(created.get("_id") or created.get("id") or "")
        if not broadcast_id:
            return JSONResponse(
                {"ok": False, "error": "Zernio no devolvió un id de broadcast."},
                status_code=502,
            )
        await client.add_broadcast_recipients(
            broadcast_id, contact_ids=ids or None, use_segment=not ids,
        )
        result = await client.send_broadcast(broadcast_id)
    except ZernioError as e:
        return JSONResponse(
            {"ok": False, "error": f"No se pudo enviar el broadcast: {e}"},
            status_code=502,
        )
    # Record the send AFTER it fires — only successful sends count
    # against the cap.
    await log.record(tenant_id, broadcast_id=broadcast_id, day=day)
    _log.info(
        "zernio.broadcast.sent tenant=%s broadcast=%s recipients=%s",
        tenant_id, broadcast_id, len(ids) if ids else "segment",
    )
    return JSONResponse({"ok": True, "broadcast_id": broadcast_id, "result": result})


# ---------- Inbox: comentarios + DMs (phase 9) ----------
# Webhook-first: the event processor writes the local store; these
# routes read it (REST is backfill). Stores are tenant-free (keyed by
# account_id), resolved to the tenant via _tenant_account_ids.


@router.get("/inbox/comments.json")
async def zernio_inbox_comments_json(
    platform_post_id: str = "",
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Comments on the tenant's clips — unified feed, or one post when
    `platform_post_id` is set. Reads the local webhook store."""
    from nexoclip.integrations.zernio.capabilities import can_hide_comment

    account_ids = await _tenant_account_ids(db, tenant_id)
    rows = await ZernioInboxRepo(db).list_comments(
        account_ids, platform_post_id=(platform_post_id or None),
    )
    for r in rows:
        r["can_hide"] = can_hide_comment(r.get("platform"))
    return JSONResponse({"ok": True, "comments": rows})


@router.post("/inbox/comments/reply")
async def zernio_inbox_comment_reply(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Reply to a comment. Body: {account_id, post_id, comment_id?,
    message}. account_id must be one the tenant owns (checked)."""
    data = await _read_json(request)
    if not isinstance(data, dict):
        return JSONResponse({"ok": False, "error": "Body must be JSON"}, status_code=400)
    account_id = str(data.get("account_id") or "")
    post_id = str(data.get("post_id") or "")
    message = str(data.get("message") or "").strip()
    if not (account_id and post_id and message):
        return JSONResponse(
            {"ok": False, "error": "account_id, post_id y message son obligatorios"},
            status_code=400,
        )
    if account_id not in await _tenant_account_ids(db, tenant_id):
        raise HTTPException(status_code=403, detail="account not owned by tenant")
    client = _build_client()
    try:
        result = await client.reply_to_comment(
            post_id, account_id=account_id, message=message,
            comment_id=str(data.get("comment_id") or "") or None,
        )
    except ZernioError as e:
        return JSONResponse(
            {"ok": False, "error": f"No se pudo responder: {e}"}, status_code=502,
        )
    return JSONResponse({"ok": True, "result": result})


@router.post("/inbox/comments/{comment_action}")
async def zernio_inbox_comment_action(
    comment_action: str,
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Like or hide a comment. comment_action ∈ {like, hide}. Body:
    {account_id, post_id, comment_id, platform?}. Hide is gated to the
    platforms that support it."""
    from nexoclip.integrations.zernio.capabilities import can_hide_comment

    if comment_action not in ("like", "hide"):
        raise HTTPException(status_code=404, detail="unknown comment action")
    data = await _read_json(request)
    if not isinstance(data, dict):
        return JSONResponse({"ok": False, "error": "Body must be JSON"}, status_code=400)
    account_id = str(data.get("account_id") or "")
    post_id = str(data.get("post_id") or "")
    comment_id = str(data.get("comment_id") or "")
    if not (account_id and post_id and comment_id):
        return JSONResponse(
            {"ok": False, "error": "account_id, post_id y comment_id obligatorios"},
            status_code=400,
        )
    if account_id not in await _tenant_account_ids(db, tenant_id):
        raise HTTPException(status_code=403, detail="account not owned by tenant")
    if comment_action == "hide" and not can_hide_comment(data.get("platform")):
        return JSONResponse(
            {
                "ok": False,
                "error": "Ocultar comentarios no está soportado en esta plataforma.",
            },
            status_code=409,
        )
    client = _build_client()
    try:
        if comment_action == "like":
            await client.like_comment(post_id, comment_id, account_id=account_id)
        else:
            await client.hide_comment(post_id, comment_id, account_id=account_id)
            await ZernioInboxRepo(db).set_comment_status(
                account_id=account_id, comment_id=comment_id, status="hidden",
            )
    except ZernioError as e:
        return JSONResponse(
            {"ok": False, "error": f"No se pudo: {e}"}, status_code=502,
        )
    return JSONResponse({"ok": True})


@router.get("/inbox/conversations.json")
async def zernio_inbox_conversations_json(
    status: str = "",
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """DM conversations across the tenant's accounts (local store)."""
    account_ids = await _tenant_account_ids(db, tenant_id)
    rows = await ZernioInboxRepo(db).list_conversations(
        account_ids, status=(status or None),
    )
    return JSONResponse({"ok": True, "conversations": rows})


@router.get("/inbox/messages.json")
async def zernio_inbox_messages_json(
    conversation_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Messages in one conversation (local store)."""
    account_ids = await _tenant_account_ids(db, tenant_id)
    rows = await ZernioInboxRepo(db).list_messages(
        account_ids, conversation_id=conversation_id,
    )
    return JSONResponse({"ok": True, "messages": rows})


@router.post("/inbox/conversations/reply")
async def zernio_inbox_send_message(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Send a DM reply. Body: {account_id, conversation_id, message,
    attachment_url?}. Attachments gated per platform capability."""
    from nexoclip.integrations.zernio.capabilities import can_send_attachment

    data = await _read_json(request)
    if not isinstance(data, dict):
        return JSONResponse({"ok": False, "error": "Body must be JSON"}, status_code=400)
    account_id = str(data.get("account_id") or "")
    conversation_id = str(data.get("conversation_id") or "")
    message = str(data.get("message") or "").strip()
    if not (account_id and conversation_id and message):
        return JSONResponse(
            {"ok": False, "error": "account_id, conversation_id y message obligatorios"},
            status_code=400,
        )
    if account_id not in await _tenant_account_ids(db, tenant_id):
        raise HTTPException(status_code=403, detail="account not owned by tenant")
    attachment_url = str(data.get("attachment_url") or "") or None
    if attachment_url and not can_send_attachment(data.get("platform")):
        return JSONResponse(
            {
                "ok": False,
                "error": "Esta plataforma solo admite texto en los DMs.",
            },
            status_code=409,
        )
    client = _build_client()
    try:
        result = await client.send_message(
            conversation_id, account_id=account_id, message=message,
            attachment_url=attachment_url,
        )
    except ZernioError as e:
        return JSONResponse(
            {"ok": False, "error": f"No se pudo enviar: {e}"}, status_code=502,
        )
    return JSONResponse({"ok": True, "result": result})


@router.post("/inbox/conversations/archive")
async def zernio_inbox_archive(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Archive (or re-activate) a conversation. Body: {account_id,
    conversation_id, status?}. status defaults to 'archived'."""
    data = await _read_json(request)
    if not isinstance(data, dict):
        return JSONResponse({"ok": False, "error": "Body must be JSON"}, status_code=400)
    account_id = str(data.get("account_id") or "")
    conversation_id = str(data.get("conversation_id") or "")
    new_status = str(data.get("status") or "archived")
    if new_status not in ("archived", "active"):
        return JSONResponse({"ok": False, "error": "status inválido"}, status_code=400)
    if not (account_id and conversation_id):
        return JSONResponse(
            {"ok": False, "error": "account_id y conversation_id obligatorios"},
            status_code=400,
        )
    if account_id not in await _tenant_account_ids(db, tenant_id):
        raise HTTPException(status_code=403, detail="account not owned by tenant")
    client = _build_client()
    try:
        await client.set_conversation_status(
            conversation_id, account_id=account_id, status=new_status,
        )
    except ZernioError as e:
        return JSONResponse(
            {"ok": False, "error": f"No se pudo archivar: {e}"}, status_code=502,
        )
    await ZernioInboxRepo(db).set_conversation_status(
        account_id=account_id, conversation_id=conversation_id, status=new_status,
    )
    return JSONResponse({"ok": True})


@router.get("/calendar.json")
async def zernio_calendar_json(
    date_from: str = "",
    date_to: str = "",
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Unified content calendar: entries from THREE sources merged and
    labeled —
      - `hub`: clips published through NexoClip (zernio_publishes)
      - `scheduled`: hub posts with a future scheduled_for
      - `external`: posts the streamer authored natively, detected via
        post.external.* webhooks (zernio_calendar), resolved to this
        tenant by matching the social account id against their
        connected accounts

    Each entry: {date, source, platform(s), content, status, url,
    post_id, clip_id?}. Deleted external posts are included with
    status='deleted' so the UI can grey them out. Date bounds are
    ISO-8601 (YYYY-MM-DD); omitted = no bound."""
    df = (date_from or "").strip() or None
    dt = (date_to or "").strip() or None
    entries: list[dict[str, Any]] = []

    # Sources 1 + 2: local hub/scheduled publishes (already tenant-scoped).
    publishes = await ZernioPublishesRepo(db).list_for_tenant(limit=200)
    for p in publishes:
        if p.status in ("draft", "deleted"):
            continue
        # Bucket by the day the post actually goes out: a future-scheduled
        # post lands on its `scheduled_for` day, an immediate publish on its
        # `created_at` day. (Before, everything landed on the batch-creation
        # day, so a day's worth of scheduled clips piled onto the day they
        # were queued instead of the day they publish.)
        date = (
            p.scheduled_for
            if p.status == "scheduled" and p.scheduled_for
            else p.created_at
        )
        if df and date and date[:10] < df:
            continue
        if dt and date and date[:10] > dt:
            continue
        entries.append(
            {
                "date": date,
                "source": "scheduled" if p.status == "scheduled" else "hub",
                "platforms": [pl for pl in p.platforms.split(",") if pl],
                "content": (p.content or "")[:120],
                "status": p.status or "published",
                "post_id": p.post_id,
                "clip_id": p.clip_id,
                "url": None,
            }
        )

    # Source 3: external posts — resolve via the tenant's connected
    # account ids (the isolation boundary for the tenant-free store).
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    settings = get_settings()
    if profile_id and settings.zernio_api_key:
        try:
            accounts = await _build_client().list_accounts(profile_id=profile_id)
            account_ids = [a.account_id for a in accounts]
        except ZernioError:
            account_ids = []
        if account_ids:
            for e in await ZernioCalendarRepo(db).list_for_accounts(
                account_ids, date_from=df, date_to=dt,
            ):
                entries.append(
                    {
                        "date": e.get("published_at"),
                        "source": "external",
                        "platforms": [e["platform"]] if e.get("platform") else [],
                        "content": (e.get("content") or "")[:120],
                        "status": e.get("status") or "active",
                        "post_id": e.get("post_id"),
                        "clip_id": None,
                        "url": e.get("url"),
                        "thumbnail_url": e.get("thumbnail_url"),
                    }
                )

    entries.sort(key=lambda x: str(x.get("date") or ""), reverse=True)
    return JSONResponse({"ok": True, "entries": entries})


@router.get("/rendimiento.json")
async def zernio_performance_json(
    days: int = 30,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Rendimiento tab data: recent clips with per-platform metrics +
    a totals header, over the last `days` (7 or 30). Live from Zernio,
    normalized (no fake zeros — absent metrics render as "—")."""
    from nexoclip.publish.analytics_service import performance_for_tenant

    days = 7 if days == 7 else 30
    settings = get_settings()
    client = _build_client() if settings.zernio_api_key else None
    view = await performance_for_tenant(db, tenant_id, days=days, client=client)
    return JSONResponse(
        {"ok": True, "days": view.days, "totals": view.totals, "posts": view.rows}
    )


@router.get("/failed.json")
async def zernio_failed_json(
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Failed posts for this tenant, each with its per-platform error +
    Spanish hint. Powers the clickable FAILED counter list."""
    from nexoclip.integrations.zernio.errors import summarize_failed_platforms

    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    settings = get_settings()
    if not profile_id or not settings.zernio_api_key:
        return JSONResponse({"ok": True, "failed": []})
    client = _build_client()
    try:
        rows = await client.list_failed(profile_id=profile_id)
    except ZernioError as e:
        _log.warning("zernio.failed.list_failed tenant=%s err=%s", tenant_id, e)
        return JSONResponse({"ok": False, "failed": []}, status_code=502)
    failed = [
        {
            "post_id": p.get("_id") or p.get("id"),
            "content": (p.get("content") or "")[:80],
            "created_at": p.get("createdAt") or p.get("scheduledFor"),
            "platforms": summarize_failed_platforms(p),
        }
        for p in rows
    ]
    return JSONResponse({"ok": True, "failed": failed})


@router.post("/retry/{post_id}")
async def zernio_retry_one(
    post_id: str,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Reintentar one failed post (POST /posts/{id}/retry)."""
    client = _build_client()
    try:
        result = await client.retry_post(post_id)
    except ZernioError as e:
        if e.status_code == 429:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "Límite de frecuencia — espera unos minutos y reintenta.",
                },
                status_code=429,
            )
        _log.warning("zernio.retry.failed tenant=%s post=%s err=%s", tenant_id, post_id, e)
        return JSONResponse(
            {"ok": False, "error": f"No se pudo reintentar: {e}"}, status_code=502,
        )
    # Best-effort local status refresh (the post may be ours).
    row = await ZernioPublishesRepo(db).get_by_post_id(post_id)
    if row is not None and row.tenant_id == tenant_id:
        await ZernioPublishesRepo(db).set_status(post_id, status="publishing")
    return JSONResponse({"ok": True, "post_id": result.post_id})


@router.post("/retry-all")
async def zernio_retry_all(
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Reintentar todos los posts fallidos del tenant. Per-post failures
    don't abort the batch; each reports ok/error."""
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    if not profile_id:
        return JSONResponse({"ok": True, "results": []})
    client = _build_client()
    try:
        rows = await client.list_failed(profile_id=profile_id)
    except ZernioError as e:
        return JSONResponse(
            {"ok": False, "error": f"No se pudo listar: {e}"}, status_code=502,
        )
    results: list[dict[str, Any]] = []
    for p in rows:
        pid = p.get("_id") or p.get("id")
        if not isinstance(pid, str):
            continue
        try:
            await client.retry_post(pid)
            results.append({"post_id": pid, "ok": True})
            row = await ZernioPublishesRepo(db).get_by_post_id(pid)
            if row is not None and row.tenant_id == tenant_id:
                await ZernioPublishesRepo(db).set_status(pid, status="publishing")
        except ZernioError as e:
            results.append({"post_id": pid, "ok": False, "error": str(e)})
    _log.info(
        "zernio.retry_all tenant=%s tried=%d ok=%d",
        tenant_id, len(results), sum(1 for r in results if r["ok"]),
    )
    return JSONResponse(
        {
            "ok": True,
            "results": results,
            "retried": sum(1 for r in results if r["ok"]),
        }
    )


async def _reprocess_failed_rows(
    request: Request, db: Database, tenant_id: str, rows: list[Any],
) -> dict[str, Any]:
    """Re-render + re-schedule failed posts through the drip gate.

    Zernio's own retry (`/retry-all`) re-fires the SAME post against the
    SAME media URL — useless when the failure was an expired signed URL
    (the 403 "could not download" class). Reprocess instead rebuilds each
    post from the local clip via `_publish_clip`: fresh render, fresh
    signed URL (correct TTL), and a NEW slot from the interval drip. The
    old failed row is tombstoned and its Zernio post deleted (best-effort).
    Per-clip failures don't abort the batch."""
    import datetime as _dt

    from nexoclip.db import AutopublishSettingsRepo
    from nexoclip.publish.hub import drip_interval_minutes, plan_drip_times

    if not rows:
        return {"ok": True, "reprocessed": 0, "results": []}

    profile_id = await _require_profile(db, tenant_id)
    client = _build_client()
    try:
        account_map = _account_map(await client.list_accounts(profile_id=profile_id))
    except ZernioError as e:
        raise HTTPException(
            status_code=502, detail=f"Zernio setup failed: {e}",
        ) from e

    s = await AutopublishSettingsRepo(db).get(tenant_id) or {}
    times = plan_drip_times(
        len(rows),
        now=_dt.datetime.now(_dt.UTC),
        interval_minutes=drip_interval_minutes(s.get("content_strategy")),
    )

    pubs = ZernioPublishesRepo(db)
    results: list[dict[str, Any]] = []
    for row, when in zip(rows, times, strict=False):
        platforms = [p for p in (row.platforms or "").split(",") if p]
        if not row.clip_id or not platforms:
            results.append({
                "post_id": row.post_id, "ok": False,
                "error": "no es un clip de NexoClip reprocesable",
            })
            continue
        # Re-schedule ONLY to platforms still connected on Zernio. The
        # operator may have disconnected one (e.g. TikTok) since the
        # original post — re-targeting it would just fail again and drag
        # the whole clip down with it. Ship to whatever remains; skip the
        # clip only when nothing is left to publish to.
        connected = [p for p in platforms if account_map.get(p.lower())]
        dropped = [p for p in platforms if p not in connected]
        if not connected:
            results.append({
                "post_id": row.post_id, "ok": False,
                "error": (
                    "ninguna de sus plataformas sigue conectada ("
                    + ", ".join(platforms) + ") — reconéctalas y reintenta"
                ),
            })
            continue
        try:
            new_post_id = await _publish_clip(
                client=client, db=db, request=request, tenant_id=tenant_id,
                profile_id=profile_id, account_map=account_map,
                clip_id=row.clip_id, platforms=connected,
                content=row.content or "", mode="schedule",
                scheduled_for=when.isoformat(),
            )
        except HTTPException as e:
            results.append({"post_id": row.post_id, "ok": False, "error": str(e.detail)})
            continue
        except Exception as e:  # one clip's failure must not abort the batch
            results.append({"post_id": row.post_id, "ok": False, "error": str(e)})
            continue
        # New post is live — retire the old failed one (best-effort delete
        # on Zernio, always tombstone locally so it leaves the failed list).
        with contextlib.suppress(ZernioError):
            await client.delete_post(row.post_id)
        await pubs.set_status(row.post_id, status="deleted")
        results.append({
            "post_id": row.post_id, "ok": True,
            "new_post_id": new_post_id, "scheduled_for": when.isoformat(),
            "platforms": connected, "dropped": dropped,
        })
    ok = sum(1 for r in results if r["ok"])
    _log.info(
        "zernio.reprocess tenant=%s tried=%d ok=%d", tenant_id, len(results), ok,
    )
    return {"ok": True, "reprocessed": ok, "results": results}


@router.post("/reprocess-failed")
async def zernio_reprocess_failed(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Reprocesar todos los posts fallidos del tenant a través del drip.

    Re-renderiza cada clip, mintea una URL firmada nueva y lo re-agenda
    con el nuevo gate (uno cada 30/60 min según la estrategia). Esto SÍ
    arregla los fallos por URL expirada — a diferencia de "Reintentar
    todos", que re-dispara la misma URL."""
    rows = await ZernioPublishesRepo(db).list_for_tenant(limit=200, status="failed")
    # Oldest-first so the earliest failure gets the earliest drip slot.
    rows = list(reversed(rows))
    return JSONResponse(await _reprocess_failed_rows(request, db, tenant_id, rows))


@router.post("/reprocess/{post_id}")
async def zernio_reprocess_one(
    request: Request,
    post_id: str,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Reprocesar un post fallido (re-render + URL nueva + re-agendar)."""
    row = await ZernioPublishesRepo(db).get_by_post_id(post_id)
    if row is None or row.tenant_id != tenant_id:
        return JSONResponse(
            {"ok": False, "error": "post no encontrado"}, status_code=404,
        )
    result = await _reprocess_failed_rows(request, db, tenant_id, [row])
    return JSONResponse(result)


@router.get("/best-time.json")
async def zernio_best_time_json(
    platform: str = "",
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Best-time slots for the "Hora óptima" panel + the "Usar hora
    óptima" prefill on Programar. 403 (Analytics add-on) or no data →
    empty list, not an error — the panel shows a "sin datos aún" state."""
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    settings = get_settings()
    if not profile_id or not settings.zernio_api_key:
        return JSONResponse({"ok": True, "slots": []})
    client = _build_client()
    try:
        slots = await client.best_time_slots(
            profile_id=profile_id, platform=(platform or None),
        )
    except ZernioError:
        # Analytics add-on missing / not enough history → no data.
        return JSONResponse({"ok": True, "slots": []})
    return JSONResponse({"ok": True, "slots": slots})


@router.get("/schedule/next-best.json")
async def zernio_next_best_json(
    platform: str = "",
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """The single next best posting datetime (UTC ISO) — backs the
    "Sugerir mejor hora" button on Programar. Uses the tenant's best-time
    analytics when available, else the sane fallback spread."""
    import datetime as _dt

    from nexoclip.publish.hub import _next_fallback_hour, next_best_time

    now_dt = _dt.datetime.now(_dt.UTC)
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    slots: list[dict[str, Any]] = []
    if profile_id and get_settings().zernio_api_key:
        with contextlib.suppress(ZernioError):
            slots = await _build_client().best_time_slots(
                profile_id=profile_id, platform=(platform or None),
            )
    best = next_best_time(slots, now=now_dt) or _next_fallback_hour(now_dt)
    return JSONResponse({"ok": True, "iso": best.isoformat()})


@router.post("/compose/{clip_id}")
async def zernio_compose_clip(
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Generate the title + caption auto-publish would post for this clip —
    backs the Single Publish "Auto-rellenar" button. Stitches the viral
    hook + caption + AI hashtags + the tenant's fixed handle/hashtag
    suffix; generates a fresh hook when the variant doesn't carry one."""
    from nexoclip.db import AutopublishSettingsRepo
    from nexoclip.publish.compose import build_post, generate_hook_line

    s = await AutopublishSettingsRepo(db).get(tenant_id) or {}
    handle_suffix = str(s.get("tag_suffix") or "")
    composed = await build_post(db, clip_id, handle_suffix=handle_suffix)
    if not composed.hook:
        hook = await generate_hook_line(db, tenant_id, clip_id)
        if hook:
            composed = await build_post(
                db, clip_id, handle_suffix=handle_suffix, hook_override=hook,
            )
    return JSONResponse({
        "ok": True,
        "title": composed.title or "",
        "caption": composed.caption,
        "hashtags": composed.hashtags,
    })


# In-memory progress for the whole-queue auto-program run, keyed by tenant.
# Single-instance state: the POST that starts a run and the GET that polls it
# must hit the same process (true on Railway's single web instance). If the
# web tier scales horizontally, move this to a shared store (DB/Redis).
_AUTOPROG: dict[str, dict[str, Any]] = {}
_AUTOPROG_TASKS: set[asyncio.Task[None]] = set()


async def _run_autoprog(
    *,
    tenant_id: str,
    schedule: list[tuple[str, str]],
    targets: list[str],
    handle_suffix: str,
    account_map: dict[str, str],
    profile_id: str,
    tenant_tier: str | None,
    base_url: str,
    session_cookie: str | None,
    db_target: str,
) -> None:
    """Background worker for /schedule/auto: enrich + publish each clip,
    updating the shared progress record as it goes. Never raises into the
    event loop — per-clip failures land on the record; the run ends 'done'
    (or 'error' on a setup failure)."""
    from nexoclip.publish.compose import build_post
    from nexoclip.tenancy import bound_tenant

    prog = _AUTOPROG[tenant_id]
    db = Database(db_target)
    try:
        client = _build_client()
        with bound_tenant(tenant_id):
            for clip_id, when_iso in schedule:
                try:
                    composed = await build_post(db, clip_id, handle_suffix=handle_suffix)
                    post_id = await _publish_clip(
                        client=client, db=db, request=None, tenant_id=tenant_id,
                        profile_id=profile_id, account_map=account_map,
                        clip_id=clip_id, platforms=targets,
                        content=composed.caption, title=composed.title,
                        mode="schedule", scheduled_for=when_iso,
                        tenant_tier=tenant_tier, base_url=base_url,
                        session_cookie=session_cookie,
                    )
                    prog["scheduled"] += 1
                    prog["results"].append({
                        "clip_id": clip_id, "ok": True, "post_id": post_id,
                        "scheduled_for": when_iso,
                    })
                except HTTPException as e:
                    prog["failed"] += 1
                    prog["results"].append(
                        {"clip_id": clip_id, "ok": False, "error": str(e.detail)}
                    )
                except Exception as e:  # one clip's failure must not abort the batch
                    prog["failed"] += 1
                    prog["results"].append(
                        {"clip_id": clip_id, "ok": False, "error": str(e)}
                    )
                prog["done"] += 1
        prog["state"] = "done"
        _log.info(
            "zernio.autoprogram tenant=%s total=%d scheduled=%d failed=%d",
            tenant_id, prog["total"], prog["scheduled"], prog["failed"],
        )
    except Exception as e:
        prog["state"] = "error"
        prog["error"] = str(e)
        _log.exception("zernio.autoprogram.run_failed tenant=%s", tenant_id)
    finally:
        with contextlib.suppress(Exception):
            await db.close()


@router.post("/schedule/auto")
async def zernio_schedule_auto(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Auto-program every approved clip as an interval drip.

    Spreads the available clips with `plan_drip_times` — one post every
    30 min (unique) or 60 min (same-video variations), per the tenant's
    content strategy, with no daily cap — and enriches each post (viral
    hook + caption + AI hashtags + the fixed handle/hashtag suffix). One
    clip's failure is collected, not fatal. Targets the autopublish target
    platforms ∩ connected, else all connected accounts."""
    import datetime as _dt

    from nexoclip.db import AutopublishSettingsRepo
    from nexoclip.publish.hub import drip_interval_minutes, plan_drip_times

    settings = get_settings()
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    if not profile_id or not settings.zernio_api_key:
        return JSONResponse(
            {"ok": False, "error": "Conecta tus redes primero."}, status_code=409,
        )

    # One auto-program run per tenant at a time — a second click while a run
    # is live would double-schedule clips.
    cur = _AUTOPROG.get(tenant_id)
    if cur and cur.get("state") == "running":
        return JSONResponse(
            {"ok": False, "error": "Ya hay una programación en curso."},
            status_code=409,
        )

    clips = await ClipsRepo(db).list_for_tenant_with_status(["approved"], limit=200)
    if not clips:
        return JSONResponse({
            "ok": True, "state": "done", "total": 0,
            "scheduled": 0, "skipped": 0, "results": [],
            "message": "No hay clips aprobados para programar.",
        })

    client = _build_client()
    try:
        accounts = await client.list_accounts(profile_id=profile_id)
    except ZernioError as e:
        _log.warning("zernio.autoprogram.accounts_failed tenant=%s err=%s", tenant_id, e)
        return JSONResponse(
            {"ok": False, "error": f"Couldn't read your accounts: {e}"},
            status_code=502,
        )
    account_map = _account_map(accounts)
    connected = _connected_platforms(accounts)
    if not connected:
        return JSONResponse(
            {"ok": False, "error": "No tienes redes conectadas."}, status_code=409,
        )

    # Target platforms: the autopublish targets ∩ connected, else all
    # connected. The per-tier account cap also caps platforms-per-post.
    s = await AutopublishSettingsRepo(db).get(tenant_id) or {}
    want = [t for t in str(s.get("targets") or "").split(",") if t.strip()]
    targets = [t for t in want if t in connected] or sorted(connected)
    limit = _account_limit(request)
    if limit is not None:
        targets = targets[:limit]
    handle_suffix = str(s.get("tag_suffix") or "")

    # Drip the approved clips one every 30/60 min from the tenant's
    # content strategy — no best-time slots, no daily cap.
    now_dt = _dt.datetime.now(_dt.UTC)
    times = plan_drip_times(
        len(clips), now=now_dt,
        interval_minutes=drip_interval_minutes(s.get("content_strategy")),
    )

    # The per-clip loop (LLM enrichment + render + Zernio publish) is slow —
    # minutes for a big queue. Run it in the background and return immediately;
    # the UI polls /schedule/auto/progress for live counts. The request's
    # tier/base-url/cookie are captured now (the request is gone by the time
    # the task runs).
    schedule = [
        (clip.id, when.isoformat())
        for clip, when in zip(clips, times, strict=False)
    ]
    _AUTOPROG[tenant_id] = {
        "state": "running", "total": len(schedule),
        "done": 0, "scheduled": 0, "failed": 0, "results": [],
    }
    task = asyncio.create_task(
        _run_autoprog(
            tenant_id=tenant_id, schedule=schedule, targets=targets,
            handle_suffix=handle_suffix, account_map=account_map,
            profile_id=profile_id,
            tenant_tier=getattr(request.state, "tenant_tier", None),
            base_url=_public_base_url(request),
            session_cookie=request.cookies.get("nexoclip_token", "") or None,
            db_target=db.target,
        )
    )
    # Hold a reference so the task isn't garbage-collected mid-run.
    _AUTOPROG_TASKS.add(task)
    task.add_done_callback(_AUTOPROG_TASKS.discard)
    return JSONResponse({"ok": True, "state": "running", "total": len(schedule)})


@router.get("/schedule/auto/progress")
async def zernio_schedule_auto_progress(
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
) -> Response:
    """Live progress for the tenant's auto-program run, polled by the UI.

    Returns {state: idle|running|done|error, total, done, scheduled, failed,
    results:[{clip_id, ok, post_id|error, ...}]}."""
    prog = _AUTOPROG.get(tenant_id)
    if not prog:
        return JSONResponse({"state": "idle"})
    return JSONResponse(prog)


@router.post("/schedule/cancel/{post_id}")
async def zernio_cancel_scheduled(
    post_id: str,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Cancel a scheduled/queued post (DELETE /posts/{id} — Zernio only
    allows it for non-published posts). Also tombstones a matching
    local row so it leaves our history immediately."""
    client = _build_client()
    try:
        await client.delete_post(post_id)
    except ZernioError as e:
        if e.status_code == 400:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "Ya se publicó — no se puede cancelar.",
                },
                status_code=409,
            )
        _log.warning("zernio.schedule.cancel_failed tenant=%s err=%s", tenant_id, e)
        return JSONResponse(
            {"ok": False, "error": f"Couldn't cancel: {e}"}, status_code=502,
        )
    # Best-effort local tombstone (the post may have been scheduled
    # directly on Zernio with no local row).
    row = await ZernioPublishesRepo(db).get_by_post_id(post_id)
    if row is not None and row.tenant_id == tenant_id:
        await ZernioPublishesRepo(db).set_status(post_id, status="cancelled")
    _log.info("zernio.schedule.cancelled tenant=%s post=%s", tenant_id, post_id)
    return JSONResponse({"ok": True})


@router.get("/accounts.json")
async def zernio_accounts_json(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Connected accounts for this tenant's profile, as JSON.

    Used by the connect popup poller: after the operator authorizes on
    Zernio (in a separate tab), the dashboard polls this until the new
    platform appears, then closes the popup + refreshes — so the
    operator never has to deal with Zernio's own dashboard."""
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_id = tenant.zernio_profile_id if tenant else None
    settings = get_settings()
    if not profile_id or not settings.zernio_api_key:
        return JSONResponse({"ok": True, "connected": [], "accounts": []})
    client = _build_client()
    try:
        accounts = await client.list_accounts(profile_id=profile_id)
    except ZernioError as e:
        _log.warning("zernio.accounts_json.failed tenant=%s err=%s", tenant_id, e)
        return JSONResponse({"ok": False, "connected": [], "accounts": []})
    return JSONResponse(
        {
            "ok": True,
            "connected": sorted({a.platform.lower() for a in accounts}),
            "accounts": [
                {"platform": a.platform, "account_id": a.account_id} for a in accounts
            ],
        }
    )


@router.post("/accounts/{account_id}/disconnect")
async def zernio_disconnect_account(
    request: Request,
    account_id: str,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Disconnect ONE connected account on Zernio (DELETE /accounts/{id}),
    so the operator manages connections without leaving NexoClip."""
    client = _build_client()
    try:
        await client.disconnect_account(account_id)
    except ZernioError as e:
        _log.warning(
            "zernio.disconnect.failed tenant=%s account=%s err=%s status=%s",
            tenant_id, account_id, e, e.status_code,
        )
        return JSONResponse(
            {"ok": False, "error": f"Disconnect failed: {e}"},
            status_code=502,
        )
    _log.info("zernio.disconnect.ok tenant=%s account=%s", tenant_id, account_id)
    return JSONResponse({"ok": True})


@router.post("/accounts/claim")
async def zernio_claim_existing(
    request: Request,
    profile_id: str = Form(..., alias="profile_id"),
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Bind an EXISTING Zernio profileId to this tenant.

    Use case: an operator who already created connections under a
    specific profileId on Zernio wants NexoClip to reuse it instead of
    deriving a fresh `ten_<ulid>` one. We validate the profileId has at
    least one connected account (so a typo surfaces immediately), then
    persist it on the tenant row.
    """
    profile_id = (profile_id or "").strip()
    if not profile_id:
        raise HTTPException(status_code=400, detail="profileId is required.")

    client = _build_client()
    try:
        accounts = await client.list_accounts(profile_id=profile_id)
    except ZernioError as e:
        _log.warning(
            "zernio.claim.lookup_failed tenant=%s profile_id=%s err=%s",
            tenant_id, profile_id, e,
        )
        raise HTTPException(
            status_code=502, detail=f"Zernio account lookup failed: {e}",
        ) from e
    if not accounts:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No connected accounts found for profileId '{profile_id}'. "
                f"Check the value or click Connect to start fresh."
            ),
        )

    # A capped tier can't sidestep the connect-time limit by claiming a
    # profile that already has more accounts than their plan allows.
    limit = _account_limit(request)
    if limit is not None and len(accounts) > limit:
        raise HTTPException(
            status_code=402,
            detail=(
                f"That profile has {len(accounts)} connected accounts. "
                + _account_limit_message(limit)
            ),
        )

    await TenantsRepo(db).set_zernio_profile(
        tenant_id, profile_id=profile_id, profile_name=None,
    )
    _log.info(
        "zernio.claim.linked tenant=%s profile_id=%s accounts=%d",
        tenant_id, profile_id, len(accounts),
    )
    return RedirectResponse(
        url=f"/dashboard/publish/zernio?claimed={profile_id}",
        status_code=303,
    )


@router.post("/unlink")
async def zernio_unlink(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    db: Database = Depends(get_db),
) -> Response:
    """Clear the tenant's Zernio profileId binding.

    Does NOT disconnect accounts on Zernio's side — it just forgets the
    local mapping so the next Create/claim can use a different profile
    without touching existing connections.
    """
    await TenantsRepo(db).set_zernio_profile(tenant_id, profile_id=None)
    _log.info("zernio.unlink tenant=%s", tenant_id)
    return RedirectResponse(
        url="/dashboard/publish/zernio?unlinked=1",
        status_code=303,
    )


# ---------- Publish ----------


@router.post("/clip/{clip_id}/mark-published")
async def zernio_mark_published(
    request: Request,
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Manually flip an approved clip to 'published' so it leaves the
    ready-to-publish grid.

    For clips that already went out — either published before NexoClip
    started recording publishes locally (pre-migration-030), or posted
    manually outside NexoClip. Local state only; nothing is sent to
    Zernio. Records a synthetic `manual_<clip_id>` history row so the
    clip shows on the Published tab with the time it was marked.
    """
    repo = ClipsRepo(db)
    clip = await repo.get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    if clip.status != "approved":
        raise HTTPException(
            status_code=409,
            detail=f"clip is {clip.status!r}, not 'approved'",
        )
    await repo.update_status(clip_id, status="published")
    await ZernioPublishesRepo(db).record(
        post_id=f"manual_{clip_id}",
        tenant_id=tenant_id,
        clip_id=clip_id,
        platforms=[],
        content=None,
    )
    _log.info("zernio.mark_published tenant=%s clip=%s", tenant_id, clip_id)
    # Stay on the publish list (no forced tab change) — the clip just
    # leaves the grid; its row waits on the Published tab.
    return RedirectResponse(url="/dashboard/publish/zernio", status_code=303)


async def _reopen_published_clip(db: Database, *, clip_id: str) -> bool:
    """Flip one `published` clip back to `approved` so it re-enters the
    publishable grid for a re-publish. Returns True if it moved.

    Used by the failed/auto-publish recovery flow: a clip can be marked
    `published` locally while the platform post never actually landed.
    Reopening leaves the old publish-history rows intact (they're the
    audit trail) — only the clip's status changes. Idempotent: a clip
    that isn't `published` is left alone and returns False.
    """
    repo = ClipsRepo(db)
    clip = await repo.get(clip_id)
    if clip is None or clip.status != "published":
        return False
    await repo.update_status(clip_id, status="approved")
    await EventsRepo(db).emit(
        type="clip.approved",
        payload={"clip_id": clip_id, "from": "published", "to": "approved",
                 "reason": "reopen_for_republish"},
    )
    return True


@router.post("/clip/{clip_id}/reopen")
async def zernio_reopen_clip(
    request: Request,
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Reopen a published clip — flip it back to `approved` so it returns
    to the Publicar uno / Publicar varios grid and can be re-published.

    For clips an auto-publish (or a manual publish) marked `published`
    even though the post failed to land. Local state only; nothing is
    sent to or removed from Zernio.
    """
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    if clip.status != "published":
        raise HTTPException(
            status_code=409,
            detail=f"clip is {clip.status!r}, not 'published'",
        )
    await _reopen_published_clip(db, clip_id=clip_id)
    _log.info("zernio.reopen tenant=%s clip=%s", tenant_id, clip_id)
    return RedirectResponse(
        url="/dashboard/publish/zernio?reopened=1", status_code=303,
    )


@router.post("/reopen-all")
async def zernio_reopen_all(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Reopen every published clip for the tenant in one shot — bulk
    recovery after an auto-publish run marked a batch `published` that
    never actually went out. Each moves `published` -> `approved`.
    """
    published = await ClipsRepo(db).list_for_tenant_with_status(
        ["published"], limit=500,
    )
    reopened = 0
    for clip in published:
        if await _reopen_published_clip(db, clip_id=clip.id):
            reopened += 1
    _log.info(
        "zernio.reopen_all tenant=%s reopened=%d", tenant_id, reopened,
    )
    return RedirectResponse(
        url=f"/dashboard/publish/zernio?reopened={reopened}", status_code=303,
    )


@router.post("/post/{clip_id}")
async def zernio_post_clip(
    request: Request,
    clip_id: str,
    platforms_csv: str = Form(..., alias="platforms"),
    title: str = Form(""),
    description: str = Form(""),
    mode: str = Form("now"),
    scheduled_for: str = Form(""),
    first_comment: str = Form(""),
    tiktok_privacy: str = Form(""),
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Publish a rendered clip to one or more platforms via Zernio.

    Inputs come from the dashboard form: clip_id (URL), platforms
    (comma-separated checkbox group), and optional title/description.
    The caption (`content`) is the description, falling back to the
    title. We mint a signed URL pointing at /api/internal/clip/{clip_id}
    (1h TTL); Zernio downloads from there, re-hosts, and publishes to
    each requested account. The page's feed polls
    /status/{post_id}.json for completion.

    Phase-4 fields: `mode` (now|draft|schedule + `scheduled_for`),
    `first_comment` (only sent to platforms that support it),
    `tiktok_privacy`, and collapsed per-platform caption overrides as
    `caption_{platform}` form keys (+ `yt_title` for YouTube's
    required title).
    """
    platforms = [p.strip() for p in platforms_csv.split(",") if p.strip()]
    if not platforms:
        raise HTTPException(status_code=400, detail="No platforms selected.")
    mode = (mode or "now").strip().lower()
    if mode not in ("now", "draft", "schedule"):
        raise HTTPException(status_code=400, detail=f"Unknown mode: {mode!r}")
    scheduled_for = (scheduled_for or "").strip()
    if mode == "schedule" and not scheduled_for:
        raise HTTPException(
            status_code=400,
            detail="Programar necesita fecha y hora (scheduled_for).",
        )

    # Collapsed per-platform caption overrides ride as caption_{key}
    # form fields; YouTube's title has its own field.
    form = await request.form()
    per_platform: dict[str, Any] = {}
    for key in _SUPPORTED_PLATFORM_IDS:
        override = str(form.get(f"caption_{key}") or "").strip()
        if override:
            per_platform[key] = override
    yt_title = str(form.get("yt_title") or "").strip()
    if yt_title:
        entry = per_platform.get("youtube")
        per_platform["youtube"] = (
            {"caption": entry, "title": yt_title}
            if isinstance(entry, str)
            else {"title": yt_title}
        )

    # YouTube requires a video title — fail BEFORE rendering/minting,
    # with a message the operator can act on.
    effective_yt_title = yt_title or (title or "").strip()
    if "youtube" in (p.lower() for p in platforms) and not effective_yt_title:
        raise HTTPException(
            status_code=400,
            detail=(
                "YouTube necesita un título — completa el campo Title "
                "o el título de YouTube en las opciones por plataforma."
            ),
        )

    profile_id = await _require_profile(db, tenant_id)
    client = _build_client()
    try:
        account_map = _account_map(
            await client.list_accounts(profile_id=profile_id),
        )
    except ZernioError as e:
        _log.warning(
            "zernio.post.profile_failed tenant=%s err=%s body=%s",
            tenant_id, e, e.body,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Zernio setup failed: {e} | Zernio response: {e.body}",
        ) from e

    content = (description or "").strip() or (title or "").strip()
    try:
        post_id = await _publish_clip(
            client=client,
            db=db,
            request=request,
            tenant_id=tenant_id,
            profile_id=profile_id,
            account_map=account_map,
            clip_id=clip_id,
            platforms=platforms,
            content=content,
            title=(title or "").strip() or None,
            mode=mode,
            scheduled_for=scheduled_for or None,
            options=PublishOptions(
                per_platform_captions=per_platform,
                first_comment=(first_comment or "").strip() or None,
            ),
            tiktok_privacy=(tiktok_privacy or "").strip() or None,
        )
    except HTTPException:
        raise
    except ZernioError as e:
        _log.warning(
            "zernio.post.publish_failed tenant=%s clip=%s err=%s body=%s",
            tenant_id, clip_id, e, e.body,
        )
        detail = f"Zernio publish failed: {e}"
        if e.body is not None:
            import json as _json
            body_str = (
                _json.dumps(e.body)
                if isinstance(e.body, dict | list) else str(e.body)
            )
            detail += f" — body: {body_str[:500]}"
        raise HTTPException(status_code=502, detail=detail) from e
    except Exception as e:
        _log.exception(
            "zernio.post.unexpected tenant=%s clip=%s", tenant_id, clip_id,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Zernio call crashed: {type(e).__name__}: {e}",
        ) from e

    _log.info(
        "zernio.post.queued tenant=%s clip=%s post_id=%s platforms=%s mode=%s",
        tenant_id, clip_id, post_id, platforms, mode,
    )
    if mode == "draft":
        return RedirectResponse(
            url="/dashboard/publish/zernio?draft=saved", status_code=303,
        )
    return RedirectResponse(
        url=f"/dashboard/publish/zernio?queued={post_id}",
        status_code=303,
    )


# ---------- Drafts (Borradores) ----------
# "Guardar como borrador" is also the agency client-approval workflow:
# the operator stages posts, the client reviews, then Publicar ahora /
# Programar fires them — or delete kills them. Drafts live on Zernio
# (isDraft) AND in zernio_publishes (status='draft', with the options
# snapshot that re-publish needs once the original signed URL expires).


def _parse_draft_options(row: Any) -> tuple[str | None, PublishOptions, str | None]:
    """(title, options, tiktok_privacy) from a draft row's snapshot."""
    import json as _json

    title: str | None = None
    options = PublishOptions()
    tiktok_privacy: str | None = None
    if row.options_json:
        try:
            data = _json.loads(row.options_json)
        except ValueError:
            data = {}
        if isinstance(data, dict):
            title = data.get("title") or None
            captions = data.get("per_platform_captions")
            options = PublishOptions(
                per_platform_captions=captions if isinstance(captions, dict) else {},
                first_comment=data.get("first_comment") or None,
            )
            tiktok_privacy = data.get("tiktok_privacy") or None
    return title, options, tiktok_privacy


async def _require_draft(
    db: Database, tenant_id: str, post_id: str
) -> Any:
    """Load a draft row, enforcing tenant ownership + draft state."""
    row = await ZernioPublishesRepo(db).get_by_post_id(post_id)
    if row is None or row.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="draft not found")
    if row.status != "draft":
        raise HTTPException(
            status_code=409,
            detail=f"post {post_id} is not a draft (status={row.status}).",
        )
    return row


@router.post("/draft/{post_id}/publish")
async def zernio_draft_publish(
    request: Request,
    post_id: str,
    scheduled_for: str = Form(""),
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Publicar ahora / Programar a saved draft.

    Zernio has no publish-a-draft endpoint (POST /posts/{id}/edit is
    X-only post-publication editing), so the hub re-creates the post
    from the LOCAL snapshot — fresh signed media URL, same captions /
    first comment / privacy — then deletes the Zernio-side draft.
    `scheduled_for` empty = ahora; set = programar.
    """
    row = await _require_draft(db, tenant_id, post_id)
    scheduled_for = (scheduled_for or "").strip()
    title, options, tiktok_privacy = _parse_draft_options(row)

    profile_id = await _require_profile(db, tenant_id)
    client = _build_client()
    try:
        account_map = _account_map(
            await client.list_accounts(profile_id=profile_id),
        )
    except ZernioError as e:
        raise HTTPException(
            status_code=502, detail=f"Zernio setup failed: {e}",
        ) from e

    new_post_id = await _publish_clip(
        client=client,
        db=db,
        request=request,
        tenant_id=tenant_id,
        profile_id=profile_id,
        account_map=account_map,
        clip_id=row.clip_id,
        platforms=[p for p in row.platforms.split(",") if p],
        content=row.content or "",
        title=title,
        mode="schedule" if scheduled_for else "now",
        scheduled_for=scheduled_for or None,
        options=options,
        tiktok_privacy=tiktok_privacy,
    )
    # The Zernio-side draft is now redundant — delete it (best-effort;
    # an orphaned draft on Zernio is cosmetic, the local row rules).
    try:
        await client.delete_post(post_id)
    except ZernioError as e:
        _log.warning(
            "zernio.draft.cleanup_failed tenant=%s post=%s err=%s",
            tenant_id, post_id, e,
        )
    await ZernioPublishesRepo(db).set_status(post_id, status="deleted")
    _log.info(
        "zernio.draft.published tenant=%s draft=%s new_post=%s scheduled=%s",
        tenant_id, post_id, new_post_id, bool(scheduled_for),
    )
    return RedirectResponse(
        url=f"/dashboard/publish/zernio?queued={new_post_id}", status_code=303,
    )


@router.post("/draft/{post_id}/delete")
async def zernio_draft_delete(
    post_id: str,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Delete a draft on Zernio + mark the local row deleted."""
    await _require_draft(db, tenant_id, post_id)
    client = _build_client()
    try:
        await client.delete_post(post_id)
    except ZernioError as e:
        # 404 = already gone on Zernio's side; anything else still
        # removes it locally (the operator asked it gone).
        _log.warning(
            "zernio.draft.delete_remote_failed tenant=%s post=%s err=%s",
            tenant_id, post_id, e,
        )
    await ZernioPublishesRepo(db).set_status(post_id, status="deleted")
    return RedirectResponse(url="/dashboard/publish/zernio", status_code=303)


@router.post("/bulk-post")
async def zernio_bulk(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_paid_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Bulk-publish entry point for the Bulk tab on the dashboard.

    JSON body shape (HTMX wire format from the bulk submit JS):

      {
        "clips": [
          {"clip_id": "clp_...", "platforms": ["tiktok", "instagram"]},
          {"clip_id": "clp_...", "platforms": ["youtube"]}
        ],
        "title": "optional",
        "description": "optional"
      }

    Returns a per-clip result list so the UI can surface "3 of 4 queued"
    feedback. Failures don't short-circuit — each clip is independent.
    """
    body = await _read_json(request)
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object.")

    items = body.get("clips") or []
    if not isinstance(items, list) or not items:
        raise HTTPException(
            status_code=400,
            detail="`clips` must be a non-empty list of {clip_id, platforms}.",
        )
    shared_title = (body.get("title") or "").strip()
    shared_desc = (body.get("description") or "").strip()
    content = shared_desc or shared_title

    profile_id = await _require_profile(db, tenant_id)
    client = _build_client()
    try:
        account_map = _account_map(
            await client.list_accounts(profile_id=profile_id),
        )
    except ZernioError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Zernio setup failed: {e} | Zernio response: {e.body}",
        ) from e

    results: list[dict[str, Any]] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        clip_id = (raw.get("clip_id") or "").strip()
        platforms = [
            p.strip() for p in (raw.get("platforms") or []) if isinstance(p, str)
        ]
        if not clip_id or not platforms:
            results.append(
                {"clip_id": clip_id, "ok": False, "error": "missing clip_id or platforms"}
            )
            continue
        try:
            post_id = await _publish_clip(
                client=client,
                db=db,
                request=request,
                tenant_id=tenant_id,
                profile_id=profile_id,
                account_map=account_map,
                clip_id=clip_id,
                platforms=platforms,
                content=content,
            )
            results.append(
                {
                    "clip_id": clip_id,
                    "ok": True,
                    "post_id": post_id,
                    "platforms": platforms,
                }
            )
        except HTTPException as e:
            results.append(
                {
                    "clip_id": clip_id,
                    "ok": False,
                    "error": str(e.detail),
                    "platforms": platforms,
                }
            )
        except ZernioError as e:
            _log.warning(
                "zernio.bulk.publish_failed tenant=%s clip=%s err=%s",
                tenant_id, clip_id, e,
            )
            results.append(
                {
                    "clip_id": clip_id,
                    "ok": False,
                    "error": str(e),
                    "platforms": platforms,
                }
            )

    _log.info(
        "zernio.bulk.done tenant=%s items=%d ok=%d",
        tenant_id, len(results), sum(1 for r in results if r.get("ok")),
    )
    return JSONResponse({"results": results})


# ---------- HTMX feed + status polling ----------


@router.get("/feed.json")
async def zernio_feed(
    request: Request,
    limit: int = 25,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Tenant-scoped publish history as JSON.

    Reads the LOCAL zernio_publishes table (never the raw company-wide
    Zernio feed — see migration 030) and joins live status from Zernio
    by post id. Status falls back to "queued" when the join fails."""
    settings = get_settings()
    if not settings.zernio_api_key:
        return JSONResponse({"history": [], "configured": False})

    publishes = await ZernioPublishesRepo(db).list_for_tenant(limit=limit)

    status_by_post: dict[str, dict[str, Any]] = {}
    if publishes:
        client = _build_client()
        try:
            feed = await client.list_posts(page=1, limit=100)
            raw = feed.get("posts")
            if isinstance(raw, list):
                for p in raw:
                    if isinstance(p, dict):
                        pid = p.get("_id") or p.get("id")
                        if isinstance(pid, str):
                            status_by_post[pid] = p
        except ZernioError as e:
            _log.warning("zernio.feed.failed tenant=%s err=%s", tenant_id, e)

    history = [
        {
            "post_id": row.post_id,
            "clip_id": row.clip_id,
            "platforms": [p for p in row.platforms.split(",") if p],
            "content": row.content,
            "created_at": row.created_at,
            "status": str(
                (status_by_post.get(row.post_id) or {}).get("status") or "queued"
            ),
        }
        for row in publishes
        # Drafts have their own panel; deleted rows are tombstones.
        if row.status not in ("draft", "deleted")
    ]
    return JSONResponse({"history": history, "configured": True, "connected": True})


def _parse_platforms_json(raw: str | None) -> Any:
    """Decode the per-platform results we snapshot from post.* webhooks
    (`zernio_publishes.platforms_json`). Returns None on absent/invalid
    JSON so callers fall through cleanly."""
    if not raw:
        return None
    import json as _json

    try:
        return _json.loads(raw)
    except ValueError:
        return None


async def _local_post_status(
    db: Database, tenant_id: str, post_id: str,
) -> tuple[str, Any] | None:
    """Last-known (status, per_platform) for a post from OUR webhook-fed
    record, scoped to the requesting tenant.

    Zernio's GET /posts/{id} can 404 a post the company key created
    (deleted/expired on Zernio's side, or published from a different
    workspace) — but the post.* webhooks already wrote the real status +
    per-platform results into zernio_publishes. Falling back to that lets
    the job page + poll degrade to data we have instead of spinning on a
    raw 404 forever. Returns None when there's no row for this tenant — or
    when the row carries no usable status yet."""
    row = await ZernioPublishesRepo(db).get_by_post_id(post_id)
    if row is None or row.tenant_id != tenant_id:
        return None
    status = (row.status or "").strip().upper()
    # A "publish now" row seeds status=None until a post.* webhook lands.
    # Returning "UNKNOWN" here is NOT terminal in the JS poller, so when
    # Zernio also 404s the post (deleted / different workspace, no webhook
    # ever coming) the page polls /status every 3s forever. Treat "no
    # usable status" as no local info so the caller falls through to the
    # terminal UNAVAILABLE branch and the poller settles.
    if not status or status == "UNKNOWN":
        return None
    return status, _parse_platforms_json(row.platforms_json)


# A post Zernio 404s AND we have no local record for. Terminal in the
# poller (the JS stops refreshing) so the page settles into a clean
# "no longer available" state instead of looping every 3s.
_STATUS_UNAVAILABLE = "UNAVAILABLE"


@router.get("/job/{post_id}", response_class=HTMLResponse)
async def zernio_job_detail(
    request: Request,
    post_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Per-post detail page.

    Reached from the queued banner ("View status →") and from feed rows.
    Renders the per-platform result of one Zernio post. Auto-polls
    /status/{post_id}.json while the post is still in a non-terminal
    state, then halts once settled.

    When Zernio can't serve the post we fall back to the webhook-fed
    local record (see _local_post_status); only a genuine 404 with no
    local row surfaces an operator-facing "no longer available" state.
    """
    client = _build_client()
    overall_status = "UNKNOWN"
    per_platform: Any = None
    fetch_error: str | None = None
    try:
        status = await client.get_post(post_id)
        overall_status = (status.status or "UNKNOWN").upper()
        per_platform = status.platforms
    except ZernioError as e:
        _log.warning(
            "zernio.job.status_fetch_failed tenant=%s post_id=%s status=%s err=%s",
            tenant_id, post_id, e.status_code, e,
        )
        local = await _local_post_status(db, tenant_id, post_id)
        if local is not None:
            overall_status, per_platform = local
            fetch_error = (
                "Live status from Zernio is unavailable right now — showing "
                "the last result we recorded for this post."
            )
        elif e.status_code == 404:
            overall_status = _STATUS_UNAVAILABLE
            fetch_error = (
                "This post is no longer available on Zernio. It may have been "
                "deleted, or it was published from a different workspace — "
                "there's nothing more to load here."
            )
        else:
            fetch_error = f"Couldn't load status from Zernio: {e}"

    return templates.TemplateResponse(
        request,
        "publish/zernio_job.html",
        {
            "post_id": post_id,
            "overall_status": overall_status,
            "per_platform": per_platform,
            "fetch_error": fetch_error,
        },
    )


@router.get("/status/{post_id}.json")
async def zernio_status(
    request: Request,
    post_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Poll a single post by post_id. Used by the toast that surfaces
    right after a publish click — flips from PUBLISHING → PUBLISHED in
    the UI without a full page reload.

    Mirrors the job page's fallback: a Zernio fetch failure tries the
    webhook-fed local record first; a genuine 404 with no local row
    settles the poller (200 + terminal UNAVAILABLE) so the page stops
    refreshing instead of looping on a 502 forever."""
    client = _build_client()
    try:
        status = await client.get_post(post_id)
    except ZernioError as e:
        _log.warning(
            "zernio.status.failed tenant=%s post_id=%s status=%s err=%s",
            tenant_id, post_id, e.status_code, e,
        )
        local = await _local_post_status(db, tenant_id, post_id)
        if local is not None:
            st, platforms = local
            return JSONResponse(
                {"post_id": post_id, "status": st, "platforms": platforms}
            )
        if e.status_code == 404:
            # Nothing to wait for. 200 (not 502) so the poller's resp.ok
            # check passes and it settles on the terminal status.
            return JSONResponse(
                {
                    "post_id": post_id,
                    "status": _STATUS_UNAVAILABLE,
                    "error": "post not found on Zernio",
                }
            )
        # Transient/other error — 502 lets the poller retry silently.
        return JSONResponse({"status": "ERROR", "error": str(e)}, status_code=502)
    return JSONResponse(
        {
            "post_id": status.post_id,
            "status": (status.status or "UNKNOWN").upper(),
            "platforms": status.platforms,
        }
    )


__all__ = ["router"]
