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
    TenantsRepo,
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
from nexoclip.settings import get_settings

from ..deps import get_db, require_full_scope, tenant_binder
from ..status_gate import require_paid_tier
from .clips import _VALID_STATUS_TRANSITIONS
from .internal import mint_signed_clip_url

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
    request: Request,
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
    limit = _account_limit(request)
    if limit is not None and len(targets) > limit:
        raise HTTPException(
            status_code=402,
            detail=(
                f"Your plan publishes with {limit} connected account"
                f"{'' if limit == 1 else 's'} — deselect the extra "
                f"platform(s) or upgrade to All-Access."
            ),
        )

    base = _public_base_url(request)

    # Make sure the edited MP4 (overlays + captions burned in) exists on
    # disk BEFORE we hand Zernio the URL — this is what closes the
    # "published clip missing hooks/subs" bug. It's the same file the
    # download path serves, so publish == download. Usually a cache hit
    # because approve pre-renders; renders inline here only when cold.
    # The /render page is auth-gated, so pass the operator's session
    # cookie through to the headless recorder.
    settings = get_settings()
    cookie_val = request.cookies.get("nexoclip_token", "") or None
    try:
        from nexoclip.api._clip_render import ensure_clip_rendered
        await ensure_clip_rendered(
            db=db,
            clip=clip,
            tenant_id=tenant_id,
            base_url=base,
            auth_cookie_value=cookie_val,
            db_path=settings.db_path,
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
        media_url = mint_signed_clip_url(
            clip_id=clip_id,
            tenant_id=tenant_id,
            base_url=base,
            ttl_seconds=3600,
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
        history.append(
            {
                "post_id": row.post_id,
                "kind": "manual" if manual else "zernio",
                "clip_id": row.clip_id,
                "title": title_by_clip.get(row.clip_id),
                "platforms": [p for p in row.platforms.split(",") if p],
                "content": row.content,
                "created_at": row.created_at,
                "status": (
                    "published" if manual
                    else str(live.get("status") or "queued")
                ),
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
    history.sort(key=lambda h: str(h.get("created_at") or ""), reverse=True)
    history = history[:25]

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
            "publishable_clips": publishable,
            "supported_platforms": _SUPPORTED_PLATFORMS,
            "stats": stats,
            # Borradores — local rows seeded status='draft' at save time
            # (publishes already tenant-filtered; limit=100 above).
            "drafts": [p for p in publishes if p.status == "draft"],
        },
    )


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
    if platform not in _SUPPORTED_PLATFORM_IDS:
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
        # zernio_publishes has no scheduled_for column (that's
        # hub_publish_jobs); created_at is the publish moment.
        date = p.created_at
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


@router.get("/job/{post_id}", response_class=HTMLResponse)
async def zernio_job_detail(
    request: Request,
    post_id: str,
    tenant_id: str = Depends(tenant_binder),
) -> Response:
    """Per-post detail page.

    Reached from the queued banner ("View status →") and from feed rows.
    Renders the per-platform result of one Zernio post. Auto-polls
    /status/{post_id}.json while the post is still in a non-terminal
    state, then halts once settled.
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
            "zernio.job.status_fetch_failed tenant=%s post_id=%s err=%s",
            tenant_id, post_id, e,
        )
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
) -> Response:
    """Poll a single post by post_id. Used by the toast that surfaces
    right after a publish click — flips from PUBLISHING → PUBLISHED in
    the UI without a full page reload."""
    client = _build_client()
    try:
        status = await client.get_post(post_id)
    except ZernioError as e:
        _log.warning(
            "zernio.status.failed tenant=%s post_id=%s err=%s",
            tenant_id, post_id, e,
        )
        return JSONResponse({"status": "ERROR", "error": str(e)}, status_code=502)
    return JSONResponse(
        {
            "post_id": status.post_id,
            "status": (status.status or "UNKNOWN").upper(),
            "platforms": status.platforms,
        }
    )


__all__ = ["router"]
