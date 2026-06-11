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

from nexoclip.db import ClipsRepo, Database, TenantsRepo, ZernioPublishesRepo
from nexoclip.integrations.zernio import (
    ZernioAccount,
    ZernioClient,
    ZernioError,
    create_profile_for_tenant,
)
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
) -> str:
    """Shared publish core for the single + bulk paths.

    Resolves each selected platform to its connected accountId, mints a
    signed clip URL, and fires one POST /posts. Returns the Zernio
    post_id. Raises HTTPException with an operator-readable message on
    every failure mode (unknown clip, unconnected platform, missing
    signing secret, Zernio error).
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
    try:
        result = await client.create_post(
            profile_id=profile_id,
            content=content,
            media_url=media_url,
            platforms=targets,
            publish_now=True,
            tiktok_settings=tiktok,
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
    await ZernioPublishesRepo(db).record(
        post_id=post_id,
        tenant_id=tenant_id,
        clip_id=clip_id,
        platforms=[p for p, _ in targets],
        content=content,
    )
    # Flip the clip out of the "ready to publish" grid. Best-effort —
    # the publish already happened; a status-write hiccup must not fail
    # the request.
    if clip.status == "approved":
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
    return RedirectResponse(
        url="/dashboard/publish/zernio?tab=published", status_code=303,
    )


@router.post("/post/{clip_id}")
async def zernio_post_clip(
    request: Request,
    clip_id: str,
    platforms_csv: str = Form(..., alias="platforms"),
    title: str = Form(""),
    description: str = Form(""),
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
    """
    platforms = [p.strip() for p in platforms_csv.split(",") if p.strip()]
    if not platforms:
        raise HTTPException(status_code=400, detail="No platforms selected.")

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
        "zernio.post.queued tenant=%s clip=%s post_id=%s platforms=%s",
        tenant_id, clip_id, post_id, platforms,
    )
    return RedirectResponse(
        url=f"/dashboard/publish/zernio?queued={post_id}",
        status_code=303,
    )


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
