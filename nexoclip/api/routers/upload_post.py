"""upload-post.com dashboard + publish actions.

Hangs off /dashboard/publish/upload-post. The Publish nav tab still
points at the existing matrix view (`dashboard.publish_view`); the
operator clicks through to this surface when they want to either:

  - Connect their socials (mints a 48h JWT magic link, 303s the
    browser to upload-post.com's hosted connect UI)
  - Publish a clip to one or more platforms
  - See history of past + scheduled posts

Endpoints
  GET  /dashboard/publish/upload-post           — the dashboard page
  POST /dashboard/publish/upload-post/connect   — mint JWT, 303 away
  POST /dashboard/publish/upload-post/post/{clip_id}
                                                — kick off a publish
  GET  /dashboard/publish/upload-post/feed.json — HTMX-polled feed
  GET  /dashboard/publish/upload-post/status/{request_id}.json
                                                — single-job poll
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from nexoclip.db import ClipsRepo, Database, TenantsRepo
from nexoclip.integrations.upload_post import (
    UploadPostClient,
    UploadPostError,
    ensure_profile_for_tenant,
)
from nexoclip.settings import get_settings

from ..deps import get_db, require_full_scope, tenant_binder
from ..status_gate import require_top_tier
from .internal import mint_signed_clip_url

_log = logging.getLogger("nexoclip.api.upload_post")

router = APIRouter(prefix="/dashboard/publish/upload-post", tags=["upload-post"])

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
from ..i18n import install_globals as _install_i18n  # noqa: E402
_install_i18n(templates)


def _build_client() -> UploadPostClient:
    """Build an UploadPostClient from settings. Raises 503 on the
    request path when the API key isn't configured so the operator
    sees a clear error instead of an opaque 500."""
    settings = get_settings()
    if not settings.upload_post_api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "UPLOAD_POST_API_KEY is not configured on this NexoClip "
                "instance. Add it to Railway env and redeploy."
            ),
        )
    return UploadPostClient(
        api_key=settings.upload_post_api_key,
        base_url=settings.upload_post_base_url,
    )


def _public_base_url(request: Request) -> str:
    """Return the externally-reachable origin so signed clip URLs
    point at the right hostname. Prefers the public_url setting
    when set (production custom domain), falls back to the request's
    own scheme+host (works for localhost dev)."""
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
    # (id used by upload-post API,  human label,           tabler icon class)
    ("tiktok",    "TikTok",    "ti-brand-tiktok"),
    ("instagram", "Instagram", "ti-brand-instagram"),
    ("youtube",   "YouTube",   "ti-brand-youtube"),
    ("x",         "X",         "ti-brand-x"),
    ("linkedin",  "LinkedIn",  "ti-brand-linkedin"),
    ("facebook",  "Facebook",  "ti-brand-facebook"),
    ("threads",   "Threads",   "ti-brand-threads"),
    ("pinterest", "Pinterest", "ti-brand-pinterest"),
    ("bluesky",   "Bluesky",   "ti-brand-bluesky"),
]


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


def _connected_platforms(social_accounts: dict[str, Any]) -> set[str]:
    """Platform keys we should consider "ready to ship to" — anything
    that comes back with a non-empty truthy value in social_accounts.
    Empty string / None / missing → not connected.
    """
    out: set[str] = set()
    for k, v in (social_accounts or {}).items():
        if v:  # truthy = real dict OR non-empty string
            out.add(k.lower())
    return out


@router.get("", response_class=HTMLResponse)
async def upload_post_dashboard(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Publish Center — single-page surface with two tabs:
      - Single Publish: thumbnail grid + selected-clip publish panel
      - Bulk Publish: per-clip rows with inline platform chip toggles

    Sections (top to bottom):
      1. Compact "Connected accounts" row (chips, no profile card)
      2. Tab switcher
      3. Active tab content
      4. History feed at the bottom
    """
    settings = get_settings()
    configured = bool(settings.upload_post_api_key)

    tenant = await TenantsRepo(db).get(tenant_id)
    profile_username = tenant.upload_post_profile_username if tenant else None

    social_accounts: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    scheduled: list[dict[str, Any]] = []
    connect_fetch_failed = False

    if configured and profile_username:
        client = _build_client()
        try:
            profile = await client.get_user_profile(profile_username)
            social_accounts = profile.social_accounts or {}
        except UploadPostError as e:
            _log.warning(
                "upload_post.dashboard.profile_fetch_failed tenant=%s err=%s",
                tenant_id, e,
            )
            connect_fetch_failed = True
        # History + scheduled failures are silent now — a brand-new
        # tenant with no posts yet legitimately gets empty arrays, and
        # the previous "Couldn't load upload history" banner was
        # alarming for what is just an empty state.
        try:
            history_body = await client.get_history(page=1, limit=25)
            history = (
                history_body.get("history")
                if isinstance(history_body.get("history"), list) else []
            )
        except UploadPostError as e:
            _log.warning(
                "upload_post.dashboard.history_fetch_failed tenant=%s err=%s",
                tenant_id, e,
            )
        try:
            scheduled = await client.get_scheduled()
        except UploadPostError as e:
            _log.warning(
                "upload_post.dashboard.scheduled_fetch_failed tenant=%s err=%s",
                tenant_id, e,
            )

    connected_set = _connected_platforms(social_accounts)

    # Publishable clips for both tabs. Decorate each with display_title
    # + thumbnail URL so the template stays dumb.
    raw_clips: list[Any] = []
    try:
        raw_clips = await ClipsRepo(db).list_for_tenant_with_status(
            ["approved", "published"], limit=60,
        )
    except Exception:  # noqa: BLE001 — display is best-effort
        pass
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

    # Stats strip — single pass over the history list (already
    # capped at 25 by get_history(limit=25)). For "all time" counts
    # we'd need to paginate, but 25 is plenty for the at-a-glance
    # tiles operators actually read. Scheduled count comes from the
    # separate /schedule endpoint which is already a complete list.
    pub_count = sum(1 for h in history if h.get("success"))
    fail_count = sum(1 for h in history if not h.get("success"))
    stats = {
        "published": pub_count,
        "failed": fail_count,
        "scheduled": len(scheduled),
        "total": len(history),
    }

    return templates.TemplateResponse(
        request,
        "publish/upload_post_dashboard.html",
        {
            "configured": configured,
            "profile_username": profile_username,
            "social_accounts": social_accounts,
            "connected_set": connected_set,
            "connect_fetch_failed": connect_fetch_failed,
            "history": history,
            "scheduled": scheduled,
            "publishable_clips": publishable,
            "supported_platforms": _SUPPORTED_PLATFORMS,
            "stats": stats,
        },
    )


# ---------- Connect ----------


@router.post("/claim")
async def upload_post_claim_existing(
    request: Request,
    username: str = Form(...),
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_top_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Bind an EXISTING upload-post profile to this tenant.

    Use case: operator already created a profile manually on
    upload-post.com (e.g. picked their personal handle like
    `aldov1llanueva` instead of letting NexoClip derive
    `ten_<ulid>` from the tenant id). Without this, our
    ensure_profile_for_tenant() would happily mint a second
    profile next to the one they hand-created.

    The flow: take the typed username, hit upload-post's
    GET /users/{username} to confirm it exists (and to surface
    a clear error if it doesn't), then persist on the tenant row.
    Subsequent connect/post clicks fast-path off the persisted value.

    upload-post usernames are CASE SENSITIVE (their docs say so
    in the create-profile confirmation banner). We pass through
    verbatim — no normalize/lowercase.
    """
    username = (username or "").strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required.")

    client = _build_client()
    try:
        profile = await client.get_user_profile(username)
    except UploadPostError as e:
        _log.warning(
            "upload_post.claim.lookup_failed tenant=%s username=%s err=%s status=%s",
            tenant_id, username, e, e.status_code,
        )
        if e.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No upload-post profile exists with username "
                    f"'{username}'. Check the spelling (case sensitive!) "
                    f"or click Connect to create a new one."
                ),
            ) from e
        raise HTTPException(
            status_code=502,
            detail=f"upload-post profile lookup failed: {e}",
        ) from e

    await TenantsRepo(db).set_upload_post_profile_username(
        tenant_id, profile.username,
    )
    _log.info(
        "upload_post.claim.linked tenant=%s username=%s",
        tenant_id, profile.username,
    )
    return RedirectResponse(
        url=f"/dashboard/publish/upload-post?claimed={profile.username}",
        status_code=303,
    )


@router.post("/connect")
async def upload_post_connect(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_top_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Mint a 48h JWT magic link on upload-post and 303 the operator
    there to connect their social accounts.

    First call for a tenant also creates the upload-post profile
    (via ensure_profile_for_tenant). Subsequent calls fast-path
    off the persisted profile username.
    """
    client = _build_client()
    try:
        username = await ensure_profile_for_tenant(
            db=db, tenant_id=tenant_id, client=client,
        )
    except UploadPostError as e:
        _log.warning(
            "upload_post.connect.profile_failed tenant=%s err=%s status=%s body=%s",
            tenant_id, e, e.status_code, e.body,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                f"upload-post profile setup failed: {e}"
                f" | upload-post response: {e.body}"
            ),
        ) from e

    # Tenant returns to our dashboard once they're done connecting.
    redirect_back = f"{_public_base_url(request)}/dashboard/publish/upload-post"

    # Localize the "back to NexoClip" affordance + page chrome so the
    # operator's locale carries through to upload-post's UI. Falls
    # back to English when state.locale isn't set (CLI / cron paths).
    locale = (getattr(request.state, "locale", None) or "en").lower()
    if locale.startswith("es"):
        button_text = "Volver a NexoClip"
        connect_title = "Conecta tus cuentas de publicación de NexoClip"
        connect_description = (
            "Vincula TikTok, Instagram, YouTube y demás. "
            "Al terminar, pulsa Volver a NexoClip arriba a la derecha."
        )
        ui_lang = "es"
    elif locale.startswith("pt"):
        button_text = "Voltar para NexoClip"
        connect_title = "Conecte suas contas de publicação do NexoClip"
        connect_description = (
            "Conecte TikTok, Instagram, YouTube e os demais. "
            "Quando terminar, clique em Voltar para NexoClip."
        )
        ui_lang = "pt"
    else:
        button_text = "Back to NexoClip"
        connect_title = "Connect your NexoClip publishing accounts"
        connect_description = (
            "Link TikTok, Instagram, YouTube, and others. "
            "When done, click Back to NexoClip in the top right."
        )
        ui_lang = "en"

    try:
        jwt_link = await client.generate_connect_jwt(
            username,
            redirect_url=redirect_back,
            redirect_button_text=button_text,
            connect_title=connect_title,
            connect_description=connect_description,
            language=ui_lang,
        )
    except UploadPostError as e:
        _log.warning(
            "upload_post.connect.jwt_failed tenant=%s err=%s status=%s body=%s",
            tenant_id, e, e.status_code, e.body,
        )
        raise HTTPException(
            status_code=502,
            detail=f"upload-post JWT mint failed: {e}",
        ) from e
    _log.info(
        "upload_post.connect.jwt_minted tenant=%s username=%s duration=%s",
        tenant_id, username, jwt_link.duration,
    )
    return RedirectResponse(url=jwt_link.access_url, status_code=303)


@router.post("/unlink")
async def upload_post_unlink(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    db: Database = Depends(get_db),
) -> Response:
    """Clear the tenant's upload-post profile binding.

    Does NOT delete the profile on upload-post's side — that's a
    separate destructive action (DELETE /users) we deliberately
    keep behind the upload-post UI. This just forgets the local
    mapping so the next Connect click can mint or claim a different
    one without touching the existing data.
    """
    await TenantsRepo(db).set_upload_post_profile_username(tenant_id, "")
    _log.info("upload_post.unlink tenant=%s", tenant_id)
    return RedirectResponse(
        url="/dashboard/publish/upload-post?unlinked=1",
        status_code=303,
    )


# ---------- Publish ----------


@router.post("/post/{clip_id}")
async def upload_post_post_clip(
    request: Request,
    clip_id: str,
    platforms_csv: str = Form(..., alias="platforms"),
    title: str = Form(""),
    description: str = Form(""),
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_top_tier),
    db: Database = Depends(get_db),
) -> Response:
    """Publish a rendered clip to one or more platforms via upload-post.

    Inputs come from the dashboard form: clip_id (URL), platforms
    (comma-separated checkbox group), and optional title/description.

    We mint a signed URL pointing at /api/internal/clip/{clip_id}
    (1h TTL); upload-post downloads from there, re-hosts, and
    publishes to each requested platform. async_upload=True so we
    return fast; the page's feed polls /status/{request_id}.json
    for completion.
    """
    client = _build_client()
    platforms = [p.strip() for p in platforms_csv.split(",") if p.strip()]
    if not platforms:
        raise HTTPException(status_code=400, detail="No platforms selected.")

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")

    # Ensure the tenant has an upload-post profile.
    try:
        username = await ensure_profile_for_tenant(
            db=db, tenant_id=tenant_id, client=client,
        )
    except UploadPostError as e:
        _log.warning(
            "upload_post.post.profile_failed tenant=%s err=%s body=%s",
            tenant_id, e, e.body,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                f"upload-post profile setup failed: {e}"
                f" | upload-post response: {e.body}"
            ),
        ) from e

    # Build the signed URL upload-post will GET to download the MP4.
    # mint_signed_clip_url raises RuntimeError when
    # NEXOCLIP_INTERNAL_SIGNING_SECRET is unset on this deploy —
    # catch + surface a clear 503 so the operator sees an
    # actionable message instead of an opaque 500.
    base = _public_base_url(request)
    try:
        video_url = mint_signed_clip_url(
            clip_id=clip_id,
            tenant_id=tenant_id,
            base_url=base,
            ttl_seconds=3600,
        )
    except RuntimeError as e:
        _log.error(
            "upload_post.post.signing_secret_missing tenant=%s clip=%s err=%s",
            tenant_id, clip_id, e,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "NEXOCLIP_INTERNAL_SIGNING_SECRET is not configured on "
                "this NexoClip instance. Add it to Railway env (same "
                "secret used for the Modal Whisper audio fetch) and "
                "redeploy. upload-post needs a signed URL to download "
                "your clip MP4."
            ),
        ) from e

    try:
        result = await client.upload_video_from_url(
            username=username,
            video_url=video_url,
            platforms=platforms,
            title=title or None,
            description=description or None,
            async_upload=True,
        )
    except UploadPostError as e:
        _log.warning(
            "upload_post.post.upload_failed tenant=%s clip=%s err=%s body=%s",
            tenant_id, clip_id, e, e.body,
        )
        # Surface upload-post's response body in the operator-facing
        # error so a 400 says exactly which field they're rejecting
        # instead of an opaque "POST /api/upload returned HTTP 400".
        detail = f"upload-post upload failed: {e}"
        if e.body is not None:
            import json as _json
            body_str = (
                _json.dumps(e.body)
                if isinstance(e.body, dict | list) else str(e.body)
            )
            detail += f" — body: {body_str[:500]}"
        raise HTTPException(
            status_code=502,
            detail=detail,
        ) from e
    except Exception as e:  # noqa: BLE001 — defensive catch-all so 500s
        # surface a real error message instead of FastAPI's opaque
        # "Internal Server Error". Unknown failure paths get logged
        # at exception level so we can diagnose from the structured logs.
        _log.exception(
            "upload_post.post.unexpected tenant=%s clip=%s", tenant_id, clip_id,
        )
        raise HTTPException(
            status_code=500,
            detail=f"upload-post call crashed: {type(e).__name__}: {e}",
        ) from e

    _log.info(
        "upload_post.post.queued tenant=%s clip=%s request_id=%s platforms=%s",
        tenant_id, clip_id, result.request_id, platforms,
    )
    return RedirectResponse(
        url=f"/dashboard/publish/upload-post?queued={result.request_id}",
        status_code=303,
    )


# ---------- HTMX feed + status polling ----------


@router.get("/feed.json")
async def upload_post_feed(
    request: Request,
    page: int = 1,
    limit: int = 25,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Lightweight JSON feed for the dashboard's HTMX-polled history
    panel. Returns the raw upload-post history + scheduled lists so
    the client can render whatever they give us back."""
    settings = get_settings()
    if not settings.upload_post_api_key:
        return JSONResponse({"history": [], "scheduled": [], "configured": False})

    tenant = await TenantsRepo(db).get(tenant_id)
    profile_username = tenant.upload_post_profile_username if tenant else None
    if not profile_username:
        return JSONResponse(
            {"history": [], "scheduled": [], "configured": True, "connected": False},
        )

    client = _build_client()
    history: list[dict[str, Any]] = []
    scheduled: list[dict[str, Any]] = []
    try:
        body = await client.get_history(page=page, limit=limit)
        if isinstance(body.get("history"), list):
            history = body["history"]
    except UploadPostError as e:
        _log.warning(
            "upload_post.feed.history_failed tenant=%s err=%s",
            tenant_id, e,
        )
    try:
        scheduled = await client.get_scheduled()
    except UploadPostError as e:
        _log.warning(
            "upload_post.feed.scheduled_failed tenant=%s err=%s",
            tenant_id, e,
        )

    return JSONResponse(
        {
            "history": history,
            "scheduled": scheduled,
            "configured": True,
            "connected": True,
        },
    )


@router.post("/bulk-post")
async def upload_post_bulk(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
    _t: None = Depends(require_top_tier),
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

    For each clip we mint a fresh signed URL + fire one /api/upload
    call to upload-post (async). Returns a per-clip result list so
    the UI can surface "3 of 4 queued" feedback. Failures don't
    short-circuit — each clip is independent.
    """
    client = _build_client()
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object.")

    items = body.get("clips") or []
    if not isinstance(items, list) or not items:
        raise HTTPException(
            status_code=400,
            detail="`clips` must be a non-empty list of {clip_id, platforms}.",
        )
    shared_title = (body.get("title") or "").strip() or None
    shared_desc = (body.get("description") or "").strip() or None

    try:
        username = await ensure_profile_for_tenant(
            db=db, tenant_id=tenant_id, client=client,
        )
    except UploadPostError as e:
        raise HTTPException(
            status_code=502,
            detail=(
                f"upload-post profile setup failed: {e}"
                f" | upload-post response: {e.body}"
            ),
        ) from e

    base = _public_base_url(request)
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
        clip = await ClipsRepo(db).get(clip_id)
        if clip is None:
            results.append({"clip_id": clip_id, "ok": False, "error": "clip not found"})
            continue
        try:
            video_url = mint_signed_clip_url(
                clip_id=clip_id, tenant_id=tenant_id, base_url=base, ttl_seconds=3600,
            )
        except RuntimeError as e:
            # Same root cause as the single-post path — surface clearly
            # on every row instead of crashing the whole batch.
            _log.error(
                "upload_post.bulk.signing_secret_missing tenant=%s clip=%s err=%s",
                tenant_id, clip_id, e,
            )
            results.append({
                "clip_id": clip_id, "ok": False,
                "error": "NEXOCLIP_INTERNAL_SIGNING_SECRET is not configured",
                "platforms": platforms,
            })
            continue
        try:
            r = await client.upload_video_from_url(
                username=username,
                video_url=video_url,
                platforms=platforms,
                title=shared_title,
                description=shared_desc,
                async_upload=True,
            )
            results.append(
                {
                    "clip_id": clip_id,
                    "ok": True,
                    "request_id": r.request_id,
                    "platforms": platforms,
                }
            )
        except UploadPostError as e:
            _log.warning(
                "upload_post.bulk.upload_failed tenant=%s clip=%s err=%s",
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
        "upload_post.bulk.done tenant=%s items=%d ok=%d",
        tenant_id, len(results), sum(1 for r in results if r.get("ok")),
    )
    return JSONResponse({"results": results})


@router.get("/job/{request_id}", response_class=HTMLResponse)
async def upload_post_job_detail(
    request: Request,
    request_id: str,
    tenant_id: str = Depends(tenant_binder),
) -> Response:
    """Per-request detail page.

    Reached from the queued banner ("View status →") and from
    history rows that have a request_id. Renders one card per
    platform that upload-post tried to publish to, with success/
    error per platform + a link to the live post when present.

    The page auto-polls /status/{id}.json every 3s while any
    platform is still PENDING/PROCESSING, then halts once they're
    all settled (FINISHED or ERROR).
    """
    client = _build_client()
    overall_status = "UNKNOWN"
    per_platform: dict[str, Any] = {}
    fetch_error: str | None = None
    try:
        status = await client.get_status(request_id=request_id)
        # upload-post returns status as lowercase ("processing",
        # "finished"), but the template's CSS selectors and the JS
        # poll-control logic key on UPPERCASE values. Normalize here
        # so the rest of the stack reads one consistent shape.
        overall_status = (status.status or "UNKNOWN").upper()
        # status.result.platforms is the canonical per-platform
        # dict. May be None until upload-post starts dispatching.
        if status.result and isinstance(status.result, dict):
            plats = status.result.get("platforms")
            if isinstance(plats, dict):
                per_platform = plats
    except UploadPostError as e:
        _log.warning(
            "upload_post.job.status_fetch_failed tenant=%s request_id=%s err=%s",
            tenant_id, request_id, e,
        )
        fetch_error = f"Couldn't load status from upload-post: {e}"

    return templates.TemplateResponse(
        request,
        "publish/upload_post_job.html",
        {
            "request_id": request_id,
            "overall_status": overall_status,
            "per_platform": per_platform,
            "fetch_error": fetch_error,
        },
    )


@router.get("/status/{request_id}.json")
async def upload_post_status(
    request: Request,
    request_id: str,
    tenant_id: str = Depends(tenant_binder),
) -> Response:
    """Poll a single async upload by request_id. Used by the toast
    that surfaces right after a publish click — flips from
    PROCESSING → FINISHED in the UI without a full page reload."""
    client = _build_client()
    try:
        status = await client.get_status(request_id=request_id)
    except UploadPostError as e:
        _log.warning(
            "upload_post.status.failed tenant=%s request_id=%s err=%s",
            tenant_id, request_id, e,
        )
        return JSONResponse(
            {"status": "ERROR", "error": str(e)},
            status_code=502,
        )
    return JSONResponse(
        {
            "request_id": status.request_id,
            # Match the /job/{id} HTML handler — upload-post returns
            # lowercase ("processing"); we surface UPPERCASE so the
            # poll JS comparisons hit consistently.
            "status": (status.status or "UNKNOWN").upper(),
            "result": status.result,
        }
    )


__all__ = ["router"]
