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


@router.get("", response_class=HTMLResponse)
async def upload_post_dashboard(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Dashboard page: connection status + history feed + clip
    publish entry point.

    Tries to fetch the connected social accounts (so the page can
    show "you've linked TikTok + IG"). Non-fatal on failure — the
    page still renders with an empty 'no connections yet' state.
    """
    settings = get_settings()
    configured = bool(settings.upload_post_api_key)

    # Look up local mapping. The profile may or may not exist on
    # upload-post's side yet — we don't create it eagerly here;
    # only the explicit Connect click does that.
    tenant = await TenantsRepo(db).get(tenant_id)
    profile_username = tenant.upload_post_profile_username if tenant else None

    social_accounts: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    scheduled: list[dict[str, Any]] = []
    fetch_error: str | None = None

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
            fetch_error = "Couldn't reach upload-post to list connections."
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
            fetch_error = fetch_error or "Couldn't load upload history."
        try:
            scheduled = await client.get_scheduled()
        except UploadPostError as e:
            _log.warning(
                "upload_post.dashboard.scheduled_fetch_failed tenant=%s err=%s",
                tenant_id, e,
            )
            # Non-fatal — scheduled feed just stays empty.

    # Per-tenant Publishable clips (last 30, approved or published).
    publishable: list[Any] = []
    try:
        publishable = await ClipsRepo(db).list_for_tenant_with_status(
            ["approved", "published"], limit=30,
        )
    except Exception:  # noqa: BLE001 — display is best-effort
        pass

    return templates.TemplateResponse(
        request,
        "publish/upload_post_dashboard.html",
        {
            "configured": configured,
            "profile_username": profile_username,
            "social_accounts": social_accounts,
            "history": history,
            "scheduled": scheduled,
            "publishable_clips": publishable,
            "fetch_error": fetch_error,
        },
    )


# ---------- Connect ----------


@router.post("/claim")
async def upload_post_claim_existing(
    request: Request,
    username: str = Form(...),
    tenant_id: str = Depends(tenant_binder),
    _: None = Depends(require_full_scope),
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
            detail=f"upload-post profile setup failed: {e}",
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
            detail=f"upload-post profile setup failed: {e}",
        ) from e

    # Build the signed URL upload-post will GET to download the MP4.
    base = _public_base_url(request)
    video_url = mint_signed_clip_url(
        clip_id=clip_id,
        tenant_id=tenant_id,
        base_url=base,
        ttl_seconds=3600,
    )

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
        raise HTTPException(
            status_code=502,
            detail=f"upload-post upload failed: {e}",
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
            "status": status.status,
            "result": status.result,
        }
    )


__all__ = ["router"]
