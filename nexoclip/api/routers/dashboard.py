"""HTMX-rendered dashboard.

Server-rendered Jinja2 + HTMX. Same auth path as the JSON API but reads
the token from the `nexoclip_token` cookie (set by `POST /dashboard/login`)
in addition to the `Authorization` header. The bearer middleware in
`auth.py` does the cookie fallback - this router just renders the HTML.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Form,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from nexoclip.db import (
    ApiTokensRepo,
    CandidatesRepo,
    ClipsRepo,
    ConnectedAccountsRepo,
    Database,
    EventsRepo,
    LLMCallsRepo,
    PersonasRepo,
    PublishJobsRepo,
    StreamsRepo,
    VariantsRepo,
)
from nexoclip.errors import NexoClipError
from nexoclip.tenancy import hash_token

from .._pipeline import PipelineKickoff
from ..deps import get_db, require_full_scope, tenant_binder
from .clips import _VALID_STATUS_TRANSITIONS

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

_COOKIE_NAME = "nexoclip_token"


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


# ---------- Login / logout (public; auth is via this form) ----------


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request, error: str | None = None) -> Response:
    return templates.TemplateResponse(
        request, "login.html", {"error": error}
    )


@router.post("/login")
async def login_submit(
    request: Request,
    token: str = Form(...),
) -> Response:
    """Validate the token, set a cookie, redirect to /dashboard/streams."""
    db: Database = request.app.state.db
    try:
        token_hash = hash_token(token.strip())
    except Exception:
        return templates.TemplateResponse(
            request, "login.html", {"error": "invalid token"}, status_code=400
        )
    row = await ApiTokensRepo(db).lookup_by_hash(token_hash)
    if row is None:
        return templates.TemplateResponse(
            request, "login.html", {"error": "unknown token"}, status_code=401
        )
    response = RedirectResponse(url="/dashboard/streams", status_code=303)
    # httponly + samesite=lax: the dashboard is same-origin, no JS needs the cookie.
    response.set_cookie(
        _COOKIE_NAME, token.strip(), httponly=True, samesite="lax", max_age=60 * 60 * 24 * 7
    )
    return response


@router.post("/logout")
async def logout() -> Response:
    response = RedirectResponse(url="/dashboard/login", status_code=303)
    response.delete_cookie(_COOKIE_NAME)
    return response


# ---------- Streams ----------


@router.get("/streams", response_class=HTMLResponse)
async def streams_list(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    streams = await StreamsRepo(db).list_for_tenant()
    personas = await PersonasRepo(db).list_for_tenant()
    return templates.TemplateResponse(
        request,
        "streams_list.html",
        {"tenant_id": tenant_id, "streams": streams, "personas": personas},
    )


@router.post("/streams", dependencies=[Depends(require_full_scope)])
async def streams_create(
    request: Request,
    background_tasks: BackgroundTasks,
    vod_url: str = Form(...),
    persona_id: str = Form(...),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    from nexoclip.db.adapters import stream_to_row
    from nexoclip.ingest import ingest_vod
    from nexoclip.settings import get_settings

    output_dir = Path(get_settings().default_output_dir)
    try:
        stream = await ingest_vod(
            vod_url=vod_url, tenant_id=tenant_id, output_dir=output_dir
        )
    except NexoClipError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    row = await StreamsRepo(db).upsert(stream_to_row(stream))
    await EventsRepo(db).emit(type="stream.created", payload={"stream_id": row.id})
    runner = request.app.state.pipeline_runner
    background_tasks.add_task(
        runner,
        PipelineKickoff(
            tenant_id=tenant_id,
            stream=stream,
            persona_id=persona_id,
            output_dir=output_dir,
        ),
    )
    return RedirectResponse(url=f"/dashboard/streams/{row.id}", status_code=303)


@router.get("/streams/{stream_id}", response_class=HTMLResponse)
async def stream_detail(
    request: Request,
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    stream = await StreamsRepo(db).get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="stream not found")
    candidates = await CandidatesRepo(db).list_for_stream(stream_id)
    clips = await ClipsRepo(db).list_for_stream(stream_id)
    return templates.TemplateResponse(
        request,
        "stream_detail.html",
        {"stream": stream, "candidates": candidates, "clips": clips},
    )


# ---------- Clips ----------


@router.get("/clips/{clip_id}", response_class=HTMLResponse)
async def clip_detail(
    request: Request,
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    variants = await VariantsRepo(db).list_for_clip(clip_id)
    accounts = await ConnectedAccountsRepo(db).list_for_tenant()
    valid_transitions = sorted(_VALID_STATUS_TRANSITIONS.get(clip.status, set()))
    return templates.TemplateResponse(
        request,
        "clip_detail.html",
        {
            "clip": clip,
            "variants": variants,
            "accounts": accounts,
            "valid_transitions": valid_transitions,
        },
    )


@router.patch(
    "/clips/{clip_id}/status",
    response_class=HTMLResponse,
    dependencies=[Depends(require_full_scope)],
)
async def clip_status_patch(
    request: Request,
    clip_id: str,
    to: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """HTMX target - returns just the status badge so the page updates in place."""
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    allowed = _VALID_STATUS_TRANSITIONS.get(clip.status, set())
    if to not in allowed:
        raise HTTPException(
            status_code=409,
            detail=f"cannot transition from {clip.status!r} to {to!r}",
        )
    conn = await db.connect()
    await conn.execute(
        "UPDATE clips SET status = ? WHERE id = ? AND tenant_id = ?",
        (to, clip_id, tenant_id),
    )
    await conn.commit()
    await EventsRepo(db).emit(
        type=f"clip.{to}",
        payload={"clip_id": clip_id, "from": clip.status, "to": to},
    )
    return HTMLResponse(
        f'<span class="status-badge status-{to}" id="clip-status">{to}</span>'
    )


@router.post(
    "/clips/{clip_id}/publish",
    dependencies=[Depends(require_full_scope)],
)
async def clip_publish(
    request: Request,
    clip_id: str,
    variant_id: str = Form(...),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    accounts = await ConnectedAccountsRepo(db).list_for_tenant()
    if not accounts:
        raise HTTPException(status_code=409, detail="no connected accounts")
    jobs_repo = PublishJobsRepo(db)
    for account in accounts:
        await jobs_repo.enqueue(
            clip_id=clip_id,
            variant_id=variant_id,
            account_id=account.id,
            platform=account.platform,
        )
    await EventsRepo(db).emit(
        type="clip.publish_requested",
        payload={"clip_id": clip_id, "variant_id": variant_id, "n_jobs": len(accounts)},
    )
    return RedirectResponse(url=f"/dashboard/clips/{clip_id}", status_code=303)


# ---------- Personas ----------


@router.get("/personas", response_class=HTMLResponse)
async def personas_list(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    personas = await PersonasRepo(db).list_for_tenant()
    return templates.TemplateResponse(request, "personas.html", {"personas": personas})


@router.post("/personas", dependencies=[Depends(require_full_scope)])
async def personas_create(
    request: Request,
    id: str = Form(...),
    name: str = Form(...),
    primary_language: str = Form(...),
    voice_prompt: str = Form(...),
    target_languages: str = Form(""),
    routing_tags: str = Form(""),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    repo = PersonasRepo(db)
    if await repo.get(id) is not None:
        raise HTTPException(status_code=409, detail=f"persona {id!r} already exists")
    await repo.create(
        persona_id=id,
        name=name,
        primary_language=primary_language,
        voice_prompt=voice_prompt,
        target_languages=_split_csv(target_languages),
        routing_tags=_split_csv(routing_tags),
    )
    return RedirectResponse(url="/dashboard/personas", status_code=303)


@router.get("/personas/{persona_id}/edit", response_class=HTMLResponse)
async def persona_edit_form(
    request: Request,
    persona_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    persona = await PersonasRepo(db).get(persona_id)
    if persona is None:
        raise HTTPException(status_code=404, detail="persona not found")
    return templates.TemplateResponse(request, "persona_edit.html", {"persona": persona})


@router.post(
    "/personas/{persona_id}",
    dependencies=[Depends(require_full_scope)],
)
async def persona_edit_submit(
    request: Request,
    persona_id: str,
    name: str = Form(...),
    primary_language: str = Form(...),
    voice_prompt: str = Form(...),
    target_languages: str = Form(""),
    routing_tags: str = Form(""),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    repo = PersonasRepo(db)
    if await repo.get(persona_id) is None:
        raise HTTPException(status_code=404, detail="persona not found")
    await repo.upsert(
        persona_id=persona_id,
        name=name,
        primary_language=primary_language,
        voice_prompt=voice_prompt,
        target_languages=_split_csv(target_languages),
        routing_tags=_split_csv(routing_tags),
    )
    return RedirectResponse(url="/dashboard/personas", status_code=303)


# ---------- Connected accounts ----------


@router.get("/connected-accounts", response_class=HTMLResponse)
async def accounts_list(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    accounts = await ConnectedAccountsRepo(db).list_for_tenant()
    return templates.TemplateResponse(
        request, "connected_accounts.html", {"accounts": accounts}
    )


@router.post(
    "/connected-accounts",
    dependencies=[Depends(require_full_scope)],
)
async def accounts_create(
    request: Request,
    platform: str = Form(...),
    external_id: str = Form(...),
    display_name: str = Form(""),
    access_token: str = Form(...),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    await ConnectedAccountsRepo(db).create(
        platform=platform,
        external_id=external_id,
        display_name=display_name or None,
        oauth_blob={"access_token": access_token},
    )
    return RedirectResponse(url="/dashboard/connected-accounts", status_code=303)


# ---------- LLM ----------


@router.get("/llm-calls", response_class=HTMLResponse)
async def llm_calls_view(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    calls = await LLMCallsRepo(db).list_for_tenant(limit=200)
    total_usd = sum(c.cost_usd_micros for c in calls) / 1_000_000.0
    return templates.TemplateResponse(
        request, "llm_calls.html", {"calls": calls, "total_usd": total_usd}
    )


@router.get("/settings/llm", response_class=HTMLResponse)
async def llm_settings_view(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
) -> Response:
    from nexoclip.llm.config import load_llm_config

    cfg = load_llm_config()
    return templates.TemplateResponse(
        request,
        "llm_settings.html",
        {"providers": cfg.providers, "routing": cfg.routing},
    )
