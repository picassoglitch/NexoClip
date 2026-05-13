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
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from nexoclip.db import (
    ApiTokensRepo,
    BrandKitsRepo,
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
from nexoclip.db.models import CustomTriggerPhrases
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


async def _merged_personas(db: Database) -> list[object]:
    """Return personas the user can pick from: DB rows union YAML defaults.

    The pipeline only persists a persona to the DB during the variants step
    (so a freshly-uploaded stream whose pipeline hasn't reached step 5 yet
    will see an empty PersonasRepo, even though `config/personas.yaml`
    defines them). Surface both sources here so the dashboard never shows
    an empty dropdown when the YAML file has options. DB rows win on id
    conflict — they reflect any in-dashboard edits.
    """
    from nexoclip.variants import load_personas

    db_personas = await PersonasRepo(db).list_for_tenant()
    seen_ids = {p.id for p in db_personas}
    out: list[object] = list(db_personas)
    try:
        yaml_personas = load_personas()
    except Exception:
        yaml_personas = {}
    for pid, p in yaml_personas.items():
        if pid in seen_ids:
            continue
        # Pico's <select> just needs `.id`, `.name`, `.primary_language`. The
        # YAML Persona has all three with the same names — pass-through is fine.
        out.append(p)
    return out


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
    from nexoclip.ingest import is_ffmpeg_available

    streams = await StreamsRepo(db).list_for_tenant()
    personas = await _merged_personas(db)
    return templates.TemplateResponse(
        request,
        "streams_list.html",
        {
            "tenant_id": tenant_id,
            "streams": streams,
            "personas": personas,
            "ffmpeg_ok": is_ffmpeg_available(),
        },
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


@router.post("/streams/upload", dependencies=[Depends(require_full_scope)])
async def streams_upload(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    persona_id: str = Form(...),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Ingest an operator-uploaded video file. The simpler path that sidesteps
    yt-dlp / Kick auth entirely — the operator drops a recorded VOD on the
    page and the pipeline runs against it.
    """
    from nexoclip.api.routers.streams import _stash_upload_to_tmp
    from nexoclip.db.adapters import stream_to_row
    from nexoclip.ingest import ingest_uploaded, is_ffmpeg_available
    from nexoclip.settings import get_settings

    if not is_ffmpeg_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "ffmpeg is not installed on the server. On Windows: "
                "`winget install --id=Gyan.FFmpeg -e` then reopen PowerShell "
                "and restart the dashboard."
            ),
        )

    output_dir = Path(get_settings().default_output_dir)
    tmp_path = await _stash_upload_to_tmp(file, output_dir)
    try:
        try:
            stream = await ingest_uploaded(
                tenant_id=tenant_id,
                source_path=tmp_path,
                output_dir=output_dir,
                title=file.filename,
            )
        except NexoClipError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

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


@router.get("/streams/{stream_id}/progress", response_class=HTMLResponse)
async def stream_progress(
    request: Request,
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """HTMX poll target — renders the live pipeline-progress card.

    Reads `pipeline.step.{start,done,failed}` events for this stream and
    surfaces what's running right now plus what's already finished. The
    parent stream-detail page polls this every 3s while the pipeline is
    in flight.
    """
    from nexoclip.db import EventsRepo

    stream = await StreamsRepo(db).get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="stream not found")

    candidates = await CandidatesRepo(db).list_for_stream(stream_id)
    clips = await ClipsRepo(db).list_for_stream(stream_id)

    # Pull every step event we've written for any stream in this tenant
    # (the events table is per-tenant; we filter by stream_id in the
    # payload). The volume is small — six steps × N streams.
    all_events = await EventsRepo(db).list_for_tenant(limit=500)
    step_events = [
        e
        for e in all_events
        if e.type.startswith("pipeline.step.")
        and e.payload.get("stream_id") == stream_id
    ]

    # Roll up by step name -> latest known status. Step order is fixed.
    step_order = [
        "ingest",
        "analyze_video",
        "diarize",
        "transcribe",
        "detect",
        "cut",
        "variants",
    ]
    step_state: dict[str, dict[str, object]] = {
        name: {
            "name": name,
            "status": "pending",
            "duration_s": None,
            "elapsed_s": None,
            "error": None,
        }
        for name in step_order
    }
    # Walk events oldest -> newest so the latest status wins.
    for ev in sorted(step_events, key=lambda e: e.ts):
        step_name = str(ev.payload.get("step", ""))
        if step_name not in step_state:
            continue
        if ev.type == "pipeline.step.start":
            step_state[step_name]["status"] = "running"
            step_state[step_name]["started_at"] = ev.ts
        elif ev.type == "pipeline.step.done":
            step_state[step_name]["status"] = "done"
            step_state[step_name]["duration_s"] = ev.payload.get("duration_s")
        elif ev.type == "pipeline.step.failed":
            step_state[step_name]["status"] = "failed"
            step_state[step_name]["error"] = ev.payload.get("error")
            step_state[step_name]["duration_s"] = ev.payload.get("duration_s")

    # Compute elapsed-time-so-far for the currently-running step. Gives the
    # user a "yes, this is taking a while" signal instead of an indeterminate
    # spinner that says nothing about progress.
    #
    # If elapsed exceeds ABANDONED_THRESHOLD_S the step is reclassified as
    # 'abandoned' — almost always means the dashboard was restarted while
    # the background task was in-flight (Ctrl+C'd, crashed, etc.). The
    # actual pipeline thread is dead and the run will never finish on its
    # own. Show that explicitly so the user knows to click 'Run pipeline'
    # rather than wait forever on a fake 'running' state.
    import datetime as _dt

    ABANDONED_THRESHOLD_S = 60 * 60  # 1 hour — generous even for hour+ VODs
    now = _dt.datetime.now(_dt.UTC)
    for s in step_state.values():
        if s["status"] == "running" and "started_at" in s:
            try:
                started = _dt.datetime.fromisoformat(str(s["started_at"]))
                elapsed = (now - started).total_seconds()
                s["elapsed_s"] = elapsed
                if elapsed > ABANDONED_THRESHOLD_S:
                    s["status"] = "abandoned"
            except ValueError:
                pass

    steps = [step_state[n] for n in step_order]
    is_running = any(s["status"] == "running" for s in steps) or all(
        s["status"] == "pending" for s in steps
    )
    is_done = all(s["status"] == "done" for s in steps)
    has_failed = any(s["status"] == "failed" for s in steps)
    is_abandoned = any(s["status"] == "abandoned" for s in steps)

    return templates.TemplateResponse(
        request,
        "_stream_progress.html",
        {
            "stream": stream,
            "steps": steps,
            "is_running": is_running,
            "is_done": is_done,
            "has_failed": has_failed,
            "is_abandoned": is_abandoned,
            "candidate_count": len(candidates),
            "clip_count": len(clips),
        },
    )


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
    personas = await _merged_personas(db)
    return templates.TemplateResponse(
        request,
        "stream_detail.html",
        {
            "stream": stream,
            "candidates": candidates,
            "clips": clips,
            "personas": personas,
        },
    )


@router.post(
    "/streams/{stream_id}/rerun",
    dependencies=[Depends(require_full_scope)],
)
async def streams_rerun(
    request: Request,
    stream_id: str,
    background_tasks: BackgroundTasks,
    persona_id: str = Form(...),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Re-trigger the pipeline for an already-ingested stream.

    Useful when (a) the original background task died with a server restart,
    (b) the user uploaded before the step-event tracking landed, or (c) they
    just want to rebuild clips after editing config. Each pipeline step is
    idempotent on its own output, so this is safe to call repeatedly.
    """
    from nexoclip.ingest import load_stream
    from nexoclip.settings import get_settings
    from nexoclip.variants import load_personas

    output_dir = Path(get_settings().default_output_dir)
    stream_row = await StreamsRepo(db).get(stream_id)
    if stream_row is None:
        raise HTTPException(status_code=404, detail="stream not found")

    # If the picked persona is YAML-only, upsert it into the DB now so the
    # variants step's FK constraint to `personas` holds. (The pipeline does
    # this itself at step 5, but we want the persona row to exist before
    # then so the dashboard's persona pages see it consistently.)
    db_personas = {p.id: p for p in await PersonasRepo(db).list_for_tenant()}
    if persona_id not in db_personas:
        try:
            yaml_personas = load_personas()
        except Exception:
            yaml_personas = {}
        if persona_id not in yaml_personas:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unknown persona {persona_id!r}; not in DB or "
                    f"config/personas.yaml"
                ),
            )
        p = yaml_personas[persona_id]
        await PersonasRepo(db).upsert(
            persona_id=p.id,
            name=p.name,
            primary_language=p.primary_language,
            target_languages=p.target_languages,
            voice_prompt=p.voice_prompt,
            routing_tags=p.routing_tags,
        )
    # Load the on-disk Stream model so PipelineKickoff can carry its full
    # metadata (the in-DB row is a different shape).
    try:
        stream = load_stream(output_dir / stream_id)
    except Exception as e:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Stream {stream_id!r} has no on-disk artifacts at "
                f"{output_dir / stream_id} — can't resume. Re-upload the file."
            ),
        ) from e

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
    await EventsRepo(db).emit(
        type="stream.rerun_requested", payload={"stream_id": stream_id}
    )
    return RedirectResponse(url=f"/dashboard/streams/{stream_id}", status_code=303)


# ---------- Clips ----------


@router.get("/clips/{clip_id}", response_class=HTMLResponse)
async def clip_detail(
    request: Request,
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    from nexoclip.clip import clip_breakdown
    from nexoclip.db import PublishJobsRepo, PublishMetricsRepo

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    variants = await VariantsRepo(db).list_for_clip(clip_id)
    accounts = await ConnectedAccountsRepo(db).list_for_tenant()
    valid_transitions = sorted(_VALID_STATUS_TRANSITIONS.get(clip.status, set()))
    breakdown = await clip_breakdown(db, clip_id)

    # Phase 3: surface engagement outcomes per published job. One row per
    # publish_job with the latest metric reading next to the platform +
    # external URL. Dashboard shows "not yet measured" rows for jobs that
    # the metrics worker hasn't touched yet.
    publish_jobs = await PublishJobsRepo(db).list_for_clip(clip_id)
    metrics_repo = PublishMetricsRepo(db)
    outcomes: list[dict[str, object]] = []
    for job in publish_jobs:
        latest = await metrics_repo.latest_for_job(job.id)
        outcomes.append(
            {
                "job": job,
                "metric": latest,
            }
        )

    return templates.TemplateResponse(
        request,
        "clip_detail.html",
        {
            "clip": clip,
            "variants": variants,
            "accounts": accounts,
            "valid_transitions": valid_transitions,
            "breakdown": breakdown,
            "outcomes": outcomes,
        },
    )


@router.get("/clips/{clip_id}/media")
async def clip_media(
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> FileResponse:
    """Stream the cut MP4 for inline <video> playback on the clip detail page.

    Returns 404 if the clip row is missing or the on-disk file disappeared
    (e.g., out/ was nuked between runs). Tenant-bound so one tenant can't
    fetch another's clip even by guessing the id.
    """
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    clip_path = Path(clip.path)
    if not clip_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"clip file missing from disk: {clip_path}",
        )
    return FileResponse(
        path=clip_path,
        media_type="video/mp4",
        # Don't set Content-Disposition: attachment — we want inline playback.
        filename=clip_path.name,
    )


@router.get("/streams/{stream_id}/source")
async def stream_source(
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> FileResponse:
    """Serve the original uploaded/downloaded source video for preview."""
    stream = await StreamsRepo(db).get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="stream not found")
    src = Path(stream.source_video_path)
    if not src.exists():
        raise HTTPException(
            status_code=404,
            detail=f"source video missing from disk: {src}",
        )
    return FileResponse(
        path=src,
        media_type="video/mp4",
        filename=src.name,
    )


@router.get("/calibration", response_class=HTMLResponse)
async def calibration_view(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Per-platform rescore-vs-views calibration table.

    Pearson r over the paired (rescore_score, views) values for each
    platform's last 30 days. Surfaces the data the operator needs to
    decide whether the scoring system has earned the right to auto-
    publish (see PHASE_3.md hard rules).
    """
    from nexoclip.metrics import compute_calibration

    reports = []
    for platform in ("youtube", "tiktok", "instagram", "buffer"):
        reports.append(await compute_calibration(db, platform=platform))
    return templates.TemplateResponse(
        request,
        "calibration.html",
        {"reports": reports},
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
    from nexoclip.cost import compute_cost_projection

    calls = await LLMCallsRepo(db).list_for_tenant(limit=200)
    projection = await compute_cost_projection(db)
    total_usd = sum(c.cost_usd_micros for c in calls) / 1_000_000.0
    return templates.TemplateResponse(
        request,
        "llm_calls.html",
        {"calls": calls, "total_usd": total_usd, "projection": projection},
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


# ---------- Brand kits (voice-markers spec slice C.3) ----------


def _parse_phrase_list(value: str) -> list[str]:
    """Split a textarea value into a phrase list (one per line, trimmed)."""
    return [line.strip() for line in value.splitlines() if line.strip()]


@router.get("/brand-kits", response_class=HTMLResponse)
async def brand_kits_list(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    kits = await BrandKitsRepo(db).list_for_tenant()
    return templates.TemplateResponse(
        request,
        "brand_kits.html",
        {"kits": kits},
    )


@router.get("/brand-kits/new", response_class=HTMLResponse)
async def brand_kits_new_form(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
) -> Response:
    return templates.TemplateResponse(
        request,
        "brand_kit_edit.html",
        {"kit": None},
    )


@router.post("/brand-kits", dependencies=[Depends(require_full_scope)])
async def brand_kits_create(
    request: Request,
    name: str = Form(...),
    primary_color: str = Form(...),
    accent_color: str = Form(...),
    text_color: str = Form("#FFFFFF"),
    font_family: str = Form("Inter"),
    font_weight: int = Form(800),
    default_layout: str = Form("pip"),
    is_default: str = Form(""),
    handle_tiktok: str = Form(""),
    handle_youtube: str = Form(""),
    handle_instagram: str = Form(""),
    handle_kick: str = Form(""),
    auto_publish_enabled: str = Form(""),
    auto_publish_platforms: str = Form(""),
    auto_publish_delay_min: int = Form(60),
    forward_phrases: str = Form(""),
    retroactive_phrases: str = Form(""),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Create a brand kit. Phrases come in as newline-separated textareas;
    color pickers and checkboxes ride the standard HTML form contract."""
    kit = await BrandKitsRepo(db).create(
        name=name,
        primary_color=primary_color,
        accent_color=accent_color,
        text_color=text_color or "#FFFFFF",
        font_family=font_family or "Inter",
        font_weight=int(font_weight) if font_weight else 800,
        default_layout=default_layout or "pip",
        is_default=bool(is_default),
        handle_tiktok=handle_tiktok or None,
        handle_youtube=handle_youtube or None,
        handle_instagram=handle_instagram or None,
        handle_kick=handle_kick or None,
        auto_publish_enabled=bool(auto_publish_enabled),
        auto_publish_platforms=_split_csv(auto_publish_platforms),
        auto_publish_delay_min=int(auto_publish_delay_min) if auto_publish_delay_min else 60,
        custom_trigger_phrases=CustomTriggerPhrases(
            forward=_parse_phrase_list(forward_phrases),
            retroactive=_parse_phrase_list(retroactive_phrases),
        ),
    )
    return RedirectResponse(url=f"/dashboard/brand-kits/{kit.id}", status_code=303)


@router.get("/brand-kits/{kit_id}", response_class=HTMLResponse)
async def brand_kits_edit_form(
    request: Request,
    kit_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    kit = await BrandKitsRepo(db).get(kit_id)
    if kit is None:
        raise HTTPException(status_code=404, detail="brand kit not found")
    return templates.TemplateResponse(
        request,
        "brand_kit_edit.html",
        {"kit": kit},
    )


@router.post(
    "/brand-kits/{kit_id}",
    dependencies=[Depends(require_full_scope)],
)
async def brand_kits_edit_submit(
    request: Request,
    kit_id: str,
    name: str = Form(...),
    primary_color: str = Form(...),
    accent_color: str = Form(...),
    text_color: str = Form("#FFFFFF"),
    font_family: str = Form("Inter"),
    font_weight: int = Form(800),
    default_layout: str = Form("pip"),
    auto_publish_enabled: str = Form(""),
    auto_publish_platforms: str = Form(""),
    auto_publish_delay_min: int = Form(60),
    forward_phrases: str = Form(""),
    retroactive_phrases: str = Form(""),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    repo = BrandKitsRepo(db)
    if await repo.get(kit_id) is None:
        raise HTTPException(status_code=404, detail="brand kit not found")
    await repo.update(
        kit_id,
        name=name,
        primary_color=primary_color,
        accent_color=accent_color,
        text_color=text_color,
        font_family=font_family,
        font_weight=int(font_weight),
        default_layout=default_layout,
        auto_publish_enabled=bool(auto_publish_enabled),
        auto_publish_platforms=_split_csv(auto_publish_platforms),
        auto_publish_delay_min=int(auto_publish_delay_min),
        custom_trigger_phrases=CustomTriggerPhrases(
            forward=_parse_phrase_list(forward_phrases),
            retroactive=_parse_phrase_list(retroactive_phrases),
        ),
    )
    return RedirectResponse(url=f"/dashboard/brand-kits/{kit_id}", status_code=303)


@router.post(
    "/brand-kits/{kit_id}/default",
    dependencies=[Depends(require_full_scope)],
)
async def brand_kits_set_default(
    request: Request,
    kit_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    try:
        await BrandKitsRepo(db).set_default(kit_id)
    except NexoClipError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return RedirectResponse(url="/dashboard/brand-kits", status_code=303)


@router.post(
    "/brand-kits/{kit_id}/delete",
    dependencies=[Depends(require_full_scope)],
)
async def brand_kits_delete(
    request: Request,
    kit_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    if await BrandKitsRepo(db).get(kit_id) is None:
        raise HTTPException(status_code=404, detail="brand kit not found")
    await BrandKitsRepo(db).delete(kit_id)
    return RedirectResponse(url="/dashboard/brand-kits", status_code=303)
