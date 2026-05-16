"""HTMX-rendered dashboard.

Server-rendered Jinja2 + HTMX. Same auth path as the JSON API but reads
the token from the `nexoclip_token` cookie (set by `POST /dashboard/login`)
in addition to the `Authorization` header. The bearer middleware in
`auth.py` does the cookie fallback - this router just renders the HTML.
"""

from __future__ import annotations

from collections.abc import Mapping
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
            # Slice F.7-G — surface "skipped" + a one-line reason so
            # near-zero durations (cached ingest, disabled visual,
            # missing HF_TOKEN diarization) don't render as the
            # confusing "0.0s" that the user thought meant "broken".
            "skipped": False,
            "note": "",
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
            step_state[step_name]["skipped"] = bool(ev.payload.get("skipped"))
            step_state[step_name]["note"] = str(ev.payload.get("note") or "")
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

    # Slice F.7-G — pass the persona's primary_language into the runner
    # so Whisper transcribes in the right language. The pipeline default
    # used to be a hardcoded "es" fallback, which silently produced
    # garbage transcripts on English/Portuguese/etc clips. Now: the
    # persona's language wins; if none, faster-whisper auto-detects.
    persona_language: str | None = None
    persona_row = await PersonasRepo(db).get(persona_id)
    if persona_row is not None and persona_row.primary_language:
        persona_language = persona_row.primary_language

    runner = request.app.state.pipeline_runner
    background_tasks.add_task(
        runner,
        PipelineKickoff(
            tenant_id=tenant_id,
            stream=stream,
            persona_id=persona_id,
            output_dir=output_dir,
            language=persona_language,
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
    from nexoclip.branding import (
        caption_style_or_default,
        preset_choices,
        resolve_brand_kit_for_candidate,
    )
    from nexoclip.clip import clip_breakdown, compute_ai_scores
    from nexoclip.db import (
        CandidatesRepo,
        PublishJobsRepo,
        PublishMetricsRepo,
    )

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    variants = await VariantsRepo(db).list_for_clip(clip_id)
    accounts = await ConnectedAccountsRepo(db).list_for_tenant()
    valid_transitions = sorted(_VALID_STATUS_TRANSITIONS.get(clip.status, set()))
    breakdown = await clip_breakdown(db, clip_id)
    ai_scores = compute_ai_scores(breakdown)

    # Resolve the brand kit + caption style so the editor can render
    # the live preview against the right defaults when overlay_config
    # is empty (or partially populated).
    speaker_label: str | None = None
    if clip.candidate_id:
        for cand in await CandidatesRepo(db).list_for_stream(clip.stream_id):
            if cand.id == clip.candidate_id:
                ev = cand.evidence or {}
                lbl = ev.get("speaker_label")
                if isinstance(lbl, str):
                    speaker_label = lbl
                break
    brand_kit = await resolve_brand_kit_for_candidate(
        db, stream_id=clip.stream_id, speaker_label=speaker_label
    )
    caption_style = caption_style_or_default(
        brand_kit.caption_style if brand_kit is not None else None
    )

    # Slice F.7-G — branding stickiness across clips.
    # When THIS clip has no overlay_config yet, prefill from the most-
    # recently-edited sibling clip on the same stream. The operator
    # types "kick.com/aldovillanueva" on clip 1, opens clip 2, and
    # the URL is already filled in. The brand_kit covers the
    # cross-stream case (different VODs from the same channel); this
    # covers the in-stream case where the brand_kit hasn't been
    # populated yet (e.g. first-time user, didn't visit Brand Kits).
    inherited_overlay: dict[str, object] | None = None
    if not clip.overlay_config:
        siblings = await ClipsRepo(db).list_for_stream(clip.stream_id)
        # Walk newest-first by created_at; .list_for_stream returns in
        # insertion order — sort defensively.
        siblings_sorted = sorted(
            (c for c in siblings if c.id != clip.id and c.overlay_config),
            key=lambda c: c.created_at,
            reverse=True,
        )
        if siblings_sorted:
            inherited = siblings_sorted[0].overlay_config
            if isinstance(inherited, dict):
                inherited_overlay = inherited

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
            "ai_scores": ai_scores,
            "outcomes": outcomes,
            "brand_kit": brand_kit,
            "caption_style": caption_style,
            "caption_preset_choices": preset_choices(),
            # Slice F.7-G — passed through so the template's form-
            # prefill block can fall back to the previous clip's
            # overlay when this clip's overlay_config is still null.
            "inherited_overlay": inherited_overlay,
        },
    )


# ---- Clip overlay editor (slice F.6) ----


def _parse_overlay_form(
    *,
    title_text: str,
    banner_enabled: str,
    banner_platform: str,
    banner_url: str,
    banner_color: str,
    banner_show_context: str = "",
    banner_show_safezones: str = "",
    captions_enabled: str,
    captions_preset: str,
    captions_highlight_color: str,
    captions_position: str = "",
    captions_font_size: str = "",
    captions_animation: str = "",
    captions_lead_ms: int = 120,
    comments_show: str,
    comments_fake_likes: int,
) -> dict[str, object]:
    """Coerce the editor form's flat key/value submission into the
    nested overlay_config shape the renderer reads.

    Empty / blank string fields collapse to None at the top level so
    the renderer falls back to brand-kit defaults — i.e. the editor
    is additive, not destructive."""

    def _bool(v: str) -> bool:
        return v.lower() in ("1", "true", "on", "yes")

    return {
        "title_text": title_text.strip() or None,
        "banner": {
            "enabled": _bool(banner_enabled),
            "platform": (banner_platform or "kick").strip().lower(),
            "url": banner_url.strip() or None,
            "color": banner_color.strip() or None,
            # Slice F.7 social-context overlay toggle. Decorative
            # only — the renderer doesn't burn the LIVE badge / chat
            # into the MP4. The HTML preview shows them so the
            # operator can see what the published clip looks like
            # inside the platform UI.
            "show_context": _bool(banner_show_context),
            # Slice F.7-F platform safe-zone guides. Editor-only —
            # never affects the burned MP4. Tells the operator which
            # regions of the frame get covered by TikTok/Reels chrome.
            "show_safezones": _bool(banner_show_safezones),
        },
        "captions": {
            "enabled": _bool(captions_enabled),
            "preset": captions_preset.strip() or None,
            "highlight_color": captions_highlight_color.strip() or None,
            # Slice F.7-H — read-ahead + visual knobs. Each is None when
            # blank so the renderer/template falls back to the brand-kit
            # caption_style defaults (no overlapping sources of truth).
            "position": (captions_position.strip().lower() or None)
                if captions_position in (
                    "upper_third", "centered", "lower_third", "bottom"
                ) else None,
            "font_size": (captions_font_size.strip().lower() or None)
                if captions_font_size in ("small", "medium", "large", "xl") else None,
            "animation": (captions_animation.strip().lower() or None)
                if captions_animation in ("pop", "slide", "typewriter", "fade") else None,
            # Clamp 0..500ms — beyond 500ms the read-ahead is so far
            # ahead it stops being "anticipation" and starts being
            # "spoiler"; pre-roll captions are no longer in sync.
            "lead_ms": max(0, min(500, int(captions_lead_ms))),
        },
        "comments": {
            "show_overlay": _bool(comments_show),
            "fake_likes": max(0, int(comments_fake_likes)),
        },
    }


# ---- Branding persistence helpers (slice F.7-G) ----
#
# On every clip-overlay save the operator's branding choices (platform
# handle, banner color, caption preset + highlight color) get mirrored
# back into the tenant's default brand_kit. The next clip the operator
# opens then prefills from the brand_kit fallback path in clip_detail.html.
#
# Why a *brand-kit* write rather than a sticky-defaults table:
#   - the kit ALREADY drives the renderer's fallback path when
#     overlay_config_json is null (see clip_detail.html ~232).
#     mirroring there means there's a single source of truth.
#   - operators editing brand kits in /dashboard/brand_kits see the
#     latest "live" handle / color without an extra migration screen.
#
# We auto-create a brand_kit with `name="Default"` and `is_default=1`
# when none exists, so first-time users don't need a kit-create UI
# detour before their second clip inherits their first clip's URL.


def _platform_handle_field(platform: str) -> str | None:
    """Map the dropdown platform → brand_kit handle column. Returns
    `None` for platforms we don't have a column for (so the caller
    doesn't fabricate a write for an unknown column)."""
    return {
        "kick": "handle_kick",
        "tiktok": "handle_tiktok",
        "youtube": "handle_youtube",
        "instagram": "handle_instagram",
    }.get(platform.lower())


async def _persist_branding_to_brand_kit(
    db: Database,
    cfg: Mapping[str, object],
) -> None:
    """Mirror the operator's editor choices back to the tenant's
    default brand_kit. Best-effort — a missing field or repo failure
    must not break the overlay save (publishers don't depend on this).
    """
    from nexoclip.branding import caption_style_or_default

    repo = BrandKitsRepo(db)
    try:
        kit = await repo.get_default()
    except Exception:  # noqa: BLE001 — best-effort
        return

    banner = cfg.get("banner") or {}
    captions = cfg.get("captions") or {}
    if not isinstance(banner, dict) or not isinstance(captions, dict):
        return

    platform_raw = banner.get("platform")
    platform = str(platform_raw).lower() if isinstance(platform_raw, str) else ""
    url_raw = banner.get("url")
    url = str(url_raw).strip() if isinstance(url_raw, str) else ""
    color_raw = banner.get("color")
    color = str(color_raw).strip() if isinstance(color_raw, str) else ""
    preset_raw = captions.get("preset")
    preset = str(preset_raw).strip() if isinstance(preset_raw, str) else ""
    hilite_raw = captions.get("highlight_color")
    hilite = str(hilite_raw).strip() if isinstance(hilite_raw, str) else ""

    # Build the caption_style dict once (used by both create and update
    # branches). None when neither preset nor highlight color changed.
    caption_style_patch: dict[str, object] | None = None
    if preset or hilite:
        base = (
            caption_style_or_default(kit.caption_style).model_dump()
            if kit is not None
            else caption_style_or_default(None).model_dump()
        )
        if preset:
            base["preset_id"] = preset
        if hilite:
            base["highlight_color"] = hilite
        caption_style_patch = base

    handle_field = _platform_handle_field(platform)
    handle_kick = url if handle_field == "handle_kick" and url else None
    handle_tiktok = url if handle_field == "handle_tiktok" and url else None
    handle_youtube = url if handle_field == "handle_youtube" and url else None
    handle_instagram = url if handle_field == "handle_instagram" and url else None

    if kit is None:
        # Auto-create the tenant's default kit on first save so
        # subsequent clips inherit the chosen branding even if the
        # operator never visits /dashboard/brand_kits.
        try:
            await repo.create(
                name="Default",
                primary_color=color or "#53FC18",
                accent_color="#FFD700",
                is_default=True,
                caption_style=caption_style_patch,
                handle_kick=handle_kick,
                handle_tiktok=handle_tiktok,
                handle_youtube=handle_youtube,
                handle_instagram=handle_instagram,
            )
        except Exception:  # noqa: BLE001
            pass
        return

    # Existing default kit — partial update of just the fields the
    # operator changed in the editor. None-valued args are ignored by
    # BrandKitsRepo.update so we only touch what was provided.
    if not any([
        color, handle_kick, handle_tiktok, handle_youtube, handle_instagram,
        caption_style_patch,
    ]):
        return
    try:
        await repo.update(
            kit.id,
            primary_color=color or None,
            handle_kick=handle_kick,
            handle_tiktok=handle_tiktok,
            handle_youtube=handle_youtube,
            handle_instagram=handle_instagram,
            caption_style=caption_style_patch,
        )
    except Exception:  # noqa: BLE001
        pass


@router.post(
    "/clips/{clip_id}/overlay",
    dependencies=[Depends(require_full_scope)],
)
async def clip_overlay_save(
    request: Request,
    clip_id: str,
    title_text: str = Form(""),
    banner_enabled: str = Form(""),
    banner_platform: str = Form("kick"),
    banner_url: str = Form(""),
    banner_color: str = Form(""),
    banner_show_context: str = Form(""),
    banner_show_safezones: str = Form(""),
    captions_enabled: str = Form(""),
    captions_preset: str = Form(""),
    captions_highlight_color: str = Form(""),
    captions_position: str = Form(""),
    captions_font_size: str = Form(""),
    captions_animation: str = Form(""),
    captions_lead_ms: int = Form(120),
    comments_show: str = Form(""),
    comments_fake_likes: int = Form(0),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Save (only) the per-clip overlay config. Idempotent — re-POSTing
    overwrites. Used by the "Save draft" button on the clip editor."""
    if await ClipsRepo(db).get(clip_id) is None:
        raise HTTPException(status_code=404, detail="clip not found")
    cfg = _parse_overlay_form(
        title_text=title_text,
        banner_enabled=banner_enabled,
        banner_platform=banner_platform,
        banner_url=banner_url,
        banner_color=banner_color,
        banner_show_context=banner_show_context,
        banner_show_safezones=banner_show_safezones,
        captions_enabled=captions_enabled,
        captions_preset=captions_preset,
        captions_highlight_color=captions_highlight_color,
        captions_position=captions_position,
        captions_font_size=captions_font_size,
        captions_animation=captions_animation,
        captions_lead_ms=captions_lead_ms,
        comments_show=comments_show,
        comments_fake_likes=comments_fake_likes,
    )
    await ClipsRepo(db).set_overlay_config(clip_id, overlay_config=cfg)
    # Slice F.7-G — mirror branding choices to the tenant brand_kit
    # so the operator doesn't re-type URL / color on every clip.
    await _persist_branding_to_brand_kit(db, cfg)
    return RedirectResponse(url=f"/dashboard/clips/{clip_id}", status_code=303)


@router.post(
    "/clips/{clip_id}/finalize",
    dependencies=[Depends(require_full_scope)],
)
async def clip_overlay_finalize(
    request: Request,
    clip_id: str,
    title_text: str = Form(""),
    banner_enabled: str = Form(""),
    banner_platform: str = Form("kick"),
    banner_url: str = Form(""),
    banner_color: str = Form(""),
    banner_show_context: str = Form(""),
    banner_show_safezones: str = Form(""),
    captions_enabled: str = Form(""),
    captions_preset: str = Form(""),
    captions_highlight_color: str = Form(""),
    captions_position: str = Form(""),
    captions_font_size: str = Form(""),
    captions_animation: str = Form(""),
    captions_lead_ms: int = Form(120),
    comments_show: str = Form(""),
    comments_fake_likes: int = Form(0),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """The 'Complete' button — saves the overlay config AND transitions
    the clip from `cut` / `ready_for_review` → `approved` (the existing
    pre-publish standby state).

    Rejects when the current status doesn't allow the transition (e.g.
    a `published` clip). The dashboard hides the button in those cases
    so the 409 only fires on a stale page."""
    repo = ClipsRepo(db)
    clip = await repo.get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")

    allowed = _VALID_STATUS_TRANSITIONS.get(clip.status, set())
    # If we're not already at `approved` and can't transition there
    # directly, walk one step (`cut` -> `ready_for_review` -> `approved`).
    if clip.status == "approved" or "approved" in allowed:
        target = "approved"
    elif clip.status == "cut" and "ready_for_review" in allowed:
        # `cut` -> `ready_for_review` first, then `approved`.
        await repo.update_status(clip_id, status="ready_for_review")
        target = "approved"
    else:
        raise HTTPException(
            status_code=409,
            detail=f"can't finalize from status {clip.status!r}",
        )

    cfg = _parse_overlay_form(
        title_text=title_text,
        banner_enabled=banner_enabled,
        banner_platform=banner_platform,
        banner_url=banner_url,
        banner_color=banner_color,
        banner_show_context=banner_show_context,
        banner_show_safezones=banner_show_safezones,
        captions_enabled=captions_enabled,
        captions_preset=captions_preset,
        captions_highlight_color=captions_highlight_color,
        captions_position=captions_position,
        captions_font_size=captions_font_size,
        captions_animation=captions_animation,
        captions_lead_ms=captions_lead_ms,
        comments_show=comments_show,
        comments_fake_likes=comments_fake_likes,
    )
    await repo.set_overlay_config(clip_id, overlay_config=cfg)
    # Slice F.7-G — see note in clip_overlay_save: persists branding
    # to the tenant brand_kit so the *next* clip prefills it.
    await _persist_branding_to_brand_kit(db, cfg)
    if target != clip.status:
        await repo.update_status(clip_id, status=target)

    # ---- Renderer-side overlay burn-in (slice F.7-E) ----
    # Re-render the clip with title / banner / captions burned into
    # the pixels. Output goes to `<clip_dir>/clip_final.mp4` next to
    # the original. Publishers prefer the final when present and
    # fall back to the original on burn failure.
    burn_outcome = await _burn_overlays_for_clip(
        db=db, clip_id=clip_id, overlay_config=cfg
    )
    await EventsRepo(db).emit(
        type="clip.finalized",
        payload={
            "clip_id": clip_id,
            "to_status": target,
            "burn_outcome": burn_outcome,
        },
    )
    return RedirectResponse(url=f"/dashboard/clips/{clip_id}", status_code=303)


@router.post(
    "/clips/{clip_id}/reject",
    dependencies=[Depends(require_full_scope)],
)
async def clip_reject(
    request: Request,
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice F.7-H — one-click reject + close editor.

    Forces the clip into `rejected` (any valid in-bound transition is
    accepted; we don't fight the operator's intent), emits the
    `clip.rejected` event for the audit trail, and redirects the
    browser back to the stream page so the editor closes immediately.
    The stream-page clip card shows the REJECTED stamp via its
    `data-status="rejected"` overlay.
    """
    repo = ClipsRepo(db)
    clip = await repo.get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    if clip.status != "rejected":
        # Skip the strict transition map — reject from ANY state is a
        # destination, not an intermediate step. The transitions table
        # only allows `cut -> rejected` and `ready_for_review -> rejected`
        # today; we extend that for the operator's one-click flow.
        conn = await db.connect()
        await conn.execute(
            "UPDATE clips SET status = ? WHERE id = ? AND tenant_id = ?",
            ("rejected", clip_id, tenant_id),
        )
        await conn.commit()
        await EventsRepo(db).emit(
            type="clip.rejected",
            payload={"clip_id": clip_id, "from": clip.status, "to": "rejected"},
        )
    return RedirectResponse(
        url=f"/dashboard/streams/{clip.stream_id}", status_code=303
    )


async def _burn_overlays_for_clip(
    *,
    db: Database,
    clip_id: str,
    overlay_config: dict[str, object],
) -> str:
    """Run the overlay burn for one clip, returning a short outcome
    string suitable for the clip.finalized event payload:

      - "skipped_no_overlays"  — nothing was enabled (no title, no
                                 banner, captions off OR no transcript)
      - "skipped_clip_missing" — the source MP4 isn't on disk
      - "burned"               — clip_final.mp4 written successfully
      - "failed:<short reason>" — ffmpeg returned non-zero; the
                                  original clip.mp4 is untouched and
                                  publishers will fall back to it

    NEVER raises — the finalize endpoint shouldn't fail just because
    a render variant didn't pan out. The operator can re-finalize to
    retry. The full ffmpeg stderr is logged regardless.
    """
    import asyncio
    import json as _json

    import structlog

    from nexoclip.clip import burn_overlays
    from nexoclip.db import TranscriptsRepo

    log = structlog.get_logger("nexoclip.api.dashboard")

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        return "skipped_clip_missing"
    source_path = Path(clip.path)
    if not source_path.exists():
        log.warning("burn.source_missing", clip_id=clip_id, path=str(source_path))
        return "skipped_clip_missing"
    target_path = source_path.parent / "clip_final.mp4"

    transcript = await TranscriptsRepo(db).get(clip.stream_id)
    segments: list[object] = []
    if transcript is not None:
        try:
            segments = _json.loads(transcript.segments_json) or []
        except (TypeError, _json.JSONDecodeError):
            segments = []

    try:
        # ffmpeg is sync + CPU-bound; offload to a thread so the
        # event loop isn't blocked while a 30s clip re-encodes.
        burned = await asyncio.to_thread(
            burn_overlays,
            source_path=source_path,
            target_path=target_path,
            overlay_config=overlay_config,
            transcript_segments=segments,
            clip_start_s=clip.start_s,
            clip_end_s=clip.end_s,
            output_w=clip.width,
            output_h=clip.height,
        )
    except Exception as e:  # noqa: BLE001 — we want the catch-all
        log.warning("burn.failed", clip_id=clip_id, error=str(e))
        # Reason capped at 80 chars so the events table doesn't get
        # ffmpeg essays.
        reason = str(e)[:80].replace("\n", " ")
        return f"failed:{reason}"
    return "burned" if burned else "skipped_no_overlays"


@router.post(
    "/clips/{clip_id}/generate-hooks",
    dependencies=[Depends(require_full_scope)],
)
async def clip_generate_hooks(
    request: Request,
    clip_id: str,
    tone: str = Form("default"),
    n: int = Form(5),
    persona_id: str = Form(""),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Generate `n` viral-hook title candidates for a clip and return
    them as JSON for the editor's hook-picker UI to render inline.

    Driven by a tone preset (default / aggressive / gen_z / corporate /
    curious) so the operator can sweep approaches without re-typing
    the prompt. Each call is a single Anthropic request — cost-tracked
    by the LLMRouter under purpose='hook_generation'.

    Returns:
        JSON: {"hooks": [{"text": "..."}, ...], "tone": "...", "n": N}
        on success.
        502 on provider failure with the LLM error inline.
    """
    import json as _json

    from nexoclip.db import CandidatesRepo, PersonasRepo, TranscriptsRepo
    from nexoclip.llm import LLMRouter, load_llm_config
    from nexoclip.settings import get_settings
    from nexoclip.variants import generate_hooks
    from nexoclip.variants.personas import (
        get_persona as get_yaml_persona,
        load_personas as load_yaml_personas,
    )

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")

    # Pick the persona: explicit form arg > first-DB-persona > YAML fallback.
    persona_voice = ""
    persona_language = "en"
    if persona_id:
        db_persona = await PersonasRepo(db).get(persona_id)
        if db_persona is not None:
            persona_voice = db_persona.voice_prompt
            persona_language = db_persona.primary_language
        else:
            try:
                yp = get_yaml_persona(persona_id)
                persona_voice = yp.voice_prompt
                persona_language = yp.primary_language
            except Exception:
                pass
    if not persona_voice:
        # No explicit persona — grab any persona the tenant has, else
        # the first YAML persona, else generate with empty voice (the
        # service tolerates it via "(no persona voice provided)").
        db_personas = await PersonasRepo(db).list_for_tenant()
        if db_personas:
            persona_voice = db_personas[0].voice_prompt
            persona_language = db_personas[0].primary_language
        else:
            try:
                yps = list(load_yaml_personas().values())
                if yps:
                    persona_voice = yps[0].voice_prompt
                    persona_language = yps[0].primary_language
            except Exception:
                pass

    # Pull a transcript snippet — the candidate evidence is the
    # cheapest source. Falls back to scanning the candidates table for
    # this clip when the candidate row carries a longer snippet.
    snippet = ""
    if clip.candidate_id:
        for cand in await CandidatesRepo(db).list_for_stream(clip.stream_id):
            if cand.id == clip.candidate_id:
                ev = cand.evidence or {}
                snippet = (
                    str(ev.get("transcript_snippet") or ev.get("phrase") or "")
                ).strip()
                break
    if not snippet:
        # Fall back to the words from the transcript that overlap
        # the clip window. Cheap heuristic — we slice the JSON segments.
        try:
            tx = await TranscriptsRepo(db).get(clip.stream_id)
            if tx is not None:
                segs = _json.loads(tx.segments_json)
                if isinstance(segs, list):
                    pieces = []
                    for s in segs:
                        if not isinstance(s, dict):
                            continue
                        s_start = s.get("start") or 0
                        s_end = s.get("end") or 0
                        text = (s.get("text") or "").strip()
                        if text and s_end >= clip.start_s and s_start <= clip.end_s:
                            pieces.append(text)
                    snippet = " ".join(pieces).strip()[:600]
        except Exception:
            snippet = ""

    output_dir = Path(get_settings().default_output_dir)
    call_log_path = output_dir / "llm_calls_hooks.jsonl"
    router_ = LLMRouter(
        config=load_llm_config(),
        call_log_path=call_log_path,
        db=db,
    )

    # Coerce + clamp form inputs.
    tone_id = tone.strip().lower() or "default"
    if tone_id not in ("default", "aggressive", "gen_z", "corporate", "curious"):
        tone_id = "default"
    try:
        n_int = int(n)
    except (TypeError, ValueError):
        n_int = 5
    n_int = max(1, min(10, n_int))

    try:
        hooks = await generate_hooks(
            tenant_id=tenant_id,
            persona_voice=persona_voice,
            persona_language=persona_language,
            transcript_snippet=snippet,
            tone=tone_id,  # type: ignore[arg-type]
            n=n_int,
            router=router_,
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"hook generation failed: {e}",
        ) from e

    return Response(
        content=_json.dumps(
            {
                "hooks": [{"text": h.text} for h in hooks],
                "tone": tone_id,
                "n": n_int,
            }
        ),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/clips/{clip_id}/media")
async def clip_media(
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> FileResponse:
    """Stream the cut MP4 for inline <video> playback on the clip detail page.

    Prefers `clip_final.mp4` when present — that's the burned-in
    version from the overlay finalize endpoint (slice F.7-E). The
    editor's preview surface immediately reflects what publishers
    will upload after the operator clicks Ship to platforms. Falls
    back to the original `clip.mp4` when no burn has run yet.

    Returns 404 if the clip row is missing or the on-disk file disappeared
    (e.g., out/ was nuked between runs). Tenant-bound so one tenant can't
    fetch another's clip even by guessing the id.
    """
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    original = Path(clip.path)
    final = original.parent / "clip_final.mp4"
    clip_path = final if final.exists() else original
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


@router.get("/clips/{clip_id}/thumbnail")
async def clip_thumbnail(
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> FileResponse:
    """Serve the per-clip thumbnail JPEG for the inbox clip cards.

    Returns 404 when the clip row has no `thumbnail_frame_path` (the
    pipeline skipped the picker step) or when the file is gone from disk.
    """
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    if not clip.thumbnail_frame_path:
        raise HTTPException(status_code=404, detail="no thumbnail for this clip")
    thumb_path = Path(clip.thumbnail_frame_path)
    if not thumb_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"thumbnail file missing from disk: {thumb_path}",
        )
    return FileResponse(
        path=thumb_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=300"},
    )


@router.get("/clips/{clip_id}/intelligence.json")
async def clip_intelligence(
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Per-clip intelligence markers (audio peaks / scene cuts /
    laughter reactions / chat-heat spikes / face-emotion changes).

    Aggregated read-only across the existing transcripts +
    visual_signals + chat_replay surfaces. Returns a flat
    `{markers: [{kind, ts, score, label}, ...]}` JSON shape — no
    caching needed (the underlying tables are already indexed by
    stream_id and the aggregation is cheap).
    """
    import json as _json

    from nexoclip.clip import compute_intelligence

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    markers = await compute_intelligence(db, clip_id=clip_id)
    return Response(
        content=_json.dumps(
            {
                "markers": [
                    {
                        "kind": m.kind,
                        "ts": m.ts,
                        "score": m.score,
                        "label": m.label,
                    }
                    for m in markers
                ],
                "duration_s": clip.duration_s,
            }
        ),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/clips/{clip_id}/captions.json")
async def clip_captions(
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Word-level captions sliced to the clip window — slice F.7-F.

    Returns the same caption lines the renderer's ASS pass burns into
    the MP4, so the live editor preview and the published clip
    match exactly. Each line carries:

      - ts / end_ts (clip-relative seconds)
      - text (joined line)
      - words: [{ts, end_ts, text, emphasis}]
      - emphasis: strongest emphasis tag on the line

    Empty {lines: []} when the transcript has no word-level data
    overlapping the clip window — the preview gracefully renders the
    fallback placeholder copy in that case.
    """
    import json as _json

    from nexoclip.clip import captions_for_clip, lines_to_json
    from nexoclip.db import TranscriptsRepo

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    transcript = await TranscriptsRepo(db).get(clip.stream_id)

    # Slice F.7-G — diagnostics block so the editor's empty-state can
    # explain *why* no captions are showing instead of just hinting
    # "re-run the pipeline" (the user asked: "i keep getting this
    # error even though im talking in spanish"). The four real
    # failure modes we want to distinguish:
    #
    #   1. transcript hasn't been written yet  → run pipeline
    #   2. transcript is present but EMPTY     → VAD ate everything,
    #                                            or audio extract was silent
    #   3. transcript has segments but no word-level data
    #                                          → pre-F.7-F run, re-transcribe
    #   4. words exist but none overlap THIS clip window
    #                                          → clip cut outside the
    #                                            transcribed span
    #
    # The editor's JS turns the diagnostics into a one-line empty-state.
    diag: dict[str, object] = {
        "transcript_present": transcript is not None,
        "language": transcript.language if transcript else None,
        "transcript_segment_count": 0,
        "transcript_word_count": 0,
        "transcript_span_s": None,
        "clip_window_s": [clip.start_s, clip.end_s],
        "reason": "no_transcript",
    }
    if transcript is None:
        body = {"lines": [], "duration_s": clip.duration_s, "diagnostics": diag}
    else:
        # Inspect the raw transcript shape without re-running the chunker
        # so the diagnostics survive even when captions_for_clip returns [].
        try:
            raw_segments = _json.loads(transcript.segments_json or "[]")
        except _json.JSONDecodeError:
            raw_segments = []
        if isinstance(raw_segments, list):
            diag["transcript_segment_count"] = len(raw_segments)
            total_words = 0
            min_ts: float | None = None
            max_ts: float | None = None
            for seg in raw_segments:
                if not isinstance(seg, dict):
                    continue
                seg_words = seg.get("words")
                if isinstance(seg_words, list):
                    total_words += len(seg_words)
                # transcript timestamps are stream-relative ("ts"/"end_ts"
                # from Pydantic, "start"/"end" from legacy fixtures).
                start_v = seg.get("ts", seg.get("start"))
                end_v = seg.get("end_ts", seg.get("end"))
                if isinstance(start_v, int | float):
                    min_ts = float(start_v) if min_ts is None else min(min_ts, float(start_v))
                if isinstance(end_v, int | float):
                    max_ts = float(end_v) if max_ts is None else max(max_ts, float(end_v))
            diag["transcript_word_count"] = total_words
            if min_ts is not None and max_ts is not None:
                diag["transcript_span_s"] = [min_ts, max_ts]

        lines = captions_for_clip(
            transcript.segments_json or "",
            clip_start_s=clip.start_s,
            clip_end_s=clip.end_s,
        )
        # Classify the failure mode for the empty-state hint.
        if lines:
            diag["reason"] = "ok"
        elif diag["transcript_segment_count"] == 0:
            diag["reason"] = "transcript_empty"
        elif diag["transcript_word_count"] == 0:
            diag["reason"] = "no_word_timestamps"
        else:
            diag["reason"] = "window_outside_transcript"
        body = {
            "lines": lines_to_json(lines),
            "duration_s": clip.duration_s,
            "diagnostics": diag,
        }
    return Response(
        content=_json.dumps(body),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/clips/{clip_id}/waveform.json")
async def clip_waveform(
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Serve the per-clip audio waveform as a list of normalized peaks.

    Computed on first request via ffmpeg PCM extraction, then cached
    next to the clip MP4 as `waveform.json` for instant subsequent
    loads. Returns `[]` (200, not 404) when extraction fails so the
    editor's scrubber gracefully degrades to a flat baseline rather
    than spamming the console with 404s on every clip without audio.
    """
    import json as _json

    from nexoclip.clip import load_or_compute_waveform

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    clip_path = Path(clip.path)
    if not clip_path.exists():
        return Response(
            content="[]",
            media_type="application/json",
            headers={"Cache-Control": "no-store"},
        )
    peaks = load_or_compute_waveform(clip_path)
    return Response(
        content=_json.dumps(peaks),
        media_type="application/json",
        # The on-disk cache means we can be aggressive with browser
        # caching too; the file changes only when ops manually delete it.
        headers={"Cache-Control": "public, max-age=3600"},
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


@router.post(
    "/publish-jobs/{job_id}/cancel",
    dependencies=[Depends(require_full_scope)],
)
async def publish_jobs_cancel(
    request: Request,
    job_id: str,
    redirect_to: str = Form("/dashboard"),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Undo a pending auto-publish job (slice E.2).

    Returns 404 when the job doesn't exist OR belongs to another tenant
    OR is already past 'pending' (sent / failed). The dashboard hides
    the Undo button in those cases so the 404 only fires on a stale page
    or a racing request — both safe to fail loudly on.
    """
    ok = await PublishJobsRepo(db).cancel(job_id)
    if not ok:
        raise HTTPException(
            status_code=404,
            detail="publish job not found or not cancelable",
        )
    await EventsRepo(db).emit(
        type="publish_job.canceled",
        payload={"publish_job_id": job_id},
    )
    return RedirectResponse(url=redirect_to or "/dashboard", status_code=303)


# ---------- Inbox (slice E.3) ----------


@router.get("/inbox", response_class=HTMLResponse)
async def inbox(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Operator inbox: clips grouped by VOD → speaker, plus a strip of
    in-window auto-publish jobs that can still be undone.

    Designed to be the one page an operator opens each morning. The strip
    at the top is time-sensitive (auto-publish about to fire); the lists
    below are the regular review queue."""
    from nexoclip.db import CandidatesRepo, VodSpeakersRepo

    streams = await StreamsRepo(db).list_for_tenant()
    jobs_repo = PublishJobsRepo(db)

    # ---- The undo strip: every pending job whose scheduled_for hasn't
    # elapsed yet. Surface across all clips so the operator doesn't have
    # to dig through individual stream pages.
    scheduled_jobs = await jobs_repo.list_scheduled(limit=20)

    # Map clip_id -> friendly label for the strip ("Stream X · @aldo · 12:34").
    # Pull names lazily — we only need clip rows that show up in
    # `scheduled_jobs`.
    scheduled_clip_ids = list({j.clip_id for j in scheduled_jobs})
    clips_by_id: dict[str, object] = {}
    if scheduled_clip_ids:
        for cid in scheduled_clip_ids:
            row = await ClipsRepo(db).get(cid)
            if row is not None:
                clips_by_id[cid] = row

    # ---- VOD-grouped review queue.
    streams_payload: list[dict[str, object]] = []
    for s in streams:
        stream_clips = await ClipsRepo(db).list_for_stream(s.id)
        if not stream_clips:
            continue
        vod_speakers = await VodSpeakersRepo(db).list_for_stream(s.id)
        speakers_by_label = {vs.speaker_label: vs for vs in vod_speakers}
        candidates_by_id = {
            c.id: c for c in await CandidatesRepo(db).list_for_stream(s.id)
        }
        # Bucket clips by their evidence.speaker_label (None bucket
        # catches non-diarized triggers).
        buckets: dict[str | None, list[object]] = {}
        for c in stream_clips:
            label = _speaker_label_for_clip(c, candidates_by_id)
            buckets.setdefault(label, []).append(c)
        groups = []
        # Stable order: known labels alphabetically, None last.
        for label in sorted(
            (lbl for lbl in buckets if lbl is not None)
        ) + ([None] if None in buckets else []):
            vs = speakers_by_label.get(label) if label is not None else None
            groups.append(
                {
                    "speaker_label": label or "(unassigned)",
                    "vod_speaker": vs,
                    "clips": buckets[label],
                }
            )
        streams_payload.append(
            {
                "stream": s,
                "groups": groups,
                "n_clips": len(stream_clips),
            }
        )

    return templates.TemplateResponse(
        request,
        "inbox.html",
        {
            "streams_payload": streams_payload,
            "scheduled_jobs": scheduled_jobs,
            "clips_by_id": clips_by_id,
        },
    )


def _speaker_label_for_clip(
    clip: object, candidates_by_id: Mapping[str, object]
) -> str | None:
    """Same logic as `nexoclip.publish.auto._speaker_label_for_clip`,
    inlined here to avoid an api -> publish dependency edge. (Both modules
    are tiny consumers of the same convention: candidate.evidence carries
    the speaker_label.)"""
    candidate_id = getattr(clip, "candidate_id", None)
    if not candidate_id:
        return None
    cand = candidates_by_id.get(candidate_id)
    if cand is None:
        return None
    evidence = getattr(cand, "evidence", None) or {}
    label = evidence.get("speaker_label") if isinstance(evidence, dict) else None
    return label if isinstance(label, str) else None


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
    from nexoclip.branding import builtin_presets, preset_choices

    return templates.TemplateResponse(
        request,
        "brand_kit_edit.html",
        {
            "kit": None,
            "caption_style": builtin_presets()["karaoke_pop"],
            "caption_preset_choices": preset_choices(),
        },
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
    caption_preset: str = Form("karaoke_pop"),
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
    from nexoclip.branding.captions import _preset_by_id

    caption_style = _preset_by_id(caption_preset).model_dump()
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
        caption_style=caption_style,
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
    from nexoclip.branding import caption_style_or_default, preset_choices

    kit = await BrandKitsRepo(db).get(kit_id)
    if kit is None:
        raise HTTPException(status_code=404, detail="brand kit not found")
    caption_style = caption_style_or_default(kit.caption_style)
    return templates.TemplateResponse(
        request,
        "brand_kit_edit.html",
        {
            "kit": kit,
            "caption_style": caption_style,
            "caption_preset_choices": preset_choices(),
        },
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
    caption_preset: str = Form("karaoke_pop"),
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
    from nexoclip.branding.captions import _preset_by_id

    repo = BrandKitsRepo(db)
    if await repo.get(kit_id) is None:
        raise HTTPException(status_code=404, detail="brand kit not found")
    caption_style = _preset_by_id(caption_preset).model_dump()
    await repo.update(
        kit_id,
        name=name,
        primary_color=primary_color,
        accent_color=accent_color,
        text_color=text_color,
        font_family=font_family,
        font_weight=int(font_weight),
        default_layout=default_layout,
        caption_style=caption_style,
        handle_tiktok=handle_tiktok,
        handle_youtube=handle_youtube,
        handle_instagram=handle_instagram,
        handle_kick=handle_kick,
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


def _brand_kit_asset_dir(output_dir: Path, kit_id: str) -> Path:
    """`<output_dir>/brand_kits/<kit_id>/` — slice D.3 asset root.

    Mirrors the per-stream layout so a future Storage abstraction
    (docs/production_deploy.md §3) can swap to S3/R2 by remapping a
    single prefix.
    """
    return output_dir / "brand_kits" / kit_id


@router.post(
    "/brand-kits/{kit_id}/generate-logo",
    dependencies=[Depends(require_full_scope)],
)
async def brand_kits_generate_logo(
    request: Request,
    kit_id: str,
    style_hint: str = Form("Minimal mark / monogram"),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Call Claude → sanitized SVG → save under brand_kits/<kit_id>/ →
    persist the relative URL + ai_* metadata on the brand kit row.

    Errors bubble up as a 500 with the provider message; the dashboard's
    next render shows the previously-generated logo unchanged (best-effort
    semantics — the kit isn't mutated until both the LLM call AND the disk
    write succeed)."""
    from nexoclip.branding import generate_logo, rasterize_svg_to_png
    from nexoclip.llm import LLMRouter, load_llm_config
    from nexoclip.settings import get_settings

    repo = BrandKitsRepo(db)
    kit = await repo.get(kit_id)
    if kit is None:
        raise HTTPException(status_code=404, detail="brand kit not found")

    output_dir = Path(get_settings().default_output_dir)
    asset_dir = _brand_kit_asset_dir(output_dir, kit_id)
    asset_dir.mkdir(parents=True, exist_ok=True)
    call_log_path = output_dir / "llm_calls_brand_kits.jsonl"
    router_ = LLMRouter(
        config=load_llm_config(),
        call_log_path=call_log_path,
        db=db,
    )
    try:
        logo = await generate_logo(
            tenant_id=tenant_id,
            brand_name=kit.name,
            primary_color=kit.primary_color,
            accent_color=kit.accent_color,
            style=style_hint or "Minimal mark / monogram",
            router=router_,
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"logo generation failed: {e}",
        ) from e

    svg_path = asset_dir / "logo.svg"
    svg_path.write_text(logo.svg, encoding="utf-8")
    # Rasterize best-effort — caller fallback is "SVG only", not failure.
    rasterize_svg_to_png(
        logo.svg, output_path=asset_dir / "logo.png", size_px=1024
    )
    # Use the file-serving endpoint as the canonical URL so the dashboard
    # template doesn't need to know about the on-disk layout.
    logo_url = f"/dashboard/brand-kits/{kit_id}/logo.svg"
    await repo.update(
        kit_id,
        logo_url=logo_url,
        ai_generated=True,
        ai_prompt=style_hint or "Minimal mark / monogram",
        ai_provider="anthropic",
    )
    return RedirectResponse(
        url=f"/dashboard/brand-kits/{kit_id}", status_code=303
    )


@router.get("/brand-kits/{kit_id}/logo.svg")
async def brand_kits_logo_svg(
    request: Request,
    kit_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Serve the saved SVG for the dashboard preview + downstream renderer."""
    from nexoclip.settings import get_settings

    if await BrandKitsRepo(db).get(kit_id) is None:
        raise HTTPException(status_code=404, detail="brand kit not found")
    output_dir = Path(get_settings().default_output_dir)
    svg_path = _brand_kit_asset_dir(output_dir, kit_id) / "logo.svg"
    if not svg_path.exists():
        raise HTTPException(status_code=404, detail="no logo for this kit")
    return FileResponse(
        path=str(svg_path),
        media_type="image/svg+xml",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/brand-kits/{kit_id}/logo.png")
async def brand_kits_logo_png(
    request: Request,
    kit_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Serve the rasterized PNG (when cairosvg was installed at generate-time).

    Falls through to 404 when the SVG was generated but rasterization was
    skipped — the dashboard's `<img>` fallback covers that case."""
    from nexoclip.settings import get_settings

    if await BrandKitsRepo(db).get(kit_id) is None:
        raise HTTPException(status_code=404, detail="brand kit not found")
    output_dir = Path(get_settings().default_output_dir)
    png_path = _brand_kit_asset_dir(output_dir, kit_id) / "logo.png"
    if not png_path.exists():
        raise HTTPException(status_code=404, detail="no logo png for this kit")
    return FileResponse(
        path=str(png_path),
        media_type="image/png",
        headers={"Cache-Control": "no-cache"},
    )
