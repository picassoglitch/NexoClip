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
    TenantsRepo,
    VariantsRepo,
)
from nexoclip.db.models import ConnectedAccount, CustomTriggerPhrases
from nexoclip.errors import NexoClipError
from nexoclip.tenancy import hash_token

from .._pipeline import PipelineKickoff
from ..deps import get_db, require_full_scope, tenant_binder
from ..status_gate import require_active_tenant, require_paid_tier
from .clips import _VALID_STATUS_TRANSITIONS

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
# Slice O.24 — i18n globals (`t`, `locale`) for dashboard templates.
# Same registry as the landing page; pages opt in by calling
# `{{ t('nav.publish') }}` etc.
from ..i18n import install_globals as _install_i18n  # noqa: E402
_install_i18n(templates)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

_COOKIE_NAME = "nexoclip_token"


@router.get("", include_in_schema=False)
@router.get("/", include_in_schema=False)
async def dashboard_root() -> Response:
    """Slice I.3 follow-up — operators kept hitting /dashboard/ in the
    address bar and getting a bare 404. There's no actual dashboard
    HOME page (the streams index is the canonical landing); just
    redirect there. The bearer-auth middleware will bounce them on to
    /dashboard/login if they're not authenticated."""
    return RedirectResponse(url="/dashboard/streams", status_code=303)


@router.get("/_balance/refresh", include_in_schema=False)
async def refresh_balance(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Force-fetch the current Nexo AI token balance + update the local cache.

    Triggered by clicking the balance chip in the nav. Useful when:
      - The tenant just signed in and hasn't run any LLM calls yet (cache is
        empty so the chip shows '— tokens'.
      - The operator wants to confirm the cross-engine balance reflects a
        recent top-up purchase that happened on another device.

    Behavior:
      - Success → 303 back to the referer (chip re-renders with fresh data).
      - Failure (env vars unset, tenant has no external_user_id, network
        error, etc.) → 303 to /dashboard/_diag/nexo_ai so the user sees
        exactly what's broken instead of staring at the same '— tokens'
        chip wondering if the click did anything.
    """
    from nexoclip.integrations.nexo_ai.balance import fetch_balance_now

    ok = await fetch_balance_now(db, tenant_id=tenant_id)
    if not ok:
        # Tell the user what's actually wrong instead of silently bouncing.
        return RedirectResponse(url="/dashboard/_diag/nexo_ai", status_code=303)
    # Bounce back to wherever the user clicked from. Falls back to streams
    # list if no Referer (direct hit, curl, etc).
    referer = request.headers.get("referer") or "/dashboard/streams"
    return RedirectResponse(url=referer, status_code=303)


@router.get("/_diag/nexo_ai", response_class=HTMLResponse, include_in_schema=False)
async def diag_nexo_ai(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Diagnostic page — surfaces every state bit that the balance pipeline
    depends on, rendered as a readable dashboard panel so we don't need to
    grep Railway logs to debug a missing token chip.

    Defensive: each fetch is wrapped in its own try/except so a broken
    upstream (e.g. NEXO_AI_BASE_URL pointing at a 404) shows up as an
    inline error on the panel rather than a generic 500. The whole route
    catches at the outer level too — the response should ALWAYS be HTML
    even if the integration is in a bad state.

    The env-var section shows the LITERAL value of NEXO_AI_BASE_URL (a URL
    is safe to display) so you can spot typos or whitespace in the Railway
    env var without copying the deploy logs. Token values stay
    presence-only (booleans) — never echo a secret back to the browser."""
    import traceback
    from nexoclip.integrations.nexo_ai.balance import (
        fetch_balance_now_verbose,
    )
    from nexoclip.settings import get_settings

    # Defaults so we can always render SOMETHING.
    base_url: str | None = None
    has_admin_token = False
    has_sso_secret = False
    external_user_id: str | None = None
    tenant_tier: str | None = None
    tenant_status: str | None = None
    cached_at: str | None = None
    cached_remaining: int | None = None
    cached_unlimited: int | None = None
    live_ok = False
    diag_error: str | None = None

    try:
        settings = get_settings()
        # Show the URL literal so the user can spot misspellings / trailing
        # whitespace / wrong port. Pydantic auto-strips outer whitespace,
        # so what shows up here is what the runtime actually uses.
        base_url = settings.nexo_ai_base_url or None
        has_admin_token = bool(settings.nexo_ai_admin_token)
        has_sso_secret = bool(settings.nexo_ai_sso_secret)
    except Exception as e:  # noqa: BLE001
        diag_error = f"settings load failed: {e!r}"

    try:
        tenant = await TenantsRepo(db).get(tenant_id)
        if tenant is not None:
            external_user_id = tenant.external_user_id
            tenant_tier = tenant.tier
            tenant_status = tenant.status
            cached_remaining = tenant.cached_balance_remaining
            cached_unlimited = tenant.cached_balance_unlimited
            cached_at = tenant.cached_balance_at
    except Exception as e:  # noqa: BLE001
        diag_error = (diag_error or "") + f" · tenant lookup failed: {e!r}"

    # Live ping. Returns a DiagPingResult with the full HTTP response info
    # (status code, body excerpt, exception type) so the diag page can
    # render the exact failure mode instead of just "ping failed". Updates
    # the cache as a side-effect on 'ok' so this page can ALSO function
    # as a manual refresh button.
    ping = None
    try:
        ping = await fetch_balance_now_verbose(db, tenant_id=tenant_id)
    except Exception as e:  # noqa: BLE001
        diag_error = (diag_error or "") + f" · ping crashed: {e!r}"
    live_ok = ping is not None and ping.outcome == "ok"

    # Re-read tenant after the ping so the rendered cache numbers reflect
    # any update that ping just produced (if it did).
    try:
        tenant_after = await TenantsRepo(db).get(tenant_id)
        if tenant_after is not None:
            external_user_id = tenant_after.external_user_id
            cached_remaining = tenant_after.cached_balance_remaining
            cached_unlimited = tenant_after.cached_balance_unlimited
            cached_at = tenant_after.cached_balance_at
    except Exception as e:  # noqa: BLE001
        diag_error = (diag_error or "") + f" · post-ping read failed: {e!r}"

    interpret = _interpret_diag(
        base_url=base_url,
        has_admin_token=has_admin_token,
        external_user_id=external_user_id,
        live_ok=live_ok,
    )

    try:
        return templates.TemplateResponse(
            request,
            "_diag_nexo_ai.html",
            {
                "tenant_id": tenant_id,
                "base_url": base_url,
                "has_admin_token": has_admin_token,
                "has_sso_secret": has_sso_secret,
                "external_user_id": external_user_id,
                "tenant_tier": tenant_tier,
                "tenant_status": tenant_status,
                "cached_remaining": cached_remaining,
                "cached_unlimited": cached_unlimited,
                "cached_at": cached_at,
                "live_ok": live_ok,
                "ping": ping,
                "interpret": interpret,
                "diag_error": diag_error,
            },
        )
    except Exception as e:  # noqa: BLE001
        # Absolute last-resort: if the template itself crashes, emit a
        # plain-text response with the traceback so the operator still
        # sees SOMETHING useful instead of FastAPI's 500 page.
        tb = traceback.format_exc()
        return HTMLResponse(
            content=(
                "<pre style='padding:24px;font-family:monospace;color:#e54;'>"
                "diag template crashed:\n\n"
                f"{tb}\n\n"
                f"diag_error: {diag_error}</pre>"
            ),
            status_code=500,
        )


def _interpret_diag(
    *,
    base_url: str | None,
    has_admin_token: bool,
    external_user_id: str | None,
    live_ok: bool,
) -> dict[str, str]:
    """Translate the raw diag bits into one human-readable status + next step.
    Returns a {severity, headline, action} dict so the template can color the
    panel appropriately."""
    if not base_url:
        return {
            "severity": "danger",
            "headline": "NEXO_AI_BASE_URL no está configurada en Railway",
            "action": (
                "Abre el dashboard de Railway → este servicio → Variables, "
                "agrega NEXO_AI_BASE_URL=https://nexo-ai.world, y redeploya. "
                "Sin esto NexoClip no sabe a dónde reportar usage."
            ),
        }
    if not has_admin_token:
        return {
            "severity": "danger",
            "headline": "NEXO_AI_ADMIN_TOKEN no está configurada en Railway",
            "action": (
                "Debe coincidir EXACTAMENTE con NEXOCLIP_ADMIN_TOKEN del lado "
                "Nexo AI (Vercel env vars). Si no son iguales, Nexo AI "
                "responde 401 a cada reporte de usage."
            ),
        }
    if not external_user_id:
        return {
            "severity": "warn",
            "headline": "Este tenant no está vinculado a un usuario de Nexo AI",
            "action": (
                "external_user_id está vacío. Probablemente el tenant se creó "
                "via CLI antes de la integración. Re-provisiona desde Nexo AI "
                "(POST /api/admin/tenants) o actualiza la columna en SQLite "
                "manualmente para apuntarlo al user_id del usuario en Supabase."
            ),
        }
    if not live_ok:
        return {
            "severity": "warn",
            "headline": "Las env vars están bien pero el ping a Nexo AI falló",
            "action": (
                "▼ Scroll abajo a la sección 'Ping en vivo a Nexo AI' — "
                "verás el HTTP status code exacto + el body de respuesta + "
                "una pista específica según el código. Es la primera cosa "
                "que tienes que mirar en esta página. (Causas más comunes: "
                "tokens no coinciden byte-por-byte entre ambos sides, "
                "external_user_id no existe en la tabla profiles de Supabase, "
                "o nexo-ai.world está caído.)"
            ),
        }
    return {
        "severity": "ok",
        "headline": "Todo en verde — el chip debería estar reportando balance",
        "action": (
            "Si el chip aún muestra '— tokens', haz click una vez más para "
            "forzar el refresh manual. Si vuelve aquí, hay un edge case que "
            "no estoy detectando — revisa Railway logs."
        ),
    }


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


# ---------- Logout ----------
#
# Slice O.23 — Login removed entirely. nexo-ai is the only gatekeeper.
# No GET /login (the page is gone), no POST /login (no token form).
# Access in: GET /auth/sso?token=<jwt-from-nexo-ai> sets a session
# cookie + redirects to /dashboard/streams. Anyone hitting any
# /dashboard/* page without that cookie gets bounced to nexo-ai's
# login URL by the auth middleware. Roles/tiers come straight from
# the SSO token's `tier` claim, synced into the tenant row on each
# login (see nexo_ai.sso_finalize).


@router.post("/logout")
async def logout() -> Response:
    """Clear our session cookie + bounce to nexo-ai. No in-house login
    page exists anymore for the redirect to land on."""
    from nexoclip.settings import get_settings
    target = (get_settings().nexo_ai_login_url or "https://nexo-ai.world/login").strip()
    response = RedirectResponse(url=target, status_code=303)
    response.delete_cookie(_COOKIE_NAME)
    return response


# ---------- Diarize health (admin) ----------
#
# Slice O.38 — operator-facing diagnostic. The pipeline frequently marks
# diarization as "skipped" with a friendly note ("speaker labels off —
# pyannote not available on this server"), which is correct UX for an
# end-user but useless for the admin trying to fix the underlying cause.
# This route probes every fail-mode in process and reports the raw
# truth: HF_TOKEN presence, pyannote / torchaudio / speechbrain
# importability + versions, the DiarizationConfig the pipeline will use,
# and (when --probe-load=1 is passed) attempts the real pyannote Pipeline
# load — the same call the worker subprocess runs. Admin-gated.


@router.get("/_health/diarize", response_class=HTMLResponse, include_in_schema=False)
async def diarize_health(
    request: Request,
    probe_load: int = 0,
) -> Response:
    """Admin-only diarize diagnostic. Surfaces real, untranslated errors."""
    if not getattr(request.state, "is_admin", False):
        raise HTTPException(status_code=404, detail="not found")

    import importlib
    import os as _os

    from nexoclip.config import load_config as _load_pipeline_config

    def _probe_import(modname: str) -> dict[str, object]:
        try:
            mod = importlib.import_module(modname)
            return {
                "ok": True,
                "version": getattr(mod, "__version__", "?"),
                "path": getattr(mod, "__file__", "?"),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001 — we want every failure
            return {
                "ok": False,
                "version": None,
                "path": None,
                "error": f"{type(e).__name__}: {e}",
            }

    pyannote = _probe_import("pyannote.audio")
    torchaudio = _probe_import("torchaudio")
    torch = _probe_import("torch")
    speechbrain = _probe_import("speechbrain")

    # torchaudio.AudioMetaData is the specific attribute pyannote 3.3.x
    # consults at import time — version skew flips this to False even
    # when both packages import successfully.
    torchaudio_audiometadata = False
    if torchaudio["ok"]:
        try:
            import torchaudio as _ta
            torchaudio_audiometadata = hasattr(_ta, "AudioMetaData")
        except Exception:  # noqa: BLE001
            torchaudio_audiometadata = False

    hf_token = _os.environ.get("HF_TOKEN", "").strip()
    hf_present = bool(hf_token)
    hf_prefix = hf_token[:6] if hf_token else ""

    cfg = _load_pipeline_config()
    diarize_cfg = {
        "enabled": cfg.diarization.enabled,
        "model": cfg.diarization.model,
        "device": cfg.diarization.device,
    }

    pipeline_load: dict[str, object] | None = None
    if probe_load == 1 and pyannote["ok"] and hf_present:
        try:
            from pyannote.audio import Pipeline  # type: ignore[import-not-found]
            pl = Pipeline.from_pretrained(
                diarize_cfg["model"], use_auth_token=hf_token
            )
            pipeline_load = {
                "ok": True,
                "summary": str(type(pl).__name__),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            pipeline_load = {
                "ok": False,
                "summary": None,
                "error": f"{type(e).__name__}: {e}",
            }

    ctx = {
        "request": request,
        "pyannote": pyannote,
        "torchaudio": torchaudio,
        "torchaudio_audiometadata": torchaudio_audiometadata,
        "torch": torch,
        "speechbrain": speechbrain,
        "hf_present": hf_present,
        "hf_prefix": hf_prefix,
        "diarize_cfg": diarize_cfg,
        "pipeline_load": pipeline_load,
        "probe_load_requested": bool(probe_load),
    }
    return templates.TemplateResponse("diarize_health.html", ctx)


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


@router.post(
    "/streams",
    dependencies=[Depends(require_full_scope), Depends(require_active_tenant)],
)
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
    dispatcher = request.app.state.job_dispatcher
    await dispatcher.dispatch_pipeline(
        PipelineKickoff(
            tenant_id=tenant_id,
            stream=stream,
            persona_id=persona_id,
            output_dir=output_dir,
        ),
        background_tasks=background_tasks,
    )
    return RedirectResponse(url=f"/dashboard/streams/{row.id}", status_code=303)


@router.post(
    "/streams/upload",
    dependencies=[Depends(require_full_scope), Depends(require_active_tenant)],
)
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
    dispatcher = request.app.state.job_dispatcher
    await dispatcher.dispatch_pipeline(
        PipelineKickoff(
            tenant_id=tenant_id,
            stream=stream,
            persona_id=persona_id,
            output_dir=output_dir,
        ),
        background_tasks=background_tasks,
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

    # Slice NX.5 — top-level pipeline failure surfacing. If a `pipeline.failed`
    # event exists for this stream, the runner caught an exception OUTSIDE
    # any individual step (typically: bad persona_id, missing config, db
    # open error). The per-step events are all 'pending' in this case, so
    # without this check the UI would spin forever. We pull the most-recent
    # such event and:
    #   1) Flip the first pending step to a synthetic 'failed' state so the
    #      progress row renders red with the error message.
    #   2) Set has_failed=True so the parent template stops polling.
    pipeline_failed_events = [
        e
        for e in all_events
        if e.type == "pipeline.failed" and e.payload.get("stream_id") == stream_id
    ]
    pipeline_failure: dict[str, object] | None = None
    if pipeline_failed_events:
        latest = max(pipeline_failed_events, key=lambda e: e.ts)
        pipeline_failure = {
            "error": str(latest.payload.get("error") or "pipeline aborted"),
            "error_type": str(latest.payload.get("error_type") or "Exception"),
        }
        # Mark the first pending step as failed so the progress card has
        # SOMETHING to point at. Walk in order so the user sees the failure
        # at the earliest stage that didn't get to run.
        for s in steps:
            if s["status"] == "pending":
                s["status"] = "failed"
                s["error"] = pipeline_failure["error"]
                break

    is_running = any(s["status"] == "running" for s in steps) or (
        all(s["status"] == "pending" for s in steps) and pipeline_failure is None
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
            "pipeline_failure": pipeline_failure,
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


@router.get("/publish", response_class=HTMLResponse)
async def publish_view(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice O.8 — global Publish page.

    Aggregates every approved (or published) clip across every stream
    for the current tenant into one matrix. Replaces the legacy
    Accounts nav tab; account management still happens inline via the
    chip strip + modal at the top.
    """
    clips = await ClipsRepo(db).list_for_tenant_with_status(
        ["approved", "published"], limit=300
    )

    # Need each clip's stream so we can group by stream and show the
    # stream title as a row header. Cache StreamsRepo.get() per id to
    # avoid N+1.
    streams_repo = StreamsRepo(db)
    streams_by_id: dict[str, object] = {}
    for c in clips:
        if c.stream_id not in streams_by_id:
            stm = await streams_repo.get(c.stream_id)
            if stm is not None:
                streams_by_id[c.stream_id] = stm

    # Order: clips appear in the order they were returned (newest first),
    # but we group them by stream so all clips from the same stream
    # render consecutively. The template walks `clip_groups` which is
    # a list of {stream, clips} dicts preserving first-seen order.
    clip_groups_dict: dict[str, dict[str, object]] = {}
    for c in clips:
        if c.stream_id not in clip_groups_dict:
            stm = streams_by_id.get(c.stream_id)
            if stm is None:
                continue
            clip_groups_dict[c.stream_id] = {"stream": stm, "clips": []}
        clip_groups_dict[c.stream_id]["clips"].append(c)  # type: ignore[index]
    clip_groups = list(clip_groups_dict.values())

    accounts = await ConnectedAccountsRepo(db).list_for_tenant()
    accounts_by_platform: dict[str, list[ConnectedAccount]] = {}
    for a in accounts:
        if a.status != "active":
            continue
        accounts_by_platform.setdefault(a.platform.lower(), []).append(a)

    pj_repo = PublishJobsRepo(db)
    existing_jobs: dict[tuple[str, str], str] = {}
    variants_by_clip: dict[str, str] = {}
    lead_variant_meta: dict[str, dict[str, object]] = {}
    for c in clips:
        for j in await pj_repo.list_for_clip(c.id):
            existing_jobs[(c.id, j.platform.lower())] = j.status
        vs = await VariantsRepo(db).list_for_clip(c.id)
        if vs:
            variants_by_clip[c.id] = vs[0].id
            lead_variant_meta[c.id] = {
                "title": vs[0].title_card_text or "",
                "caption": vs[0].caption or "",
                "hashtags": " ".join(vs[0].hashtags or []),
            }

    return templates.TemplateResponse(
        request,
        "publish.html",
        {
            "clip_groups": clip_groups,
            "total_clips": len(clips),
            "accounts_by_platform": accounts_by_platform,
            "existing_jobs": existing_jobs,
            "variants_by_clip": variants_by_clip,
            "lead_variant_meta": lead_variant_meta,
            "supported_platforms": [
                ("tiktok",     "TikTok",  "ti-brand-tiktok"),
                ("reels",      "Reels",   "ti-brand-instagram"),
                ("shorts",     "Shorts",  "ti-brand-youtube"),
                ("kick",       "Kick",    "ti-flame"),
                ("twitch",     "Twitch",  "ti-brand-twitch"),
            ],
        },
    )


@router.get("/publish/status.json")
async def publish_status_json(
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice O.8 — tenant-wide publish status feed for the global page.

    Same payload shape as the per-stream endpoint so the JS poller
    can be shared verbatim — same `cells` dict keyed `<clip>__<platform>`,
    same `failures` list, same `summary` counts.
    """
    import json as _json

    clips = await ClipsRepo(db).list_for_tenant_with_status(
        ["approved", "published"], limit=300
    )
    clip_by_id = {c.id: c for c in clips}
    pj_repo = PublishJobsRepo(db)
    cells: dict[str, dict[str, object]] = {}
    failures: list[dict[str, object]] = []
    summary: dict[str, int] = {"pending": 0, "running": 0, "sent": 0, "failed": 0}
    for c in clips:
        for j in await pj_repo.list_for_clip(c.id):
            key = f"{c.id}__{j.platform.lower()}"
            cells[key] = {
                "status": j.status,
                "job_id": j.id,
                "last_error": j.last_error,
                "external_url": j.external_url,
                "attempts": j.attempts,
            }
            summary[j.status] = summary.get(j.status, 0) + 1
            if j.status == "failed":
                clp = clip_by_id.get(c.id)
                failures.append({
                    "job_id": j.id,
                    "clip_id": c.id,
                    "clip_label": (
                        f"{clp.start_s:.1f}s → {clp.end_s:.1f}s" if clp else c.id
                    ),
                    "platform": j.platform,
                    "error": j.last_error or "(no error message recorded)",
                    "attempts": j.attempts,
                })

    return Response(
        content=_json.dumps({
            "cells": cells,
            "summary": summary,
            "failures": failures,
        }),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


@router.post(
    "/publish",
    dependencies=[
        Depends(require_full_scope),
        Depends(require_active_tenant),
        Depends(require_paid_tier),
    ],
)
async def publish_submit(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice O.8 — accept the global publish form.

    Same form shape as the per-stream POST: `clip_<id>_<platform>=1`
    ticks + `meta_<id>_<platform>_<field>` overrides. We just look up
    each clip's stream on demand instead of getting it from the URL.
    """
    form = await request.form()
    pj_repo = PublishJobsRepo(db)
    accounts = await ConnectedAccountsRepo(db).list_for_tenant()
    active_by_platform: dict[str, str] = {}
    for a in accounts:
        if a.status == "active":
            active_by_platform.setdefault(a.platform.lower(), a.id)

    all_clips = await ClipsRepo(db).list_for_tenant_with_status(
        ["approved", "published"], limit=500
    )
    eligible_ids = {c.id for c in all_clips}

    def _gather_meta(clip_id: str, platform: str) -> dict[str, str]:
        prefix = f"meta_{clip_id}_{platform}_"
        bundle: dict[str, str] = {}
        for fld in ("title", "description", "hashtags"):
            raw = form.get(prefix + fld)
            if isinstance(raw, str) and raw.strip():
                bundle[fld] = raw.strip()
        return bundle

    created = 0
    for key, value in form.items():
        if value != "1" or not key.startswith("clip_"):
            continue
        parts = key[len("clip_"):].rsplit("_", 1)
        if len(parts) != 2:
            continue
        clip_id, platform = parts
        if clip_id not in eligible_ids:
            continue
        account_id = active_by_platform.get(platform.lower())
        if account_id is None:
            continue
        variants = await VariantsRepo(db).list_for_clip(clip_id)
        if not variants:
            continue
        meta = _gather_meta(clip_id, platform.lower())
        try:
            await pj_repo.enqueue(
                clip_id=clip_id,
                variant_id=variants[0].id,
                account_id=account_id,
                platform=platform.lower(),
                platform_metadata=meta or None,
            )
            created += 1
        except Exception:  # noqa: BLE001
            continue

    await EventsRepo(db).emit(
        type="publish.batch_enqueued",
        payload={"scope": "global", "job_count": created},
    )
    return RedirectResponse(
        url=f"/dashboard/publish?queued={created}",
        status_code=303,
    )


@router.get("/streams/{stream_id}/publish", response_class=HTMLResponse)
async def stream_publish_view(
    request: Request,
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice O.2 — publishing page.

    The page after Approve. Shows every approved/published clip on
    this stream alongside the tenant's connected social accounts,
    and lets the operator pick which clips ship to which platforms.

    Lists pre-existing publish_jobs per clip so the operator sees
    what's already queued / sent / failed — `Publish to TikTok`
    becomes greyed out for clips that already have a TikTok job.
    """
    stream = await StreamsRepo(db).get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="stream not found")

    all_clips = await ClipsRepo(db).list_for_stream(stream_id)
    # Only ship clips the operator has approved (or already published).
    clips = [c for c in all_clips if c.status in ("approved", "published")]

    # Connected accounts → group by platform so the editor can render
    # one section per platform with the account it'd post to.
    accounts = await ConnectedAccountsRepo(db).list_for_tenant()
    accounts_by_platform: dict[str, list[ConnectedAccount]] = {}
    for a in accounts:
        if a.status != "active":
            continue
        accounts_by_platform.setdefault(a.platform.lower(), []).append(a)

    # Existing publish jobs per clip → operator sees what's already
    # queued. Keyed (clip_id, platform) → status.
    pj_repo = PublishJobsRepo(db)
    existing_jobs: dict[tuple[str, str], str] = {}
    for c in clips:
        for j in await pj_repo.list_for_clip(c.id):
            existing_jobs[(c.id, j.platform.lower())] = j.status

    # Pull the first variant per clip so we have a variant_id ready
    # for the enqueue path (publish_jobs requires a variant_id).
    # Operators can refine variant choice in the per-clip editor;
    # the bulk page uses the lead variant by default.
    variants_by_clip: dict[str, str] = {}
    # Slice O.6 — also surface the lead variant's title / caption /
    # hashtags so the per-platform overrides editor can prefill its
    # fields. Operator typed a great caption in the variant editor →
    # the publish page should reuse it instead of forcing a re-type.
    lead_variant_meta: dict[str, dict[str, object]] = {}
    for c in clips:
        vs = await VariantsRepo(db).list_for_clip(c.id)
        if vs:
            variants_by_clip[c.id] = vs[0].id
            lead_variant_meta[c.id] = {
                "title": vs[0].title_card_text or "",
                "caption": vs[0].caption or "",
                "hashtags": " ".join(vs[0].hashtags or []),
            }

    return templates.TemplateResponse(
        request,
        "stream_publish.html",
        {
            "stream": stream,
            "clips": clips,
            "accounts_by_platform": accounts_by_platform,
            "existing_jobs": existing_jobs,
            "variants_by_clip": variants_by_clip,
            "lead_variant_meta": lead_variant_meta,
            # Convenience: the five platforms we support today, in the
            # order we want to render them. Account presence drives
            # whether each section is enabled or shows a "connect" CTA.
            # Slice O.3 — Twitch joined the list. Clips routed to
            # Twitch ship as channel highlights / clip uploads via the
            # Helix API (operator pastes a user access token w/
            # clips:edit + channel:manage:videos scopes).
            "supported_platforms": [
                ("tiktok",     "TikTok",  "ti-brand-tiktok"),
                ("reels",      "Reels",   "ti-brand-instagram"),
                ("shorts",     "Shorts",  "ti-brand-youtube"),
                ("kick",       "Kick",    "ti-flame"),
                ("twitch",     "Twitch",  "ti-brand-twitch"),
            ],
        },
    )


@router.get("/streams/{stream_id}/publish-status.json")
async def stream_publish_status_json(
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice O.5 — live publish-status polling for the matrix view.

    Returns a flat dict keyed `<clip_id>__<platform>` → job status
    string ("pending" / "running" / "sent" / "failed"). The publish
    page polls this every few seconds and rewrites the matrix cells
    in place so the operator sees progress without a hard refresh.

    Also returns `summary` (counts per status) so the page header
    can show "3 sent · 1 failed · 2 pending" rolling totals.
    """
    import json as _json

    clips = await ClipsRepo(db).list_for_stream(stream_id)
    clip_by_id = {c.id: c for c in clips}
    pj_repo = PublishJobsRepo(db)
    cells: dict[str, dict[str, object]] = {}
    # Slice O.7 — `failures` is a flat list of {job_id, clip_id,
    # clip_label, platform, error} the publish page renders in a panel
    # below the matrix. Lets the operator see WHY something failed
    # without bouncing into the clip detail screen.
    failures: list[dict[str, object]] = []
    summary: dict[str, int] = {"pending": 0, "running": 0, "sent": 0, "failed": 0}
    for c in clips:
        if c.status not in ("approved", "published"):
            continue
        for j in await pj_repo.list_for_clip(c.id):
            key = f"{c.id}__{j.platform.lower()}"
            cells[key] = {
                "status": j.status,
                "job_id": j.id,
                "last_error": j.last_error,
                "external_url": j.external_url,
                "attempts": j.attempts,
            }
            summary[j.status] = summary.get(j.status, 0) + 1
            if j.status == "failed":
                clp = clip_by_id.get(c.id)
                failures.append({
                    "job_id": j.id,
                    "clip_id": c.id,
                    "clip_label": (
                        f"{clp.start_s:.1f}s → {clp.end_s:.1f}s" if clp else c.id
                    ),
                    "platform": j.platform,
                    "error": j.last_error or "(no error message recorded)",
                    "attempts": j.attempts,
                })

    return Response(
        content=_json.dumps({
            "cells": cells,
            "summary": summary,
            "failures": failures,
        }),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


@router.post(
    "/streams/{stream_id}/publish",
    dependencies=[
        Depends(require_full_scope),
        Depends(require_active_tenant),
        Depends(require_paid_tier),
    ],
)
async def stream_publish_submit(
    request: Request,
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice O.2 — accept the publish form.

    The form posts a flat list of `clip_<clip_id>_<platform>=1` flags
    (one per checkbox the operator ticked). For each tick we create
    a `publish_jobs` row in `pending` state; the existing publish
    worker drains them on its next pass.
    """
    form = await request.form()
    pj_repo = PublishJobsRepo(db)
    accounts = await ConnectedAccountsRepo(db).list_for_tenant()
    active_by_platform: dict[str, str] = {}
    for a in accounts:
        if a.status == "active":
            # If multiple accounts per platform, the first wins for now.
            active_by_platform.setdefault(a.platform.lower(), a.id)

    clips = await ClipsRepo(db).list_for_stream(stream_id)
    eligible_ids = {c.id for c in clips if c.status in ("approved", "published")}

    # Slice O.6 — read per-platform metadata override fields. Form
    # keys: meta_<clip_id>_<platform>_<field>. Fields we surface in
    # the publish-page editor: title, description, hashtags. The
    # worker merges these onto the platform API call (TikTok caption,
    # IG Reels caption, YT title/description, Twitch clip title, etc).
    def _gather_meta(clip_id: str, platform: str) -> dict[str, str]:
        prefix = f"meta_{clip_id}_{platform}_"
        bundle: dict[str, str] = {}
        for fld in ("title", "description", "hashtags"):
            raw = form.get(prefix + fld)
            if isinstance(raw, str) and raw.strip():
                bundle[fld] = raw.strip()
        return bundle

    created = 0
    for key, value in form.items():
        if value != "1" or not key.startswith("clip_"):
            continue
        # key shape: clip_<clip_id>_<platform>
        parts = key[len("clip_"):].rsplit("_", 1)
        if len(parts) != 2:
            continue
        clip_id, platform = parts
        if clip_id not in eligible_ids:
            continue
        account_id = active_by_platform.get(platform.lower())
        if account_id is None:
            continue
        # Lead variant per clip — same fallback as the GET view.
        variants = await VariantsRepo(db).list_for_clip(clip_id)
        if not variants:
            continue
        meta = _gather_meta(clip_id, platform.lower())
        try:
            await pj_repo.enqueue(
                clip_id=clip_id,
                variant_id=variants[0].id,
                account_id=account_id,
                platform=platform.lower(),
                platform_metadata=meta or None,
            )
            created += 1
        except Exception:  # noqa: BLE001 — best-effort; one failure
            # shouldn't abort the whole batch
            continue

    await EventsRepo(db).emit(
        type="publish.batch_enqueued",
        payload={
            "stream_id": stream_id,
            "job_count": created,
        },
    )
    return RedirectResponse(
        url=f"/dashboard/streams/{stream_id}/publish?queued={created}",
        status_code=303,
    )


@router.post(
    "/streams/{stream_id}/rerun",
    dependencies=[Depends(require_full_scope), Depends(require_active_tenant)],
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

    dispatcher = request.app.state.job_dispatcher
    await dispatcher.dispatch_pipeline(
        PipelineKickoff(
            tenant_id=tenant_id,
            stream=stream,
            persona_id=persona_id,
            output_dir=output_dir,
            language=persona_language,
        ),
        background_tasks=background_tasks,
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
        platform_choices,
        platform_target_choices,
        preset_choices,
        style_choices,
        zones_for_platform,
        resolve_brand_kit_for_candidate,
    )
    from nexoclip.clip import (
        clip_breakdown,
        compute_ai_scores,
        compute_publishability,
        director_lines,
    )
    from nexoclip.db import (
        CandidatesRepo,
        PublishJobsRepo,
        PublishMetricsRepo,
    )

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    variants = await VariantsRepo(db).list_for_clip(clip_id)
    # Slice I.4 — enrich each variant with tone / fit / retention /
    # risk metadata for the new card-grid layout.
    enriched_variants: list[object] = []
    try:
        from nexoclip.llm import Variant as _LLMVariant
        from nexoclip.variants import enrich_variants as _enrich

        variant_rows = [
            _LLMVariant(
                id=v.id,
                language=getattr(v, "language", "en"),
                caption=getattr(v, "caption", ""),
                title_card_text=getattr(v, "title_card_text", "") or "",
                hashtags=list(getattr(v, "hashtags", None) or []),
            )
            for v in variants
        ]
        enriched_variants = list(_enrich(variant_rows))
    except Exception:  # noqa: BLE001 — best-effort enrichment
        enriched_variants = []
    accounts = await ConnectedAccountsRepo(db).list_for_tenant()
    valid_transitions = sorted(_VALID_STATUS_TRANSITIONS.get(clip.status, set()))
    breakdown = await clip_breakdown(db, clip_id)
    ai_scores = compute_ai_scores(breakdown)

    # Slice I.3 — publishability verdict + AI Director narration.
    # The verdict reads BOTH the breakdown AND the operator's overlay
    # config (current banner / hook / caption choices) so it scores
    # what's about to ship, not the raw clip. Safe-zone target falls
    # back to TikTok (the strictest mainstream chrome).
    safe_zone_target = "tiktok"
    if isinstance(clip.overlay_config, dict):
        szp = clip.overlay_config.get("safe_zone_platform")
        if isinstance(szp, str) and szp:
            safe_zone_target = szp
    publishability = compute_publishability(
        breakdown=breakdown,
        overlay_config=(
            clip.overlay_config if isinstance(clip.overlay_config, dict) else {}
        ),
        safe_zone_platform=safe_zone_target,
    )
    director = director_lines(breakdown=breakdown, verdict=publishability)

    # Resolve the brand kit + caption style so the editor can render
    # the live preview against the right defaults when overlay_config
    # is empty (or partially populated).
    speaker_label: str | None = None
    # Slice K.7 — surface trigger-phrase detection on the verdict card.
    # The operator's #1 confusion was "did the AI actually detect the
    # phrase I said?". Pull the matched phrase / snippet / kind out of
    # the candidate evidence so the template can render a "Detected
    # trigger: 'clip it'" line.
    candidate_trigger: dict[str, object] | None = None
    # Slice L.4b — face-aware hook positioning. Derive a coarse
    # "face_zone" hint from the G.3 framing verdict's safe_crop_box so
    # the editor's main hook + captions can dodge the face instead of
    # statically anchoring at 18% (and sometimes covering it). Only
    # three coarse zones — the precision isn't there to justify five —
    # plus None when framing wasn't run on this clip (degrades to the
    # L.4 static defaults).
    face_zone: str | None = None
    if clip.candidate_id:
        for cand in await CandidatesRepo(db).list_for_stream(clip.stream_id):
            if cand.id == clip.candidate_id:
                ev = cand.evidence or {}
                lbl = ev.get("speaker_label")
                if isinstance(lbl, str):
                    speaker_label = lbl
                if isinstance(ev, dict):
                    phrase = ev.get("phrase")
                    snippet = ev.get("transcript_snippet")
                    kind = ev.get("trigger_kind")
                    if isinstance(phrase, str) and phrase:
                        candidate_trigger = {
                            "phrase": phrase,
                            "snippet": snippet if isinstance(snippet, str) else "",
                            "kind": kind if isinstance(kind, str) else "forward",
                            "language": (
                                ev.get("language")
                                if isinstance(ev.get("language"), str)
                                else None
                            ),
                            "confidence": (
                                float(ev.get("confidence"))
                                if isinstance(ev.get("confidence"), int | float)
                                else None
                            ),
                        }
                    # Slice L.4b — extract face_zone from framing.
                    # The framing evidence persists `safe_crop_box`
                    # (the chosen 9:16 crop region within the source).
                    # `crop_box.y` near 0 means the crop is anchored at
                    # the TOP of the source = face appears HIGH in the
                    # 9:16 output. Map to three zones:
                    #   top    → face in upper third → hook conflicts
                    #   mid    → face mid-frame → L.4 default is fine
                    #   lower  → face near bottom → hook is fully clear
                    framing = ev.get("framing")
                    if isinstance(framing, dict):
                        crop = framing.get("safe_crop_box")
                        if isinstance(crop, dict):
                            cy = crop.get("y")
                            if isinstance(cy, int | float):
                                cy_f = float(cy)
                                if cy_f < 0.10:
                                    face_zone = "top"
                                elif cy_f < 0.35:
                                    face_zone = "mid"
                                else:
                                    face_zone = "lower"
                break
    brand_kit = await resolve_brand_kit_for_candidate(
        db, stream_id=clip.stream_id, speaker_label=speaker_label
    )
    caption_style = caption_style_or_default(
        brand_kit.caption_style if brand_kit is not None else None
    )

    # Slice L.6 — viral-intelligence decision engine. Pure function
    # over the breakdown + framing + platform inputs we already have.
    # Outputs:
    #   - intel_lines: ✨ phrases for the verdict card
    #   - caption_position: cascades into the "Auto Viral" picker
    #   - needs_hook / needs_subtitles / secondary_hook_recommended:
    #     drive editor visibility hints
    # The decision runs every request — it's a pure function with no
    # I/O so per-request cost is negligible (microseconds).
    from nexoclip.clip.ai_decisions import decide as _ai_decide

    # Resolve operator's target platform (per-clip override wins over
    # brand-kit default).
    _ai_target_platform: str | None = None
    if isinstance(clip.overlay_config, dict):
        _tp = clip.overlay_config.get("target_platform")
        if isinstance(_tp, str) and _tp:
            _ai_target_platform = _tp
    if not _ai_target_platform and brand_kit is not None:
        _bk_tp = getattr(brand_kit, "target_platform", None)
        if isinstance(_bk_tp, str) and _bk_tp:
            _ai_target_platform = _bk_tp

    ai_decisions = _ai_decide(
        face_presence=breakdown.face_presence,
        speaking_intensity=breakdown.speaking_intensity,
        reaction_confidence=breakdown.reaction_confidence,
        heuristic_reason=breakdown.heuristic_reason,
        duration_s=clip.duration_s,
        face_zone=face_zone,
        target_platform=_ai_target_platform,
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
            # Slice I.4 — per-variant tone / fit / retention / risk for the
            # card-grid template. Parallel array; same index as `variants`.
            "enriched_variants": enriched_variants,
            "accounts": accounts,
            "valid_transitions": valid_transitions,
            "breakdown": breakdown,
            "ai_scores": ai_scores,
            # Slice I.3 — Publishability verdict + AI Director narrative.
            "publishability": publishability,
            "director_lines": director,
            # Slice K.7 — surface the voice-trigger match that fired
            # this candidate. None when the clip wasn't voice-triggered
            # (e.g. chat-heat-only candidates).
            "candidate_trigger": candidate_trigger,
            # Slice L.4b — coarse "where is the face" hint derived from
            # G.3 framing. Template wires it onto the preview as
            # data-face-zone="top|mid|lower" so the hook + captions
            # avoid the face. None when framing wasn't run.
            "face_zone": face_zone,
            # Slice L.6 — AI auto-decisions (hook needed / subtitles
            # needed / caption position / face vs gameplay priority /
            # reaction-wins-over-captions). Drives the ✨ intel lines
            # on the verdict card AND the "Auto Viral" cascade in the
            # editor JS.
            "ai_decisions": ai_decisions,
            "outcomes": outcomes,
            "brand_kit": brand_kit,
            "caption_style": caption_style,
            "caption_preset_choices": preset_choices(),
            # Slice I.1 — Clip Style preset cards (Repost Page Viral /
            # Clean Creator / Gaming Chaos / Documentary / Minimal Native).
            "clip_style_choices": style_choices(),
            # Slice K.5 — Target-platform chip row (TikTok / Reels /
            # Shorts / Kick / All). Picking one auto-applies every
            # downstream editor setting via PLATFORM_PRESETS.
            "platform_target_choices": platform_target_choices(),
            # Slice I.2 — platform overlay simulation + safe zones.
            # `platform_choices` populates the simulation dropdown;
            # `platform_zone_specs` is the full zone catalog, indexed
            # by platform id, so the editor JS can draw dashed zones
            # without a round-trip and run collision warnings client
            # side. The server-side `detect_collisions` is reserved
            # for headless validation (export checklist in I.3).
            "platform_choices": platform_choices(),
            "platform_zone_specs": {
                p: [
                    {
                        "kind": z.kind,
                        "x": z.x, "y": z.y, "w": z.w, "h": z.h,
                        "reason": z.reason,
                    }
                    for z in zones_for_platform(p)
                ]
                for p, _label in platform_choices()
            },
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
    # Slice I.1 — clip style + banner variant + top hook.
    clip_style: str = "",
    banner_variant: str = "",
    banner_live_badge: str = "",
    top_hook_enabled: str = "",
    top_hook_text: str = "",
    top_hook_style: str = "",
    # Slice I.2 — platform overlay simulation + safe-zone target +
    # preview mode. All three are EDITOR-only state — they never feed
    # the burn, but they DO persist to the brand_kit so the operator's
    # preferred simulation target survives across clips.
    platform_overlay_preview: str = "",
    safe_zone_platform: str = "",
    preview_mode: str = "",
    # Slice K.5 — target-platform auto-config. ONE chip at the top of
    # the editor (tiktok / reels / shorts / kick / all) that the JS
    # uses to fan-out every other field (safe_zone, captions,
    # banner_variant, clip_style…). We just need to remember *which*
    # platform the operator picked so the next clip opens to the
    # same chip.
    target_platform: str = "",
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
        # Slice I.1 — clip style preset (Repost Page Viral / etc).
        "clip_style": (clip_style or "").strip().lower() or None,
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
            # Slice I.1 — Kick banner variant (repost_page / classic /
            # green_block / minimal_url) + LIVE NOW pill toggle.
            "variant": (banner_variant or "").strip().lower() or None,
            "live_badge": _bool(banner_live_badge),
        },
        # Slice I.1 — top hook box (white rounded headline above face).
        "top_hook": {
            "enabled": _bool(top_hook_enabled),
            "text": top_hook_text.strip(),
            "style": (top_hook_style or "white_rounded").strip().lower(),
        },
        # Slice I.2 — editor-only platform sim + safe-zone target.
        "platform_overlay_preview":
            (platform_overlay_preview or "").strip().lower() or None,
        "safe_zone_platform": (safe_zone_platform or "").strip().lower() or None,
        "preview_mode": (preview_mode or "").strip().lower() or None,
        # Slice K.5 — picked target platform (tiktok/reels/shorts/kick/all).
        # JS uses this to fan-out the rest of the editor; we persist it so
        # the next clip remembers the operator's last pick.
        "target_platform": (target_platform or "").strip().lower() or None,
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
    top_hook = cfg.get("top_hook") or {}
    if not isinstance(banner, dict) or not isinstance(captions, dict):
        return
    if not isinstance(top_hook, dict):
        top_hook = {}

    # Slice I.1 — clip style preset (Repost Page Viral / etc) +
    # banner variant + top hook are all user-level prefs.
    clip_style_raw = cfg.get("clip_style")
    clip_style = (
        str(clip_style_raw).strip().lower() if isinstance(clip_style_raw, str) else ""
    )
    # Slice K.5 — target platform (tiktok/reels/shorts/kick/all). One
    # chip drives the whole editor; remembered as the operator's
    # default so the next clip auto-applies the same preset bundle.
    target_platform_raw = cfg.get("target_platform")
    target_platform_val = (
        str(target_platform_raw).strip().lower()
        if isinstance(target_platform_raw, str)
        else ""
    )
    banner_variant_raw = banner.get("variant")
    banner_variant = (
        str(banner_variant_raw).strip().lower()
        if isinstance(banner_variant_raw, str)
        else ""
    )
    banner_live_badge_val = (
        bool(banner.get("live_badge")) if "live_badge" in banner else None
    )
    top_hook_enabled_val = (
        bool(top_hook.get("enabled")) if "enabled" in top_hook else None
    )
    top_hook_style_raw = top_hook.get("style")
    top_hook_style = (
        str(top_hook_style_raw).strip().lower()
        if isinstance(top_hook_style_raw, str)
        else ""
    )

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
    # Slice H.1 — caption knobs from F.7-H + banner toggles now also
    # persist to the brand_kit so they survive across clips.
    position_raw = captions.get("position")
    position = str(position_raw).strip() if isinstance(position_raw, str) else ""
    font_size_raw = captions.get("font_size")
    font_size = str(font_size_raw).strip() if isinstance(font_size_raw, str) else ""
    animation_raw = captions.get("animation")
    animation = str(animation_raw).strip() if isinstance(animation_raw, str) else ""
    lead_ms_raw = captions.get("lead_ms")
    lead_ms_val: int | None = None
    if isinstance(lead_ms_raw, int | float):
        lead_ms_val = int(lead_ms_raw)
    elif isinstance(lead_ms_raw, str) and lead_ms_raw.strip().lstrip("-").isdigit():
        lead_ms_val = int(lead_ms_raw)
    banner_enabled_val = bool(banner.get("enabled")) if "enabled" in banner else None
    banner_show_context_val = (
        bool(banner.get("show_context")) if "show_context" in banner else None
    )
    banner_show_safezones_val = (
        bool(banner.get("show_safezones")) if "show_safezones" in banner else None
    )

    # Build the caption_style dict — now carries ALL four F.7-H knobs
    # plus the legacy preset/highlight_color. The renderer + preview
    # JS both read from this same dict via caption_style_or_default.
    caption_style_patch: dict[str, object] | None = None
    if preset or hilite or position or font_size or animation or lead_ms_val is not None:
        base = (
            caption_style_or_default(kit.caption_style).model_dump()
            if kit is not None
            else caption_style_or_default(None).model_dump()
        )
        if preset:
            base["preset_id"] = preset
        if hilite:
            base["highlight_color"] = hilite
        if position:
            base["position"] = position
        if font_size:
            base["font_size"] = font_size
        if animation:
            base["animation"] = animation
        if lead_ms_val is not None:
            base["lead_ms"] = lead_ms_val
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
            new_kit = await repo.create(
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
            return
        # The new toggle/platform fields aren't in `create()`'s signature
        # (they're update-only). Apply them on the fresh row immediately.
        if any(v is not None for v in [
            platform or None, banner_enabled_val,
            banner_show_context_val, banner_show_safezones_val,
            clip_style or None, banner_variant or None,
            banner_live_badge_val, top_hook_enabled_val, top_hook_style or None,
            target_platform_val or None,
        ]):
            try:
                await repo.update(
                    new_kit.id,
                    default_platform=platform or None,
                    banner_enabled_default=banner_enabled_val,
                    banner_show_context_default=banner_show_context_val,
                    banner_show_safezones_default=banner_show_safezones_val,
                    # Slice I.1.
                    clip_style=clip_style or None,
                    bottom_banner_style=banner_variant or None,
                    banner_live_badge_default=banner_live_badge_val,
                    top_hook_enabled_default=top_hook_enabled_val,
                    top_hook_style_default=top_hook_style or None,
                    # Slice K.5.
                    target_platform=target_platform_val or None,
                )
            except Exception:  # noqa: BLE001
                pass
        return

    # Existing default kit — partial update of just the fields the
    # operator changed in the editor. None-valued args are ignored by
    # BrandKitsRepo.update so we only touch what was provided.
    has_change = any([
        color, handle_kick, handle_tiktok, handle_youtube, handle_instagram,
        caption_style_patch, platform,
        banner_enabled_val is not None,
        banner_show_context_val is not None,
        banner_show_safezones_val is not None,
        clip_style, banner_variant,
        banner_live_badge_val is not None,
        top_hook_enabled_val is not None,
        top_hook_style,
        target_platform_val,
    ])
    if not has_change:
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
            default_platform=platform or None,
            banner_enabled_default=banner_enabled_val,
            banner_show_context_default=banner_show_context_val,
            banner_show_safezones_default=banner_show_safezones_val,
            # Slice I.1.
            clip_style=clip_style or None,
            bottom_banner_style=banner_variant or None,
            banner_live_badge_default=banner_live_badge_val,
            top_hook_enabled_default=top_hook_enabled_val,
            top_hook_style_default=top_hook_style or None,
            # Slice K.5.
            target_platform=target_platform_val or None,
        )
    except Exception:  # noqa: BLE001
        pass


@router.post(
    "/clips/{clip_id}/apply-ai-fixes",
    dependencies=[Depends(require_full_scope)],
)
async def clip_apply_ai_fixes(
    request: Request,
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice I.3 + N.1 — apply non-destructive AI-recommended fixes.

    Re-runs `apply_ai_fixes` against the clip's current overlay_config,
    writes the updated dict back, and (N.1) recomputes publishability
    before + after so the dashboard can animate the retention score
    moving up. Also auto-promotes a clip from `cut` to
    `ready_for_review` when the post-fix score crosses the
    publish-ready threshold (75) — operator's "Auto fix & optimize"
    button should produce a tangible status change, not just a
    silent toast.
    """
    import json as _json

    from nexoclip.clip import (
        apply_ai_fixes,
        clip_breakdown,
        compute_publishability,
    )

    repo = ClipsRepo(db)
    clip = await repo.get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")

    safe_zone_target = "tiktok"
    if isinstance(clip.overlay_config, dict):
        szp = clip.overlay_config.get("safe_zone_platform")
        if isinstance(szp, str) and szp:
            safe_zone_target = szp

    # --- Snapshot the BEFORE state for the response delta -----
    before_overlay = (
        clip.overlay_config if isinstance(clip.overlay_config, dict) else {}
    )
    breakdown = await clip_breakdown(db, clip_id)
    before_verdict = compute_publishability(
        breakdown=breakdown,
        overlay_config=before_overlay,
        safe_zone_platform=safe_zone_target,
    )

    # --- Apply fixes ------------------------------------------
    # N.1 — pull the operator's saved brand-kit URL/handle so the
    # auto-fixer can fill an empty banner.url. Avoids the "no banner
    # because no URL was typed" trap that kept clips below the
    # publish-ready threshold.
    bk_url: str | None = None
    try:
        from nexoclip.branding import resolve_brand_kit_for_candidate
        _bk = await resolve_brand_kit_for_candidate(
            db, stream_id=clip.stream_id, speaker_label=None
        )
        if _bk is not None:
            for attr in ("handle_kick", "handle_tiktok", "handle_youtube",
                         "handle_instagram"):
                v = getattr(_bk, attr, None)
                if isinstance(v, str) and v.strip():
                    bk_url = v.strip()
                    break
    except Exception:  # noqa: BLE001 — best-effort
        bk_url = None

    result = apply_ai_fixes(
        overlay_config=before_overlay,
        safe_zone_platform=safe_zone_target,
        brand_kit_url=bk_url,
    )

    # --- Compute the AFTER state ------------------------------
    after_verdict = compute_publishability(
        breakdown=breakdown,
        overlay_config=result.new_overlay_config,
        safe_zone_platform=safe_zone_target,
    )

    auto_promoted = False
    if result.fixes:
        await repo.set_overlay_config(clip_id, overlay_config=result.new_overlay_config)
        await EventsRepo(db).emit(
            type="clip.ai_fixes_applied",
            payload={
                "clip_id": clip_id,
                "fix_count": len(result.fixes),
                "fields": [f.field for f in result.fixes],
                "score_before": before_verdict.score,
                "score_after": after_verdict.score,
            },
        )

        # Auto-promote `cut` → `ready_for_review` when the AI's fix
        # lands us in publish_ready territory. The operator's clip
        # status moves from "draft" to "ready" without a manual click
        # — Auto Fix becomes a real one-click ship lane.
        allowed = _VALID_STATUS_TRANSITIONS.get(clip.status, set())
        if (
            after_verdict.status == "publish_ready"
            and clip.status == "cut"
            and "ready_for_review" in allowed
        ):
            await repo.update_status(clip_id, status="ready_for_review")
            auto_promoted = True
            await EventsRepo(db).emit(
                type="clip.auto_promoted",
                payload={
                    "clip_id": clip_id,
                    "from_status": "cut",
                    "to_status": "ready_for_review",
                    "score": after_verdict.score,
                    "trigger": "ai_fixes",
                },
            )

    return Response(
        content=_json.dumps(
            {
                "ok": True,
                "fix_count": len(result.fixes),
                "fixes": [
                    {
                        "field": f.field,
                        "before": _json_safe(f.before),
                        "after": _json_safe(f.after),
                        "why": f.why,
                    }
                    for f in result.fixes
                ],
                # N.1 — score delta + status info for the editor JS
                # to animate. Both scores are 0-100 ints.
                "score_before": before_verdict.score,
                "score_after": after_verdict.score,
                "status_before": before_verdict.status,
                "status_after": after_verdict.status,
                "auto_promoted": auto_promoted,
                "auto_promoted_to": "ready_for_review" if auto_promoted else None,
            }
        ),
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


def _json_safe(value: object) -> object:
    """Coerce a fix's before/after into JSON-serializable scalars."""
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    return str(value)


@router.post(
    "/me/brand-kit-prefs",
    dependencies=[Depends(require_full_scope)],
)
async def me_brand_kit_prefs(
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice H.1 — auto-save endpoint for the editor's right panel.

    The clip-editor JS debounce-calls this on every form change so the
    operator's setup (URL, banner toggle, caption position / font size
    / animation / lead-time, etc.) lands in the tenant's default
    brand_kit immediately. The brand_kit becomes the canonical source
    of truth for user-level setup; every new clip's editor reads from
    it on render.

    Body is a single-field JSON patch:

        { "field": "banner_url",  "value": "aldovillanueva" }
        { "field": "banner_enabled", "value": true }
        { "field": "captions_lead_ms", "value": 150 }

    Returns `{"ok": true}` on success. Failures return a 400 so the JS
    can surface the error inline without breaking the page.

    SINGLE source of truth: this endpoint and the existing
    `_persist_branding_to_brand_kit` (which fires on Save Draft /
    Finalize) both write to the same brand_kit columns — no risk of
    drift because the brand_kit row is the only writer destination.
    """
    import json as _json

    body_bytes = await request.body()
    try:
        body = _json.loads(body_bytes or b"{}")
    except _json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    field = str(body.get("field") or "").strip()
    value: object = body.get("value")
    if not field:
        raise HTTPException(status_code=400, detail="missing 'field'")

    # Wrap the single-field patch into the same `cfg` shape that
    # `_persist_branding_to_brand_kit` already knows how to consume.
    # That keeps the brand-kit-write logic in exactly one place.
    cfg: dict[str, object] = {"banner": {}, "captions": {}, "top_hook": {}}
    banner_block: dict[str, object] = cfg["banner"]   # type: ignore[assignment]
    captions_block: dict[str, object] = cfg["captions"]  # type: ignore[assignment]
    top_hook_block: dict[str, object] = cfg["top_hook"]  # type: ignore[assignment]

    _BANNER_FIELDS = {
        "banner_enabled": ("enabled", bool),
        "banner_platform": ("platform", str),
        "banner_url": ("url", str),
        "banner_color": ("color", str),
        "banner_show_context": ("show_context", bool),
        "banner_show_safezones": ("show_safezones", bool),
        # Slice I.1.
        "banner_variant": ("variant", str),
        "banner_live_badge": ("live_badge", bool),
    }
    _CAPTION_FIELDS = {
        "captions_preset": ("preset", str),
        "captions_highlight_color": ("highlight_color", str),
        "captions_position": ("position", str),
        "captions_font_size": ("font_size", str),
        "captions_animation": ("animation", str),
        "captions_lead_ms": ("lead_ms", int),
    }
    # Slice I.1 — top hook + clip_style live at the top level / their
    # own block. clip_style is a flat top-level key on the cfg dict.
    _TOP_HOOK_FIELDS = {
        "top_hook_enabled": ("enabled", bool),
        "top_hook_text": ("text", str),
        "top_hook_style": ("style", str),
    }

    # `clip_style` is the only top-level brand-kit-bound editor field.
    # The I.2 editor-only fields (platform_overlay_preview / safe_zone_
    # platform / preview_mode) are persisted client-side via localStorage
    # rather than written to the brand_kit — they're pure editor state
    # and don't belong in the user's branding record. They still ride
    # the Save Draft path into clips.overlay_config_json so a per-clip
    # override is honored when set, but day-to-day persistence is JS.
    # Slice K.5 — `target_platform` is also brand-kit-bound: it's the
    # operator's "where I post" sticky default, so the next clip opens
    # to the same chip pre-selected and every downstream field
    # auto-applies from PLATFORM_PRESETS.
    if field == "clip_style":
        cfg["clip_style"] = _coerce(value, str)
    elif field == "target_platform":
        cfg["target_platform"] = _coerce(value, str)
    elif field in _BANNER_FIELDS:
        key, kind = _BANNER_FIELDS[field]
        banner_block[key] = _coerce(value, kind)
    elif field in _CAPTION_FIELDS:
        key, kind = _CAPTION_FIELDS[field]
        captions_block[key] = _coerce(value, kind)
    elif field in _TOP_HOOK_FIELDS:
        key, kind = _TOP_HOOK_FIELDS[field]
        top_hook_block[key] = _coerce(value, kind)
    else:
        raise HTTPException(status_code=400, detail=f"unknown field {field!r}")

    await _persist_branding_to_brand_kit(db, cfg)
    return Response(
        content='{"ok": true}',
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


def _coerce(value: object, kind: type) -> object:
    """Best-effort JSON value → target type. Falsy strings collapse to
    None so an empty input clears a setting instead of writing ''."""
    if kind is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "on", "yes")
        return bool(value)
    if kind is int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int | float):
            return int(value)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value)
        return 0
    if kind is str:
        return str(value).strip() if value is not None else ""
    return value


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
    # Slice I.1 — clip style + Kick banner variant + top hook.
    clip_style: str = Form(""),
    banner_variant: str = Form(""),
    banner_live_badge: str = Form(""),
    top_hook_enabled: str = Form(""),
    top_hook_text: str = Form(""),
    top_hook_style: str = Form(""),
    # Slice I.2 — editor-only platform simulation + safe-zone target.
    platform_overlay_preview: str = Form(""),
    safe_zone_platform: str = Form(""),
    preview_mode: str = Form(""),
    # Slice K.5 — target-platform auto-config (one chip → all settings).
    target_platform: str = Form(""),
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
        clip_style=clip_style,
        banner_variant=banner_variant,
        banner_live_badge=banner_live_badge,
        top_hook_enabled=top_hook_enabled,
        top_hook_text=top_hook_text,
        top_hook_style=top_hook_style,
        platform_overlay_preview=platform_overlay_preview,
        safe_zone_platform=safe_zone_platform,
        preview_mode=preview_mode,
        target_platform=target_platform,
        comments_show=comments_show,
        comments_fake_likes=comments_fake_likes,
    )
    repo = ClipsRepo(db)
    clip_before = await repo.get(clip_id)
    await repo.set_overlay_config(clip_id, overlay_config=cfg)
    # Slice F.7-G — mirror branding choices to the tenant brand_kit
    # so the operator doesn't re-type URL / color on every clip.
    await _persist_branding_to_brand_kit(db, cfg)

    # Slice O.17 — burn-on-save REMOVED. The previous O.12 behavior
    # re-encoded clip_final.mp4 on every editor save, which was both
    # wasteful (operator might save 10× while iterating) and useless
    # (any subsequent save invalidates the prior burn anyway). The
    # download / publish path already lazy-regenerates clip_final.mp4
    # if it's missing — so saves become DB-only writes, and the burn
    # happens exactly when an MP4 is actually about to leave the
    # system.
    #
    # We MUST nuke any cached export here — otherwise the download
    # endpoint sees a file on disk + skips the lazy regen + serves
    # an MP4 produced with the OLD overlay config. Both the new
    # Playwright-rendered cache AND the legacy ffmpeg burn cache get
    # invalidated.
    if clip_before is not None:
        try:
            clip_dir = Path(clip_before.path).parent
            for stale in (clip_dir / "clip_render.mp4", clip_dir / "clip_final.mp4"):
                if stale.exists():
                    stale.unlink()
        except Exception:  # noqa: BLE001 — invalidation is best-effort
            pass
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
    # Slice I.1 — clip style + Kick banner variant + top hook.
    clip_style: str = Form(""),
    banner_variant: str = Form(""),
    banner_live_badge: str = Form(""),
    top_hook_enabled: str = Form(""),
    top_hook_text: str = Form(""),
    top_hook_style: str = Form(""),
    # Slice I.2 — editor-only platform simulation + safe-zone target.
    platform_overlay_preview: str = Form(""),
    safe_zone_platform: str = Form(""),
    preview_mode: str = Form(""),
    # Slice K.5 — target-platform auto-config (one chip → all settings).
    target_platform: str = Form(""),
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
        clip_style=clip_style,
        banner_variant=banner_variant,
        banner_live_badge=banner_live_badge,
        top_hook_enabled=top_hook_enabled,
        top_hook_text=top_hook_text,
        top_hook_style=top_hook_style,
        platform_overlay_preview=platform_overlay_preview,
        safe_zone_platform=safe_zone_platform,
        preview_mode=preview_mode,
        target_platform=target_platform,
        comments_show=comments_show,
        comments_fake_likes=comments_fake_likes,
    )
    await repo.set_overlay_config(clip_id, overlay_config=cfg)
    # Slice F.7-G — see note in clip_overlay_save: persists branding
    # to the tenant brand_kit so the *next* clip prefills it.
    await _persist_branding_to_brand_kit(db, cfg)
    if target != clip.status:
        await repo.update_status(clip_id, status=target)

    # Slice O.37 — Approve no longer blocks on ffmpeg burn. O.17 took
    # the burn out of the save path; finalize was still doing a
    # synchronous re-encode, which is what made "Continue & close"
    # spin for 30-90s. The download path lazy-regenerates a fresh
    # MP4 (Playwright recorder, fallback ffmpeg burn) from the saved
    # overlay config, so we just nuke any stale cache and bounce.
    try:
        clip_dir = Path(clip.path).parent
        for stale in (clip_dir / "clip_render.mp4", clip_dir / "clip_final.mp4"):
            if stale.exists():
                stale.unlink()
    except Exception:  # noqa: BLE001 — invalidation is best-effort
        pass
    await EventsRepo(db).emit(
        type="clip.finalized",
        payload={
            "clip_id": clip_id,
            "to_status": target,
            "burn_outcome": "deferred_to_download",
        },
    )

    # Slice N.2 — Approve & continue. After finalize, walk to the
    # NEXT clip on the same stream that's still in the editor queue
    # (status in `cut` / `ready_for_review`). If none remain, fall
    # back to the stream detail page so the operator sees they're
    # done. The "next clip" definition: earliest created_at among
    # remaining draft clips on this stream, excluding the one we
    # just approved.
    siblings = await repo.list_for_stream(clip.stream_id)
    next_clip = None
    for c in sorted(siblings, key=lambda x: x.created_at):
        if c.id == clip_id:
            continue
        if c.status in ("cut", "ready_for_review"):
            next_clip = c
            break

    if next_clip is not None:
        return RedirectResponse(
            url=f"/dashboard/clips/{next_clip.id}?from_approve=1",
            status_code=303,
        )
    return RedirectResponse(
        url=f"/dashboard/streams/{clip.stream_id}?queue_done=1",
        status_code=303,
    )


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
    from nexoclip.db import TenantsRepo, TranscriptsRepo

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

    # Slice O.1 — resolve whether to burn the nexoclip.com watermark.
    # Free tier always; pro+ respects the brand_kit toggle. Best-effort
    # — we never block a burn on missing tier/kit data.
    render_wm = True
    try:
        tenant = await TenantsRepo(db).get(clip.tenant_id)
        tier = (tenant.tier if tenant else "free") or "free"
        if tier != "free":
            from nexoclip.branding import resolve_brand_kit_for_candidate
            kit = await resolve_brand_kit_for_candidate(
                db, stream_id=clip.stream_id, speaker_label=None
            )
            if kit is not None and not getattr(kit, "show_nexoclip_credit", True):
                render_wm = False
    except Exception:  # noqa: BLE001 — best-effort
        render_wm = True

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
            render_watermark=render_wm,
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


@router.get("/streams/{stream_id}/download-approved")
async def stream_download_approved(
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice O.1 — bulk extract: zip every approved/published clip
    on this stream into a single download. Operator's "Download all"
    flow at the top of the Clips section.

    Builds the zip in-memory (clips are short, the entire bundle
    rarely exceeds a few hundred MB). For a stream with hundreds of
    finalized clips this would warrant streaming-zip-on-disk, but
    that's a J.2 follow-up — short-form clip pipelines max out
    around 20-40 clips per stream.
    """
    import io
    import zipfile

    repo = ClipsRepo(db)
    clips = await repo.list_for_stream(stream_id)
    if not clips:
        raise HTTPException(status_code=404, detail="no clips on this stream")
    # Pick the ones the operator would actually want to ship —
    # `approved` (ready to publish) + `published` (already shipped
    # but operator wants a backup copy).
    ready = [c for c in clips if c.status in ("approved", "published")]
    if not ready:
        raise HTTPException(
            status_code=404,
            detail="no approved clips yet — approve some first",
        )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        # Lower compresslevel = MP4s are already compressed; we only
        # want zip's container behavior, not double-compression CPU.
        for i, c in enumerate(ready, 1):
            original = Path(c.path)
            final = original.parent / "clip_final.mp4"
            src = final if final.exists() else original
            if not src.exists():
                continue
            # Index-prefixed name so the order in the zip mirrors the
            # order on the stream page.
            zf.write(src, arcname=f"{i:02d}_nexoclip_{c.id}.mp4")

    buf.seek(0)
    body = buf.read()
    fname = f"nexoclip_stream_{stream_id}.zip"
    return Response(
        content=body,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{fname}"',
            "Content-Length": str(len(body)),
            "Cache-Control": "no-store",
        },
    )


@router.get("/clips/{clip_id}/download")
async def clip_download(
    request: Request,
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> FileResponse:
    """Slice O.21 — download the clip's headless-Chrome rendered MP4.

    Replaces the previous overlay_burn.py pipeline. The exported MP4 is
    now produced by Playwright recording the `/clips/<id>/render` page
    at 1080×1920 native, then muxing the source clip's audio track
    lossless. By construction the output is pixel-identical to the
    editor preview — same browser engine renders both.

    `overlay_burn.py` is intentionally kept in the repo (slice O.21
    decision) but no longer called from any endpoint. We retain it as
    a fallback option in case Playwright fails at runtime — see the
    except branch below.

    Cache: the recorder writes `clip_render.mp4` next to the source
    clip; subsequent downloads serve the cached file unless the
    operator saved new overlay settings (which deletes the cache via
    `clip_overlay_save`).
    """
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    original = Path(clip.path)
    rendered = original.parent / "clip_render.mp4"

    # Generate on-demand if we don't have a cache. The cache is
    # invalidated by `clip_overlay_save` (slice O.17), so this regen
    # only happens once per (clip × overlay_config) pair.
    if not rendered.exists() and original.exists():
        # Pass the same session cookie the operator just used so the
        # render page (which sits behind the dashboard auth middleware)
        # loads correctly. Cookie value comes from the inbound request.
        from nexoclip.settings import get_settings
        from nexoclip.clip.preview_recorder import (
            record_clip_to_mp4, PreviewRecordingError,
        )
        settings = get_settings()
        cookie_val = request.cookies.get("nexoclip_token", "")

        # Slice O.31 — derive the recorder's base URL from the inbound
        # request's Host header by default. The previous version relied
        # on `settings.public_url` which defaults to `http://localhost:8000`
        # and breaks the moment we deploy anywhere; on Railway it caused
        # Playwright to dial localhost (nothing listening) → timeout →
        # PreviewRecordingError → fallback to legacy ffmpeg burn (the
        # plain-captions output the operator complained about).
        # Using the request's own host is correct by construction — the
        # browser already routed to this server, so it can route to
        # itself with the same URL. `NEXOCLIP_PUBLIC_URL` still wins
        # when explicitly set so reverse-proxy setups can pin it.
        explicit_base = (settings.public_url or "").strip()
        if explicit_base and explicit_base != "http://localhost:8000":
            base_url = explicit_base
        else:
            scheme = (
                request.headers.get("x-forwarded-proto")
                or request.url.scheme
                or "https"
            )
            host = request.headers.get("host") or request.url.netloc
            base_url = f"{scheme}://{host}"

        try:
            await record_clip_to_mp4(
                clip_id=clip_id,
                duration_s=float(clip.duration_s),
                audio_source_path=original,
                output_path=rendered,
                base_url=base_url,
                auth_cookie_value=cookie_val or None,
            )
        except PreviewRecordingError as e:
            # Slice O.31 — surface the recorder's error LOUDLY now so we
            # can finally see why it falls back. Previously this was a
            # warning log + silent legacy burn; the operator saw the
            # legacy MP4 + no idea Playwright had failed.
            import structlog
            structlog.get_logger("nexoclip.api.dashboard").error(
                "clip_download.recorder_failed",
                clip_id=clip_id,
                base_url=base_url,
                error=str(e),
            )
            legacy_final = original.parent / "clip_final.mp4"
            if not legacy_final.exists():
                try:
                    cfg = clip.overlay_config or {}
                    await _burn_overlays_for_clip(
                        db=db, clip_id=clip_id, overlay_config=cfg,
                    )
                except Exception:  # noqa: BLE001 — best-effort fallback
                    pass
            rendered = legacy_final  # serve burned MP4 if it exists

    clip_path = rendered if rendered.exists() else original
    if not clip_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"clip file missing from disk: {clip_path}",
        )
    pretty = f"nexoclip_{clip_id}.mp4"
    return FileResponse(
        path=clip_path,
        media_type="video/mp4",
        filename=pretty,
        headers={"Content-Disposition": f'attachment; filename="{pretty}"'},
    )


@router.get("/clips/{clip_id}/render", response_class=HTMLResponse)
async def clip_render_view(
    request: Request,
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice O.19 — minimal render-mode page for headless capture.

    Returns a stripped-down HTML page rendering ONLY the .nc-preview
    frame at 1080×1920 native. No editor chrome (nav, sidebar, tabs,
    controls). Playwright (slice O.20) opens this URL at viewport
    1080×1920 and screen-records the playback. What you see here IS
    what gets exported — by construction, no separate burn pipeline.

    Tenant-scoped: the underlying ClipsRepo.get filters by current
    tenant context, so cross-tenant clip-id guessing 404s.
    """
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")

    # Resolve banner URL like the editor template does, so the chrome
    # banner shows the canonical "KICK.COM/HANDLE" form.
    ov = clip.overlay_config or {}
    banner = ov.get("banner") if isinstance(ov.get("banner"), dict) else {}
    banner_url_display = ""
    if isinstance(banner, dict) and banner.get("enabled", False):
        from nexoclip.clip.overlay_burn import _format_kick_url  # type: ignore[attr-defined]
        try:
            banner_url_display = _format_kick_url(str(banner.get("url") or ""))
        except Exception:  # noqa: BLE001
            banner_url_display = str(banner.get("url") or "").upper()

    # Tier-aware watermark: free always; pro+ honors brand_kit toggle.
    render_watermark = True
    try:
        from nexoclip.db import TenantsRepo
        tenant = await TenantsRepo(db).get(clip.tenant_id)
        tier = (tenant.tier if tenant else "free") or "free"
        if tier != "free":
            from nexoclip.branding import resolve_brand_kit_for_candidate
            kit = await resolve_brand_kit_for_candidate(
                db, stream_id=clip.stream_id, speaker_label=None
            )
            if kit is not None and not getattr(kit, "show_nexoclip_credit", True):
                render_watermark = False
    except Exception:  # noqa: BLE001 — best-effort
        render_watermark = True

    return templates.TemplateResponse(
        request,
        "clip_render.html",
        {
            "clip": clip,
            "banner_url_display": banner_url_display,
            "render_watermark": render_watermark,
        },
    )


@router.get("/clips/{clip_id}/media")
async def clip_media(
    clip_id: str,
    source: str = "original",
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> FileResponse:
    """Stream the cut MP4 for inline <video> playback on the clip detail page.

    Slice M.6 — defaults to the ORIGINAL `clip.mp4` (no burns).
    Operator-reported bug: the editor was serving `clip_final.mp4`
    (which has captions BAKED INTO PIXELS) while the live preview
    ALSO renders styled overlay captions on top — operators saw
    every caption line twice: the burned plain-white one underneath
    + the karaoke-styled overlay on top.

    The editor's job is to compose; the burn is what we SHIP. To see
    the burned-final pixels, pass `?source=final` (used only by the
    "Final" preview-mode tab, if/when wired). Everywhere else gets
    the source so the styled overlays don't conflict with anything.

    Returns 404 if the clip row is missing or the on-disk file disappeared
    (e.g., out/ was nuked between runs). Tenant-bound so one tenant can't
    fetch another's clip even by guessing the id.
    """
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")
    original = Path(clip.path)
    final = original.parent / "clip_final.mp4"

    # Slice O.13 — when the editor's "Final" tab requests the burned
    # version and it's missing on disk, regenerate on the fly using
    # the saved overlay_config. Guarantees the Final tab is never a
    # broken video element. Same auto-burn pattern as the download
    # endpoint (slice O.12).
    if source == "final" and not final.exists() and original.exists():
        try:
            cfg = clip.overlay_config or {}
            await _burn_overlays_for_clip(
                db=db, clip_id=clip_id, overlay_config=cfg
            )
        except Exception:  # noqa: BLE001 — non-fatal, fall back below
            pass

    if source == "final" and final.exists():
        clip_path = final
    else:
        # Default: ORIGINAL (no captions burned in pixels). Fall back
        # to final only if the original is missing for some reason.
        clip_path = original if original.exists() else final
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

    # Slice M.1 — surface the voice-trigger phrase that fired this
    # clip as a first-class timeline marker. Operator reported the
    # missing signal: "en el video dije clipea esto y no lo marco
    # abajo como que escucho al usuario y como genero el clip basado
    # en eso". The candidate's evidence carries the phrase + the
    # timestamp; the timeline now shows it as a green dot labeled
    # 🎯 with the matched phrase so the operator FEELS the AI's
    # detection.
    #
    # Slice M.3 — `compute_intelligence` now ALSO scans the clip's
    # transcript for trigger phrases (independent of how the candidate
    # was generated). When that scan already found the phrase, this
    # candidate-evidence path is a duplicate — skip it.
    transcript_already_found_trigger = any(
        m.kind == "voice_trigger" for m in markers
    )
    trigger_marker: dict[str, object] | None = None
    if clip.candidate_id and not transcript_already_found_trigger:
        for cand in await CandidatesRepo(db).list_for_stream(clip.stream_id):
            if cand.id == clip.candidate_id:
                ev = cand.evidence or {}
                if isinstance(ev, dict):
                    phrase = ev.get("phrase")
                    if isinstance(phrase, str) and phrase:
                        # Convert absolute stream timestamp to clip-
                        # relative. cand.timestamp is the phrase START
                        # in the stream; the clip starts at clip.start_s.
                        clip_rel = max(0.0, float(cand.timestamp) - clip.start_s)
                        kind_label = (
                            "Voice trigger fired"
                            if ev.get("trigger_kind") != "retroactive"
                            else "Retroactive trigger fired"
                        )
                        trigger_marker = {
                            "kind": "voice_trigger",
                            "ts": clip_rel,
                            "score": float(ev.get("confidence") or 0.9),
                            "label": f"🎯 {kind_label}: \"{phrase}\"",
                        }
                break

    out_markers: list[dict[str, object]] = [
        {
            "kind": m.kind,
            "ts": m.ts,
            "score": m.score,
            "label": m.label,
        }
        for m in markers
    ]
    # Sort by ts and put the trigger first so it leads the legend.
    if trigger_marker is not None:
        out_markers.insert(0, trigger_marker)

    return Response(
        content=_json.dumps(
            {
                "markers": out_markers,
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
        # Slice N.2 — "window_outside_transcript" was a catch-all
        # that incorrectly fired even when the clip window WAS inside
        # the transcript span but just happened to have no spoken
        # words (silence between sentences, music-only stretches,
        # crossfade gaps). Now we actually CHECK the span boundaries
        # before claiming the window is outside; otherwise the
        # diagnostic is simply "silent stretch" — the clip's audio
        # band had no transcribable speech.
        if lines:
            diag["reason"] = "ok"
        elif diag["transcript_segment_count"] == 0:
            diag["reason"] = "transcript_empty"
        elif diag["transcript_word_count"] == 0:
            diag["reason"] = "no_word_timestamps"
        else:
            span = diag.get("transcript_span_s")
            if (
                isinstance(span, list)
                and len(span) == 2
                and (clip.end_s < span[0] or clip.start_s > span[1])
            ):
                diag["reason"] = "window_outside_transcript"
            else:
                # Clip window IS inside the transcript span but no
                # words fall in it — the audio band itself is silent /
                # non-verbal across this stretch.
                diag["reason"] = "silent_stretch"
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


@router.post(
    "/streams/{stream_id}/delete",
    dependencies=[Depends(require_full_scope)],
)
async def stream_delete(
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice O.11 — hard-delete a stream + cascade everything beneath it.

    Two-phase: DB cascade first (FKs handle clips/candidates/variants/
    publish_jobs/etc), then best-effort filesystem cleanup of the
    per-stream output directory derived from `source_video_path`.

    Blocks deletion while the pipeline is running so we don't kneecap
    a worker mid-flight (the cascade would yank rows it's still
    writing to). Ingested / done / failed streams are fair game.
    """
    repo = StreamsRepo(db)
    stream = await repo.get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="stream not found")
    if stream.status == "running":
        raise HTTPException(
            status_code=409,
            detail="stream is currently running — wait for it to finish or fail before deleting",
        )

    # Resolve the on-disk stream directory BEFORE the DB row goes
    # away. The convention is `<output_dir>/<stream_id>/source.<ext>`,
    # so the parent of source_video_path IS the per-stream dir. Using
    # the row's own path is safer than reconstructing — it survives
    # output_dir config drift between ingest time and delete time.
    stream_dir: Path | None = None
    try:
        src = Path(stream.source_video_path).resolve()
        parent = src.parent
        # Guard: only nuke the directory if its name is the stream id.
        # Belt-and-suspenders against a misconfigured stream pointing
        # at, say, `/Users/picasso/Movies/source.mp4` whose parent is
        # NOT a disposable stream dir.
        if parent.name == stream_id:
            stream_dir = parent
    except Exception:  # noqa: BLE001 — path parsing failures are non-fatal
        stream_dir = None

    deleted = await repo.delete(stream_id)
    if not deleted:
        # Shouldn't happen — we just fetched the row — but if a parallel
        # delete beat us, treat it as a no-op success.
        return RedirectResponse(url="/dashboard/streams", status_code=303)

    # Best-effort filesystem cleanup. Failures here don't fail the
    # delete — the DB row is already gone, the operator's goal is
    # met. Worst case there's a stale folder of mp4s the operator
    # can rm by hand.
    if stream_dir and stream_dir.exists():
        import shutil
        try:
            shutil.rmtree(stream_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass

    await EventsRepo(db).emit(
        type="stream.deleted",
        payload={"stream_id": stream_id, "fs_dir": str(stream_dir) if stream_dir else None},
    )
    return RedirectResponse(url="/dashboard/streams?deleted=1", status_code=303)


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


# Slice H.1 — manual status transitions removed.
# Status now derives entirely from the lifecycle handlers:
#   - cut/ready_for_review → approved : via clip_overlay_finalize ("Ship")
#   - any              → rejected : via clip_reject ("Reject & close")
#   - approved         → published: via the publish worker after a
#                                   successful platform post
# The legacy PATCH /clips/{id}/status endpoint + dashboard panel are
# gone. The full lifecycle table still lives in
# nexoclip.api.routers.clips._VALID_STATUS_TRANSITIONS for the
# bearer-token JSON API path (PATCH /api/clips/{id}/status) — that
# stays available for headless automations; only the dashboard's
# manual-clicker is removed.


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
    redirect_to: str = Form(""),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice O.3 — accept a `redirect_to` form field so the inline
    "Connect TikTok" modal on the stream publish page can drop the
    operator right back on the matrix view after they finish the
    connection (instead of bouncing them to the standalone
    /connected-accounts page). Defaults to the legacy redirect for
    requests that don't pass it."""
    await ConnectedAccountsRepo(db).create(
        platform=platform,
        external_id=external_id,
        display_name=display_name or None,
        oauth_blob={"access_token": access_token},
    )
    # Only honor same-origin paths so nobody can use this as an open
    # redirect. Whitelisted prefixes match real dashboard routes.
    safe_redirect = "/dashboard/connected-accounts"
    if redirect_to.startswith("/dashboard/"):
        safe_redirect = redirect_to
    return RedirectResponse(url=safe_redirect, status_code=303)


@router.post(
    "/connected-accounts/{account_id}/update",
    dependencies=[Depends(require_full_scope)],
)
async def accounts_update(
    account_id: str,
    external_id: str = Form(""),
    display_name: str = Form(""),
    access_token: str = Form(""),
    redirect_to: str = Form(""),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice O.7 — edit an existing connected account.

    Called from the "Edit connection" modal on the publish page when
    the operator clicks an already-connected platform chip. Partial
    update: blank `access_token` keeps the existing one (so the
    operator can rename without re-pasting a long-lived secret).
    """
    repo = ConnectedAccountsRepo(db)
    existing = await repo.get(account_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="connection not found")

    await repo.update_meta(
        account_id,
        display_name=display_name if display_name else None,
        external_id=external_id if external_id else None,
    )
    # Replace the access token only if the operator typed something.
    # Empty string is the deliberate "leave it alone" signal — that's
    # what the modal placeholder explains to the user.
    if access_token.strip():
        await repo.update_oauth(
            account_id,
            oauth_blob={"access_token": access_token.strip()},
        )
    # If the account had failed auth previously, a token replacement
    # should flip it back to active. We do this unconditionally on
    # update since the operator just confirmed the connection.
    if existing.status != "active":
        await repo.mark_status(account_id, "active")

    safe_redirect = "/dashboard/connected-accounts"
    if redirect_to.startswith("/dashboard/"):
        safe_redirect = redirect_to
    return RedirectResponse(url=safe_redirect, status_code=303)


@router.post(
    "/connected-accounts/{account_id}/disconnect",
    dependencies=[Depends(require_full_scope)],
)
async def accounts_disconnect(
    account_id: str,
    redirect_to: str = Form(""),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice O.7 — mark a connection as `disabled`.

    We don't hard-delete because there may be publish_jobs that point
    at it; the schema declares ON DELETE RESTRICT for that exact reason.
    Disabling hides it from the publish UI without orphaning anything.
    """
    repo = ConnectedAccountsRepo(db)
    existing = await repo.get(account_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="connection not found")
    await repo.mark_status(account_id, "disabled")
    safe_redirect = "/dashboard/connected-accounts"
    if redirect_to.startswith("/dashboard/"):
        safe_redirect = redirect_to
    return RedirectResponse(url=safe_redirect, status_code=303)


@router.post(
    "/publish-jobs/{job_id}/retry",
    dependencies=[Depends(require_full_scope)],
)
async def publish_job_retry(
    job_id: str,
    redirect_to: str = Form(""),
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> Response:
    """Slice O.7 — flip a failed job back to `pending`.

    Surfaced from the publish-page failures panel ("Retry" button next
    to each error). Doesn't reset the attempts counter — we want the
    history visible so chronic failures are spottable.
    """
    flipped = await PublishJobsRepo(db).retry(job_id)
    if not flipped:
        # Job either doesn't exist or wasn't in `failed` state. Return
        # 404 so the JS poller can surface the issue if it ever calls
        # this directly.
        raise HTTPException(status_code=404, detail="job not retryable")
    safe_redirect = "/dashboard/connected-accounts"
    if redirect_to.startswith("/dashboard/"):
        safe_redirect = redirect_to
    return RedirectResponse(url=safe_redirect, status_code=303)


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
