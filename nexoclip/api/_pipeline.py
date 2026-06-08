"""PipelineKickoff bundle + default runner — re-export shim.

The canonical home for these types is `nexoclip.jobs` (slice F.8).
Existing imports `from nexoclip.api._pipeline import PipelineKickoff`
keep working via the re-exports below; new code should import from
`nexoclip.jobs` directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nexoclip.jobs import PipelineKickoff, PipelineRunner

if TYPE_CHECKING:
    from pathlib import Path

_log = logging.getLogger("nexoclip.pipeline.runner")


async def default_pipeline_runner(kickoff: PipelineKickoff) -> None:
    """Real `process_vod` invocation, lazily imported so tests that don't
    kick off the pipeline don't pay for Whisper / CV / yt-dlp module-load
    cost.

    Forwards the configured `db_path` so `_step()` can write
    pipeline.step.{start,done,failed} rows to the events table — without
    this, the dashboard's progress card has no events to read and stays
    permanently stuck on 'all six steps pending'.

    ERROR SURFACING (slice NX.5):
        process_vod raises in two distinct ways:
          (a) Inside a `_step(...)` context — that context manager already
              emits `pipeline.step.failed` with the error. The progress
              endpoint reads those events and renders the step in red.
          (b) BEFORE entering any step (e.g. persona-not-found at
              pipeline.py:316, db_path bad, etc). No event gets emitted,
              the progress endpoint sees all-pending, the UI spins
              forever, the user has no idea what happened.

        We catch (b) here and write a `pipeline.failed` top-level event
        scoped to the stream_id. The progress endpoint flags this as a
        terminal failure even when no step has been started. Then we
        re-raise so FastAPI's BackgroundTasks logs the full traceback
        for the operator.
    """
    from nexoclip.pipeline import process_vod
    from nexoclip.settings import get_settings

    db_path = get_settings().db_path

    try:
        await process_vod(
            tenant_id=kickoff.tenant_id,
            vod_url=kickoff.stream.vod_url,
            output_dir=kickoff.output_dir,
            persona_id=kickoff.persona_id,
            stream_id=kickoff.stream.id,
            language=kickoff.language,
            db_path=db_path,
        )
    except Exception as e:
        # Best-effort: emit a top-level failure event so the dashboard's
        # progress card surfaces the error instead of spinning. We catch
        # broadly because we can't anticipate every internal exception type
        # — VariantError, IngestError, ValidationError, RuntimeError,
        # KeyError on bad config, etc.
        try:
            await _emit_top_level_failure(
                db_path=db_path,
                tenant_id=kickoff.tenant_id,
                stream_id=kickoff.stream.id,
                error=e,
            )
        except Exception:
            # If even the event write fails, fall through to re-raise. The
            # original error matters more than the dashboard UX.
            _log.exception(
                "pipeline.failed event write failed for stream=%s",
                kickoff.stream.id,
            )
        # Always re-raise so FastAPI logs the full traceback for the
        # operator to debug from Railway logs.
        raise

    # Token T2 — run succeeded (a failure re-raises above). Charge the
    # per-run base fee + force a live Nexo AI balance refresh so the token
    # chip reflects what this run just consumed without the operator
    # clicking it.
    await _refresh_balance_after_run(
        db_path=db_path,
        tenant_id=kickoff.tenant_id,
        stream_id=kickoff.stream.id,
    )


# Grace delay before the post-run balance pull. The pipeline's final LLM /
# transcription usage reports are fire-and-forget (3s timeout each); this
# lets the last ones land at Nexo AI so the fetched balance reflects the
# full run, not a mid-deduction snapshot.
_BALANCE_REFRESH_GRACE_S = 4.0


async def _refresh_balance_after_run(
    *, db_path: str, tenant_id: str, stream_id: str | None = None,
) -> None:
    """Token T2/T3 — after a successful run: (1) charge the per-run base
    fee, (2) pull the live Nexo AI balance into the cache so the chip is
    fresh without a manual click.

    Best-effort: the base charge is awaited (so it lands before we read
    the balance), then a short grace delay lets the run's fire-and-forget
    provider reports land, then ONE live fetch. Every error is swallowed —
    the run already succeeded and the 30s chip poll is the fallback."""
    import asyncio

    from nexoclip.db import Database
    from nexoclip.integrations.nexo_ai.balance import fetch_balance_now

    try:
        db = Database(db_path)
        await db.connect()
        try:
            # (1) Per-run base charge — covers server/render/storage
            #     overhead so a near-free-API run still draws down quota.
            if stream_id:
                await _charge_run_base_fee(db, tenant_id, stream_id)
            # (2) Let the run's fire-and-forget provider reports land.
            await asyncio.sleep(_BALANCE_REFRESH_GRACE_S)
            # (3) Live balance fetch → cache → chip.
            await fetch_balance_now(db, tenant_id=tenant_id)
        finally:
            await db.close()
    except Exception:  # noqa: BLE001 — observability, never affects the run
        _log.warning(
            "post-run balance refresh failed for tenant=%s", tenant_id,
        )


async def _charge_run_base_fee(
    db: "Database", tenant_id: str, stream_id: str
) -> None:
    """Token T3 — report the per-run base charge (engine.base) to Nexo AI.

    A flat fee covering server time / render / storage that the raw API
    cost doesn't capture. Idempotent: source_id is base_<stream_id>, so a
    re-run of the same stream is a no-op upstream. Amount is configurable
    via NEXOCLIP_PIPELINE_BASE_CHARGE_USD_MICROS (0 disables). Awaited (not
    fire-and-forget) so the balance fetch right after reflects it."""
    import datetime as _dt

    from nexoclip.integrations.nexo_ai.reporter import report_usage
    from nexoclip.settings import get_settings

    base = int(getattr(get_settings(), "pipeline_base_charge_usd_micros", 0) or 0)
    if base <= 0:
        return
    try:
        await report_usage(
            db,
            tenant_id=tenant_id,
            kind="engine.base",
            amount=1,
            cost_usd_micros=base,
            source_id=f"base_{stream_id}",
            occurred_at_iso=_dt.datetime.now(_dt.UTC).isoformat(),
            provider="nexoclip",
            operation="pipeline_run",
        )
    except Exception:  # noqa: BLE001 — never affects the run
        _log.warning(
            "base-charge report failed for tenant=%s stream=%s",
            tenant_id, stream_id,
        )


async def _emit_top_level_failure(
    *,
    db_path: str,
    tenant_id: str,
    stream_id: str,
    error: Exception,
) -> None:
    """Write a `pipeline.failed` event to the tenant's events table.

    Different from `pipeline.step.failed` (which is scoped to one of the six
    pipeline steps): this fires when the runner's outermost try/except
    catches an exception that escaped the per-step contexts entirely.
    The progress endpoint treats this as a terminal pipeline failure even
    if no step is in 'running' state.
    """
    from nexoclip.db import Database
    from nexoclip.events import emit
    from nexoclip.tenancy import bound_tenant

    # process_vod already opens + closes its own DB session; that one is
    # gone by the time we get here. Open a fresh one just for this write.
    db = Database(db_path)
    await db.connect()
    with bound_tenant(tenant_id):
        await emit(
            db,
            "pipeline.failed",
            {
                "stream_id": stream_id,
                "error_type": type(error).__name__,
                # Keep the message short — the full traceback is in the
                # Railway log. The UI only needs enough text to display.
                "error": str(error)[:500],
            },
        )


async def upload_pipeline_runner(
    *,
    tenant_id: str,
    stream_id: str,
    persona_id: str,
    tmp_path: "Path",
    output_dir: "Path",
    title: str | None,
    language: str | None = None,
) -> None:
    """Run the full pipeline for an uploaded file in the background.

    The HTTP upload endpoint stashes the request body to a tempfile
    inline (unavoidable — the bytes have to arrive before we can
    respond) and then schedules THIS runner so the operator gets a
    303 to the live progress page immediately. Audio extraction +
    transcribe + detect + cut all happen here, asynchronously.

    Phases inside this runner:

      1. `ingest_uploaded()` — moves the tempfile to the canonical
         <output_dir>/<stream_id>/source/video.mp4, extracts audio,
         writes stream.json.
      2. UPSERT the StreamRow with the real values now that we know
         the duration + on-disk paths.
      3. `process_vod()` — picks up the cached stream.json (skipping
         ingest_vod's URL path entirely) and runs the rest of the
         pipeline.

    Failure surfacing matches default_pipeline_runner: any exception
    that escapes the per-step contexts gets a `pipeline.failed` event
    so the dashboard's progress card explains the failure instead of
    spinning forever. We then re-raise so FastAPI logs the traceback.
    """
    from nexoclip.db import Database, StreamsRepo
    from nexoclip.db.adapters import stream_to_row
    from nexoclip.ingest import ingest_uploaded
    from nexoclip.pipeline import process_vod
    from nexoclip.settings import get_settings
    from nexoclip.tenancy import bound_tenant

    settings = get_settings()
    db_path = settings.db_path

    try:
        # Phase 1 — finish ingest (move file + extract audio + persist
        # stream.json). Idempotent on stream.json: if the operator
        # somehow re-triggers this run, the second call returns the
        # cached Stream without re-extracting.
        stream = await ingest_uploaded(
            tenant_id=tenant_id,
            source_path=tmp_path,
            output_dir=output_dir,
            stream_id=stream_id,
            title=title,
        )

        # Phase 2 — promote the placeholder StreamRow we inserted in
        # the endpoint to the real values (duration_s, title from
        # ffprobe, etc.). The pipeline's StreamsRepo.upsert further
        # along would also do this, but doing it here means the
        # dashboard's progress page shows real metadata as soon as
        # ingest finishes instead of waiting for transcribe to start.
        db = Database(db_path)
        await db.connect()
        try:
            with bound_tenant(tenant_id):
                await StreamsRepo(db).upsert(stream_to_row(stream))
        finally:
            await db.close()

        # Phase 3 — run the full pipeline. process_vod sees the
        # cached stream.json and skips ingest_vod entirely (the
        # upload:// pseudo-URL never has to hit yt-dlp).
        await process_vod(
            tenant_id=tenant_id,
            vod_url=stream.vod_url,
            output_dir=output_dir,
            persona_id=persona_id,
            stream_id=stream.id,
            language=language,
            db_path=db_path,
        )
    except Exception as e:
        try:
            await _emit_top_level_failure(
                db_path=db_path,
                tenant_id=tenant_id,
                stream_id=stream_id,
                error=e,
            )
        except Exception:
            _log.exception(
                "upload pipeline.failed event write failed for stream=%s",
                stream_id,
            )
        raise

    # Token T2 — run succeeded; charge the base fee + refresh the balance.
    await _refresh_balance_after_run(
        db_path=db_path, tenant_id=tenant_id, stream_id=stream_id,
    )


__all__ = [
    "PipelineKickoff",
    "PipelineRunner",
    "default_pipeline_runner",
    "upload_pipeline_runner",
]
