"""PipelineKickoff bundle + default runner — re-export shim.

The canonical home for these types is `nexoclip.jobs` (slice F.8).
Existing imports `from nexoclip.api._pipeline import PipelineKickoff`
keep working via the re-exports below; new code should import from
`nexoclip.jobs` directly.
"""

from __future__ import annotations

import logging

from nexoclip.jobs import PipelineKickoff, PipelineRunner

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


__all__ = ["PipelineKickoff", "PipelineRunner", "default_pipeline_runner"]
