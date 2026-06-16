"""Production ingest callback for channel watches.

The drive watcher left its ingest callback stubbed; channel watches get
the real wiring: detect a VOD → `ingest_vod` (idempotent download + audio
extract) → persist the stream row → kick off the full pipeline via the
configured `JobDispatcher`. Downstream (clip → variants → auto-publish)
runs unchanged, with publishing gated by the safe trap.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from nexoclip.db import Database, EventsRepo, StreamsRepo
from nexoclip.jobs import JobDispatcher, PipelineKickoff
from nexoclip.tenancy import bound_tenant

from .service import ChannelIngestCallback

_log = structlog.get_logger(__name__)


def make_channel_ingest_callback(
    db: Database,
    dispatcher: JobDispatcher,
    *,
    output_dir: Path,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> ChannelIngestCallback:
    """Build the ingest callback the channel poller hands new VODs to.

    Each call ingests the VOD synchronously (fast — yt-dlp metadata + audio
    extract), persists the stream, emits `channel.vod_detected`, then
    dispatches the pipeline. `ingest_vod` is idempotent, so re-detecting an
    already-ingested VOD is a no-op download.
    """
    output_dir = Path(output_dir).resolve()

    async def _callback(
        tenant_id: str,
        vod_url: str,
        video_id: str,
        persona_id: str,
        language: str | None,
    ) -> None:
        from nexoclip.db.adapters import stream_to_row
        from nexoclip.ingest import ingest_vod

        with bound_tenant(tenant_id):
            # Idempotency on (tenant, vod_url): if this VOD was already
            # ingested, reuse its stream id so ingest_vod cache-hits (no
            # re-download) and the upsert is a no-op. Without this, every
            # re-detection mints a fresh ULID and re-downloads the whole
            # VOD into a duplicate stream row — which is what a lost
            # seen-set or a retried pipeline failure was producing.
            existing = await StreamsRepo(db).find_by_vod_url(vod_url)
            stream = await ingest_vod(
                tenant_id=tenant_id,
                vod_url=vod_url,
                output_dir=output_dir,
                stream_id=existing.id if existing is not None else None,
                cookies_from_browser=cookies_from_browser,
                cookies_file=cookies_file,
                db=db,
            )
            row = await StreamsRepo(db).upsert(stream_to_row(stream))
            if existing is None:
                await EventsRepo(db).emit(
                    type="channel.vod_detected",
                    payload={
                        "stream_id": row.id,
                        "video_id": video_id,
                        "vod_url": vod_url,
                    },
                )

        kickoff = PipelineKickoff(
            tenant_id=tenant_id,
            stream=stream,
            persona_id=persona_id,
            output_dir=output_dir,
            language=language,
        )
        # Decouple ingest from pipeline outcome. A successful download +
        # stream row IS the ingest as far as the poller's dedup is
        # concerned; the pipeline runs with its own retry/resume (step
        # events + dashboard re-run). If a pipeline step fails, it must NOT
        # propagate here — otherwise the poller counts the VOD as
        # un-ingested, never adds it to the seen-set, and re-runs the whole
        # thing next poll (the duplicate-ingest + re-download loop that also
        # amplifies host load). Download failures still raise from
        # ingest_vod above, so a genuinely failed ingest is still retried.
        try:
            await dispatcher.dispatch_pipeline(kickoff)
        except Exception as e:  # pipeline failure is not an ingest failure
            _log.warning(
                "channel.pipeline_dispatch_failed",
                tenant_id=tenant_id,
                stream_id=stream.id,
                video_id=video_id,
                error=str(e),
            )
            return
        _log.info(
            "channel.vod_dispatched",
            tenant_id=tenant_id,
            stream_id=stream.id,
            video_id=video_id,
        )

    return _callback
