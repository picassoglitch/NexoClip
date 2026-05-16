"""PipelineKickoff bundle + default runner — re-export shim.

The canonical home for these types is `nexoclip.jobs` (slice F.8).
Existing imports `from nexoclip.api._pipeline import PipelineKickoff`
keep working via the re-exports below; new code should import from
`nexoclip.jobs` directly.
"""

from __future__ import annotations

from nexoclip.jobs import PipelineKickoff, PipelineRunner


async def default_pipeline_runner(kickoff: PipelineKickoff) -> None:
    """Real `process_vod` invocation, lazily imported so tests that don't
    kick off the pipeline don't pay for Whisper / CV / yt-dlp module-load
    cost.

    Forwards the configured `db_path` so `_step()` can write
    pipeline.step.{start,done,failed} rows to the events table — without
    this, the dashboard's progress card has no events to read and stays
    permanently stuck on 'all six steps pending'.
    """
    from nexoclip.pipeline import process_vod
    from nexoclip.settings import get_settings

    db_path = get_settings().db_path

    await process_vod(
        tenant_id=kickoff.tenant_id,
        vod_url=kickoff.stream.vod_url,
        output_dir=kickoff.output_dir,
        persona_id=kickoff.persona_id,
        stream_id=kickoff.stream.id,
        language=kickoff.language,
        db_path=db_path,
    )


__all__ = ["PipelineKickoff", "PipelineRunner", "default_pipeline_runner"]
