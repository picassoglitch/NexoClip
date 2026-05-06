"""PipelineKickoff bundle + default runner.

Lives in its own module so `nexoclip.api.app` and the streams router can
both import it without forming a circular dependency.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nexoclip.ingest import Stream


@dataclass(frozen=True)
class PipelineKickoff:
    """Inputs the API endpoint hands to `pipeline_runner` after ingest.

    The stream is already on disk + persisted; the runner finishes the
    pipeline (transcribe, detect, cut, variants).
    """

    tenant_id: str
    stream: Stream
    persona_id: str
    output_dir: Path
    language: str | None = None


PipelineRunner = Callable[[PipelineKickoff], Awaitable[None]]


async def default_pipeline_runner(kickoff: PipelineKickoff) -> None:
    """Real `process_vod` invocation, lazily imported so tests that don't
    kick off the pipeline don't pay for Whisper / CV / yt-dlp module-load
    cost."""
    from nexoclip.pipeline import process_vod

    await process_vod(
        tenant_id=kickoff.tenant_id,
        vod_url=kickoff.stream.vod_url,
        output_dir=kickoff.output_dir,
        persona_id=kickoff.persona_id,
        stream_id=kickoff.stream.id,
        language=kickoff.language,
    )
