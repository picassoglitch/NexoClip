"""ModalJobDispatcher — stub for the nexo-ai production deploy.

NOT YET WIRED. Documents the shape a Modal-backed dispatcher will take
so the swap-in is a config change, not a code change.

How Modal will plug in (planned for slice F.10+):

    # nexoclip/modal_app.py  (NEW)
    import modal
    app = modal.App("nexoclip-pipeline")

    @app.function(gpu="A10G", timeout=1800, mounts=[...], secrets=[...])
    async def run_pipeline(kickoff_dict: dict) -> None:
        from nexoclip.pipeline import process_vod
        from nexoclip.jobs import PipelineKickoff
        kickoff = PipelineKickoff(**kickoff_dict)
        await process_vod(
            tenant_id=kickoff.tenant_id,
            vod_url=kickoff.stream.vod_url,
            output_dir=kickoff.output_dir,
            persona_id=kickoff.persona_id,
            stream_id=kickoff.stream.id,
            language=kickoff.language,
        )

Then this dispatcher would do:

    def __init__(self):
        from nexoclip.modal_app import run_pipeline
        self._run_pipeline = run_pipeline

    async def dispatch_pipeline(self, kickoff, *, background_tasks=None):
        # `spawn` enqueues + returns instantly (vs `remote` which blocks).
        # The Modal Web UI tracks runs; our dashboard polls Modal for
        # status via the function call ID.
        call = self._run_pipeline.spawn(asdict(kickoff))
        await self._record_modal_call_id(kickoff.stream.id, call.object_id)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nexoclip.errors import NexoClipError

from .base import JobDispatcher, PipelineKickoff

if TYPE_CHECKING:
    from fastapi import BackgroundTasks


class ModalJobDispatcher(JobDispatcher):
    """Stub. Set `settings.job_dispatcher = "in_process"` until wired."""

    @property
    def name(self) -> str:
        return "modal"

    async def dispatch_pipeline(
        self,
        kickoff: PipelineKickoff,
        *,
        background_tasks: BackgroundTasks | None = None,
    ) -> None:
        raise NexoClipError(
            "ModalJobDispatcher is a stub. Wire nexoclip/modal_app.py "
            "in slice F.10+ before setting NEXOCLIP_JOB_DISPATCHER=modal."
        )
