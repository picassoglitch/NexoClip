"""InProcessJobDispatcher — runs the pipeline on this host via
FastAPI's BackgroundTasks. Default for local dev + single-host deploys.

NOT durable: if the dashboard process dies mid-run, the job is lost.
The pipeline emits step-event rows along the way so the dashboard can
detect "abandoned" runs and show the operator a re-run button — that's
the current best we can do without a real queue.

For the nexo-ai deployment, replace this with `ModalJobDispatcher`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import JobDispatcher, PipelineKickoff, PipelineRunner

if TYPE_CHECKING:
    from fastapi import BackgroundTasks


class InProcessJobDispatcher(JobDispatcher):
    """Wraps a PipelineRunner so the API layer doesn't reach into
    `app.state.pipeline_runner` directly anymore."""

    def __init__(self, runner: PipelineRunner) -> None:
        self._runner = runner

    @property
    def name(self) -> str:
        return "in-process"

    async def dispatch_pipeline(
        self,
        kickoff: PipelineKickoff,
        *,
        background_tasks: BackgroundTasks | None = None,
    ) -> None:
        if background_tasks is not None:
            # Normal API path — defer until after the response is sent.
            background_tasks.add_task(self._runner, kickoff)
            return
        # CLI / test path — run inline.
        await self._runner(kickoff)
