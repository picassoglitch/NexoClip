"""InProcessJobDispatcher — runs the pipeline on this host via
FastAPI's BackgroundTasks. Default for local dev + single-host deploys.

NOT durable: if the dashboard process dies mid-run, the job is lost.
The pipeline emits step-event rows along the way so the dashboard can
detect "abandoned" runs and show the operator a re-run button — that's
the current best we can do without a real queue.

Concurrency IS bounded: a single global semaphore gates how many heavy
pipelines (download + Whisper + ffmpeg render) execute at once, so a
redeploy-recovery backlog or a channel-poll burst can't stampede the box.
Excess launches queue on the semaphore and run as slots free.

For the nexo-ai deployment, replace this with `ModalJobDispatcher`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from .base import JobDispatcher, PipelineKickoff, PipelineRunner

if TYPE_CHECKING:
    from fastapi import BackgroundTasks


class InProcessJobDispatcher(JobDispatcher):
    """Wraps a PipelineRunner so the API layer doesn't reach into
    `app.state.pipeline_runner` directly anymore.

    `max_concurrency` bounds simultaneous pipeline execution across EVERY
    launch path (API, channel-poll, recovery sweep) — they all funnel through
    `dispatch_pipeline`. Defaults to `Settings.max_concurrent_pipelines`.
    """

    def __init__(
        self, runner: PipelineRunner, *, max_concurrency: int | None = None
    ) -> None:
        self._runner = runner
        if max_concurrency is None:
            from nexoclip.settings import get_settings

            max_concurrency = int(
                getattr(get_settings(), "max_concurrent_pipelines", 2) or 2
            )
        self._sem = asyncio.Semaphore(max(1, max_concurrency))

    @property
    def name(self) -> str:
        return "in-process"

    async def _guarded_run(self, kickoff: PipelineKickoff) -> None:
        """Execute the runner under the global concurrency semaphore. A
        launch over the cap simply waits here for a slot — it holds a cheap
        coroutine, not download/render resources."""
        async with self._sem:
            await self._runner(kickoff)

    async def dispatch_pipeline(
        self,
        kickoff: PipelineKickoff,
        *,
        background_tasks: BackgroundTasks | None = None,
    ) -> None:
        if background_tasks is not None:
            # Normal API path — defer until after the response is sent. The
            # semaphore still applies once the background task runs.
            background_tasks.add_task(self._guarded_run, kickoff)
            return
        # CLI / recovery / test path — run inline (callers wrap in a task).
        await self._guarded_run(kickoff)
