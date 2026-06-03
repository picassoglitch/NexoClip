"""Background clip-render runner — Render Migration T1.

The legacy download path awaited `record_clip_to_mp4` inline inside
the HTTP request handler. On Railway a 35s clip rendered for ~4.7
min (seek-and-shoot at ~3.8 fps for 1077 frames), well past the
request timeout. Operators saw a permanent "Preparing…" spinner;
the actual render had been SIGKILL'd by the proxy.

This module is the background path. The download endpoint now:

  1. If the rendered MP4 is on disk → serve it (cache hit).
  2. Else flip the clip row to render_state='rendering' atomically
     (the SQL guards against double-dispatch).
  3. Schedule `render_clip_in_background(...)` as a FastAPI
     BackgroundTask.
  4. Return JSON status so the UI's polling Download button can
     show progress + swap to "Download MP4" when ready.

The runner itself:

  1. Calls record_clip_to_mp4 with a `progress_callback` that
     writes back to clips.render_progress_pct so the UI doesn't
     need to read events directly.
  2. On success: marks state='ready', progress=100.
  3. On failure: marks state='failed' with the truncated error.

Failures are absorbed here — we never propagate into FastAPI's
BackgroundTask runner because that just logs a traceback and the
operator sees nothing. The UI's status poll reads
render_state='failed' + render_error and surfaces a clear retry
button.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_log = logging.getLogger("nexoclip.clip_render")

# How often the background task is allowed to write progress updates
# back to the DB. Capping at 1/s prevents a 30-fps capture from
# hammering SQLite with thousands of UPDATE statements.
_PROGRESS_UPDATE_DEBOUNCE_S = 1.0


async def render_clip_in_background(
    *,
    clip_id: str,
    tenant_id: str,
    duration_s: float,
    audio_source_path: Path,
    output_path: Path,
    base_url: str,
    auth_cookie_value: str | None,
    width: int,
    height: int,
    db_path: str,
) -> None:
    """Run the headless-Chromium recorder and update clip.render_state.

    This is the function the download endpoint schedules via
    BackgroundTasks. It opens its own DB connection (the request's
    connection closes as soon as the response is sent) so the state
    updates land even after the HTTP layer is gone.

    Failure surfacing: any exception here is caught + recorded on the
    clip row as render_state='failed' + render_error. The UI poll
    surfaces it. We never re-raise — FastAPI's BackgroundTask runner
    would just log a traceback and the operator would see a stuck
    spinner.
    """
    from nexoclip.clip.preview_recorder import (
        PreviewRecordingError,
        record_clip_to_mp4,
    )
    from nexoclip.db import ClipsRepo, Database
    from nexoclip.tenancy import bound_tenant

    db = Database(db_path)
    await db.connect()
    try:
        # Throttled progress callback. The recorder calls this many
        # times per second; we debounce to 1 Hz so SQLite writes don't
        # contend with everything else.
        last_pct_written: dict = {"pct": -1, "ts": 0.0}

        async def _on_progress(pct: int) -> None:
            import time as _time
            now = _time.monotonic()
            if (
                pct == last_pct_written["pct"]
                or now - last_pct_written["ts"] < _PROGRESS_UPDATE_DEBOUNCE_S
            ):
                return
            last_pct_written["pct"] = pct
            last_pct_written["ts"] = now
            try:
                with bound_tenant(tenant_id):
                    await ClipsRepo(db).update_render_progress(
                        clip_id, pct=pct,
                    )
            except Exception:  # noqa: BLE001 — observability never blocks
                _log.warning(
                    "render_clip.progress_write_failed clip=%s pct=%s",
                    clip_id, pct,
                )

        try:
            await record_clip_to_mp4(
                clip_id=clip_id,
                duration_s=duration_s,
                audio_source_path=audio_source_path,
                output_path=output_path,
                base_url=base_url,
                auth_cookie_value=auth_cookie_value,
                width=width,
                height=height,
                progress_callback=_on_progress,
            )
        except PreviewRecordingError as e:
            with bound_tenant(tenant_id):
                await ClipsRepo(db).mark_render_failed(
                    clip_id, error=f"recorder: {e}",
                )
            _log.warning(
                "render_clip.failed clip=%s error=%s", clip_id, str(e)[:300],
            )
            return
        except Exception as e:  # noqa: BLE001 — catch-all so state lands
            with bound_tenant(tenant_id):
                await ClipsRepo(db).mark_render_failed(
                    clip_id, error=f"{type(e).__name__}: {e}",
                )
            _log.exception(
                "render_clip.crashed clip=%s", clip_id,
            )
            return

        # Sanity: the file is supposed to exist on success. If it
        # somehow doesn't, surface that instead of marking ready and
        # then 404'ing on the download.
        if not output_path.exists():
            with bound_tenant(tenant_id):
                await ClipsRepo(db).mark_render_failed(
                    clip_id,
                    error=(
                        f"recorder returned ok but {output_path.name} "
                        f"is missing on disk"
                    ),
                )
            return

        with bound_tenant(tenant_id):
            await ClipsRepo(db).mark_render_ready(clip_id)
        _log.info(
            "render_clip.ready clip=%s output=%s", clip_id, output_path,
        )
    finally:
        await db.close()


__all__ = ["render_clip_in_background"]
