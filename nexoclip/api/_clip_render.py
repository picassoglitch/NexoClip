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

from nexoclip.api._render_validation import validate_rendered_mp4

if TYPE_CHECKING:
    pass

_log = logging.getLogger("nexoclip.clip_render")

# How often the background task is allowed to write progress updates
# back to the DB. Capping at 1/s prevents a 30-fps capture from
# hammering SQLite with thousands of UPDATE statements.
_PROGRESS_UPDATE_DEBOUNCE_S = 1.0


async def _build_ass_for_clip(
    *,
    db: object,
    tenant_id: str,
    clip_id: str,
    output_path: Path,
    width: int,
    height: int,
) -> Path | None:
    """Generate an ASS subtitle file for the clip's captions.

    Reads the transcript via TranscriptsRepo, slices to the clip's
    window via `captions_for_clip` (same code path the editor's
    /captions.json endpoint uses, so the burn matches the preview
    byte-for-byte), then writes the ASS file sibling to the cache
    MP4 as `clip_render_<res>.ass`. Returns the path on success,
    None when there's nothing to burn (no transcript / disabled /
    empty result) — caller treats None as "skip the libass step".

    Pulls captions config off the clip's overlay_config so a
    clip-level disable, position pick, or highlight-color override
    flows through. Defaults match the operator-visible editor
    preview when no per-clip override is set.
    """
    from nexoclip.clip.captions_ass import generate_ass
    from nexoclip.clip import captions_for_clip, lines_to_json
    from nexoclip.db import ClipsRepo, TranscriptsRepo
    from nexoclip.tenancy import bound_tenant

    with bound_tenant(tenant_id):
        clip = await ClipsRepo(db).get(clip_id)
        if clip is None:
            return None
        ov = clip.overlay_config or {}
        captions_cfg = ov.get("captions") if isinstance(ov.get("captions"), dict) else {}
        if not captions_cfg.get("enabled", True):
            return None

        transcript = await TranscriptsRepo(db).get(clip.stream_id)
        if transcript is None:
            return None

        lines = captions_for_clip(
            transcript.segments_json or "",
            clip_start_s=clip.start_s,
            clip_end_s=clip.end_s,
        )
        if not lines:
            return None

        # captions_for_clip returns Pydantic / dataclass objects; the
        # ASS generator needs plain dict. lines_to_json gives us the
        # exact serializable shape the /captions.json endpoint exposes
        # — same shape the editor renders from, so the ASS burn and
        # the editor preview stay in lockstep.
        lines_json = lines_to_json(lines)

    highlight = captions_cfg.get("highlight_color") or "#ffd84a"
    position = captions_cfg.get("position") or "lower_third"
    font_size_key = captions_cfg.get("font_size") or "medium"
    # Match the CSS scale * the base size. CSS base is 4.231cqw on a
    # 1080-px-wide container = ~46 px. Scale to absolute pixels and
    # scale by the operator's size pick.
    base_size = int(width * 0.0423)
    scale = {"small": 0.78, "medium": 1.0, "large": 1.24, "xl": 1.5}.get(
        font_size_key, 1.0,
    )
    font_size = max(16, int(base_size * scale))

    ass_path = output_path.with_suffix(".ass")
    generate_ass(
        lines=lines_json,
        width=width,
        height=height,
        output_path=ass_path,
        highlight_color_hex=highlight,
        position=position,
        font_size=font_size,
    )
    _log.info(
        "render_clip.ass_built clip=%s lines=%d ass=%s "
        "highlight=%s position=%s font_size=%d",
        clip_id, len(lines_json), ass_path, highlight, position, font_size,
    )
    return ass_path


def _safe_unlink(path: Path) -> None:
    """Best-effort delete used by the failure paths so a partial /
    corrupt file is never left for the next download click to serve
    as a cache hit. We swallow OSError because the only thing worse
    than a partial file is failing the failure path."""
    try:
        path.unlink(missing_ok=True)
    except OSError as e:  # noqa: BLE001 — cleanup never raises
        _log.warning(
            "render_clip.unlink_failed path=%s error=%s", path, e,
        )


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
    # Render Migration R5 — outer guard. ANY unhandled exception below
    # leaves state stuck at 'rendering' → operator's UI polls forever
    # with no way to retry. Wrap the whole body so a stray crash (in
    # validation, mark_render_ready DB write, anywhere) still ends up
    # at mark_render_failed before this coroutine returns.
    crashed_with: BaseException | None = None
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

        # Render Migration T2 — try the hybrid ffmpeg + overlay-alpha
        # path first (~10-30s for a typical 35s clip). On any failure
        # we fall back to the legacy seek-and-shoot recorder for one
        # retry (~5min) — the operator still gets their MP4, just
        # slower. The fallback path is the same code T1 backgrounded;
        # the failure-recovery semantics are unchanged.
        from nexoclip.clip.hybrid_recorder import (
            HybridRecordingError, record_clip_hybrid,
        )

        # Render Migration R16 — generate the ASS subtitle file once,
        # pass it to whichever recorder runs (hybrid first, legacy
        # fallback). The page no longer renders captions in any form;
        # libass burns them after the alpha composite. If captions
        # are disabled on this clip, or the transcript has no
        # word-level data overlapping the window, we pass None and
        # the composite skips the ass=filter chain.
        ass_file_path: Path | None = None
        try:
            ass_file_path = await _build_ass_for_clip(
                db=db,
                tenant_id=tenant_id,
                clip_id=clip_id,
                output_path=output_path,
                width=width,
                height=height,
            )
        except Exception as e:  # noqa: BLE001 — ASS is enrichment, not gating
            _log.warning(
                "render_clip.ass_build_failed clip=%s error=%s",
                clip_id, str(e)[:300],
            )

        hybrid_failed_with: str | None = None
        try:
            await record_clip_hybrid(
                clip_id=clip_id,
                duration_s=duration_s,
                audio_source_path=audio_source_path,
                output_path=output_path,
                base_url=base_url,
                auth_cookie_value=auth_cookie_value,
                width=width,
                height=height,
                progress_callback=_on_progress,
                ass_file_path=ass_file_path,
            )
        except HybridRecordingError as e:
            hybrid_failed_with = str(e)
            _log.warning(
                "render_clip.hybrid_failed_falling_back clip=%s error=%s",
                clip_id, str(e)[:300],
            )
        except Exception as e:  # noqa: BLE001
            # Render Migration R5 — catch ANY exception the hybrid
            # recorder leaked. Previously only HybridRecordingError
            # was caught, so a FileNotFoundError (Railway's silent
            # ffmpeg disk-full failure landed on `output_path.stat()`
            # in the success-log line) propagated all the way out
            # of render_clip_in_background → state stuck at
            # 'rendering' → UI polled /render-status forever with
            # no way to retry. Treat any non-HybridRecordingError
            # exception as a hybrid failure → fall through to the
            # legacy seek-and-shoot path, same as if it had raised
            # HybridRecordingError properly.
            hybrid_failed_with = f"{type(e).__name__}: {e}"
            _log.exception(
                "render_clip.hybrid_unexpected_exception_falling_back "
                "clip=%s",
                clip_id,
            )
            # The hybrid path may have left a partial composite file
            # on disk; wipe before legacy gets a chance to skip-because-
            # exists or otherwise interact with stale bytes.
            _safe_unlink(output_path)

        if hybrid_failed_with is not None:
            # Reset progress so the legacy path's 0-100% animation
            # doesn't read as "going backwards" after the hybrid bailed.
            try:
                with bound_tenant(tenant_id):
                    await ClipsRepo(db).update_render_progress(
                        clip_id, pct=0,
                    )
            except Exception:  # noqa: BLE001
                pass

        try:
            if hybrid_failed_with is None:
                # Hybrid succeeded — output is on disk, skip legacy.
                pass
            else:
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
            # Render Migration R2 — wipe any partial bytes the recorder
            # left on disk BEFORE marking failed. Otherwise the download
            # endpoint's rendered.exists() branch serves the fragment
            # ahead of the 409 + retry hint.
            _safe_unlink(output_path)
            with bound_tenant(tenant_id):
                await ClipsRepo(db).mark_render_failed(
                    clip_id, error=f"recorder: {e}",
                )
            _log.warning(
                "render_clip.failed clip=%s error=%s", clip_id, str(e)[:300],
            )
            return
        except Exception as e:  # noqa: BLE001 — catch-all so state lands
            _safe_unlink(output_path)
            with bound_tenant(tenant_id):
                await ClipsRepo(db).mark_render_failed(
                    clip_id, error=f"{type(e).__name__}: {e}",
                )
            _log.exception(
                "render_clip.crashed clip=%s", clip_id,
            )
            return

        # Render Migration R1 — validate the produced MP4 BEFORE
        # marking ready. ffmpeg can return exit 0 and still emit a
        # malformed container (missing moov atom, broken mux, weird
        # timestamps). The download path served those as cache hits
        # → operator got a file Windows refused to open
        # (0xC00D36C4). Validation now gates the ready flip: a file
        # that doesn't parse / has no video / has no audio / is
        # truncated → gets unlinked + the render is marked failed
        # with a clear reason. Partial bytes never survive into the
        # download endpoint's cache-hit branch.
        ok, reason = validate_rendered_mp4(
            output_path,
            expected_duration_s=duration_s,
        )
        if not ok:
            _safe_unlink(output_path)
            with bound_tenant(tenant_id):
                await ClipsRepo(db).mark_render_failed(
                    clip_id,
                    error=f"output failed validation: {reason}",
                )
            _log.warning(
                "render_clip.validation_failed clip=%s reason=%s",
                clip_id, (reason or "")[:200],
            )
            return

        with bound_tenant(tenant_id):
            await ClipsRepo(db).mark_render_ready(clip_id)
        _log.info(
            "render_clip.ready clip=%s output=%s renderer=%s "
            "bytes=%d duration_s=%.2f",
            clip_id,
            output_path,
            "legacy" if hybrid_failed_with else "hybrid",
            output_path.stat().st_size if output_path.exists() else 0,
            duration_s,
        )
    except BaseException as e:  # noqa: BLE001
        # R5 outer guard. We caught nested exceptions for each step
        # above, but ANY uncaught path (validation crash, DB write
        # crash, import error, etc.) lands here so state always
        # exits 'rendering' before the coroutine returns. We swallow
        # rather than re-raise — FastAPI's BackgroundTask runner
        # would log + drop, leaving the operator with a stuck
        # spinner and no error message. Only re-raise for
        # KeyboardInterrupt / SystemExit / asyncio.CancelledError
        # so the worker can still shut down cleanly.
        crashed_with = e
        import asyncio as _asyncio
        if isinstance(e, (KeyboardInterrupt, SystemExit, _asyncio.CancelledError)):
            raise
        _safe_unlink(output_path)
        try:
            with bound_tenant(tenant_id):
                await ClipsRepo(db).mark_render_failed(
                    clip_id,
                    error=f"runner crashed: {type(e).__name__}: {e}",
                )
        except BaseException:  # noqa: BLE001 — even the DB write may fail
            _log.exception(
                "render_clip.outer_guard_mark_failed_crashed clip=%s",
                clip_id,
            )
        _log.exception(
            "render_clip.outer_guard_caught clip=%s", clip_id,
        )
    finally:
        await db.close()
        if crashed_with is not None:
            _log.warning(
                "render_clip.finished_via_outer_guard clip=%s",
                clip_id,
            )


__all__ = ["render_clip_in_background"]
