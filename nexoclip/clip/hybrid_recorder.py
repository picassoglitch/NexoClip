"""Hybrid ffmpeg + Playwright-overlay-alpha recorder — Render Migration T2.

The legacy seek-and-shoot recorder (preview_recorder.record_clip_to_mp4)
took a full-composite screenshot of source video + overlay per frame.
For a 35.9 s clip at 30 fps that's 1077 sequential
`page.screenshot()` calls — observed throughput ~3.8 fps on Railway,
so a single render took ~4.7 min. T1 backgrounded the work; T2 fixes
the slowness.

Hybrid pipeline:

  1. Source video is NEVER screenshotted. ffmpeg handles it directly:
     copies the existing 1080×1920 clip.mp4 frames (already in the
     right shape from the cut step) and re-encodes at the export
     framerate. Sub-second cost.

  2. Playwright renders ONLY the overlay layer on a transparent
     background (`?capture=alpha` mode in clip_render.html). The
     recorder samples at the export framerate but DEDUPS via a
     state-hash string emitted by the page — captions tick at
     word-frequency (~3-5 changes/sec), not 30/sec, so ~85% of the
     "screenshots" are skipped (hardlinked / copied from the
     previous unique PNG). Per-frame cost: state_hash compare
     (~5 ms) instead of full screenshot (~250 ms).

  3. ffmpeg composites the alpha PNG sequence over the re-encoded
     source video and muxes the original audio losslessly. One
     filtergraph pass, leverages all CPU cores.

Result on a 35s clip:
  Legacy seek-and-shoot: ~280 s wall (~1077 screenshots × 250ms +
                         encode)
  Hybrid:                ~10-30 s wall (~150 unique screenshots
                         × 250ms + ffmpeg overlay pass)

The state-machine + UI from T1 are unchanged — only the runner
internals here change. record_clip_to_mp4 still exists; this
module wraps it with the hybrid path as the default, falling back
to the legacy seek-and-shoot if any stage blows up.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import structlog

_log = structlog.get_logger(__name__)

# Same encode knobs as the legacy path so the output is visually
# identical aside from the speed difference.
FFMPEG_CRF = "14"
FFMPEG_PRESET = "medium"
DEFAULT_FPS = 30


class HybridRecordingError(RuntimeError):
    """Raised on any unrecoverable failure inside the hybrid pipeline.

    Caller (the background runner) catches this and either marks the
    clip's render_state="failed" + surfaces the message to the UI, OR
    falls back to the legacy seek-and-shoot recorder for one retry.
    """


async def record_clip_hybrid(
    *,
    clip_id: str,
    duration_s: float,
    audio_source_path: Path,
    output_path: Path,
    base_url: str,
    auth_cookie_name: str = "nexoclip_token",
    auth_cookie_value: str | None = None,
    width: int = 1080,
    height: int = 1920,
    fps: int = DEFAULT_FPS,
    progress_callback: Any = None,
) -> Path:
    """Run the hybrid pipeline. Same signature shape as the legacy
    record_clip_to_mp4 so the background runner can swap one for the
    other without other plumbing.

    Phases (each emits a coarse progress %):
      0-50%   — Overlay PNG sequence capture (the dominant cost, gets
                most of the bar).
      50-95%  — ffmpeg composite + audio mux.
      95-100% — Final flush, return.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError as e:
        raise HybridRecordingError(
            "Playwright not installed. Run "
            "`pip install playwright && playwright install chromium`."
        ) from e

    if not audio_source_path.exists():
        raise HybridRecordingError(
            f"audio source missing on disk: {audio_source_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    record_url = (
        f"{base_url.rstrip('/')}/dashboard/clips/{clip_id}/render"
        f"?capture=alpha"
    )

    with tempfile.TemporaryDirectory(prefix="nexoclip_hybrid_") as tmpdir:
        frames_dir = Path(tmpdir) / "overlay_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=[
                    "--autoplay-policy=no-user-gesture-required",
                    "--disable-gpu-rasterization",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )
            try:
                context = await browser.new_context(
                    viewport={"width": width, "height": height},
                    device_scale_factor=1,
                )
                if auth_cookie_value:
                    from urllib.parse import urlparse
                    parsed = urlparse(base_url)
                    await context.add_cookies([
                        {
                            "name": auth_cookie_name,
                            "value": auth_cookie_value,
                            "domain": parsed.hostname or "127.0.0.1",
                            "path": "/",
                            "httpOnly": False,
                            "secure": parsed.scheme == "https",
                        }
                    ])

                page = await context.new_page()
                _log.info(
                    "hybrid_recorder.navigate",
                    clip_id=clip_id, url=record_url,
                    viewport=[width, height], fps=fps,
                )
                await page.goto(record_url, wait_until="domcontentloaded")
                # Wait for captions + animator to be ready. The alpha
                # path skips the `playing` gate (no <video>), so this
                # lands as soon as captions.json fetches.
                try:
                    await page.wait_for_function(
                        "window.__nexoclipRender"
                        " && window.__nexoclipRender.captureFrameAtAlpha"
                        " && window.__nexoclipRender.getOverlayStateHash"
                        " && window.__nexoclipRender.captionsReady === true",
                        timeout=10_000,
                    )
                except Exception as e:  # noqa: BLE001
                    raise HybridRecordingError(
                        f"render page never reached alpha-capture-ready: {e}"
                    ) from e

                # R11 debug — save the first caption-bearing alpha PNG
                # sibling to the final clip MP4 so operators can fetch
                # it via /dashboard/clips/<id>/debug-alpha-caption.png
                # and SEE whether captions are reaching the PNG layer.
                debug_png = output_path.parent / "debug_alpha_caption.png"
                unique_count, total_count = await _capture_overlay_alpha_sequence(
                    page=page,
                    duration_s=duration_s,
                    fps=fps,
                    frames_dir=frames_dir,
                    progress_callback=progress_callback,
                    debug_caption_png_path=debug_png,
                )
                _log.info(
                    "hybrid_recorder.capture_done",
                    clip_id=clip_id,
                    unique_frames=unique_count,
                    total_frames=total_count,
                    dedup_ratio=(
                        round(1.0 - unique_count / total_count, 3)
                        if total_count > 0 else 0.0
                    ),
                )

                await page.close()
                await context.close()
            finally:
                await browser.close()

        # 50% mark — capture done, start composite.
        await _emit_progress(progress_callback, 50)

        await asyncio.to_thread(
            _ffmpeg_composite_alpha_over_source,
            source_video=audio_source_path,
            frames_dir=frames_dir,
            output_path=output_path,
            fps=fps,
            duration_s=duration_s,
        )

        await _emit_progress(progress_callback, 100)
        # Telemetry stat — failure here is non-fatal. The post-write
        # check inside _ffmpeg_composite_alpha_over_source already
        # raised if the file was missing / 0 bytes, so by the time we
        # get here we expect it to be on disk. If something deleted it
        # between now and that check (unlikely but observable on shared
        # volumes), don't propagate — the runner's validate step in
        # _clip_render.py is the gatekeeper now.
        try:
            output_bytes = output_path.stat().st_size
        except OSError as e:
            _log.warning(
                "hybrid_recorder.stat_failed",
                clip_id=clip_id, output=str(output_path), error=str(e),
            )
            output_bytes = -1
        _log.info(
            "hybrid_recorder.done",
            clip_id=clip_id, output=str(output_path),
            output_bytes=output_bytes,
        )
    return output_path


async def _capture_overlay_alpha_sequence(
    *,
    page: Any,
    duration_s: float,
    fps: int,
    frames_dir: Path,
    progress_callback: Any,
    debug_caption_png_path: Path | None = None,
) -> tuple[int, int]:
    """Walk frame 0..N at fps. For each frame, read the overlay state
    hash from the page; only call page.screenshot when the hash changes.
    Unchanged frames hardlink to the previous unique PNG so ffmpeg's
    image2 demuxer sees a complete frame sequence.

    Returns (unique_screenshots_taken, total_frames_written) so the
    caller can log a dedup ratio.

    debug_caption_png_path — when set, the FIRST screenshot whose state
    hash contains a `nc-pv-caption__word` span gets copied to this path
    AS WELL AS the frames_dir. The download endpoint exposes it at
    `/dashboard/clips/<id>/debug-alpha-caption.png` so operators can
    diagnose "captions missing from export" by eyeballing the actual
    rasterized PNG that ffmpeg overlays. Add R11 (live audit). If the
    caption pixels are absent here, it's a Playwright/Chromium capture
    bug, not a ffmpeg overlay bug; if present, the overlay step is the
    culprit. Either way the PNG tells us which.
    """
    frame_count = max(1, int(round(duration_s * fps)))
    frame_dt = 1.0 / float(fps)

    prev_state: str | None = None
    prev_png_path: Path | None = None
    unique_screenshots = 0
    debug_capture_saved = False

    # The recorder emits progress in the 0-50% range during capture
    # (the composite step owns 50-95%; final flush 95-100%). Avoid
    # spamming the callback — emit at most every 5%.
    last_emitted_pct = -10

    for i in range(frame_count):
        target_t = i * frame_dt
        # captureFrameAtAlpha paints the overlay at time t; returns
        # once the layout is committed.
        try:
            await page.evaluate(
                "(t) => window.__nexoclipRender.captureFrameAtAlpha(t)",
                target_t,
            )
            state = await page.evaluate(
                "() => window.__nexoclipRender.getOverlayStateHash()",
            )
        except Exception as e:  # noqa: BLE001
            raise HybridRecordingError(
                f"page evaluate failed on frame {i}: {e}"
            ) from e

        png_path = frames_dir / f"overlay_{i:08d}.png"

        if state == prev_state and prev_png_path is not None:
            # Reuse: hardlink (POSIX) or copy (Windows / cross-device).
            try:
                os.link(prev_png_path, png_path)
            except OSError:
                # Hardlink not supported (Windows, different FS) —
                # cp gives the same demuxer-visible result, just costs
                # extra disk space.
                shutil.copy2(prev_png_path, png_path)
        else:
            # R13 — compositor settle. captureFrameAtAlpha already
            # forced a layout flush + 2 rAF + setTimeout(0) before
            # resolving, but on Railway's headless Chromium with
            # --disable-gpu the screenshot can still land on a
            # pre-commit surface. Hold for one vsync frame here so
            # the compositor has wall clock to incorporate the new
            # caption layout into the screenshot pipeline.
            await asyncio.sleep(0.020)
            try:
                png_bytes = await page.screenshot(
                    type="png",
                    omit_background=True,
                    full_page=False,
                )
            except Exception as e:  # noqa: BLE001
                raise HybridRecordingError(
                    f"page.screenshot failed on frame {i}: {e}"
                ) from e
            png_path.write_bytes(png_bytes)
            unique_screenshots += 1
            prev_state = state
            prev_png_path = png_path

            # R11 debug: keep one caption-bearing PNG for the operator
            # to inspect. The state-hash check is a cheap proxy for
            # "this frame's caption row has spans" — if the DOM has a
            # caption painted, its outerHTML appears in the hash.
            if (
                debug_caption_png_path is not None
                and not debug_capture_saved
                and "nc-pv-caption__word" in (state or "")
            ):
                try:
                    debug_caption_png_path.parent.mkdir(
                        parents=True, exist_ok=True,
                    )
                    debug_caption_png_path.write_bytes(png_bytes)
                    debug_capture_saved = True
                    _log.info(
                        "hybrid_recorder.debug_caption_saved",
                        path=str(debug_caption_png_path),
                        frame_idx=i,
                        target_t=round(target_t, 3),
                        png_bytes=len(png_bytes),
                    )
                except OSError as e:
                    _log.warning(
                        "hybrid_recorder.debug_caption_save_failed",
                        path=str(debug_caption_png_path),
                        error=str(e),
                    )

        # Progress 0-50% during capture.
        pct = int(50 * (i + 1) / frame_count)
        if pct - last_emitted_pct >= 5:
            await _emit_progress(progress_callback, pct)
            last_emitted_pct = pct

    if debug_caption_png_path is not None and not debug_capture_saved:
        _log.warning(
            "hybrid_recorder.debug_caption_never_saved",
            note=(
                "no frame's state hash contained nc-pv-caption__word — "
                "either captions disabled, no transcript, or the page's "
                "_renderCaptions never painted any line. Check the "
                "captions.json endpoint for this clip."
            ),
        )

    return unique_screenshots, frame_count


def _ffmpeg_composite_alpha_over_source(
    *,
    source_video: Path,
    frames_dir: Path,
    output_path: Path,
    fps: int,
    duration_s: float,
) -> None:
    """Composite the alpha PNG sequence over the source video and mux
    the original audio. One ffmpeg pass; one file out.

    Filter graph:
      [0:v]        — source video (already 1080×1920 from the cut step)
      [1:v]        — alpha PNG sequence at the export framerate
      overlay=0:0  — alpha-blend the PNGs on top of the source frames
      -map [v]     — composited video stream
      -map 0:a?    — original audio (the `?` makes it optional so
                     a video-only source doesn't fail)
      -c:a aac     — re-encode to AAC@48k for social-platform compat;
                     the source's audio is already AAC from the cut
                     step's accurate-seek encode (O.55) so this is
                     near-lossless.
      -shortest    — clip to the shorter of source video / overlay
                     sequence (defends against off-by-one frame counts).

    NOT used: `-c:v copy` on the source — once we overlay we have to
    re-encode the video anyway, and pinning libx264 + the same CRF the
    legacy recorder used keeps output bit-equivalent for caller-visible
    purposes.
    """
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    overlay_pattern = str(frames_dir / "overlay_%08d.png")
    # Render Migration R9 — atomic publish. ffmpeg writes to a sibling
    # `.partial.mp4` and we `os.replace` into the final cache path only
    # after the post-conditions pass. Previously ffmpeg wrote directly
    # to `output_path`, so a download click that landed mid-render hit
    # FileResponse against a growing file: Content-Length sampled at
    # response-start, more bytes written before streaming finished →
    # `Response content longer than Content-Length` + a truncated MP4
    # the operator's player rejected with 0xC00D36C4. With the rename,
    # `output_path` only ever exists as a fully-flushed render.
    partial_path = output_path.with_suffix(".partial.mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        # Source video (audio + already-shaped frames).
        "-i", str(source_video),
        # Alpha PNG sequence at the export framerate.
        "-framerate", str(fps),
        "-i", overlay_pattern,
        # Composite.
        "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto[v]",
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", FFMPEG_PRESET,
        "-crf", FFMPEG_CRF,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        # Clamp output duration so a rounding mismatch in the PNG
        # count doesn't extend the clip a fraction of a second past
        # the audio.
        "-t", f"{duration_s:.3f}",
        # Render Migration R8 — +faststart REMOVED. On Railway's
        # persistent volume, ffmpeg's two-pass faststart fails the
        # re-open step ("Unable to re-open output file for shifting
        # data, Error writing trailer") and ffmpeg's cleanup deletes
        # the file as it bails — producing rc=0 + missing output and
        # exactly the symptom that haunted us for two sessions.
        # faststart is only useful for in-browser progressive
        # streaming of an MP4 hosted at a URL; for the operator's
        # local download + platform upload flow it's irrelevant
        # (every player handles moov-at-end fine). If we ever need
        # streaming preview of the rendered MP4, run a SEPARATE
        # `qt-faststart` pass that doesn't have ffmpeg's
        # delete-on-failure semantics.
        str(partial_path),
    ]
    _log.info(
        "hybrid_recorder.composite_start",
        cmd=" ".join(cmd),
        partial=str(partial_path),
        final=str(output_path),
    )
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            stderr_tail = proc.stderr.decode("utf-8", errors="replace")[-800:]
            raise HybridRecordingError(
                f"ffmpeg composite failed (rc={proc.returncode}): "
                f"{stderr_tail.strip()}"
            )

        # Render Migration R5 — post-condition. ffmpeg's exit code is
        # NOT a reliable signal that the output landed: on Railway's
        # persistent volume we've observed ffmpeg return 0 with the
        # output file missing (suspected disk-full / mount race; needs
        # more data to confirm). Without this check, the success path
        # then calls `output_path.stat()` for telemetry →
        # FileNotFoundError propagates out of record_clip_hybrid →
        # past _clip_render.py's narrow exception handler → kills the
        # background task → state stays 'rendering' → UI polls forever
        # with no way to retry. Turn the silent failure into a loud
        # one.
        if not partial_path.exists():
            stderr_tail = (
                proc.stderr or b""
            ).decode("utf-8", errors="replace")[-800:]
            raise HybridRecordingError(
                f"ffmpeg returned rc=0 but {partial_path.name} is "
                f"missing on disk (likely disk-full or mount race). "
                f"stderr tail: {stderr_tail.strip() or '(empty)'}"
            )
        try:
            size = partial_path.stat().st_size
        except OSError as e:
            raise HybridRecordingError(
                f"stat({partial_path.name}) failed after ffmpeg rc=0: {e}"
            ) from e
        if size == 0:
            raise HybridRecordingError(
                f"ffmpeg returned rc=0 but {partial_path.name} is 0 bytes"
            )

        # Atomic publish. Same-FS rename is atomic on POSIX and Windows;
        # both paths share a parent so the rename can't cross devices.
        os.replace(partial_path, output_path)
    except BaseException:
        # Any failure path: scrub the partial so it can't accumulate as
        # debris that a later operator-facing tool mistakes for cache.
        try:
            partial_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


async def _emit_progress(callback: Any, pct: int) -> None:
    """Fire-and-forget progress update. Tolerates sync, async, or
    None callbacks."""
    if callback is None:
        return
    try:
        if asyncio.iscoroutinefunction(callback):
            await callback(int(pct))
        else:
            callback(int(pct))
    except Exception:  # noqa: BLE001 — observability never blocks render
        _log.warning("hybrid_recorder.progress_callback_failed", pct=pct)


__all__ = ["HybridRecordingError", "record_clip_hybrid"]
