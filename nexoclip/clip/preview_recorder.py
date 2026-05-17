"""Slice O.20 — headless-Chrome recorder for the clip preview.

Renders the same HTML/CSS that the operator sees in the editor (the
`/dashboard/clips/<id>/render` page from slice O.19), captures it as
video bytes via Playwright's built-in WebM recorder, and muxes the
original clip's audio track on top. Output: an H.264 MP4 byte-perfect
visually identical to the editor preview.

Why this over the previous overlay_burn.py approach:
  - One renderer (Chromium), not two (Chromium for preview + ffmpeg
    drawtext for export). No font-fallback drift, no manual coordinate
    translation, no missing CSS features (border-radius, backdrop-blur,
    multi-pass drop-shadow).
  - WYSIWYG by construction: same browser engine renders both.

Tradeoffs:
  - Real-time playback. A 15-second clip takes ~15 seconds to record.
  - Playwright + Chromium dependency. ~110 MB on disk for chromium
    headless shell.

Audio handling:
  - Playwright's `record_video` captures VIDEO ONLY (no audio).
  - We post-process: ffmpeg muxes the recorded WebM video stream with
    the source clip's audio track (which is the same audio playing in
    the `<video>` element). Audio is copied bit-perfect (-c:a copy),
    not re-encoded. Video re-encodes WebM → H.264 (single pass,
    veryfast preset, CRF 20).
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import structlog

_log = structlog.get_logger(__name__)


# Recording knobs — kept module-level so callers can override via
# kwargs if a specific delivery target (e.g. Reels 60fps) ever needs it.
DEFAULT_FPS = 30
DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920
TAIL_PAD_S = 0.3  # extra seconds we record past clip.duration_s — absorbs
                  # the slight latency between video element's 'ended' event
                  # and Playwright actually flushing the last frames.
# Wait at most this long for the page's <video> element to fire 'playing'.
# Healthy load is <500 ms; 10 s is a generous ceiling for slow disks.
READY_TIMEOUT_S = 10.0


async def record_clip_to_mp4(
    *,
    clip_id: str,
    duration_s: float,
    audio_source_path: Path,
    output_path: Path,
    base_url: str,
    auth_cookie_name: str = "nexoclip_token",
    auth_cookie_value: str | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
) -> Path:
    """Record the render page for `clip_id` and produce an MP4 at
    `output_path`. Returns `output_path` on success; raises
    `PreviewRecordingError` on any failure with a short reason.

    Pipeline:
      1. Launch Playwright + Chromium headless at viewport `width`×`height`.
      2. Navigate to `<base_url>/dashboard/clips/<clip_id>/render`.
         Auth via the same `nexoclip_token` cookie the dashboard uses.
      3. Wait for the `<video>` element's `playing` event (not just for
         `play()` to resolve — `play()` resolves the moment the call is
         accepted, BEFORE the first frame is decoded; `playing` fires
         when actual playback starts).
      4. Start the wall-clock timer THEN, record for
         `duration_s + TAIL_PAD_S` seconds.
      5. Close the page → Playwright flushes the WebM to disk.
      6. ffmpeg muxes the WebM video stream + the source clip's audio
         track into the final MP4. Audio is copied (lossless); video
         is re-encoded H.264 (libx264, veryfast, CRF 20).

    `audio_source_path` is the source clip MP4 — same file that the
    `<video>` element streams. Same audio stream the operator just
    heard in the editor.
    """
    # Lazy-import Playwright so the rest of the package still imports
    # on hosts without it installed (e.g. test environments). The
    # caller gets a clean error message if invoked without the dep.
    try:
        from playwright.async_api import async_playwright
    except ImportError as e:
        raise PreviewRecordingError(
            "Playwright not installed. Run "
            "`pip install playwright && playwright install chromium`."
        ) from e

    if not audio_source_path.exists():
        raise PreviewRecordingError(
            f"Audio source missing on disk: {audio_source_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    record_url = f"{base_url.rstrip('/')}/dashboard/clips/{clip_id}/render"

    # Playwright writes each recorded video to a randomly-named WebM
    # file inside the context's `record_video_dir`. We use a temp dir
    # so cleanup is trivial.
    with tempfile.TemporaryDirectory(prefix="nexoclip_render_") as tmpdir:
        record_dir = Path(tmpdir)
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=[
                    # Allow autoplay without user gesture (the <video>
                    # element starts muted so this is safe).
                    "--autoplay-policy=no-user-gesture-required",
                    # Disable GPU rendering shortcuts that can introduce
                    # tearing on Windows headless builds.
                    "--disable-gpu-rasterization",
                    # Reduce memory + CPU noise.
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )
            try:
                context = await browser.new_context(
                    viewport={"width": width, "height": height},
                    device_scale_factor=1,
                    record_video_dir=str(record_dir),
                    record_video_size={"width": width, "height": height},
                )
                # Auth: dashboard pages live behind cookie-based session.
                # Pass the same cookie the operator's browser carries.
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
                    "preview_recorder.navigate",
                    clip_id=clip_id, url=record_url,
                    viewport=[width, height], fps=fps,
                )
                await page.goto(record_url, wait_until="domcontentloaded")

                # Wait for the `playing` flag the render page exposes
                # on window.__nexoclipRender. We poll because Playwright
                # doesn't have a built-in "wait for media event" API.
                # Auto-play kicks off because the <video> is muted +
                # the autoplay-policy flag we set above.
                try:
                    await page.wait_for_function(
                        "window.__nexoclipRender && window.__nexoclipRender.playing === true",
                        timeout=READY_TIMEOUT_S * 1000,
                    )
                except Exception as e:  # noqa: BLE001
                    # Some browsers won't fire 'playing' until play() is
                    # explicitly called. Try once, then re-wait.
                    try:
                        await page.evaluate(
                            "() => { const v = document.getElementById('preview-video');"
                            " if (v) { v.muted = true; return v.play(); } }"
                        )
                        await page.wait_for_function(
                            "window.__nexoclipRender && window.__nexoclipRender.playing === true",
                            timeout=READY_TIMEOUT_S * 1000,
                        )
                    except Exception as inner:  # noqa: BLE001
                        raise PreviewRecordingError(
                            f"<video> never fired 'playing' within "
                            f"{READY_TIMEOUT_S:.1f}s. "
                            f"Last error: {inner}"
                        ) from e

                # Cronómetro starts NOW — first frame is on screen.
                _log.info(
                    "preview_recorder.recording_start",
                    clip_id=clip_id, duration_s=duration_s,
                )
                await asyncio.sleep(duration_s + TAIL_PAD_S)

                # Closing the page flushes the recording. We do not
                # close the context first — Playwright requires the page
                # to be closed before the WebM is finalized on disk.
                await page.close()
                # `record_video.path()` resolves the file path inside
                # `record_video_dir`; we have to ask it AFTER close.
                # Easier: just glob the dir.
                await context.close()
            finally:
                await browser.close()

        # Locate the WebM Playwright wrote. There's exactly one inside
        # the temp dir for a single-page session.
        webm_files = list(record_dir.glob("*.webm"))
        if not webm_files:
            raise PreviewRecordingError(
                f"No WebM produced inside {record_dir}. "
                "Playwright recording failed silently."
            )
        webm_path = webm_files[0]
        _log.info(
            "preview_recorder.recorded",
            clip_id=clip_id, webm_path=str(webm_path),
            webm_size=webm_path.stat().st_size,
        )

        # Mux: video re-encode H.264, audio copy lossless from the source.
        _mux_video_and_audio(
            video_webm=webm_path,
            audio_mp4=audio_source_path,
            output_mp4=output_path,
            duration_s=duration_s,
            fps=fps,
        )

    _log.info(
        "preview_recorder.done",
        clip_id=clip_id, output=str(output_path),
        output_size=output_path.stat().st_size,
    )
    return output_path


def _mux_video_and_audio(
    *,
    video_webm: Path,
    audio_mp4: Path,
    output_mp4: Path,
    duration_s: float,
    fps: int,
) -> None:
    """ffmpeg mux step — see module docstring for rationale.

    - `-map 0:v:0` takes the video stream from the WebM.
    - `-map 1:a:0?` takes the audio stream from the source MP4 (the `?`
      makes the map optional, so a clip without audio doesn't fail).
    - `-c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p` re-encodes
      the WebM (which uses VP8/VP9) to H.264 for universal compatibility.
    - `-c:a copy` keeps the audio bit-perfect — no quality loss.
    - `-r {fps}` enforces output framerate (Playwright records at a
      variable framerate; we standardize to 30).
    - `-t {duration_s}` clamps output duration to exactly the clip
      length, in case the recording overshot from TAIL_PAD_S.
    - `-movflags +faststart` writes the MOOV atom at the head of the
      MP4 so social platforms can start streaming before download
      completes.

    Raises `PreviewRecordingError` on non-zero ffmpeg exit.
    """
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-i", str(video_webm),
        "-i", str(audio_mp4),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-c:a", "copy",
        "-t", f"{duration_s:.3f}",
        "-movflags", "+faststart",
        str(output_mp4),
    ]
    _log.info("preview_recorder.mux_start", cmd=" ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        stderr_tail = proc.stderr.decode("utf-8", errors="replace")[-800:]
        raise PreviewRecordingError(
            f"ffmpeg mux failed (rc={proc.returncode}): {stderr_tail.strip()}"
        )


class PreviewRecordingError(RuntimeError):
    """Raised on any failure inside the recorder pipeline.

    Caller decides what to do — typical handling: log + return 500 with
    a short detail, or fall back to the legacy overlay_burn pipeline
    (still in the repo, slice O.21 left it intact but disconnected).
    """


__all__ = ["record_clip_to_mp4", "PreviewRecordingError"]
