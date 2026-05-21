"""Headless-Chrome recorder for the clip preview (slice O.50).

Renders the same HTML/CSS that the operator sees in the editor (the
`/dashboard/clips/<id>/render` page from slice O.19) and captures it
straight to MP4 via Chrome DevTools Protocol screencast — bypassing
Playwright's built-in MediaRecorder.

Why CDP screencast vs MediaRecorder:

  Playwright's `record_video` uses Chromium's MediaRecorder API which
  emits VP8/VP9 WebM at a fixed (lowish) bitrate that we can't control.
  For a 4K-source workflow that intermediate compressed the picture
  to ~3 Mbps before our final H.264 mux could touch it — visible
  pixelation operators reported.

  CDP `Page.startScreencast` emits raw JPEG-per-frame events. We pipe
  those JPEGs straight into ffmpeg's image2pipe demuxer and encode
  H.264 at OUR chosen bitrate (CRF 14, medium preset). No intermediate
  lossy step. Visually indistinguishable from the editor preview at
  any export resolution.

Trade-offs:
  - Real-time playback. A 15-second clip still takes ~15s to record.
  - Slightly more CPU than MediaRecorder (JPEG encode on Chrome side).
  - Frame rate is "best effort" from Chrome's screencast — we ask for
    every frame but Chrome batches based on repaints. ffmpeg's input
    `-r 30` smooths to constant frame rate on output.

Audio handling:
  - CDP screencast captures VIDEO ONLY. Same as before.
  - ffmpeg muxes the source clip's audio track on top of the encoded
    H.264 stream. Audio is `-c:a copy` so it's bit-perfect — no
    re-encode.
"""
from __future__ import annotations

import asyncio
import base64
import shutil
import subprocess
from pathlib import Path

import structlog

_log = structlog.get_logger(__name__)


# Recording knobs — kept module-level so callers can override via
# kwargs if a specific delivery target (e.g. Reels 60fps) ever needs it.
DEFAULT_FPS = 30
DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920
TAIL_PAD_S = 0.3  # extra seconds we record past clip.duration_s
# Wait at most this long for the page's <video> element to be ready.
READY_TIMEOUT_S = 10.0
# JPEG quality for the screencast frames. 95 is "visually transparent"
# — at 100 the file sizes balloon for no visible benefit, and Chrome's
# JPEG encoder gets noticeably slower above 95.
SCREENCAST_JPEG_QUALITY = 95
# ffmpeg encode settings — applied once to the screencast frames.
# CRF 14 is mathematically "visually transparent for high-motion
# content" on libx264. The medium preset trades ~2× encode time vs
# veryfast for noticeably better detail retention; output is cached
# per (clip, resolution) so the cost amortizes across re-downloads.
FFMPEG_CRF = "14"
FFMPEG_PRESET = "medium"


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
    """Record the render page for `clip_id` to an MP4 at `output_path`.

    Pipeline:
      1. Launch Playwright + Chromium headless at viewport width × height.
      2. Navigate to `<base_url>/dashboard/clips/<clip_id>/render`,
         authenticate via the operator's session cookie.
      3. Wait for the page's `window.__nexoclipRender.allReady` (video
         playing + caption animator running) before recording the first
         frame.
      4. Start an ffmpeg subprocess that consumes JPEG frames from stdin
         and emits an H.264 MP4 with the source audio muxed in.
      5. Start a CDP screencast on the page; for each `screencastFrame`
         event, write the decoded JPEG bytes to ffmpeg's stdin and ACK
         the frame so Chrome sends the next one.
      6. After `duration_s + TAIL_PAD_S`, stop the screencast, close
         ffmpeg's stdin, wait for it to flush, return.

    Returns `output_path` on success. Raises `PreviewRecordingError`
    with a short reason on any failure (recorder caller decides
    fallback strategy).
    """
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
                "preview_recorder.navigate",
                clip_id=clip_id, url=record_url,
                viewport=[width, height], fps=fps, mode="cdp",
            )
            await page.goto(record_url, wait_until="domcontentloaded")

            try:
                await page.wait_for_function(
                    "window.__nexoclipRender && window.__nexoclipRender.allReady === true",
                    timeout=READY_TIMEOUT_S * 1000,
                )
            except Exception as e:  # noqa: BLE001
                # Kick playback explicitly, then re-wait.
                try:
                    await page.evaluate(
                        "() => { const v = document.getElementById('preview-video');"
                        " if (v) { v.muted = true; return v.play(); } }"
                    )
                    await page.wait_for_function(
                        "window.__nexoclipRender && window.__nexoclipRender.allReady === true",
                        timeout=READY_TIMEOUT_S * 1000,
                    )
                except Exception as inner:  # noqa: BLE001
                    try:
                        await page.wait_for_function(
                            "window.__nexoclipRender && window.__nexoclipRender.playing === true",
                            timeout=2000,
                        )
                    except Exception:
                        raise PreviewRecordingError(
                            f"<video> never reached `allReady` within "
                            f"{READY_TIMEOUT_S:.1f}s. Last error: {inner}"
                        ) from e

            # All ready — kick off the CDP screencast + ffmpeg sink.
            _log.info(
                "preview_recorder.recording_start",
                clip_id=clip_id, duration_s=duration_s,
            )
            await _record_via_cdp(
                page=page,
                context=context,
                audio_source_path=audio_source_path,
                output_path=output_path,
                width=width,
                height=height,
                fps=fps,
                duration_s=duration_s,
            )
            await page.close()
            await context.close()
        finally:
            await browser.close()

    _log.info(
        "preview_recorder.done",
        clip_id=clip_id, output=str(output_path),
        output_size=output_path.stat().st_size if output_path.exists() else 0,
    )
    return output_path


async def _record_via_cdp(
    *,
    page: object,  # playwright Page — typed loosely to avoid the import
    context: object,  # playwright BrowserContext
    audio_source_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    duration_s: float,
) -> None:
    """Open a CDP session, screencast JPEG frames, pipe to ffmpeg.

    Slice O.51 — fixed-cadence writer. Chrome's screencast sends frames
    at a variable rate (one per repaint, which depends on system load
    + how busy the page is). If we naively forward each frame into
    ffmpeg with `-framerate 30`, ffmpeg assumes every input frame is
    1/30s long — so a clip recorded at 15 effective fps comes out
    half-length and plays back at 2× speed.

    Fix: decouple Chrome's variable input rate from ffmpeg's fixed
    output rate. A background writer task wakes every 1/fps seconds
    and writes whatever the "latest frame" reference holds; the
    screencast callback just updates that reference. If Chrome lags
    we duplicate the last frame (output stays real-time, no fast-
    forward); if Chrome is faster than 1/fps we drop intermediate
    frames (output stays at the requested rate, no slow-mo).
    """
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        # Input 0: JPEG frames on stdin at a strict fps. Because the
        # writer task below feeds exactly fps frames/sec of wall time,
        # this is now accurate.
        "-f", "image2pipe",
        "-framerate", str(fps),
        "-i", "pipe:0",
        # Input 1: audio from the source clip MP4.
        "-i", str(audio_source_path),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-preset", FFMPEG_PRESET,
        "-crf", FFMPEG_CRF,
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-c:a", "copy",
        # Clamp output to exactly the requested duration. Chrome can
        # send a stray frame after we ask it to stop; -t prevents
        # desync between the video + audio tracks.
        "-t", f"{duration_s:.3f}",
        "-movflags", "+faststart",
        str(output_path),
    ]
    _log.info("preview_recorder.ffmpeg_start", cmd=" ".join(cmd))
    proc = subprocess.Popen(  # noqa: S603 — args list, not shell
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    cdp = await context.new_cdp_session(page)

    # Shared state between the screencast callback (Chrome side) and
    # the writer task (ffmpeg side).
    latest_frame: dict[str, bytes | None] = {"jpeg": None}
    frames_received = 0  # how many Chrome sent (i.e. how many repaints)
    frames_written = 0   # how many we forwarded to ffmpeg (= duration * fps)
    write_failed = False
    done_event = asyncio.Event()

    def on_screencast_frame(event: dict) -> None:
        nonlocal frames_received
        try:
            latest_frame["jpeg"] = base64.b64decode(event["data"])
            frames_received += 1
        except Exception as e:  # noqa: BLE001
            _log.warning(
                "preview_recorder.frame_decode_failed", error=str(e)
            )
            return
        # ACK so Chrome sends the next frame. Fire-and-forget.
        asyncio.create_task(
            cdp.send(
                "Page.screencastFrameAck",
                {"sessionId": event["sessionId"]},
            )
        )

    cdp.on("Page.screencastFrame", on_screencast_frame)

    async def writer_loop() -> None:
        """Feed ffmpeg's stdin at a strict 1/fps cadence.

        Runs from the moment the screencast starts until done_event is
        set. On every tick: if we have a frame, write it; if not, write
        nothing (ffmpeg's first input frame must exist or the encode
        fails — `_wait_first_frame` guards that before this loop starts).
        """
        nonlocal frames_written, write_failed
        loop = asyncio.get_event_loop()
        frame_interval = 1.0 / float(fps)
        next_tick = loop.time()
        while not done_event.is_set():
            frame = latest_frame["jpeg"]
            if frame is not None and not write_failed:
                try:
                    assert proc.stdin is not None
                    proc.stdin.write(frame)
                    frames_written += 1
                except (BrokenPipeError, OSError) as e:
                    write_failed = True
                    _log.warning(
                        "preview_recorder.ffmpeg_stdin_closed",
                        frames_written=frames_written, error=str(e),
                    )
                    return
            next_tick += frame_interval
            sleep_for = next_tick - loop.time()
            if sleep_for > 0:
                try:
                    await asyncio.wait_for(
                        done_event.wait(), timeout=sleep_for
                    )
                    # done_event fired during the sleep — exit cleanly.
                    return
                except asyncio.TimeoutError:
                    pass
            else:
                # We fell behind — skip the catch-up sleep and write
                # next frame immediately. Output stays close to fps.
                next_tick = loop.time()

    # Kick off the screencast. format=jpeg + quality=95 = small frames
    # with no visible compression artifacts. maxWidth/Height match the
    # viewport so Chrome doesn't downscale.
    await cdp.send(
        "Page.startScreencast",
        {
            "format": "jpeg",
            "quality": SCREENCAST_JPEG_QUALITY,
            "maxWidth": width,
            "maxHeight": height,
            "everyNthFrame": 1,
        },
    )

    # Wait briefly for the first frame to land before starting the
    # writer — otherwise the first 1/fps tick writes nothing and ffmpeg
    # errors out with "no input data".
    wait_start = asyncio.get_event_loop().time()
    while latest_frame["jpeg"] is None:
        if asyncio.get_event_loop().time() - wait_start > 5.0:
            await cdp.send("Page.stopScreencast")
            if proc.stdin is not None:
                proc.stdin.close()
            proc.kill()
            raise PreviewRecordingError(
                "CDP screencast first frame did not arrive within 5s"
            )
        await asyncio.sleep(0.05)

    writer_task = asyncio.create_task(writer_loop())

    try:
        await asyncio.sleep(duration_s + TAIL_PAD_S)
    finally:
        done_event.set()
        try:
            await asyncio.wait_for(writer_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        try:
            await cdp.send("Page.stopScreencast")
        except Exception:  # noqa: BLE001
            pass
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass

    try:
        proc.wait(timeout=max(60.0, duration_s + 30.0))
    except subprocess.TimeoutExpired as e:
        proc.kill()
        raise PreviewRecordingError(
            f"ffmpeg encode hung past {duration_s + 30:.0f}s timeout"
        ) from e

    _log.info(
        "preview_recorder.ffmpeg_done",
        rc=proc.returncode,
        frames_received=frames_received,
        frames_written=frames_written,
        expected_frames=int(duration_s * fps),
    )
    if proc.returncode != 0:
        stderr_tail = (proc.stderr.read() if proc.stderr else b"").decode(
            "utf-8", errors="replace"
        )[-800:]
        raise PreviewRecordingError(
            f"ffmpeg encode failed (rc={proc.returncode}, "
            f"received={frames_received}, written={frames_written}): "
            f"{stderr_tail.strip()}"
        )
    if frames_written == 0:
        raise PreviewRecordingError(
            "CDP screencast produced zero frames. Render page never repainted "
            "or the writer never ran."
        )


class PreviewRecordingError(RuntimeError):
    """Raised on any failure inside the recorder pipeline.

    Caller decides what to do — typical handling: log + return 500 with
    a short detail, or fall back to the legacy overlay_burn pipeline
    (still in the repo, slice O.21 left it intact but disconnected).
    """


__all__ = ["record_clip_to_mp4", "PreviewRecordingError"]
