"""Headless-Chrome recorder for the clip preview (slices O.50 + O.53).

Renders the same HTML/CSS that the operator sees in the editor (the
`/dashboard/clips/<id>/render` page from slice O.19) and captures it
to MP4 via Chrome DevTools Protocol screencast.

Why CDP screencast vs Playwright's MediaRecorder (slice O.50):
  Playwright's `record_video` uses Chromium's MediaRecorder API which
  emits VP8/VP9 WebM at a fixed (lowish) bitrate we can't control.
  CDP `Page.startScreencast` emits raw JPEG-per-frame events; we
  encode H.264 at OUR chosen bitrate (CRF 14, medium preset).
  No intermediate lossy step.

Why per-frame timestamps via concat demuxer (slice O.53):
  Chrome sends screencast frames at a VARIABLE rate (one per repaint,
  which depends on system load + page activity). Earlier versions
  forced a constant fps input which made variable-rate captures look
  fast-forward (O.51 fixed that via duplication) OR jittery (this
  slice fixes that).

  The correct architecture: save each frame to disk with Chrome's
  reported timestamp, then build a `concat` demuxer playlist where
  each frame's duration is the gap between its timestamp and the
  next frame's. ffmpeg then plays back the captured stream at EXACTLY
  the wall-clock rate Chrome produced — no duplication, no skipping,
  no fast-forward. Output is encoded at a constant 30fps (ffmpeg
  resamples from the variable concat input to constant output).

Audio sync:
  Chrome timestamps are normalized to start at 0 (first-frame ts is
  subtracted from all). The audio source already starts at 0. ffmpeg's
  `-t duration_s` clamps both to the same end. Result: captions
  (which are part of the video frames) stay in lockstep with the
  audio they're transcribing.
"""
from __future__ import annotations

import asyncio
import base64
import shutil
import subprocess
import tempfile
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
    """Open a CDP session, save JPEG frames to disk with their real
    timestamps, then ffmpeg-encode using the concat demuxer.

    Slice O.53 — timestamp-accurate concat. Earlier versions either:
      - Forced constant fps -> variable Chrome output looked fast-
        forward or jittery (mismatch between assumed and actual frame
        intervals)
      - Duplicated the last frame on missed ticks -> stutter when
        Chrome repaint cadence dropped below the target fps
    Neither matched what Chrome actually rendered. The correct fix is
    to honor Chrome's per-frame `metadata.timestamp` (seconds since
    epoch, monotonic) and let ffmpeg's concat demuxer do the math.

    Procedure:
      1. Save each screencast JPEG to <tmpdir>/frame_NNNN.jpg
      2. Track (filename, chrome_timestamp) tuples
      3. After capture ends, normalize timestamps so first frame is
         at t=0 (audio source already starts at t=0)
      4. Write a concat playlist: each `file ...` line followed by a
         `duration` line equal to (next_ts - this_ts)
      5. ffmpeg consumes the playlist + resamples to constant 30 fps
         output, mux audio bit-perfect from the source clip
    """
    cdp = await context.new_cdp_session(page)

    # Frame storage. tempfile context auto-cleans on exit; the
    # playlist file lives inside the same dir so all paths are
    # relative + portable across OS path separators.
    with tempfile.TemporaryDirectory(prefix="nexoclip_frames_") as tmpdir:
        tmp_path = Path(tmpdir)
        frames_meta: list[tuple[Path, float]] = []  # (filename, ts_seconds)
        decode_failures = 0
        done_event = asyncio.Event()

        def on_screencast_frame(event: dict) -> None:
            nonlocal decode_failures
            try:
                jpeg = base64.b64decode(event["data"])
            except Exception as e:  # noqa: BLE001
                decode_failures += 1
                _log.warning(
                    "preview_recorder.frame_decode_failed", error=str(e)
                )
                return
            # Chrome's metadata.timestamp is seconds since epoch in
            # the page's monotonic clock. Absolute value doesn't
            # matter — we'll normalize the first frame to 0 below.
            md = event.get("metadata") or {}
            ts = float(md.get("timestamp") or 0.0)
            idx = len(frames_meta)
            path = tmp_path / f"frame_{idx:08d}.jpg"
            try:
                path.write_bytes(jpeg)
            except OSError as e:
                decode_failures += 1
                _log.warning(
                    "preview_recorder.frame_write_failed",
                    error=str(e), idx=idx,
                )
                return
            frames_meta.append((path, ts))
            # ACK so Chrome sends the next frame. Fire-and-forget.
            asyncio.create_task(
                cdp.send(
                    "Page.screencastFrameAck",
                    {"sessionId": event["sessionId"]},
                )
            )

        cdp.on("Page.screencastFrame", on_screencast_frame)

        # Start the screencast. format=jpeg + quality=95 = visually
        # transparent at this resolution. maxWidth/Height match the
        # viewport so Chrome doesn't downscale before encoding.
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

        # Wait for the first frame so we can fail fast if the
        # screencast never gets going.
        wait_start = asyncio.get_event_loop().time()
        while not frames_meta:
            if asyncio.get_event_loop().time() - wait_start > 5.0:
                try:
                    await cdp.send("Page.stopScreencast")
                except Exception:  # noqa: BLE001
                    pass
                raise PreviewRecordingError(
                    "CDP screencast first frame did not arrive within 5s"
                )
            await asyncio.sleep(0.05)

        # Now record for the clip's duration + tail pad. Chrome keeps
        # firing screencastFrame events; our callback writes them to
        # disk with their real timestamps.
        try:
            await asyncio.sleep(duration_s + TAIL_PAD_S)
        finally:
            done_event.set()
            try:
                await cdp.send("Page.stopScreencast")
            except Exception:  # noqa: BLE001
                pass

        if not frames_meta:
            raise PreviewRecordingError(
                "CDP screencast produced zero frames"
            )

        # Normalize timestamps to start at 0 — audio source already
        # starts at 0, so this keeps video + audio aligned.
        t0 = frames_meta[0][1]
        normalized = [(p, ts - t0) for p, ts in frames_meta]

        # Build the concat playlist. Format:
        #   file 'frame_00000000.jpg'
        #   duration 0.033
        #   file 'frame_00000001.jpg'
        #   duration 0.041
        #   ...
        #   file 'frame_NNNNNNNN.jpg'    <- last frame listed twice
        # The last file is listed without a duration AND repeated so
        # ffmpeg's concat demuxer picks up its frame for the tail.
        playlist_lines: list[str] = []
        for i, (path, ts) in enumerate(normalized):
            playlist_lines.append(f"file '{path.name}'")
            if i + 1 < len(normalized):
                next_ts = normalized[i + 1][1]
                dur = max(0.001, next_ts - ts)
                playlist_lines.append(f"duration {dur:.6f}")
        # Concat-demuxer quirk: the last file's duration is taken from
        # the previous duration line OR ignored entirely. Listing the
        # final file twice (no duration on the dup) makes the demuxer
        # emit a final frame for the trailing gap. Otherwise the last
        # frame can get dropped.
        if normalized:
            playlist_lines.append(f"file '{normalized[-1][0].name}'")
        playlist_path = tmp_path / "concat.txt"
        playlist_path.write_text("\n".join(playlist_lines), encoding="utf-8")

        # Capture stats for telemetry BEFORE invoking ffmpeg so the
        # log line shows even on encode failure.
        captured_duration = normalized[-1][1] - normalized[0][1]
        effective_fps = (
            len(normalized) / captured_duration if captured_duration > 0 else 0
        )
        _log.info(
            "preview_recorder.capture_done",
            frames=len(normalized),
            captured_s=round(captured_duration, 3),
            effective_fps=round(effective_fps, 2),
            target_duration_s=duration_s,
            decode_failures=decode_failures,
        )

        await _run_ffmpeg_concat_encode(
            playlist_path=playlist_path,
            audio_source_path=audio_source_path,
            output_path=output_path,
            duration_s=duration_s,
            fps=fps,
        )


async def _run_ffmpeg_concat_encode(
    *,
    playlist_path: Path,
    audio_source_path: Path,
    output_path: Path,
    duration_s: float,
    fps: int,
) -> None:
    """Concat-demuxer ffmpeg encode. Runs sync in a thread so the
    async event loop isn't blocked while x264 chews."""
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        # Input 0: concat playlist with per-frame durations from
        # Chrome's actual timestamps. -safe 0 allows the playlist to
        # reference files by relative name without a fixed-set guard.
        "-f", "concat",
        "-safe", "0",
        "-i", str(playlist_path),
        # Input 1: audio source.
        "-i", str(audio_source_path),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        # Video encode: H.264, visually-transparent CRF + medium
        # preset. -r forces constant 30 fps output; ffmpeg interpolates
        # / drops frames from the variable concat input to match.
        # -vsync cfr makes the resample explicit (some ffmpeg builds
        # default to passthrough which preserves the variable rate
        # but breaks player seekability).
        "-c:v", "libx264",
        "-preset", FFMPEG_PRESET,
        "-crf", FFMPEG_CRF,
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-vsync", "cfr",
        "-c:a", "copy",
        # Hard clamp to the operator's requested clip duration so we
        # never overshoot if Chrome streamed past the stop signal.
        "-t", f"{duration_s:.3f}",
        "-movflags", "+faststart",
        str(output_path),
    ]
    _log.info("preview_recorder.ffmpeg_start", cmd=" ".join(cmd))
    loop = asyncio.get_event_loop()
    proc = await loop.run_in_executor(
        None,
        lambda: subprocess.run(  # noqa: S603 — args list, not shell
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=max(120.0, duration_s + 60.0),
            check=False,
        ),
    )
    _log.info("preview_recorder.ffmpeg_done", rc=proc.returncode)
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or b"").decode("utf-8", errors="replace")[-800:]
        raise PreviewRecordingError(
            f"ffmpeg encode failed (rc={proc.returncode}): {stderr_tail.strip()}"
        )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise PreviewRecordingError(
            "ffmpeg returned 0 but no output file was written"
        )


class PreviewRecordingError(RuntimeError):
    """Raised on any failure inside the recorder pipeline.

    Caller decides what to do — typical handling: log + return 500 with
    a short detail, or fall back to the legacy overlay_burn pipeline
    (still in the repo, slice O.21 left it intact but disconnected).
    """


__all__ = ["record_clip_to_mp4", "PreviewRecordingError"]
