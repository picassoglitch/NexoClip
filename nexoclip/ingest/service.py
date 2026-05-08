"""VOD ingest: download via yt-dlp, extract a 16 kHz mono WAV via ffmpeg.

The output layout for a stream is:

    <output_dir>/<stream_id>/
        stream.json                # serialized Stream model
        source/
            video.mp4              # downloaded VOD (remuxed to mp4)
            audio.wav              # 16 kHz mono PCM extract

Idempotency: if `stream.json` exists, the function returns the saved Stream
without re-downloading or re-extracting unless `force=True`.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path
from typing import Any

import yt_dlp

from nexoclip.errors import IngestError
from nexoclip.ids import new_id

from .models import Platform, Stream

_PLATFORM_PATTERNS: dict[Platform, re.Pattern[str]] = {
    "kick": re.compile(r"(?:^|//|\.)kick\.com/", re.IGNORECASE),
    "twitch": re.compile(r"(?:^|//|\.)twitch\.tv/", re.IGNORECASE),
    "youtube": re.compile(r"(?:^|//|\.)(?:youtube\.com|youtu\.be)/", re.IGNORECASE),
}


def detect_platform(vod_url: str) -> Platform:
    """Map a VOD URL to a known platform tag (or `unknown`)."""
    for platform, pattern in _PLATFORM_PATTERNS.items():
        if pattern.search(vod_url):
            return platform
    return "unknown"


async def ingest_vod(
    tenant_id: str,
    vod_url: str,
    output_dir: Path,
    *,
    stream_id: str | None = None,
    force: bool = False,
    chat_replay_source: Path | None = None,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> Stream:
    """Download a VOD and extract its audio. Returns a Stream.

    Args:
        tenant_id: Tenant owning the stream. Phase 0 hardcodes `"default"`.
        vod_url: VOD URL on Kick / Twitch / YouTube.
        output_dir: Root directory; a `<stream_id>/` subdirectory is created.
        stream_id: Optional existing stream ID for resumes. New ULID if omitted.
        force: If true, overwrite existing files even when `stream.json` exists.
        chat_replay_source: Optional path to a JSONL of chat messages
            (one `ChatMessage` per line). Phase 1 doesn't fetch chat replay
            from platforms automatically - pass a pre-fetched file or skip.
        cookies_from_browser: Pass cookies from a logged-in browser session
            ("chrome" / "edge" / "firefox" / "brave" / "chromium") through to
            yt-dlp. Required for Kick. Falls back to `Settings.cookies_from_browser`
            when omitted. Conflicts with `cookies_file`; if both are set,
            `cookies_file` wins.
        cookies_file: Alternative to `cookies_from_browser` — absolute path
            to a Netscape-format cookies.txt file (exported via a browser
            extension). Browser can stay open. Falls back to
            `Settings.cookies_file` when omitted.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sid = stream_id or new_id("str")
    stream_dir = output_dir / sid
    source_dir = stream_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    stream_json = stream_dir / "stream.json"
    video_path = source_dir / "video.mp4"
    audio_path = source_dir / "audio.wav"

    if not force and stream_json.exists():
        cached = Stream.model_validate_json(stream_json.read_text("utf-8"))
        # Resume: still import chat replay if a new source is provided this run.
        if chat_replay_source is not None:
            from .chat_replay import chat_replay_path, import_chat_replay

            import_chat_replay(
                source=chat_replay_source,
                stream_dir=stream_dir,
                stream_id=cached.id,
                tenant_id=cached.tenant_id,
            )
            cached = cached.model_copy(update={"source_chat_path": chat_replay_path(stream_dir)})
            stream_json.write_text(cached.model_dump_json(indent=2), encoding="utf-8")
        return cached

    platform = detect_platform(vod_url)

    # Resolve cookie auth: explicit kwargs win; otherwise fall back to
    # Settings (env vars / .env). cookies_file takes precedence over
    # cookies_from_browser when both are set, since the file path is the
    # workaround for the "Chrome holds the cookie DB lock" failure mode.
    if cookies_file is None or cookies_from_browser is None:
        from nexoclip.settings import get_settings

        settings = get_settings()
        if cookies_file is None:
            cookies_file = settings.cookies_file or None
        if cookies_from_browser is None:
            cookies_from_browser = settings.cookies_from_browser or None

    info = await asyncio.to_thread(
        _download_vod,
        vod_url=vod_url,
        target_path=video_path,
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
        platform=platform,
    )

    if force or not audio_path.exists():
        await asyncio.to_thread(_extract_audio, video_path, audio_path)

    duration_s = float(info.get("duration") or 0.0)
    if duration_s <= 0.0:
        duration_s = await asyncio.to_thread(_ffprobe_duration, video_path)

    chat_path: Path | None = None
    if chat_replay_source is not None:
        from .chat_replay import chat_replay_path, import_chat_replay

        import_chat_replay(
            source=chat_replay_source,
            stream_dir=stream_dir,
            stream_id=sid,
            tenant_id=tenant_id,
        )
        chat_path = chat_replay_path(stream_dir)

    stream = Stream(
        id=sid,
        tenant_id=tenant_id,
        vod_url=vod_url,
        platform=platform,
        title=info.get("title"),
        channel=info.get("uploader") or info.get("channel"),
        duration_s=duration_s,
        source_video_path=video_path,
        source_audio_path=audio_path,
        source_chat_path=chat_path,
    )
    stream_json.write_text(stream.model_dump_json(indent=2), encoding="utf-8")
    return stream


async def ingest_uploaded(
    tenant_id: str,
    source_path: Path,
    output_dir: Path,
    *,
    stream_id: str | None = None,
    title: str | None = None,
    chat_replay_source: Path | None = None,
) -> Stream:
    """Ingest a video the operator uploaded (no yt-dlp involved).

    Mirrors `ingest_vod` but skips the download step entirely. The on-disk
    layout is identical so every downstream step (transcribe, detect, cut,
    variants) finds its inputs in the usual places:

        <output_dir>/<stream_id>/
            stream.json
            source/
                video.mp4
                audio.wav

    `source_path` may be any container ffmpeg understands. We do NOT remux —
    just copy/move the file to `source/video.mp4`. ffmpeg reads it for the
    audio extract and ffprobe reads it for duration; both work on whatever
    actual container is inside.

    Args:
        tenant_id: Tenant owning the stream.
        source_path: An on-disk file the user just uploaded. The caller is
            responsible for streaming the upload to a tempfile first; this
            function moves it into place.
        output_dir: Root directory; a `<stream_id>/` subdirectory is created.
        stream_id: Optional existing stream ID for resumes. New ULID otherwise.
        title: Display title — usually the original filename.
        chat_replay_source: Same semantics as `ingest_vod`. Most uploads
            won't have one and this stays None.

    Returns:
        The persisted Stream. Idempotent on `stream_id`: a re-call with the
        same id and an existing stream.json returns the cached row.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_path).resolve()
    if not source_path.exists():
        raise IngestError(f"upload source not found: {source_path}")

    sid = stream_id or new_id("str")
    stream_dir = output_dir / sid
    source_dir = stream_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    stream_json = stream_dir / "stream.json"
    video_path = source_dir / "video.mp4"
    audio_path = source_dir / "audio.wav"

    if stream_json.exists():
        cached = Stream.model_validate_json(stream_json.read_text("utf-8"))
        if chat_replay_source is not None:
            from .chat_replay import chat_replay_path, import_chat_replay

            import_chat_replay(
                source=chat_replay_source,
                stream_dir=stream_dir,
                stream_id=cached.id,
                tenant_id=cached.tenant_id,
            )
            cached = cached.model_copy(
                update={"source_chat_path": chat_replay_path(stream_dir)}
            )
            stream_json.write_text(cached.model_dump_json(indent=2), encoding="utf-8")
        return cached

    # Move the uploaded tempfile into place. Move beats copy because the
    # uploaded file is in a temp dir we own and it's likely 100s of MB.
    if video_path.exists():
        video_path.unlink()
    try:
        source_path.replace(video_path)
    except OSError as e:
        # Cross-device move on Windows — fall back to copy.
        import shutil

        shutil.copy2(source_path, video_path)
        try:
            source_path.unlink()
        except OSError:
            pass
        del e

    await asyncio.to_thread(_extract_audio, video_path, audio_path)
    duration_s = await asyncio.to_thread(_ffprobe_duration, video_path)

    chat_path: Path | None = None
    if chat_replay_source is not None:
        from .chat_replay import chat_replay_path, import_chat_replay

        import_chat_replay(
            source=chat_replay_source,
            stream_dir=stream_dir,
            stream_id=sid,
            tenant_id=tenant_id,
        )
        chat_path = chat_replay_path(stream_dir)

    # `vod_url` is required by the Stream schema but uploads have no canonical
    # URL. Stash an `upload://` pseudo-URL so logs / dashboards don't show an
    # empty string and downstream `ingest_vod(stream_id=...)` cache-resume
    # calls have something stable to dedupe on.
    pseudo_url = f"upload://{title or video_path.name}"

    stream = Stream(
        id=sid,
        tenant_id=tenant_id,
        vod_url=pseudo_url,
        platform="upload",
        title=title,
        channel=None,
        duration_s=duration_s,
        source_video_path=video_path,
        source_audio_path=audio_path,
        source_chat_path=chat_path,
    )
    stream_json.write_text(stream.model_dump_json(indent=2), encoding="utf-8")
    return stream


def _download_vod(
    *,
    vod_url: str,
    target_path: Path,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
    platform: Platform = "unknown",
) -> dict[str, Any]:
    """Run yt-dlp; ensure the resulting file lives at exactly `target_path`.

    yt-dlp picks the file extension from the chosen format, so we let it
    write `video.<ext>` and rename to `video.mp4` once we know the actual
    path. `merge_output_format=mp4` keeps the container consistent for
    downstream ffmpeg work.

    Cookie auth (used by Kick + age-gated YouTube):
      * `cookies_file` — Netscape-format cookies.txt (browser stays open;
        most reliable on Windows where Chrome locks its cookie DB).
      * `cookies_from_browser` — pull cookies live from a browser profile.
        Faster setup but breaks if Chrome is running.
    When both are set, `cookies_file` wins.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    outtmpl = str(target_path.with_suffix("")) + ".%(ext)s"

    ydl_opts: dict[str, Any] = {
        "outtmpl": outtmpl,
        "format": "bestvideo*+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "writeinfojson": False,
        "writethumbnail": False,
        "overwrites": True,
    }
    if cookies_file:
        ydl_opts["cookiefile"] = str(cookies_file)
    elif cookies_from_browser:
        # yt-dlp's option is a tuple: (BROWSER, [PROFILE, [KEYRING, [CONTAINER]]]).
        # We only ever pass the browser name; users with multiple profiles can
        # extend this later.
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser.strip().lower(),)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info: dict[str, Any] = ydl.extract_info(vod_url, download=True) or {}
    except yt_dlp.utils.DownloadError as e:
        raise _explain_download_failure(
            err=e, vod_url=vod_url, platform=platform,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
        ) from e

    actual = _resolve_downloaded_path(info, fallback=target_path)
    if actual.resolve() != target_path.resolve():
        if target_path.exists():
            target_path.unlink()
        actual.replace(target_path)
    return info


def _explain_download_failure(
    *,
    err: Exception,
    vod_url: str,
    platform: Platform,
    cookies_from_browser: str | None,
    cookies_file: str | None = None,
) -> IngestError:
    """Wrap yt-dlp's raw error with an actionable hint when we recognize the
    pattern. The two most common Windows foot-guns:
      * Kick 403 because no cookies were configured at all.
      * "Could not copy Chrome cookie database" because Chrome is running
        and holds an exclusive lock on its cookies SQLite file.
    """
    raw = str(err)
    # Chrome cookie-DB lock — yt-dlp issue #7271.
    if "Could not copy" in raw and "cookie" in raw.lower() and not cookies_file:
        return IngestError(
            f"yt-dlp can't read your browser's cookie database while the "
            f"browser is running (yt-dlp #7271). Three fixes, in order of "
            f"least friction:\n"
            f"  1. Switch to a browser that doesn't lock its cookie DB: "
            f"set NEXOCLIP_COOKIES_FROM_BROWSER=edge (or firefox) in .env, "
            f"visit kick.com once in that browser, restart the server.\n"
            f"  2. Close Chrome completely (every chrome.exe process), then "
            f"retry. Chrome must stay closed during the run.\n"
            f"  3. Export cookies to a Netscape cookies.txt file (browser "
            f"extension 'Get cookies.txt LOCALLY'), set "
            f"NEXOCLIP_COOKIES_FILE=C:/path/to/cookies.txt, restart the "
            f"server. Chrome can stay open.\n"
            f"Raw error: {raw}"
        )
    if (
        platform == "kick"
        and "403" in raw
        and not cookies_from_browser
        and not cookies_file
    ):
        return IngestError(
            f"yt-dlp 403'd on Kick for {vod_url}. Kick blocks unauthenticated "
            f"scraping; pass logged-in browser cookies through. Set "
            f"NEXOCLIP_COOKIES_FROM_BROWSER=chrome (or edge / firefox / brave / "
            f"chromium) in your .env or shell, then retry. The browser must "
            f"have visited Kick at least once. "
            f"Raw error: {raw}"
        )
    return IngestError(f"yt-dlp failed for {vod_url}: {raw}")


def _resolve_downloaded_path(info: dict[str, Any], *, fallback: Path) -> Path:
    """Best-effort lookup of the actual file yt-dlp wrote on disk."""
    requested = info.get("requested_downloads")
    if isinstance(requested, list) and requested:
        first = requested[0]
        if isinstance(first, dict):
            for key in ("filepath", "_filename"):
                value = first.get(key)
                if isinstance(value, str):
                    return Path(value)
    filename = info.get("_filename")
    if isinstance(filename, str):
        return Path(filename)
    return fallback


def _extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract a 16 kHz mono PCM WAV from `video_path` using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(audio_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as e:
        raise IngestError("ffmpeg binary not found on PATH") from e
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", "replace") if e.stderr else ""
        raise IngestError(f"ffmpeg audio extraction failed: {stderr}") from e


def _ffprobe_duration(video_path: Path) -> float:
    """Return the duration in seconds, or 0.0 if ffprobe fails."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError):
        return 0.0


def load_stream(stream_dir: Path) -> Stream:
    """Rehydrate a Stream from `<stream_dir>/stream.json`."""
    stream_json = Path(stream_dir) / "stream.json"
    if not stream_json.exists():
        raise IngestError(f"no stream at {stream_dir} (missing stream.json)")
    return Stream.model_validate_json(stream_json.read_text("utf-8"))
