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
) -> Stream:
    """Download a VOD and extract its audio. Returns a Stream.

    Args:
        tenant_id: Tenant owning the stream. Phase 0 hardcodes `"default"`.
        vod_url: VOD URL on Kick / Twitch / YouTube.
        output_dir: Root directory; a `<stream_id>/` subdirectory is created.
        stream_id: Optional existing stream ID for resumes. New ULID if omitted.
        force: If true, overwrite existing files even when `stream.json` exists.
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
        return Stream.model_validate_json(stream_json.read_text("utf-8"))

    platform = detect_platform(vod_url)

    info = await asyncio.to_thread(_download_vod, vod_url=vod_url, target_path=video_path)

    if force or not audio_path.exists():
        await asyncio.to_thread(_extract_audio, video_path, audio_path)

    duration_s = float(info.get("duration") or 0.0)
    if duration_s <= 0.0:
        duration_s = await asyncio.to_thread(_ffprobe_duration, video_path)

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
    )
    stream_json.write_text(stream.model_dump_json(indent=2), encoding="utf-8")
    return stream


def _download_vod(*, vod_url: str, target_path: Path) -> dict[str, Any]:
    """Run yt-dlp; ensure the resulting file lives at exactly `target_path`.

    yt-dlp picks the file extension from the chosen format, so we let it
    write `video.<ext>` and rename to `video.mp4` once we know the actual
    path. `merge_output_format=mp4` keeps the container consistent for
    downstream ffmpeg work.
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

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info: dict[str, Any] = ydl.extract_info(vod_url, download=True) or {}
    except yt_dlp.utils.DownloadError as e:
        raise IngestError(f"yt-dlp failed for {vod_url}: {e}") from e

    actual = _resolve_downloaded_path(info, fallback=target_path)
    if actual.resolve() != target_path.resolve():
        if target_path.exists():
            target_path.unlink()
        actual.replace(target_path)
    return info


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
