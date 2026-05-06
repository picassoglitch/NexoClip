"""Clip cutting + 9:16 reformat via two ffmpeg invocations.

Pipeline per candidate:
    1. Fast cut         → `_cut.mp4` (`-ss` before `-i`, `-c copy`; may snap to keyframe).
    2. Reformat 9:16    → `clip.mp4` (`crop=ih*9/16:ih,scale=W:H`, libx264 + aac).

Caption burning is deliberately skipped in Phase 0 — see PHASE_0.md.

Layout:
    <output_dir>/<stream_id>/
        clips_manifest.json
        clips/
            <clip_id>/
                clip.mp4
                metadata.json
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from nexoclip.config import ClipConfig
from nexoclip.detect import Candidate
from nexoclip.errors import ClipError
from nexoclip.ids import new_id
from nexoclip.ingest import Stream

from .models import Clip, ClipManifest


def cut_window(
    *,
    timestamp: float,
    pre_roll_s: float,
    post_roll_s: float,
    stream_duration_s: float,
) -> tuple[float, float, float]:
    """Compute (start, end, duration) for one candidate, clamped to the VOD bounds."""
    if stream_duration_s <= 0:
        raise ClipError(f"non-positive stream duration: {stream_duration_s}")
    start = max(0.0, timestamp - pre_roll_s)
    naive_end = min(stream_duration_s, timestamp + post_roll_s)
    duration = max(0.0, naive_end - start)
    return start, start + duration, duration


async def cut_clips(
    tenant_id: str,
    stream: Stream,
    candidates: list[Candidate],
    output_dir: Path,
    *,
    config: ClipConfig | None = None,
    force: bool = False,
) -> list[Clip]:
    """Cut + reformat one clip per candidate. Idempotent on `clips_manifest.json`.

    Args:
        tenant_id: Must match `stream.tenant_id` (CLAUDE.md hard rule #1).
        stream: Source stream produced by `ingest_vod`.
        candidates: Detected candidates produced by `detect_voice_triggers`.
        output_dir: Root output dir (`./out`); clips go to `<output_dir>/<stream.id>/clips/`.
        config: Cut/reformat parameters. Defaults to `ClipConfig()`.
        force: Re-cut every clip even when the manifest exists.
    """
    if tenant_id != stream.tenant_id:
        raise ClipError(f"tenant mismatch: caller={tenant_id!r}, stream={stream.tenant_id!r}")

    cfg = config or ClipConfig()
    output_dir = Path(output_dir).resolve()
    stream_dir = output_dir / stream.id
    clips_dir = stream_dir / "clips"
    manifest_path = stream_dir / "clips_manifest.json"
    clips_dir.mkdir(parents=True, exist_ok=True)

    if not force and manifest_path.exists():
        return ClipManifest.model_validate_json(manifest_path.read_text("utf-8")).clips

    if not stream.source_video_path.exists():
        raise ClipError(f"source video missing: {stream.source_video_path}")

    clips = await asyncio.to_thread(
        _cut_all,
        tenant_id=tenant_id,
        stream=stream,
        candidates=candidates,
        clips_dir=clips_dir,
        cfg=cfg,
    )

    manifest = ClipManifest(stream_id=stream.id, tenant_id=tenant_id, clips=clips)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return clips


def _cut_all(
    *,
    tenant_id: str,
    stream: Stream,
    candidates: list[Candidate],
    clips_dir: Path,
    cfg: ClipConfig,
) -> list[Clip]:
    """Synchronous helper kept off the event loop via `asyncio.to_thread`."""
    clips: list[Clip] = []
    for candidate in candidates:
        start, end, duration = cut_window(
            timestamp=candidate.timestamp,
            pre_roll_s=cfg.pre_roll_s,
            post_roll_s=cfg.post_roll_s,
            stream_duration_s=stream.duration_s,
        )
        if duration <= 0.0:
            continue

        clip_id = new_id("clp")
        clip_dir = clips_dir / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        intermediate = clip_dir / "_cut.mp4"
        final = clip_dir / "clip.mp4"

        try:
            _ffmpeg_fast_cut(
                video_path=stream.source_video_path,
                start_s=start,
                duration_s=duration,
                out_path=intermediate,
            )
            _ffmpeg_reformat_9_16(
                in_path=intermediate,
                out_path=final,
                cfg=cfg,
            )
        finally:
            if intermediate.exists():
                intermediate.unlink()

        clip = Clip(
            id=clip_id,
            tenant_id=tenant_id,
            stream_id=stream.id,
            candidate=candidate,
            start_s=start,
            end_s=end,
            duration_s=duration,
            width=cfg.output_width,
            height=cfg.output_height,
            path=final,
        )
        (clip_dir / "metadata.json").write_text(clip.model_dump_json(indent=2), encoding="utf-8")
        clips.append(clip)
    return clips


def _ffmpeg_fast_cut(
    *, video_path: Path, start_s: float, duration_s: float, out_path: Path
) -> None:
    """Stream-copy a window out of `video_path`. Fast but keyframe-aligned."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration_s:.3f}",
        "-c",
        "copy",
        str(out_path),
    ]
    _run_ffmpeg(cmd, what=f"fast cut at {start_s:.3f}s")


def _ffmpeg_reformat_9_16(*, in_path: Path, out_path: Path, cfg: ClipConfig) -> None:
    """Center-crop to 9:16 and scale to the configured resolution."""
    vf = f"crop=ih*9/16:ih,scale={cfg.output_width}:{cfg.output_height}"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(in_path),
        "-vf",
        vf,
        "-c:v",
        cfg.encoder,
        "-preset",
        cfg.preset,
        "-crf",
        str(cfg.crf),
        "-c:a",
        "aac",
        str(out_path),
    ]
    _run_ffmpeg(cmd, what=f"9:16 reformat -> {out_path.name}")


def _run_ffmpeg(cmd: list[str], *, what: str) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as e:
        raise ClipError("ffmpeg binary not found on PATH") from e
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", "replace") if e.stderr else ""
        raise ClipError(f"ffmpeg {what} failed: {stderr}") from e


def load_clips(stream_dir: Path) -> ClipManifest:
    """Read the saved clips manifest back from disk."""
    path = Path(stream_dir) / "clips_manifest.json"
    if not path.exists():
        raise ClipError(f"clips manifest not found at {path}")
    return ClipManifest.model_validate_json(path.read_text("utf-8"))
