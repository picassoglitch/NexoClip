"""Clip cutting + 9:16 reformat via two ffmpeg invocations.

Pipeline per candidate:
    1. Fast cut         → `_cut.mp4` (`-ss` before `-i`, `-c copy`; may snap to keyframe).
    2. Smart-crop       → choose a face-centered 9:16 box on the source frame.
    3. Auto-thumbnail   → pick the sharpest face-bearing frame, save JPEG.
    4. Reformat 9:16    → `clip.mp4` (`crop=W:H:X:Y,scale=Wo:Ho`, libx264 + aac).

Caption burning is deliberately skipped in Phase 0/1 — see PHASE_0.md.

Layout:
    <output_dir>/<stream_id>/
        clips_manifest.json
        clips/
            <clip_id>/
                clip.mp4
                thumbnail.jpg
                metadata.json
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from nexoclip.config import ClipConfig
from nexoclip.detect import Candidate
from nexoclip.errors import ClipError
from nexoclip.ids import new_id
from nexoclip.ingest import Stream
from nexoclip.logging import get_logger

if TYPE_CHECKING:
    from nexoclip.transcribe import Transcript

from .models import Clip, ClipManifest, SmartCropBox
from .smart_crop import compute_smart_crop_box, crop_box_to_ffmpeg_filter
from .thumbnail import pick_thumbnail, save_thumbnail
from .thumbnail_brand import pick_brand_kit_handle, render_branded_thumbnails

_log = get_logger("nexoclip.clip")


def cut_window(
    *,
    timestamp: float,
    pre_roll_s: float,
    post_roll_s: float,
    stream_duration_s: float,
    trigger_kind: str = "forward",
    retroactive_lookback_s: float | None = None,
) -> tuple[float, float, float]:
    """Compute (start, end, duration) for one candidate, clamped to VOD bounds.

    Two modes:
      * `trigger_kind="forward"` (default) — symmetric window centered loosely
        around the timestamp: [ts - pre_roll, ts + post_roll]. Matches the
        original behavior. Use for audio spikes, viral picks, forward voice
        triggers ('clipea esto').
      * `trigger_kind="retroactive"` — clip extends BACKWARD from the
        timestamp: [ts - retroactive_lookback, ts]. The natural shape for
        post-hoc voice markers ('clipeaste eso' — the moment ended, then
        the streamer flagged it). Falls back to forward semantics if
        `retroactive_lookback_s` isn't provided.
    """
    if stream_duration_s <= 0:
        raise ClipError(f"non-positive stream duration: {stream_duration_s}")
    if trigger_kind == "retroactive" and retroactive_lookback_s is not None:
        start = max(0.0, timestamp - retroactive_lookback_s)
        end = min(stream_duration_s, timestamp)
        duration = max(0.0, end - start)
        return start, end, duration
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
    brand_kits: list[object] | None = None,
    transcript: "Transcript | None" = None,
) -> list[Clip]:
    """Cut + reformat one clip per candidate. Idempotent on `clips_manifest.json`.

    Args:
        tenant_id: Must match `stream.tenant_id` (CLAUDE.md hard rule #1).
        stream: Source stream produced by `ingest_vod`.
        candidates: Detected candidates produced by `detect_voice_triggers`.
        output_dir: Root output dir (`./out`); clips go to `<output_dir>/<stream.id>/clips/`.
        config: Cut/reformat parameters. Defaults to `ClipConfig()`.
        force: Re-cut every clip even when the manifest exists.
        brand_kits: Optional parallel-to-candidates list of resolved BrandKitRow
            (or None per candidate). When set, the renderer burns the kit's
            primary social handle into the top-left of each clip. Typed loosely
            as `list[object] | None` to avoid a clip → db type cycle; the
            renderer reads attributes via `getattr` and falls back to no overlay
            on any missing field. Voice-markers spec slice D.1.
        transcript: Optional Transcript — when present (and the new
            `ClipConfig.dynamic_windowing` flag is True), each clip's
            start/end snaps to sentence boundaries instead of using the
            fixed pre_roll/post_roll. Slice G.1. Pass `None` to keep the
            legacy static-window behavior (existing tests use this path).
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
        brand_kits=brand_kits,
        transcript=transcript,
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
    brand_kits: list[object] | None = None,
    transcript: "Transcript | None" = None,
) -> list[Clip]:
    """Synchronous helper kept off the event loop via `asyncio.to_thread`."""
    clips: list[Clip] = []
    for idx, candidate in enumerate(candidates):
        kit = brand_kits[idx] if brand_kits and idx < len(brand_kits) else None
        ev = candidate.evidence or {}
        trigger_kind = str(ev.get("trigger_kind", "forward"))
        retro_lookback = ev.get("retroactive_lookback_s")
        retro_lookback_f = float(retro_lookback) if isinstance(retro_lookback, int | float) else None

        # Slice G.1 — dynamic windowing when a transcript is available
        # AND the operator hasn't disabled it. Falls back to the legacy
        # static cut_window() so existing tests + callers (without
        # transcript) keep producing the same boundaries they did before.
        if transcript is not None and getattr(cfg, "dynamic_windowing", True):
            from .windowing import plan_clip_window

            plan = plan_clip_window(
                candidate=candidate,
                transcript=transcript,
                stream_duration_s=stream.duration_s,
                fallback_pre_roll_s=cfg.pre_roll_s,
                fallback_post_roll_s=cfg.post_roll_s,
            )
            start, end, duration = plan.start_s, plan.end_s, plan.duration_s
            window_plan_evidence: dict[str, object] | None = {
                "kind": plan.kind,
                "reason": plan.reason,
                "duration_s": round(plan.duration_s, 3),
            }
        else:
            start, end, duration = cut_window(
                timestamp=candidate.timestamp,
                pre_roll_s=cfg.pre_roll_s,
                post_roll_s=cfg.post_roll_s,
                stream_duration_s=stream.duration_s,
                trigger_kind=trigger_kind,
                retroactive_lookback_s=retro_lookback_f,
            )
            window_plan_evidence = None

        if duration <= 0.0:
            continue

        clip_id = new_id("clp")
        clip_dir = clips_dir / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        intermediate = clip_dir / "_cut.mp4"
        final = clip_dir / "clip.mp4"

        # Smart crop + thumbnail: pre-decode pass on the source video.
        # Failures here fall back gracefully — vision deps may be absent
        # (test stubs with placeholder bytes) or the source may not have
        # any detectable faces. The clip still ships either way.
        smart_box = _safe_smart_crop(
            video_path=stream.source_video_path, start_s=start, end_s=end
        )
        thumbnail_path, raw_jpeg = _safe_thumbnail(
            video_path=stream.source_video_path,
            start_s=start,
            end_s=end,
            clip_dir=clip_dir,
        )
        branded = (
            _safe_branded_thumbnails(
                source_jpeg=raw_jpeg, clip_dir=clip_dir, brand_kit=kit
            )
            if raw_jpeg is not None
            else {}
        )

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
                smart_box=smart_box,
                brand_kit=kit,
            )
        finally:
            if intermediate.exists():
                intermediate.unlink()

        # Slice G.1 — when the new dynamic-windowing path picked the
        # boundaries, stamp the plan's metadata onto the candidate's
        # evidence so it persists with the clip and the dashboard can
        # display "reaction band 10-22s; snapped end to sentence boundary".
        if window_plan_evidence is not None:
            candidate = candidate.model_copy(
                update={
                    "evidence": {
                        **(candidate.evidence or {}),
                        "window_plan": window_plan_evidence,
                    }
                }
            )

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
            smart_crop_box=smart_box,
            thumbnail_path=thumbnail_path,
            thumbnail_16x9_path=branded.get("16x9"),
            thumbnail_9x16_path=branded.get("9x16"),
            thumbnail_1x1_path=branded.get("1x1"),
        )
        (clip_dir / "metadata.json").write_text(clip.model_dump_json(indent=2), encoding="utf-8")
        clips.append(clip)
    return clips


def _safe_smart_crop(
    *, video_path: Path, start_s: float, end_s: float
) -> SmartCropBox | None:
    """Run smart_crop, log + skip on failure (e.g. unreadable test stub video)."""
    try:
        return compute_smart_crop_box(video_path, start_s=start_s, end_s=end_s)
    except ClipError as e:
        _log.warning("smart_crop.skipped", reason=str(e))
        return None


def _safe_thumbnail(
    *, video_path: Path, start_s: float, end_s: float, clip_dir: Path
) -> tuple[Path | None, bytes | None]:
    """Run pick_thumbnail + save_thumbnail; log + skip on failure.

    Returns `(path, raw_jpeg_bytes)` so the branded compositor can reuse
    the same decoded frame without another OpenCV pass over the source
    video. On failure both values are None and the caller skips the
    branded-variant step too."""
    try:
        jpeg, _ts, _bd = pick_thumbnail(video_path, start_s=start_s, end_s=end_s)
        return save_thumbnail(clip_dir, jpeg), jpeg
    except ClipError as e:
        _log.warning("thumbnail.skipped", reason=str(e))
        return None, None


def _safe_branded_thumbnails(
    *, source_jpeg: bytes, clip_dir: Path, brand_kit: object | None
) -> dict[str, Path]:
    """Composite the 16:9 / 9:16 / 1:1 brand-kit thumbnail variants.

    `brand_kit` is loosely typed as `object | None` to keep the clip
    module free of a hard dependency on `nexoclip.db` (same approach
    `_brand_kit_drawtext_filter` uses for the slice D.1 handle overlay).
    When `brand_kit` is None we fall back to a neutral grey scheme so
    the variants still render — the publisher upload still benefits
    from the aspect-correct thumbnails even without brand colors.

    Returns a `{aspect: path}` dict keyed by `"16x9"` / `"9x16"` / `"1x1"`.
    Missing keys mean that variant failed to render (per-variant try/
    except in the compositor)."""
    primary = (
        str(getattr(brand_kit, "primary_color", None) or "#1F2937")
        if brand_kit is not None
        else "#1F2937"
    )
    accent = (
        str(getattr(brand_kit, "accent_color", None) or "#F59E0B")
        if brand_kit is not None
        else "#F59E0B"
    )
    text = (
        str(getattr(brand_kit, "text_color", None) or "#FFFFFF")
        if brand_kit is not None
        else "#FFFFFF"
    )
    handle = pick_brand_kit_handle(
        handle_tiktok=getattr(brand_kit, "handle_tiktok", None),
        handle_youtube=getattr(brand_kit, "handle_youtube", None),
        handle_instagram=getattr(brand_kit, "handle_instagram", None),
        handle_kick=getattr(brand_kit, "handle_kick", None),
    )
    paths = render_branded_thumbnails(
        source_jpeg=source_jpeg,
        clip_dir=clip_dir,
        handle=handle,
        primary_color=primary,
        accent_color=accent,
        text_color=text,
    )
    out: dict[str, Path] = {}
    for p in paths:
        # render_branded_thumbnails names files thumb_<aspect>.jpg — peel
        # the aspect token out of the filename so we don't re-import the
        # private VARIANTS tuple just to map back.
        stem = p.stem  # 'thumb_16x9'
        if stem.startswith("thumb_"):
            out[stem.removeprefix("thumb_")] = p
    return out


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


def _ffmpeg_reformat_9_16(
    *,
    in_path: Path,
    out_path: Path,
    cfg: ClipConfig,
    smart_box: SmartCropBox | None = None,
    brand_kit: object | None = None,
) -> None:
    """Crop to 9:16 (smart-box or center) and scale to the configured resolution.

    When `brand_kit` is provided and the kit carries a primary social handle
    + we can resolve a font on this OS, append a `drawtext` filter that burns
    the handle into the top-left corner using the kit's accent color. Slice
    D.1 of the voice-markers spec. Logo burn-in lands in D.3 alongside the
    AI logo generator.

    Failures to resolve a font or read kit attributes are silent — the
    clip still renders, just without the overlay.
    """
    if smart_box is not None:
        vf = crop_box_to_ffmpeg_filter(
            smart_box, output_w=cfg.output_width, output_h=cfg.output_height
        )
    else:
        vf = f"crop=ih*9/16:ih,scale={cfg.output_width}:{cfg.output_height}"

    overlay = _brand_kit_drawtext_filter(brand_kit, output_w=cfg.output_width)
    if overlay:
        vf = f"{vf},{overlay}"

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


def _brand_kit_drawtext_filter(
    brand_kit: object | None, *, output_w: int
) -> str | None:
    """Build the ffmpeg `drawtext` filter chunk for a brand kit's handle.

    Returns the filter expression (without leading comma) or None when:
      * no kit was supplied
      * the kit has no primary handle on any platform
      * we can't locate a system font (drawtext requires a file path on
        most ffmpeg builds; bundling a font is a slice E upgrade)

    Handle resolution priority (matches the renderer's spec §3.5):
      tiktok > youtube > instagram > kick
    """
    if brand_kit is None:
        return None
    handle = (
        getattr(brand_kit, "handle_tiktok", None)
        or getattr(brand_kit, "handle_youtube", None)
        or getattr(brand_kit, "handle_instagram", None)
        or getattr(brand_kit, "handle_kick", None)
    )
    if not handle:
        return None

    fontfile = _find_system_font()
    if fontfile is None:
        _log.warning(
            "brand_kit.drawtext_skipped",
            reason="no system font found",
            handle=handle,
        )
        return None

    accent = getattr(brand_kit, "accent_color", "#FFD700") or "#FFD700"
    # ffmpeg drawtext needs a path with forward slashes + escaped colons on
    # Windows. The escape rules: literal colons in the filter graph value
    # delimit filter options, so `C:` becomes `C\\:`. Forward slashes survive.
    fontfile_ff = str(fontfile).replace("\\", "/").replace(":", "\\:")
    # Single quotes inside the value need their own dance. Stripping them
    # from the handle is fine — handles rarely have them.
    safe_handle = (handle or "").replace("'", "")
    fontsize = max(24, int(output_w * 0.028))
    margin = max(12, int(output_w * 0.022))
    return (
        f"drawtext=fontfile='{fontfile_ff}'"
        f":text='{safe_handle}'"
        f":fontcolor={accent}"
        f":fontsize={fontsize}"
        f":x={margin}:y={margin}"
        f":box=1:boxcolor=black@0.45:boxborderw=8"
    )


def _find_system_font() -> Path | None:
    """Resolve a usable TTF/OTF for ffmpeg drawtext across OSes.

    Cached after first hit. None when nothing matches — caller must
    tolerate it. The bundled-font path (ship our own Inter.ttf) is a
    slice E upgrade for cross-machine consistency.
    """
    global _CACHED_FONT
    if not isinstance(_CACHED_FONT, _UnsetFont):
        return _CACHED_FONT
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
        Path("/System/Library/Fonts/HelveticaNeue.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ]
    for c in candidates:
        if c.exists():
            _CACHED_FONT = c
            return c
    _CACHED_FONT = None
    return None


class _UnsetFont:
    pass


_UNSET_FONT = _UnsetFont()
_CACHED_FONT: Path | None | _UnsetFont = _UNSET_FONT


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
