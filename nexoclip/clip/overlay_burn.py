"""Renderer-side overlay burn-in — slice F.7-E.

The clip editor saves a per-clip `overlay_config` (title text,
KICK-style platform banner, captions toggle, etc) and shows an
HTML preview. THIS module is the second pass that takes the
already-cut MP4 and produces a fresh MP4 with those overlays
actually burned into the pixels:

    <clip_dir>/clip.mp4          (cut by the pipeline; brand handle only)
        +
    <clip_dir>/.captions.srt     (regenerated each burn from transcript)
        ↓
    ffmpeg with chained drawbox + drawtext + subtitles filters
        ↓
    <clip_dir>/clip_final.mp4    (what publishers should upload)

Publishers (auto-publish dispatcher + manual publish endpoint)
prefer `clip_final.mp4` when present, fall back to `clip.mp4`
otherwise. No migration / model field needed — file presence IS
the contract.

Designed pure-function-where-possible so the filter-graph
construction is unit-testable without invoking ffmpeg:

    build_filter_graph(...)   → str of chained filters
    build_srt(...)            → SRT body string
    burn_overlays(...)        → drives ffmpeg (the only impure part)

Failures bubble up as RuntimeError; the caller (the finalize
endpoint) maps that to HTTP 502 with the ffmpeg stderr inline so
the operator can see what went wrong without leaving the page.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import structlog

from .service import _find_system_font

_log = structlog.get_logger(__name__)


# Platform → default banner color (kept consistent with the editor's
# preview overlay so the burned MP4 matches what the operator saw).
PLATFORM_COLORS: dict[str, str] = {
    "kick":      "#53FC18",
    "twitch":    "#9146FF",
    "tiktok":    "#000000",
    "youtube":   "#FF0000",
    "instagram": "#E1306C",
}

# Title overlay sizing (relative to output width, computed in
# build_filter_graph). Tuned to feel right at 1080x1920 — the values
# scale linearly when the output is smaller / larger.
_TITLE_FONTSIZE_FRAC = 0.038
_TITLE_BOX_PADDING = 14
_TITLE_TOP_MARGIN = 28

# Banner sizing — height as a fraction of total output height.
_BANNER_HEIGHT_FRAC = 0.05
_BANNER_PLATFORM_FONTSIZE_FRAC = 0.025
_BANNER_URL_FONTSIZE_FRAC = 0.018


class _OverlaySpec(NamedTuple):
    """Resolved overlay parameters extracted from the editor's config.

    Lives as an intermediate so the filter-graph builder doesn't need
    to know about the dict shape — it operates on typed values."""

    title_text: str | None
    banner_enabled: bool
    banner_platform: str
    banner_url: str
    banner_color: str
    captions_enabled: bool


def _spec_from_overlay_config(
    overlay: dict[str, object] | None,
) -> _OverlaySpec:
    """Pull the burnable fields out of overlay_config with defensive
    defaults — every field is optional / nested, so missing keys
    collapse to "skip this overlay" rather than raising."""
    if not isinstance(overlay, dict):
        return _OverlaySpec(
            title_text=None,
            banner_enabled=False,
            banner_platform="kick",
            banner_url="",
            banner_color="",
            captions_enabled=True,  # default-ON to match the editor checkbox
        )
    title = overlay.get("title_text")
    banner = overlay.get("banner") if isinstance(overlay.get("banner"), dict) else {}
    captions = overlay.get("captions") if isinstance(overlay.get("captions"), dict) else {}
    assert isinstance(banner, dict)
    assert isinstance(captions, dict)
    return _OverlaySpec(
        title_text=(title.strip() if isinstance(title, str) and title.strip() else None),
        banner_enabled=bool(banner.get("enabled", False)),
        banner_platform=str(banner.get("platform") or "kick").lower(),
        banner_url=str(banner.get("url") or "").strip(),
        banner_color=str(banner.get("color") or "").strip(),
        captions_enabled=bool(captions.get("enabled", True)),
    )


# ---- helpers --------------------------------------------------


def _ff_escape_text(s: str) -> str:
    """Escape a string for inclusion inside an ffmpeg drawtext `text=`
    value. The filter-graph parser eats: backslash, colon, single
    quote, square brackets, comma, semicolon. The drawtext value
    itself eats: percent (printf-style format codes).

    Order matters — backslash first so we don't double-escape later
    insertions of `\\:`."""
    return (
        s.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace(",", "\\,")
        .replace(";", "\\;")
        .replace("%", "\\%")
    )


def _ff_escape_path(p: Path) -> str:
    """ffmpeg drawtext / subtitles need forward-slashed paths, with
    Windows colons escaped (`C:` → `C\\:`)."""
    return str(p).replace("\\", "/").replace(":", "\\:")


def _hex_to_ff_color(hex_color: str, *, fallback: str = "white") -> str:
    """`#FFD700` → `0xFFD700`. ffmpeg accepts named colors (white,
    black, red, ...), `0xRRGGBB`, and `0xRRGGBB@alpha`. Hex with `#`
    prefix is NOT accepted — has to be `0x` or named."""
    h = hex_color.strip().lstrip("#")
    if len(h) in (3, 6) and all(c in "0123456789abcdefABCDEF" for c in h):
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return f"0x{h.upper()}"
    return fallback


# ---- SRT generation --------------------------------------------


def build_srt(
    segments: Iterable[object],
    *,
    clip_start_s: float,
    clip_end_s: float,
) -> str:
    """Generate SRT subtitles from transcript segments, sliced to the
    clip's window and with timestamps shifted to clip-relative.

    Returns "" when no segments overlap the window — caller should
    treat that as "no captions to burn" and skip the subtitles filter.

    Each segment dict needs:
      - start (float, stream-relative seconds)
      - end   (float, stream-relative seconds)
      - text  (str)
    Anything missing is skipped.
    """
    rows: list[tuple[float, float, str]] = []
    for s in segments:
        if not isinstance(s, dict):
            continue
        try:
            seg_start = float(s.get("start") or 0)
            seg_end = float(s.get("end") or seg_start)
        except (TypeError, ValueError):
            continue
        text = str(s.get("text") or "").strip()
        if not text:
            continue
        # Filter to clip window.
        if seg_end < clip_start_s or seg_start > clip_end_s:
            continue
        # Clip to window edges + shift to clip-relative timing.
        rel_start = max(0.0, seg_start - clip_start_s)
        rel_end = max(rel_start + 0.1, min(clip_end_s, seg_end) - clip_start_s)
        rows.append((rel_start, rel_end, text))

    if not rows:
        return ""

    out: list[str] = []
    for i, (start, end, text) in enumerate(rows, start=1):
        out.append(str(i))
        out.append(f"{_srt_ts(start)} --> {_srt_ts(end)}")
        out.append(text)
        out.append("")
    return "\n".join(out)


def _srt_ts(seconds: float) -> str:
    """SRT timestamps: HH:MM:SS,mmm (comma, not period, before ms)."""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    rem = seconds - h * 3600
    m = int(rem // 60)
    s = rem - m * 60
    whole = int(s)
    ms = round((s - whole) * 1000)
    if ms == 1000:
        ms = 0
        whole += 1
    return f"{h:02d}:{m:02d}:{whole:02d},{ms:03d}"


# ---- filter-graph construction --------------------------------


def build_filter_graph(
    spec: _OverlaySpec,
    *,
    output_w: int,
    output_h: int,
    fontfile: Path | None,
    srt_path: Path | None,
) -> str:
    """Compose the chained ffmpeg `-vf` filter expression for the
    overlay burn pass. Returns "" when nothing's enabled — caller
    can skip the re-render entirely in that case.

    Order matters — earlier filters render first (i.e. underneath
    later ones in the final pixel composition):
      1. title text (top, white box card)
      2. captions (subtitles= filter, ASS / SRT)
      3. platform banner (bottom strip — drawn LAST so it covers
         any caption that drifted into the bottom band)
    """
    chunks: list[str] = []

    # 1. Title overlay — top, white box, dark text.
    if spec.title_text and fontfile is not None:
        chunks.append(_title_filter(spec.title_text, output_w=output_w, fontfile=fontfile))

    # 2. Captions — subtitles= filter against an SRT we wrote next
    # to the clip. Skipped when captions_enabled is False, when no
    # transcript segments overlapped the clip window (srt_path is
    # None), or when the SRT body was empty.
    if spec.captions_enabled and srt_path is not None and srt_path.exists():
        chunks.append(_captions_filter(srt_path))

    # 3. Platform banner — drawbox + drawtext (platform left, URL right).
    if spec.banner_enabled and fontfile is not None:
        chunks.extend(
            _banner_filters(
                platform=spec.banner_platform,
                url=spec.banner_url,
                color_hex=(
                    spec.banner_color
                    or PLATFORM_COLORS.get(spec.banner_platform, "#53FC18")
                ),
                output_w=output_w,
                output_h=output_h,
                fontfile=fontfile,
            )
        )

    return ",".join(chunks)


def _title_filter(
    text: str, *, output_w: int, fontfile: Path
) -> str:
    """White-card title at the top — bold dark text, padded white box,
    soft drop shadow (achieved via box opacity + border)."""
    fontsize = max(20, int(output_w * _TITLE_FONTSIZE_FRAC))
    return (
        f"drawtext=fontfile='{_ff_escape_path(fontfile)}'"
        f":text='{_ff_escape_text(text)}'"
        f":fontcolor=black"
        f":fontsize={fontsize}"
        f":x=(w-text_w)/2"
        f":y={_TITLE_TOP_MARGIN}"
        f":box=1:boxcolor=white@0.95:boxborderw={_TITLE_BOX_PADDING}"
    )


def _captions_filter(srt_path: Path) -> str:
    """Burn segment-level captions via the subtitles filter.

    Uses `force_style` to override the SRT default (Arial yellow)
    with a high-contrast white-on-black-stroke that reads on any
    background — matches the editor's caption preview vibe.
    """
    return (
        f"subtitles='{_ff_escape_path(srt_path)}'"
        f":force_style='"
        f"FontName=Arial,FontSize=22,"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"BorderStyle=1,Outline=3,Shadow=0,"
        f"Bold=1,Alignment=2,MarginV=80"
        f"'"
    )


def _banner_filters(
    *,
    platform: str,
    url: str,
    color_hex: str,
    output_w: int,
    output_h: int,
    fontfile: Path,
) -> list[str]:
    """Bottom strip — drawbox for the colored band, then two
    drawtext layers (platform name on the left, URL on the right).

    Returns a list of filter chunks the caller joins with commas."""
    band_h = max(28, int(output_h * _BANNER_HEIGHT_FRAC))
    band_y = output_h - band_h
    color = _hex_to_ff_color(color_hex, fallback="0x53FC18")
    chunks: list[str] = [
        # Colored bottom band.
        (
            f"drawbox=x=0:y={band_y}"
            f":w={output_w}:h={band_h}"
            f":color={color}@0.95:t=fill"
        )
    ]

    plat_size = max(16, int(output_w * _BANNER_PLATFORM_FONTSIZE_FRAC))
    chunks.append(
        f"drawtext=fontfile='{_ff_escape_path(fontfile)}'"
        f":text='{_ff_escape_text(platform.upper())}'"
        f":fontcolor=black"
        f":fontsize={plat_size}"
        f":x=20"
        f":y={band_y}+({band_h}-text_h)/2"
    )

    if url:
        url_size = max(12, int(output_w * _BANNER_URL_FONTSIZE_FRAC))
        chunks.append(
            f"drawtext=fontfile='{_ff_escape_path(fontfile)}'"
            f":text='{_ff_escape_text(url)}'"
            f":fontcolor=black@0.85"
            f":fontsize={url_size}"
            f":x=w-text_w-20"
            f":y={band_y}+({band_h}-text_h)/2"
        )

    return chunks


# ---- driver ---------------------------------------------------


def burn_overlays(
    *,
    source_path: Path,
    target_path: Path,
    overlay_config: dict[str, object] | None,
    transcript_segments: Iterable[object],
    clip_start_s: float,
    clip_end_s: float,
    output_w: int,
    output_h: int,
) -> bool:
    """Re-render `source_path` with overlays burned in, writing to
    `target_path`. Returns True on success, False when there's
    nothing to burn (no overlays enabled and no captions to render).

    Raises RuntimeError on ffmpeg failure with the stderr inline so
    the caller can show the operator what went wrong.
    """
    if not source_path.exists():
        raise RuntimeError(f"source clip missing: {source_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    spec = _spec_from_overlay_config(overlay_config)

    # Write SRT next to the target (deterministic path so the
    # captions filter can resolve it; cleaned up after burn).
    srt_path: Path | None = None
    if spec.captions_enabled:
        srt_body = build_srt(
            transcript_segments,
            clip_start_s=clip_start_s,
            clip_end_s=clip_end_s,
        )
        if srt_body:
            srt_path = target_path.parent / ".captions.srt"
            srt_path.write_text(srt_body, encoding="utf-8")

    fontfile = _find_system_font()
    filter_graph = build_filter_graph(
        spec,
        output_w=output_w,
        output_h=output_h,
        fontfile=fontfile,
        srt_path=srt_path,
    )

    if not filter_graph:
        # Nothing to burn — caller should fall back to source_path.
        # We still return False so the caller can decide what to do.
        if srt_path and srt_path.exists():
            srt_path.unlink(missing_ok=True)
        return False

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", str(source_path),
        "-vf", filter_graph,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "copy",  # no re-encode of audio — saves time
        str(target_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)

    # Clean up the SRT regardless of outcome — it's a per-burn artifact.
    if srt_path and srt_path.exists():
        srt_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        # Surface the LAST ~600 chars of stderr so the operator can
        # see what ffmpeg was unhappy about without drowning in it.
        err = proc.stderr.decode("utf-8", errors="replace")[-600:]
        raise RuntimeError(f"ffmpeg burn failed: {err.strip()}")

    return True
