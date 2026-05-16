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
from .word_captions import CaptionLine, captions_for_clip

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
    # Slice I.1 — clip style + Kick banner variant + top hook box.
    clip_style: str = "repost_page_viral"
    banner_variant: str = "kick_repost_page"
    banner_live_badge: bool = False
    top_hook_enabled: bool = False
    top_hook_text: str = ""
    top_hook_style: str = "white_rounded"


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
    top_hook = overlay.get("top_hook") if isinstance(overlay.get("top_hook"), dict) else {}
    assert isinstance(banner, dict)
    assert isinstance(captions, dict)
    assert isinstance(top_hook, dict)
    # Slice I.1 — clip_style + banner_variant + top hook overlay.
    clip_style = str(overlay.get("clip_style") or "repost_page_viral")
    banner_variant = str(banner.get("variant") or "kick_repost_page")
    return _OverlaySpec(
        title_text=(title.strip() if isinstance(title, str) and title.strip() else None),
        banner_enabled=bool(banner.get("enabled", False)),
        banner_platform=str(banner.get("platform") or "kick").lower(),
        banner_url=str(banner.get("url") or "").strip(),
        banner_color=str(banner.get("color") or "").strip(),
        captions_enabled=bool(captions.get("enabled", True)),
        clip_style=clip_style,
        banner_variant=banner_variant,
        banner_live_badge=bool(banner.get("live_badge", False)),
        top_hook_enabled=bool(top_hook.get("enabled", False)),
        top_hook_text=str(top_hook.get("text") or "").strip(),
        top_hook_style=str(top_hook.get("style") or "white_rounded"),
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

    Tolerates BOTH the real Whisper-serialized shape `{ts, end_ts}`
    (Pydantic field names from `transcribe.models.Segment`) AND the
    legacy test-fixture shape `{start, end}`. This is the foundational
    fix — production data uses `ts`/`end_ts` so the previous version
    of this function silently produced empty SRTs on real clips.

    Returns "" when no segments overlap the window — caller treats
    that as "no captions to burn" and skips the subtitles filter.
    """
    rows: list[tuple[float, float, str]] = []
    for s in segments:
        if not isinstance(s, dict):
            continue
        # Real DB shape uses `ts` / `end_ts`; test fixtures + legacy
        # callers use `start` / `end`. Read whichever is present.
        seg_start = _as_float(s.get("ts"), s.get("start"))
        seg_end = _as_float(s.get("end_ts"), s.get("end"))
        if seg_start is None or seg_end is None:
            continue
        if seg_end < seg_start:
            seg_end = seg_start
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


def _as_float(*candidates: object) -> float | None:
    """First non-None numeric coercible to float. Returns None when
    nothing in the chain is usable."""
    for c in candidates:
        if isinstance(c, int | float):
            return float(c)
        if isinstance(c, str):
            try:
                return float(c)
            except ValueError:
                continue
    return None


# ---- ASS (Advanced SubStation Alpha) generation ----------------
#
# ASS is the format ffmpeg's `subtitles=` filter renders with the
# richest tag support — per-word karaoke fill, scale, color, position.
# This is what burns into clip_final.mp4 to match the editor preview.

# ASS expects integer (width, height) for PlayResX/Y in the Script Info
# block. Renderer scales positioning + font-size relative to these.
_ASS_PLAYRES_W = 1080
_ASS_PLAYRES_H = 1920

# Bottom UI band on TikTok/Reels/Shorts measured in PlayRes units —
# captions sit ABOVE this so platform chrome doesn't eat them. ~220
# of 1920 = bottom ~11% of the frame is "danger zone".
_ASS_BOTTOM_SAFE_MARGIN_V = 220

# Per-emphasis font-size multiplier on a baseline of 56pt (renders
# tall + readable at 1080x1920). Shout > reaction > emphasis > normal.
_ASS_BASE_FONTSIZE = 56
_ASS_EMPHASIS_SCALE: dict[str, float] = {
    "shout": 1.55,
    "reaction": 1.30,
    "emphasis": 1.18,
    "normal": 1.00,
}
# Per-emphasis fill color (ASS &HBBGGRR — BGR + alpha-first hex).
_ASS_EMPHASIS_FILL: dict[str, str] = {
    "shout": "&H003B3BFF",      # red-ish (FF3B3B in RGB)
    "reaction": "&H006BD6FF",   # warm gold-orange
    "emphasis": "&H0000D7FF",   # gold (FFD700 in RGB)
    "normal": "&H00FFFFFF",     # white
}


def _ass_ts(seconds: float) -> str:
    """ASS timestamp — H:MM:SS.cs (centiseconds, NOT ms)."""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    rem = seconds - h * 3600
    m = int(rem // 60)
    s = rem - m * 60
    whole = int(s)
    cs = round((s - whole) * 100)
    if cs == 100:
        cs = 0
        whole += 1
    return f"{h:d}:{m:02d}:{whole:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    """Escape user text for inclusion in an ASS dialogue. ASS treats
    `{` / `}` as override-tag delimiters and `\\` as a continuation
    escape. Hard linebreaks via `\\N`; we don't use them here but
    must not let raw `\\N` leak in from transcript text."""
    return (
        text.replace("\\", "\\\\")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\r", " ")
        .replace("\n", " ")
    )


_ASS_FONT_SIZE_SCALE: dict[str, float] = {
    "small": 0.78,
    "medium": 1.0,
    "large": 1.24,
    "xl": 1.5,
}

# Maps the editor's position dropdown → ASS MarginV. The values are
# tuned for a 1080×1920 vertical canvas (`_ASS_PLAYRES_H = 1920`).
# Larger MarginV = more space from the bottom = caption sits higher.
_ASS_MARGIN_V_BY_POSITION: dict[str, int] = {
    "bottom":      220,    # above the platform UI band, default
    "lower_third": 700,    # ~63% from top — under the face
    "centered":    960,    # dead center
    "upper_third": 1280,   # ~33% from top
}


def build_ass(
    lines: Iterable[CaptionLine],
    *,
    playres_w: int = _ASS_PLAYRES_W,
    playres_h: int = _ASS_PLAYRES_H,
    lead_ms: int = 0,
    position: str = "bottom",
    font_size: str = "medium",
    animation: str = "pop",
) -> str:
    """Generate an ASS subtitle file with per-word karaoke + emphasis.

    Each `CaptionLine` becomes ONE Dialogue entry. Inside the dialogue
    text, each word is wrapped with `{\\kf<cs>}` (karaoke fill over
    `<cs>` centiseconds) so the active word lights up in real-time as
    the clip plays back. Emphasized words also get `{\\fs<size>\\c<col>}`
    overrides so SHOUT lands BIG + red, REACTION lands warm gold,
    EMPHASIS lands gold, NORMAL stays white.

    Captions sit ABOVE the TikTok/Reels/Shorts bottom UI band (default
    margin_v = 220 of 1920 → ~11.5% from bottom).

    Returns "" when no lines were passed — caller should treat that
    as "no captions to burn" and skip the subtitles filter entirely.
    """
    lines = list(lines)
    if not lines:
        return ""

    # Slice F.7-H — operator-controlled rendering knobs.
    lead_s = max(0, min(500, int(lead_ms))) / 1000.0
    size_scale = _ASS_FONT_SIZE_SCALE.get(font_size, 1.0)
    base_fontsize = round(_ASS_BASE_FONTSIZE * size_scale)
    margin_v = _ASS_MARGIN_V_BY_POSITION.get(position, _ASS_BOTTOM_SAFE_MARGIN_V)
    # Alignment 2 = bottom-center for "bottom" / "lower_third"; 5 = mid
    # for "centered"; 8 = top-center for "upper_third". Anchor matters
    # because MarginV is measured FROM the anchor — wrong anchor and
    # the caption flips off-frame.
    alignment = {"bottom": 2, "lower_third": 2, "centered": 5, "upper_third": 8}.get(
        position, 2
    )
    # \fad(in_ms, out_ms) gives the "fade" animation per Dialogue.
    # "pop" / "slide" / "typewriter" stay with the default \kf karaoke
    # (the per-word fill animation handles "pop" implicitly; the
    # template's CSS handles "slide" + "typewriter" for the preview;
    # the burned MP4 keeps the karaoke fill as a sensible fallback).
    line_fad = "{\\fad(180,0)}" if animation == "fade" else ""

    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {playres_w}",
        f"PlayResY: {playres_h}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
        "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        # Base style: white-fill, black-outline, alignment from above,
        # MarginV from above.
        f"Style: Default,Arial,{base_fontsize},&H00FFFFFF,"
        f"&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,5,0,{alignment},"
        f"40,40,{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text",
    ]

    events: list[str] = []
    for line in lines:
        # Shift all timings BACKWARD by lead_s so the caption appears
        # `lead_s` seconds before the word is spoken (matches the
        # preview's `t + lead/1000` lookup). Clamp to 0 so we never
        # produce negative ASS timestamps.
        ts = max(0.0, line.ts - lead_s)
        end_ts = max(ts + 0.05, line.end_ts - lead_s)
        text = _ass_line_text(line, base_fontsize=base_fontsize, lead_s=lead_s)
        events.append(
            f"Dialogue: 0,{_ass_ts(ts)},{_ass_ts(end_ts)},"
            f"Default,,0,0,0,,{line_fad}{text}"
        )

    return "\n".join(header + events) + "\n"


def _ass_line_text(
    line: CaptionLine,
    *,
    base_fontsize: int = _ASS_BASE_FONTSIZE,
    lead_s: float = 0.0,
) -> str:
    """Compose the ASS dialogue text for one line — per-word
    karaoke timings + emphasis-driven overrides.

    `base_fontsize` is the after-scale base (font_size knob has
    already been applied at the caller). `lead_s` matches the
    Dialogue-level shift so the per-word \\kf values stay in
    sync with the shifted line start.
    """
    pieces: list[str] = []
    line_start = max(0.0, line.ts - lead_s)
    for i, w in enumerate(line.words):
        # Centiseconds the word holds (used by \kf for fill duration).
        cs = max(1, round((w.end_ts - w.ts) * 100))
        scale = _ASS_EMPHASIS_SCALE.get(w.emphasis, 1.0)
        fontsize = round(base_fontsize * scale)
        color = _ASS_EMPHASIS_FILL.get(w.emphasis, "&H00FFFFFF")
        text = _ass_escape(w.text)
        # \kf needs to advance through the entire dialogue's duration.
        # Insert a leading silent-skip if there's a gap between the
        # line start and the word start (rare but happens).
        w_start = max(0.0, w.ts - lead_s)
        if i == 0 and w_start > line_start:
            gap_cs = max(0, round((w_start - line_start) * 100))
            if gap_cs > 0:
                pieces.append(f"{{\\kf{gap_cs}}}")
        pieces.append(f"{{\\kf{cs}\\fs{fontsize}\\c{color}}}{text} ")
    return "".join(pieces).rstrip()


def captions_artifact_for_clip(
    transcript_segments_json: str | None,
    *,
    clip_start_s: float,
    clip_end_s: float,
    lead_ms: int = 0,
    position: str = "bottom",
    font_size: str = "medium",
    animation: str = "pop",
) -> tuple[str, str]:
    """Return `(ass_or_srt_body, extension)` for the captions file
    the burn-in should write.

    Prefers ASS (word-level karaoke + emphasis) when the transcript
    has word-level data; falls back to SRT (segment-level only) when
    it doesn't. The ASS / SRT filter is identical from ffmpeg's
    perspective — same `subtitles=` directive.

    Slice F.7-H — the four caption knobs (`lead_ms`, `position`,
    `font_size`, `animation`) flow through to `build_ass` so the
    burned MP4 matches what the operator sees in the editor preview.
    """
    if not transcript_segments_json:
        return "", "srt"
    lines = captions_for_clip(
        transcript_segments_json,
        clip_start_s=clip_start_s,
        clip_end_s=clip_end_s,
    )
    if not lines:
        return "", "srt"
    # Did any line carry word-level timings? If yes → ASS for karaoke.
    has_words = any(
        len(line.words) > 0 and line.words[0].end_ts > line.words[0].ts
        for line in lines
    )
    if has_words:
        return build_ass(
            lines,
            lead_ms=lead_ms,
            position=position,
            font_size=font_size,
            animation=animation,
        ), "ass"

    # Fall back to SRT (legacy segment-level) by going through the
    # same JSON the build_srt function consumes.
    import json

    try:
        segments = json.loads(transcript_segments_json)
    except json.JSONDecodeError:
        return "", "srt"
    if not isinstance(segments, list):
        return "", "srt"
    return build_srt(
        segments, clip_start_s=clip_start_s, clip_end_s=clip_end_s
    ), "srt"


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
    srt_path: Path | None = None,
    captions_path: Path | None = None,
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

    # 1. Title overlay (legacy "title_text" — kept for backward compat).
    #    Slice I.1's `top_hook` is its own renderer below.
    if spec.title_text and fontfile is not None:
        chunks.append(_title_filter(spec.title_text, output_w=output_w, fontfile=fontfile))

    # 1b. Top hook box — Slice I.1 white-rounded headline overlay.
    #     Lives ABOVE the subject's face, below the platform top safe zone.
    if spec.top_hook_enabled and spec.top_hook_text and fontfile is not None:
        chunks.append(
            _top_hook_filter(
                text=spec.top_hook_text,
                style=spec.top_hook_style,
                output_w=output_w,
                output_h=output_h,
                fontfile=fontfile,
            )
        )

    # 2. Captions — subtitles= filter against an ASS (word-level
    # karaoke, slice F.7-F) or SRT (segment-level fallback) file we
    # wrote next to the clip. Skipped when captions_enabled is False,
    # when no transcript data overlapped the clip window, or when the
    # body was empty.
    cap_path = captions_path or srt_path
    if spec.captions_enabled and cap_path is not None and cap_path.exists():
        chunks.append(_captions_filter(cap_path))

    # 3. Platform banner — variant-dispatched. The chosen Kick banner
    #    variant (repost_page / black_bar_classic / green_block /
    #    minimal_url) renders a slightly different chrome. Falls back
    #    to the legacy renderer for unknown variants so older clips
    #    that don't carry banner.variant still publish identically.
    if spec.banner_enabled and fontfile is not None:
        chunks.extend(
            _banner_filters_by_variant(
                variant=spec.banner_variant,
                platform=spec.banner_platform,
                url=spec.banner_url,
                color_hex=(
                    spec.banner_color
                    or PLATFORM_COLORS.get(spec.banner_platform, "#53FC18")
                ),
                live_badge=spec.banner_live_badge,
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


def _captions_filter(captions_path: Path) -> str:
    """Burn captions via the subtitles filter.

    Two paths the captions_path can take:

      * `.ass` — Advanced SubStation Alpha with word-level karaoke
                 (slice F.7-F). Carries its own styling so we DON'T
                 pass force_style — the ASS file's own Style line
                 controls font / colors / position / margins.
      * `.srt` — segment-level fallback. We layer force_style so it
                 picks up our white-on-black-stroke baseline.
    """
    if captions_path.suffix.lower() == ".ass":
        return f"subtitles='{_ff_escape_path(captions_path)}'"
    return (
        f"subtitles='{_ff_escape_path(captions_path)}'"
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


# ---- Slice I.1 — variant-aware banner dispatcher --------------


def _banner_filters_by_variant(
    *,
    variant: str,
    platform: str,
    url: str,
    color_hex: str,
    live_badge: bool,
    output_w: int,
    output_h: int,
    fontfile: Path,
) -> list[str]:
    """Dispatch to the right Kick banner renderer.

    Falls back to the legacy `_banner_filters` for any unknown variant
    so clips saved before slice I.1 (no `banner.variant` set) keep
    publishing identically — non-breaking.
    """
    if variant == "kick_repost_page":
        return _banner_repost_page(
            url=url, color_hex=color_hex, live_badge=live_badge,
            output_w=output_w, output_h=output_h, fontfile=fontfile,
        )
    if variant == "kick_green_block":
        return _banner_green_block(
            url=url, color_hex=color_hex,
            output_w=output_w, output_h=output_h, fontfile=fontfile,
        )
    if variant == "kick_minimal_url":
        return _banner_minimal_url(
            url=url,
            output_w=output_w, output_h=output_h, fontfile=fontfile,
        )
    # "kick_black_bar_classic" + unknown → legacy renderer.
    return _banner_filters(
        platform=platform, url=url, color_hex=color_hex,
        output_w=output_w, output_h=output_h, fontfile=fontfile,
    )


def _banner_repost_page(
    *,
    url: str,
    color_hex: str,
    live_badge: bool,
    output_w: int,
    output_h: int,
    fontfile: Path,
) -> list[str]:
    """Repost-page Kick banner — short BLACK bottom bar with a huge
    accent-color "KICK" wordmark on the left, blocky URL center, and
    an optional LIVE NOW pill on the right. Matches the look operators
    actually see when their clip ricochets through repost pages.
    """
    band_h = max(56, int(output_h * 0.07))
    band_y = output_h - band_h
    accent = _hex_to_ff_color(color_hex, fallback="0x53FC18")

    # The black bottom bar itself.
    chunks: list[str] = [
        f"drawbox=x=0:y={band_y}"
        f":w={output_w}:h={band_h}"
        f":color=black@0.98:t=fill"
    ]

    # Huge accent-colored "KICK" wordmark on the left.
    logo_size = max(28, int(output_w * 0.058))
    chunks.append(
        f"drawtext=fontfile='{_ff_escape_path(fontfile)}'"
        f":text='KICK'"
        f":fontcolor={accent}"
        f":fontsize={logo_size}"
        f":x=24"
        f":y={band_y}+({band_h}-text_h)/2"
    )

    # Centered URL — defaults to "KICK.COM/USERNAME" when blank.
    if url:
        display_url = url if "." in url else f"kick.com/{url}".upper()
        url_text = display_url.upper()
    else:
        url_text = "KICK.COM/YOURHANDLE"
    url_size = max(20, int(output_w * 0.034))
    chunks.append(
        f"drawtext=fontfile='{_ff_escape_path(fontfile)}'"
        f":text='{_ff_escape_text(url_text)}'"
        f":fontcolor=white"
        f":fontsize={url_size}"
        f":x=(w-text_w)/2"
        f":y={band_y}+({band_h}-text_h)/2"
    )

    # Optional LIVE NOW pill on the right.
    if live_badge:
        pill_h = max(22, int(band_h * 0.55))
        pill_w = max(120, int(output_w * 0.13))
        pill_y = band_y + (band_h - pill_h) // 2
        pill_x = output_w - pill_w - 24
        chunks.append(
            f"drawbox=x={pill_x}:y={pill_y}"
            f":w={pill_w}:h={pill_h}"
            f":color=red@0.95:t=fill"
        )
        live_size = max(14, int(output_w * 0.022))
        chunks.append(
            f"drawtext=fontfile='{_ff_escape_path(fontfile)}'"
            f":text='LIVE NOW'"
            f":fontcolor=white"
            f":fontsize={live_size}"
            f":x={pill_x}+({pill_w}-text_w)/2"
            f":y={pill_y}+({pill_h}-text_h)/2"
        )

    return chunks


def _banner_green_block(
    *,
    url: str,
    color_hex: str,
    output_w: int,
    output_h: int,
    fontfile: Path,
) -> list[str]:
    """Gaming Chaos variant — wide colored block on the LEFT carrying
    the platform name, URL flows to the right of it on a black bar.
    Looks like an esports/gaming overlay rather than a SaaS export.
    """
    band_h = max(60, int(output_h * 0.08))
    band_y = output_h - band_h
    block_w = max(140, int(output_w * 0.20))
    accent = _hex_to_ff_color(color_hex, fallback="0x53FC18")

    chunks: list[str] = [
        # Black backing bar.
        f"drawbox=x=0:y={band_y}"
        f":w={output_w}:h={band_h}"
        f":color=black@0.95:t=fill",
        # Accent block.
        f"drawbox=x=0:y={band_y}"
        f":w={block_w}:h={band_h}"
        f":color={accent}@0.98:t=fill",
    ]
    plat_size = max(20, int(output_w * 0.036))
    chunks.append(
        f"drawtext=fontfile='{_ff_escape_path(fontfile)}'"
        f":text='KICK'"
        f":fontcolor=black"
        f":fontsize={plat_size}"
        f":x=({block_w}-text_w)/2"
        f":y={band_y}+({band_h}-text_h)/2"
    )
    if url:
        url_size = max(16, int(output_w * 0.028))
        chunks.append(
            f"drawtext=fontfile='{_ff_escape_path(fontfile)}'"
            f":text='{_ff_escape_text(url.upper())}'"
            f":fontcolor=white"
            f":fontsize={url_size}"
            f":x={block_w}+24"
            f":y={band_y}+({band_h}-text_h)/2"
        )
    return chunks


def _banner_minimal_url(
    *,
    url: str,
    output_w: int,
    output_h: int,
    fontfile: Path,
) -> list[str]:
    """Clean Creator / Minimal Native variant — no banner box, just a
    small URL stamped in the bottom-left over a subtle dark gradient
    for legibility."""
    if not url:
        return []
    # Subtle shadow box behind the URL for legibility on bright frames.
    pad = 8
    url_size = max(16, int(output_w * 0.026))
    text = url.upper() if "." in url else f"kick.com/{url}".upper()
    # No full-width band — just the text with its own outline box.
    return [
        f"drawtext=fontfile='{_ff_escape_path(fontfile)}'"
        f":text='{_ff_escape_text(text)}'"
        f":fontcolor=white"
        f":fontsize={url_size}"
        f":x=24"
        f":y=h-text_h-24"
        f":box=1:boxcolor=black@0.55:boxborderw={pad}"
    ]


# ---- Slice I.1 — top hook title box (white rounded headline) ----


def _top_hook_filter(
    *,
    text: str,
    style: str,
    output_w: int,
    output_h: int,
    fontfile: Path,
) -> str:
    """White-rounded headline box above the subject's face.

    Sits below the platform top safe zone (~8% from top) so TikTok /
    Reels chrome doesn't cover it. The "rounded corners" are emulated
    via a chunky `boxborderw` — ffmpeg drawtext doesn't natively render
    rounded box corners, but a fat padded box reads as a chip badge.

    `style` toggles fill / text color combos. White-rounded is the
    viral repost-page default.
    """
    if style == "black_solid":
        fill, fg = "black@0.92", "white"
    elif style == "subtle":
        fill, fg = "white@0.75", "black"
    else:  # white_rounded
        fill, fg = "white@0.96", "black"
    fontsize = max(20, int(output_w * 0.040))
    padding = max(14, int(output_w * 0.020))
    # Position: ~8% from top so we clear the platform's status / nav.
    top_y = max(40, int(output_h * 0.08))
    return (
        f"drawtext=fontfile='{_ff_escape_path(fontfile)}'"
        f":text='{_ff_escape_text(text)}'"
        f":fontcolor={fg}"
        f":fontsize={fontsize}"
        f":x=(w-text_w)/2"
        f":y={top_y}"
        f":box=1:boxcolor={fill}:boxborderw={padding}"
    )


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

    # Write captions file (ASS for word-level karaoke when available,
    # SRT otherwise) next to the target. Deterministic path so the
    # subtitles filter can resolve it; cleaned up after the burn.
    captions_path: Path | None = None
    if spec.captions_enabled:
        # Serialize the segments to JSON so we can reuse the same
        # captions_artifact_for_clip path the editor's preview uses —
        # one source of truth for what gets rendered.
        import json as _json

        segments_list = [s for s in transcript_segments if isinstance(s, dict)]
        if segments_list:
            # Slice F.7-H — pull the four caption knobs out of the
            # overlay_config so the burn matches the editor preview
            # (lead_ms shifts timings, position/font_size adjust the
            # ASS style header, animation controls the entrance).
            captions_cfg = overlay_config.get("captions") if isinstance(overlay_config, dict) else {}
            captions_cfg = captions_cfg if isinstance(captions_cfg, dict) else {}
            lead_ms_raw = captions_cfg.get("lead_ms", 0)
            lead_ms_int = int(lead_ms_raw) if isinstance(lead_ms_raw, int | float | str) and str(lead_ms_raw).strip() else 0
            position_str = str(captions_cfg.get("position") or "bottom")
            font_size_str = str(captions_cfg.get("font_size") or "medium")
            animation_str = str(captions_cfg.get("animation") or "pop")

            segments_json = _json.dumps(segments_list)
            body, ext = captions_artifact_for_clip(
                segments_json,
                clip_start_s=clip_start_s,
                clip_end_s=clip_end_s,
                lead_ms=lead_ms_int,
                position=position_str,
                font_size=font_size_str,
                animation=animation_str,
            )
            if body:
                captions_path = target_path.parent / f".captions.{ext}"
                captions_path.write_text(body, encoding="utf-8")

    fontfile = _find_system_font()
    filter_graph = build_filter_graph(
        spec,
        output_w=output_w,
        output_h=output_h,
        fontfile=fontfile,
        captions_path=captions_path,
    )

    if not filter_graph:
        # Nothing to burn — caller should fall back to source_path.
        # We still return False so the caller can decide what to do.
        if captions_path and captions_path.exists():
            captions_path.unlink(missing_ok=True)
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

    # Clean up the captions file regardless of outcome — it's a
    # per-burn artifact.
    if captions_path and captions_path.exists():
        captions_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        # Surface the LAST ~600 chars of stderr so the operator can
        # see what ffmpeg was unhappy about without drowning in it.
        err = proc.stderr.decode("utf-8", errors="replace")[-600:]
        raise RuntimeError(f"ffmpeg burn failed: {err.strip()}")

    return True
