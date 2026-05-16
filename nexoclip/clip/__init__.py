"""Clip cutting + 9:16 reformat module."""

from .breakdown import ClipBreakdown, clip_breakdown
from .intelligence import Marker, MarkerKind, compute_intelligence
from .models import Clip, ClipManifest
from .overlay_burn import (
    build_ass,
    build_filter_graph,
    build_srt,
    burn_overlays,
    captions_artifact_for_clip,
)
from .scoring import AIScoreCard, compute_ai_scores
from .service import cut_clips, cut_window, load_clips
from .waveform import compute_waveform, load_or_compute as load_or_compute_waveform
from .word_captions import (
    CaptionLine,
    CaptionWord,
    Emphasis,
    captions_for_clip,
    chunk_words_to_lines,
    classify_emphasis,
    lines_to_json,
)

__all__ = [
    "AIScoreCard",
    "CaptionLine",
    "CaptionWord",
    "Clip",
    "ClipBreakdown",
    "ClipManifest",
    "Emphasis",
    "Marker",
    "MarkerKind",
    "build_ass",
    "build_filter_graph",
    "build_srt",
    "burn_overlays",
    "captions_artifact_for_clip",
    "captions_for_clip",
    "chunk_words_to_lines",
    "classify_emphasis",
    "clip_breakdown",
    "compute_ai_scores",
    "compute_intelligence",
    "compute_waveform",
    "cut_clips",
    "cut_window",
    "lines_to_json",
    "load_clips",
    "load_or_compute_waveform",
]
