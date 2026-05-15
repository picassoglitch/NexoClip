"""Per-clip intelligence timeline — slice F.7-C.

Aggregates the multimodal signals NexoClip already collects into
labeled markers the editor renders under the waveform:

    00:04  Audio peak
    00:11  Scene cut
    00:17  Reaction (laughter spike)
    00:22  Chat heat spike
    00:28  Face-emotion change

Pipeline (read-only across existing tables):
    transcripts.segments_json   → laughter / words-per-sec spikes
    visual_signals (per-second) → motion_energy peaks, scene_cuts,
                                   face_emotion changes
    chat_replay (per-stream)    → message-density per-second buckets

Each signal is detected on a "rolling baseline + threshold" rule —
e.g. an audio peak is a 1-second bucket whose word-density is
>= 1.8x the clip's median wps. Tunables live as module constants
so we can iterate the heuristics without touching callers.

The output is a flat list of markers, each tagged with:
    - kind  (audio_peak | scene_cut | reaction | chat_heat | face_emotion)
    - ts    (seconds from clip start, NOT from stream start)
    - score (0-1 — how strong the signal is)
    - label (short human-readable string for the marker tooltip)

Markers are sorted by ts. Empty list when no signals fire — the UI
falls back to the waveform-only view in that case.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from typing import Literal, NamedTuple

from nexoclip.db import (
    ClipsRepo,
    Database,
    TranscriptsRepo,
    VisualSignalsRepo,
)
from nexoclip.ingest import load_chat_replay

# ---- tunables -------------------------------------------------

# Words like "haha", "lmao", "lol" inside a 1-second window count as a
# laughter signal. Cheap enough that the proxy is fine in practice.
_LAUGHTER_TOKENS: frozenset[str] = frozenset(
    {
        "haha", "hahaha", "jaja", "jajaja", "jajajaja",
        "lol", "lmao", "lmaoo", "rofl", "kekw",
    }
)

# Audio-peak threshold — a 1-second bucket whose words-per-sec is
# at least this multiple of the clip's median fires a marker.
_AUDIO_PEAK_THRESHOLD_X_MEDIAN = 1.8

# Motion-peak threshold — the 90th-percentile motion bucket fires.
_MOTION_PEAK_PERCENTILE = 90

# Chat-heat threshold — a 1-second bucket whose message count is
# ≥ this multiple of the clip's median fires.
_CHAT_HEAT_THRESHOLD_X_MEDIAN = 2.0

# Cap on emitted markers per kind — keeps a wild detector from
# burying the timeline under hundreds of dots.
_MAX_MARKERS_PER_KIND = 6


MarkerKind = Literal[
    "audio_peak",
    "scene_cut",
    "reaction",
    "chat_heat",
    "face_emotion",
]


class Marker(NamedTuple):
    """One signal marker on the clip's intelligence timeline."""

    kind: MarkerKind
    ts: float  # seconds from clip start
    score: float  # 0-1
    label: str  # short human label for the tooltip


# ---- public entry --------------------------------------------


async def compute_intelligence(
    db: Database, *, clip_id: str
) -> list[Marker]:
    """Aggregate per-clip markers from every signal source available.

    Read-only — no caching, the data is cheap to pull from the
    already-indexed tables. Re-running is idempotent.

    Returns `[]` when the clip doesn't exist or every signal source
    is empty (no transcript, no visual_signals, no chat replay).
    """
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        return []

    markers: list[Marker] = []

    # ---- audio peaks + reaction (laughter) — both from the transcript.
    transcript = await TranscriptsRepo(db).get(clip.stream_id)
    if transcript is not None:
        try:
            segments = json.loads(transcript.segments_json)
        except json.JSONDecodeError:
            segments = []
        if isinstance(segments, list):
            audio = _audio_peaks_from_segments(
                segments, clip_start=clip.start_s, clip_end=clip.end_s
            )
            laughter = _reactions_from_segments(
                segments, clip_start=clip.start_s, clip_end=clip.end_s
            )
            markers.extend(audio)
            markers.extend(laughter)

    # ---- scene cuts + motion peaks + face-emotion changes — visual_signals.
    vs_rows = await VisualSignalsRepo(db).list_for_stream(clip.stream_id)
    if vs_rows:
        markers.extend(
            _scene_cuts_and_motion_peaks(
                vs_rows, clip_start=clip.start_s, clip_end=clip.end_s
            )
        )
        markers.extend(
            _face_emotion_changes(
                vs_rows, clip_start=clip.start_s, clip_end=clip.end_s
            )
        )

    # ---- chat-heat spikes — from the on-disk chat replay file.
    try:
        from pathlib import Path

        from nexoclip.settings import get_settings

        output_dir = get_settings().default_output_dir
        stream_dir = Path(output_dir) / clip.stream_id
        replay = load_chat_replay(
            stream_dir, stream_id=clip.stream_id, tenant_id=clip.tenant_id
        )
        if replay is not None and replay.messages:
            markers.extend(
                _chat_heat_spikes(
                    replay.messages,
                    clip_start=clip.start_s,
                    clip_end=clip.end_s,
                )
            )
    except Exception:
        pass

    markers.sort(key=lambda m: m.ts)
    return markers


# ---- transcript-derived signals ----------------------------------


def _audio_peaks_from_segments(
    segments: list[object], *, clip_start: float, clip_end: float
) -> list[Marker]:
    """Bucket transcript words into 1-sec windows; emit peaks."""
    bucket_words: dict[int, int] = {}
    for s in segments:
        if not isinstance(s, dict):
            continue
        s_start = float(s.get("start") or 0)
        s_end = float(s.get("end") or s_start)
        if s_end < clip_start or s_start > clip_end:
            continue
        text = (s.get("text") or "").strip()
        if not text:
            continue
        n_words = len(text.split())
        # Walk 1-second buckets touched by this segment.
        first_b = max(0, int(s_start - clip_start))
        last_b = max(first_b, int(min(clip_end, s_end) - clip_start))
        per_bucket = max(1, n_words // max(1, last_b - first_b + 1))
        for b in range(first_b, last_b + 1):
            bucket_words[b] = bucket_words.get(b, 0) + per_bucket

    if not bucket_words:
        return []
    counts = sorted(bucket_words.values())
    median = counts[len(counts) // 2] or 1
    threshold = max(2, median * _AUDIO_PEAK_THRESHOLD_X_MEDIAN)
    peaks = sorted(
        (
            (b, n)
            for b, n in bucket_words.items()
            if n >= threshold
        ),
        key=lambda x: -x[1],
    )[:_MAX_MARKERS_PER_KIND]
    out: list[Marker] = []
    for ts, count in peaks:
        # score normalized by max possible words/sec (~6)
        score = min(1.0, count / 6.0)
        out.append(
            Marker(
                kind="audio_peak",
                ts=float(ts),
                score=score,
                label=f"Audio peak · {count} words",
            )
        )
    return out


def _reactions_from_segments(
    segments: list[object], *, clip_start: float, clip_end: float
) -> list[Marker]:
    """Laughter heuristic — segments containing any of the laughter
    tokens are tagged as reactions. The score is the fraction of
    laughter tokens to total tokens in that segment."""
    out: list[Marker] = []
    for s in segments:
        if not isinstance(s, dict):
            continue
        s_start = float(s.get("start") or 0)
        if s_start < clip_start or s_start > clip_end:
            continue
        text = (s.get("text") or "").strip().lower()
        if not text:
            continue
        tokens = text.split()
        n_laughs = sum(1 for t in tokens if t.strip(".,!?¡¿") in _LAUGHTER_TOKENS)
        if n_laughs == 0:
            continue
        score = min(1.0, n_laughs / max(1, len(tokens)))
        out.append(
            Marker(
                kind="reaction",
                ts=s_start - clip_start,
                score=score,
                label=f"Reaction · {n_laughs} laugh token{'s' if n_laughs != 1 else ''}",
            )
        )
    return out[:_MAX_MARKERS_PER_KIND]


# ---- visual_signals-derived signals -------------------------------


def _scene_cuts_and_motion_peaks(
    vs_rows: Iterable[dict[str, object]],
    *,
    clip_start: float,
    clip_end: float,
) -> list[Marker]:
    """Scene cuts get one marker each (binary); motion peaks fire
    when the bucket's motion_energy is in the top 10% of the clip."""
    in_window = [
        r
        for r in vs_rows
        if isinstance(r, dict)
        and clip_start <= _as_float(r.get("ts_offset_s")) <= clip_end
    ]
    if not in_window:
        return []

    out: list[Marker] = []

    # Scene cuts.
    cuts = [r for r in in_window if r.get("scene_cut")]
    for r in cuts[:_MAX_MARKERS_PER_KIND]:
        ts_rel = float(r["ts_offset_s"]) - clip_start  # type: ignore[arg-type]
        out.append(
            Marker(
                kind="scene_cut",
                ts=ts_rel,
                score=1.0,
                label="Scene cut",
            )
        )

    # Motion peaks — top _MOTION_PEAK_PERCENTILE-th percentile.
    motions = [
        (float(r["ts_offset_s"]) - clip_start, float(r["motion_energy"]))  # type: ignore[arg-type]
        for r in in_window
        if r.get("motion_energy") is not None
    ]
    if motions:
        sorted_motions = sorted(motions, key=lambda x: x[1])
        p_idx = int(len(sorted_motions) * _MOTION_PEAK_PERCENTILE / 100)
        threshold = sorted_motions[min(p_idx, len(sorted_motions) - 1)][1]
        peaks = sorted(
            ((ts, m) for ts, m in motions if m >= threshold and m > 0),
            key=lambda x: -x[1],
        )[:_MAX_MARKERS_PER_KIND]
        max_m = max((m for _, m in motions), default=1.0) or 1.0
        for ts, m in peaks:
            out.append(
                Marker(
                    kind="audio_peak",  # treat motion peak as audio_peak in UI
                    ts=ts,
                    score=min(1.0, m / max_m),
                    label=f"Motion peak · {m:.2f}",
                )
            )

    return out


def _face_emotion_changes(
    vs_rows: Iterable[dict[str, object]],
    *,
    clip_start: float,
    clip_end: float,
) -> list[Marker]:
    """Emit a marker each time the detected face_emotion CHANGES
    inside the clip window. Same-emotion repeats are folded."""
    in_window = sorted(
        (r for r in vs_rows if isinstance(r, dict)),
        key=lambda r: _as_float(r.get("ts_offset_s")),
    )
    out: list[Marker] = []
    last: str | None = None
    for r in in_window:
        ts = _as_float(r.get("ts_offset_s"))
        if ts < clip_start or ts > clip_end:
            continue
        emo = r.get("face_emotion")
        if not isinstance(emo, str):
            continue
        if emo and emo != last:
            out.append(
                Marker(
                    kind="face_emotion",
                    ts=ts - clip_start,
                    score=0.6,
                    label=f"Face emotion → {emo}",
                )
            )
            last = emo
            if len(out) >= _MAX_MARKERS_PER_KIND:
                break
    return out


# ---- chat-replay derived signals -----------------------------------


def _chat_heat_spikes(
    messages: Iterable[object], *, clip_start: float, clip_end: float
) -> list[Marker]:
    """Bucket chat messages into 1-sec windows in the clip's range,
    emit markers where the bucket count is >= N times the median."""
    bucket_counts: dict[int, int] = {}
    for m in messages:
        ts = float(getattr(m, "ts", 0))
        if ts < clip_start or ts > clip_end:
            continue
        bucket = int(ts - clip_start)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    if not bucket_counts:
        return []
    counts = sorted(bucket_counts.values())
    median = counts[len(counts) // 2] or 1
    threshold = max(3, median * _CHAT_HEAT_THRESHOLD_X_MEDIAN)
    peaks = sorted(
        (
            (b, n)
            for b, n in bucket_counts.items()
            if n >= threshold
        ),
        key=lambda x: -x[1],
    )[:_MAX_MARKERS_PER_KIND]
    out: list[Marker] = []
    max_count = max(counts)
    for ts, count in peaks:
        out.append(
            Marker(
                kind="chat_heat",
                ts=float(ts),
                score=min(1.0, count / max(1, max_count)),
                label=f"Chat spike · {count} messages",
            )
        )
    return out


def _is_finite(x: float) -> bool:
    """Cheap sanity guard against NaN/inf making it into JSON."""
    return isinstance(x, int | float) and math.isfinite(x)


def _as_float(value: object) -> float:
    """Best-effort coerce to float; 0.0 on anything weird.

    Used by the row-readers above to satisfy mypy --strict when
    walking heterogeneous `dict[str, object]` rows out of the DB.
    """
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
