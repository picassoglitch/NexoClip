"""Confidence breakdown for a single clip — the "why this clip?" panel.

Joins three already-persisted tables for the bound tenant:
  * `visual_signals` -> motion_score (avg motion_energy across the clip's
    seconds) + face_presence (frac of seconds with a detected face).
  * `transcripts.segments_json` -> speaking_intensity (words per second
    during the clip's window).
  * `candidates` -> reaction_confidence (rescore_score) +
    rescore_delta (rescore_score - heuristic_score).

This is the observability hook your scope-adjusted PHASE_2 mandated:
without it, "why did the system pick THIS clip?" stays a guessing game.
The dashboard renders the dataclass; tests pin the math.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from nexoclip.db import (
    CandidatesRepo,
    ClipsRepo,
    Database,
    TranscriptsRepo,
    VisualSignalsRepo,
)
from nexoclip.errors import ClipError
from nexoclip.tenancy import current_tenant_id


@dataclass(frozen=True)
class ClipBreakdown:
    """All the "why" numbers the dashboard's panel renders."""

    clip_id: str
    duration_s: float
    motion_score: float | None  # avg motion_energy across the clip seconds
    face_presence: float | None  # fraction of seconds with a face detected
    speaking_intensity: float | None  # words per second over the clip window
    reaction_confidence: float | None  # candidates.rescore_score
    rescore_delta: float | None  # rescore_score - heuristic_score
    rescore_reason: str | None
    heuristic_reason: str  # candidates.reason (voice / chat / audio / visual)
    heuristic_score: float


async def clip_breakdown(db: Database, clip_id: str) -> ClipBreakdown:
    """Compute the breakdown panel data for `clip_id` under the bound tenant.

    Raises `ClipError` if the clip doesn't exist (or belongs to another
    tenant - tenancy enforcement happens at the repo layer).
    """
    tenant_id = current_tenant_id()  # noqa: F841 - asserted via repo calls

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise ClipError(f"clip not found: {clip_id}")

    # ---- visual signals: per-second rows in [floor(start_s), ceil(end_s)) ----
    visuals = await VisualSignalsRepo(db).list_for_stream(clip.stream_id)
    in_window = [
        v
        for v in visuals
        if clip.start_s <= float(v["ts_offset_s"]) < clip.end_s
    ]
    motion_score: float | None
    face_presence: float | None
    if in_window:
        motion_values = [
            float(v["motion_energy"])
            for v in in_window
            if v["motion_energy"] is not None
        ]
        motion_score = (
            sum(motion_values) / len(motion_values) if motion_values else None
        )
        face_count = sum(
            1 for v in in_window if v["face_emotion"] is not None
        )
        face_presence = face_count / len(in_window)
    else:
        motion_score = None
        face_presence = None

    # ---- transcript: count words whose ts falls in the clip window ----
    transcript = await TranscriptsRepo(db).get(clip.stream_id)
    speaking_intensity: float | None = None
    if transcript is not None:
        try:
            segments = json.loads(transcript.segments_json or "[]")
        except json.JSONDecodeError:
            segments = []
        word_count = 0
        for seg in segments:
            for word in seg.get("words") or []:
                ts = word.get("ts")
                if isinstance(ts, int | float) and clip.start_s <= ts < clip.end_s:
                    word_count += 1
        if clip.duration_s > 0:
            speaking_intensity = word_count / clip.duration_s

    # ---- candidate: rescore + heuristic_reason from the source candidate ----
    reaction_confidence: float | None = None
    rescore_delta: float | None = None
    rescore_reason: str | None = None
    heuristic_reason = "unknown"
    heuristic_score = 0.0
    if clip.candidate_id is not None:
        # `list_for_stream` filters by tenant + stream; we then pick the row
        # we want by id.
        candidates = await CandidatesRepo(db).list_for_stream(clip.stream_id)
        match = next((c for c in candidates if c.id == clip.candidate_id), None)
        if match is not None:
            heuristic_reason = match.reason
            heuristic_score = float(match.score)
            if match.rescore_score is not None:
                reaction_confidence = float(match.rescore_score)
                rescore_delta = float(match.rescore_score) - float(match.score)
            rescore_reason = match.rescore_reason

    return ClipBreakdown(
        clip_id=clip_id,
        duration_s=clip.duration_s,
        motion_score=motion_score,
        face_presence=face_presence,
        speaking_intensity=speaking_intensity,
        reaction_confidence=reaction_confidence,
        rescore_delta=rescore_delta,
        rescore_reason=rescore_reason,
        heuristic_reason=heuristic_reason,
        heuristic_score=heuristic_score,
    )
