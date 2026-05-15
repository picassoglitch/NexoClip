"""AI-scoring helper — composite scores derived from `ClipBreakdown`.

Surfaces a small set of human-readable scores for the clip-editor's
"AI insights" strip. The breakdown already aggregates every per-clip
signal we have (heuristic score, motion, face presence, speaking
intensity, vision-LLM rescore); this module turns those raw numbers
into the four creator-facing scores the UI shows:

  * Viral score (0-100)        — composite "is this clip cookable?"
  * Hook strength              — HIGH / MEDIUM / DEVELOPING
  * Caption readability        — GOOD / OK / CHECK
  * Dead-air risk              — LOW / MED / HIGH

These are HEURISTICS, not magic. Their job is to make the operator's
"should I publish this?" call faster — they should never replace
manual review entirely. Each score is also annotated with a one-line
"why" the UI can show on hover so the operator understands what
pushed it up or down.

The scoring formulas are deliberately simple + tunable. When
`reaction_confidence` (vision-LLM rescore) is available the viral
score weights it heavily — the rescore is the strongest single
signal we have. When it's not present we fall back to the heuristic
score plus face-presence boost.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

from .breakdown import ClipBreakdown

HookStrength = Literal["HIGH", "MEDIUM", "DEVELOPING"]
ReadabilityLabel = Literal["GOOD", "OK", "CHECK"]
DeadAirLabel = Literal["LOW", "MED", "HIGH"]


class AIScoreCard(NamedTuple):
    """One-shot bundle of all four scores + their explanation strings."""

    viral_score: int  # 0-100
    viral_why: str

    hook_strength: HookStrength
    hook_why: str

    caption_readability: ReadabilityLabel
    readability_why: str

    dead_air_risk: DeadAirLabel
    dead_air_why: str


# ---- viral score ----


def _viral_score(b: ClipBreakdown) -> tuple[int, str]:
    """Composite 0-100 score weighted toward the strongest signal we have.

    Priority order:
      1. vision-LLM rescore (`reaction_confidence`) — the most accurate
         single signal when present
      2. heuristic_score — what the trigger scanner produced
      3. face_presence boost — up to +10 when a face is in frame for
         most of the clip (talking-head clips publish better)
    """
    if b.reaction_confidence is not None:
        # Rescore is the truth when present. Heuristic gets half-weight.
        base = b.reaction_confidence * 70 + b.heuristic_score * 30
        why = f"vision-LLM rescore = {b.reaction_confidence:.2f}, heuristic = {b.heuristic_score:.2f}"
    else:
        base = b.heuristic_score * 90  # leave 10pt headroom for the face boost
        why = f"heuristic ({b.heuristic_reason}) = {b.heuristic_score:.2f}; not yet vision-rescored"

    if b.face_presence is not None and b.face_presence > 0.4:
        boost = min(10, int((b.face_presence - 0.4) * 20))
        base += boost
        why += f" · face-presence boost +{boost}"

    return int(max(0, min(100, round(base)))), why


# ---- hook strength ----


def _hook_strength(viral: int) -> tuple[HookStrength, str]:
    """Banded label for the viral score — operator-facing label that
    reads as "should I write a strong hook?" rather than as a raw number."""
    if viral >= 75:
        return "HIGH", f"viral {viral} ≥ 75 — push this clip first"
    if viral >= 55:
        return "MEDIUM", f"viral {viral} in 55-74 — solid mid-tier candidate"
    return "DEVELOPING", f"viral {viral} < 55 — needs a strong title to lift it"


# ---- caption readability ----


def _caption_readability(b: ClipBreakdown) -> tuple[ReadabilityLabel, str]:
    """Words-per-second sweet spot for burned-in captions.

    Industry-standard subtitle pacing maxes out around 2.5 wps before
    viewers stop reading. Below 0.5 wps the captions feel sparse and
    the operator should consider trimming dead air or shortening the clip.
    """
    wps = b.speaking_intensity
    if wps is None:
        return "CHECK", "no transcript yet — can't measure caption pacing"
    if 0.8 <= wps <= 2.2:
        return "GOOD", f"{wps:.1f} words/sec — comfortable read pace"
    if 0.5 <= wps < 0.8 or 2.2 < wps <= 3.0:
        return "OK", f"{wps:.1f} words/sec — viewers may struggle"
    return "CHECK", f"{wps:.1f} words/sec — outside the readable band"


# ---- dead-air risk ----


def _dead_air_risk(b: ClipBreakdown) -> tuple[DeadAirLabel, str]:
    """Inverse of speaking_intensity — short-form viewers bounce on quiet."""
    wps = b.speaking_intensity
    if wps is None:
        return "MED", "no transcript — can't measure dead-air"
    if wps >= 0.8:
        return "LOW", f"{wps:.1f} words/sec — clip is talking through"
    if wps >= 0.4:
        return "MED", f"{wps:.1f} words/sec — some quiet stretches"
    return "HIGH", f"{wps:.1f} words/sec — long quiet stretches; trim or skip"


# ---- public entry ----


def compute_ai_scores(b: ClipBreakdown) -> AIScoreCard:
    """Compute all four AI scores from a clip's breakdown."""
    viral, viral_why = _viral_score(b)
    hook, hook_why = _hook_strength(viral)
    readability, readability_why = _caption_readability(b)
    dead_air, dead_air_why = _dead_air_risk(b)
    return AIScoreCard(
        viral_score=viral,
        viral_why=viral_why,
        hook_strength=hook,
        hook_why=hook_why,
        caption_readability=readability,
        readability_why=readability_why,
        dead_air_risk=dead_air,
        dead_air_why=dead_air_why,
    )
