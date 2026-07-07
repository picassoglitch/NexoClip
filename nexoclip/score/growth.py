"""Pre-publish Growth Score — the killer feature.

Given a clip's signals (motion, face, speaking, reaction confidence,
publishability), its composed caption/hashtags/hook, the target platforms, and
the content tags of what we recently published, the LLM returns a
`GrowthScoreCard`: an overall growth score, a per-platform fit + publish/skip
verdict, optimization recommendations, and the clip's own content tags.

The model is steered by a strict hierarchy (CLAUDE.md hard rule #5 — structured
output only):

    1. Never hurt account health  — respect pacing, avoid repetitive patterns.
    2. Maximize impressions       — only "publish" where the clip fits.
    3. Maximize watch time        — reward strong hooks / retention signals.
    4. Maximize engagement        — reward share/comment-bait captions.

Never raises into the pipeline: on any LLM failure it returns a deterministic
`fallback_card` derived from the publishability score, so publishing keeps
working with the LLM offline.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from nexoclip.errors import BudgetExceeded, LLMError
from nexoclip.llm import GrowthScoreCard, LLMRouter, PlatformGrowthScore
from nexoclip.publish.pacing import canonical_platform, default_rule_for

_log = logging.getLogger("nexoclip.score.growth")

_PURPOSE = "growth_score"

_SYSTEM_PROMPT = """\
You are the publishing brain of a short-form clip growth engine. For ONE clip \
you score how well it will GROW each target platform's account, and decide \
whether it should be published there at all.

Honour this hierarchy, in order:
1. NEVER hurt account health. Penalise clips that look repetitive vs what was \
recently posted, or that don't fit the platform's audience.
2. Maximise impressions: only mark a platform 'publish' when the clip genuinely \
fits it. A 12s funny reaction is ★ on TikTok and near-zero on LinkedIn.
3. Maximise watch time: reward a strong hook and clear payoff.
4. Maximise engagement: reward captions/CTAs that invite real interaction.

Scoring bands (per platform): 90-100 Excellent, 80-89 Very Good, 65-79 Good, \
40-64 Weak, 0-39 Skip. Set verdict='skip' for anything you'd hold back; \
'publish' otherwise. The overall decision: 'publish_all' (strong everywhere), \
'publish_select' (good on some), 'archive' (weak everywhere, save for later), \
'skip' (don't post).

Give 0-5 concrete recommendations (delay, trim_hashtags, shorten_caption, \
new_thumbnail, change_first_frame, rewrite_hook). Always return content_tags: \
3-6 lowercase descriptors of WHAT the clip is (game, map, weapon, creator, bit, \
emotion) — these must be stable and comparable across clips so we can space \
similar content. Expected impressions are rough ranges, not promises."""


@dataclass(frozen=True)
class GrowthInput:
    """Everything the Growth Score model sees about one clip."""

    clip_id: str
    duration_s: float
    caption: str
    platforms: list[str]
    hashtags: list[str] = field(default_factory=list)
    hook: str = ""
    stream_title: str | None = None
    heuristic_reason: str | None = None
    # Signal numbers from ClipBreakdown (all optional; absent = unknown).
    motion_score: float | None = None
    face_presence: float | None = None
    speaking_intensity: float | None = None
    reaction_confidence: float | None = None
    publishability_score: int | None = None
    # Content tags of recently published clips for this tenant — the fatigue
    # context. The model uses these to penalise near-duplicate content.
    recent_content_tags: list[list[str]] = field(default_factory=list)


def _band_label(score: int) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 80:
        return "Very Good"
    if score >= 65:
        return "Good"
    if score >= 40:
        return "Weak"
    return "Skip"


def fallback_card(inp: GrowthInput) -> GrowthScoreCard:
    """Deterministic card when the LLM is unavailable. Uses the publishability
    score as the overall growth score (50 when unknown) and gives every target
    platform that same score with a 'publish' verdict unless it lands in the
    Skip band — so the engine degrades to "post the publishable clips" rather
    than stalling."""
    base = inp.publishability_score if inp.publishability_score is not None else 50
    base = max(0, min(100, int(base)))
    verdict = "publish" if base >= 40 else "skip"
    platforms = [
        PlatformGrowthScore(
            platform=canonical_platform(p),
            score=base,
            verdict=verdict,
            label=_band_label(base),
            reason="Heuristic fallback (Growth Score model unavailable).",
        )
        for p in inp.platforms
    ]
    decision = "publish_select" if base >= 40 else "archive"
    return GrowthScoreCard(
        overall_score=base,
        decision=decision,
        platforms=platforms,
        recommendations=[],
        content_tags=[],
        summary="Fallback score from publishability (no LLM).",
    )


def _build_user_prompt(inp: GrowthInput) -> str:
    targets = [canonical_platform(p) for p in inp.platforms]
    rules = {
        p: {
            "max_per_day": default_rule_for(p).max_per_day,
            "caption_style": default_rule_for(p).caption_style,
            "hashtag_max": default_rule_for(p).hashtag_max,
        }
        for p in targets
    }
    payload = {
        "clip": {
            "duration_s": round(inp.duration_s, 1),
            "caption": inp.caption,
            "hashtags": inp.hashtags,
            "hook": inp.hook,
            "stream_title": inp.stream_title,
            "detected_reason": inp.heuristic_reason,
        },
        "signals": {
            "motion_score": inp.motion_score,
            "face_presence": inp.face_presence,
            "speaking_intensity_wps": inp.speaking_intensity,
            "reaction_confidence": inp.reaction_confidence,
            "publishability_score": inp.publishability_score,
        },
        "target_platforms": targets,
        "platform_rulebook": rules,
        "recently_published_content_tags": inp.recent_content_tags,
    }
    return (
        "Score this clip for growth. Return a verdict for EACH target platform.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


async def compute_growth_score(
    inp: GrowthInput,
    *,
    tenant_id: str,
    router: LLMRouter,
) -> GrowthScoreCard:
    """Compute the Growth Score card for one clip. Never raises — returns a
    deterministic `fallback_card` on any LLM error or budget cutoff."""
    if not inp.platforms:
        return fallback_card(inp)
    try:
        card = await router.complete(
            tenant_id=tenant_id,
            purpose=_PURPOSE,
            system=_SYSTEM_PROMPT,
            user=_build_user_prompt(inp),
            schema=GrowthScoreCard,
        )
    except BudgetExceeded as e:
        _log.warning("growth_score.skipped_budget clip=%s reason=%s", inp.clip_id, e)
        return fallback_card(inp)
    except LLMError as e:
        _log.warning("growth_score.skipped_llm_error clip=%s reason=%s", inp.clip_id, e)
        return fallback_card(inp)

    # Normalise platform ids to canonical keys so downstream allocation matches
    # the connected-account ids without alias drift.
    card.platforms = [
        p.model_copy(update={"platform": canonical_platform(p.platform)})
        for p in card.platforms
    ]
    return card
