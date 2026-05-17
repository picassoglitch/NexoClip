"""Variant card metadata — slice I.4.

The base `Variant` is intentionally minimal (id, language, caption,
title_card_text, hashtags). The clip editor's card layout needs more:
predicted-fit platforms, tone label, retention-confidence estimate,
risk flag. Computing those at generation time would require touching
the LLM generator schema — bigger surface, longer feedback loop.
Instead we DERIVE them from the existing fields:

  - language → which Spanish / English filter the variant belongs to
  - caption length + hashtag count → predicted platform fit
  - caption keywords + emojis → tone classification
  - caption shape (hook word count, presence of "!", "?", caps) →
    retention confidence
  - generic vs distinctive → risk label

This is heuristic on purpose — the goal is good-enough labels so the
operator can scan + pick fast, not a perfect LLM-graded score. A real
retention-prediction model lands in a future slice (G.5+).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from nexoclip.llm import Variant

Tone = Literal["aggressive", "funny", "clean", "motivational", "controversial", "neutral"]
RiskLabel = Literal["safe", "spicy", "generic"]
PlatformFit = Literal["tiktok", "reels", "shorts", "kick"]

# Slice K.2 — emotional variant labels. Operators don't think in
# "aggressive vs funny vs clean" — those are editor terms. Map each
# detected tone to an outcome the creator emotionally understands.
EmotionalLabel = Literal[
    "high_viral",       # 🔥 High Viral Potential
    "authority",        # 🧠 Authority Builder
    "meme",             # 😂 Meme Format
    "conversion",       # 💰 Conversion Style
    "safe_brand",       # 🎯 Safe Brand Format
]


@dataclass(frozen=True)
class EnrichedVariant:
    """One variant with all the card metadata the editor wants."""

    variant: Variant
    tone: Tone
    fit: list[PlatformFit]
    retention_confidence: int  # 0-100
    risk: RiskLabel
    risk_reason: str
    # Slice K.2 — emotional outcome label (replaces tone in the UI).
    emotional_label: EmotionalLabel
    # Filter facets the card-grid filter tabs read.
    facets: list[str]


# ---- keyword corpora -----------------------------------------

# These are deliberately small + obvious. The goal is "this caption
# leans funny" not "perfect comedy detection". Easy to extend.
_FUNNY_TOKENS = frozenset({
    "lol", "lmao", "lmaoo", "jaja", "jajaja", "jajajaja",
    "muerto", "muertísimo", "muerta", "muerte",
    "rip", "kekw", "haha", "hahaha",
    "broma", "chiste", "comedia", "wtf",
})
_FUNNY_EMOJI = frozenset({"😭", "💀", "🤣", "😂", "🤡", "🫡"})

_AGGRESSIVE_TOKENS = frozenset({
    "destruyó", "destruye", "humillación", "humilló", "rekt",
    "expuesto", "destroza", "destrozó", "violación",
    "destroyed", "rekt", "exposed", "obliterated", "wrecked",
})
_AGGRESSIVE_EMOJI = frozenset({"🔥", "💥", "🚨", "🩸", "⚔️"})

_MOTIVATIONAL_TOKENS = frozenset({
    "logra", "logró", "consigue", "consiguió", "supera", "superó",
    "achievement", "made it", "earned", "respect", "legendary",
    "leyenda", "héroe", "campeón",
})
_MOTIVATIONAL_EMOJI = frozenset({"🙏", "💪", "🏆", "🎯", "✨"})

_CONTROVERSIAL_TOKENS = frozenset({
    "polémica", "controversial", "drama", "shock", "exposed",
    "verdad", "secreto", "revelación", "leak", "filtrado",
    "scandal", "escándalo",
})
_CONTROVERSIAL_EMOJI = frozenset({"👀", "🚨", "🔥", "⚡"})

# Generic-caption indicators — phrases LLMs over-emit when they have
# nothing specific to say. A caption matching multiple gets `risk=generic`.
_GENERIC_PATTERNS = (
    re.compile(r"check (this|it) out", re.IGNORECASE),
    re.compile(r"this is (insane|crazy|wild|amazing)", re.IGNORECASE),
    re.compile(r"watch (this|until the end)", re.IGNORECASE),
    re.compile(r"wait (for|until) it", re.IGNORECASE),
    re.compile(r"you won.?t believe", re.IGNORECASE),
    re.compile(r"^(epic|amazing|wild|insane) (moment|clip|reaction)", re.IGNORECASE),
)

# Spicy: heavy profanity / inflammatory tokens — content the operator
# may want to think twice about before shipping. NOT a moral judgement,
# just a flag.
_SPICY_TOKENS = frozenset({
    "fuck", "shit", "wtf", "damn", "asshole", "fuckin",
    "mierda", "puta", "joder", "cojones",
})

# Slice K.2 — conversion-CTA detection. Variants that ask viewers to
# DO something (visit / follow / link in bio / swipe up / etc) get
# the 💰 Conversion Style emotional label.
_CTA_PATTERNS = (
    re.compile(r"link\s+in\s+bio", re.IGNORECASE),
    re.compile(r"swipe\s+up", re.IGNORECASE),
    re.compile(r"\bvisit\b", re.IGNORECASE),
    re.compile(r"\bfollow\b", re.IGNORECASE),
    re.compile(r"\bjoin\s+(my|the|us)", re.IGNORECASE),
    re.compile(r"\bsubscribe\b", re.IGNORECASE),
    re.compile(r"\bcheck\s+out\b", re.IGNORECASE),
    re.compile(r"sign\s+up", re.IGNORECASE),
    re.compile(r"comprar|compra|comprá", re.IGNORECASE),  # ES: buy
    re.compile(r"síguenos|síguete|sígueme", re.IGNORECASE),
)


# Emotional-label display metadata. Order matters: the renderer
# picks the FIRST matching label from a small ranking heuristic.
EMOTIONAL_LABELS: dict[EmotionalLabel, tuple[str, str]] = {
    "high_viral":  ("🔥", "High Viral Potential"),
    "authority":   ("🧠", "Authority Builder"),
    "meme":        ("😂", "Meme Format"),
    "conversion":  ("💰", "Conversion Style"),
    "safe_brand":  ("🎯", "Safe Brand Format"),
}


def _emotional_label(text: str, tone: Tone) -> EmotionalLabel:
    """Map (caption text, detected tone) → outcome-first label the
    creator can decide from at a glance.

    Priority:
      1. CTA keywords win — conversion is the most actionable label
      2. Aggressive / controversial → high_viral (high-energy, polarizing)
      3. Motivational → authority
      4. Funny → meme
      5. Clean / neutral → safe_brand
    """
    if any(p.search(text) for p in _CTA_PATTERNS):
        return "conversion"
    if tone in ("aggressive", "controversial"):
        return "high_viral"
    if tone == "motivational":
        return "authority"
    if tone == "funny":
        return "meme"
    return "safe_brand"


# ---- enrichment ----------------------------------------------


def enrich_variant(variant: Variant) -> EnrichedVariant:
    """Compute card metadata for one variant."""
    text = (variant.caption or "").strip()
    title = (variant.title_card_text or "").strip()
    blob = f"{text}\n{title}".lower()
    char_count = len(text)
    word_count = len(text.split())
    hashtag_count = len(variant.hashtags or [])
    emojis = _extract_emojis(text + title)
    exclaims = text.count("!")
    questions = text.count("?")
    caps_words = sum(1 for w in text.split() if w.isupper() and len(w) >= 3)

    tone = _classify_tone(blob, emojis)
    fit = _predict_fit(char_count, hashtag_count, variant.language)
    retention = _retention_confidence(
        word_count=word_count,
        char_count=char_count,
        emojis_count=len(emojis),
        exclaims=exclaims,
        questions=questions,
        caps_words=caps_words,
        has_title=bool(title),
        tone=tone,
    )
    risk, risk_reason = _classify_risk(text, blob, tone)

    emotional_label = _emotional_label(text, tone)

    facets: list[str] = []
    facets.append(variant.language.lower())  # 'es' / 'en' / 'pt' / …
    facets.append(tone)
    facets.append(emotional_label)  # Slice K.2 — facet for filter chips
    if risk == "safe":
        facets.append("safe")
    if risk == "spicy":
        facets.append("spicy")
    if risk == "generic":
        facets.append("generic")

    return EnrichedVariant(
        variant=variant,
        tone=tone,
        fit=fit,
        retention_confidence=retention,
        risk=risk,
        risk_reason=risk_reason,
        emotional_label=emotional_label,
        facets=facets,
    )


def enrich_variants(variants: list[Variant]) -> list[EnrichedVariant]:
    return [enrich_variant(v) for v in variants]


# ---- internals ------------------------------------------------


def _extract_emojis(text: str) -> list[str]:
    """Pull emoji characters out of a string. Crude but adequate —
    we just want presence + count for the tone classifier."""
    return [c for c in text if _is_emoji_char(c)]


def _is_emoji_char(c: str) -> bool:
    """Cheap emoji-ish detector — anything outside the BMP plus a few
    common emoji-range plane points. Good enough for tone scoring."""
    if not c:
        return False
    cp = ord(c)
    return (
        cp > 0xFFFF
        or 0x2600 <= cp <= 0x27BF       # misc symbols + dingbats
        or 0x1F300 <= cp <= 0x1FAFF     # main emoji blocks
    )


def _classify_tone(blob: str, emojis: list[str]) -> Tone:
    """Score tone by keyword + emoji presence. Highest-scoring tone wins."""
    emoji_set = set(emojis)
    scores: dict[Tone, int] = {
        "funny": 0, "aggressive": 0, "motivational": 0,
        "controversial": 0, "clean": 0, "neutral": 0,
    }
    for token in _FUNNY_TOKENS:
        if token in blob:
            scores["funny"] += 2
    for e in emoji_set & _FUNNY_EMOJI:
        scores["funny"] += 1
        _ = e  # explicit ack so the loop's bookkeeping is obvious
    for token in _AGGRESSIVE_TOKENS:
        if token in blob:
            scores["aggressive"] += 2
    for e in emoji_set & _AGGRESSIVE_EMOJI:
        scores["aggressive"] += 1
        _ = e
    for token in _MOTIVATIONAL_TOKENS:
        if token in blob:
            scores["motivational"] += 2
    for e in emoji_set & _MOTIVATIONAL_EMOJI:
        scores["motivational"] += 1
        _ = e
    for token in _CONTROVERSIAL_TOKENS:
        if token in blob:
            scores["controversial"] += 2
    for e in emoji_set & _CONTROVERSIAL_EMOJI:
        scores["controversial"] += 1
        _ = e

    # "clean" is the default when nothing else fires AND no emojis are
    # present — short, neutral captions read clean.
    if not emojis and max(scores.values()) == 0:
        return "clean"

    best = max(scores.items(), key=lambda kv: kv[1])
    if best[1] == 0:
        return "neutral"
    return best[0]


def _predict_fit(char_count: int, hashtag_count: int, language: str) -> list[PlatformFit]:
    """Score which platforms a variant fits. Looseness on purpose:
    most clips fit multiple — the card just highlights the best 1-2."""
    fits: list[PlatformFit] = []
    # TikTok / Reels / Shorts all eat 100-150 char captions well.
    if char_count <= 200:
        fits.append("tiktok")
        fits.append("reels")
        fits.append("shorts")
    # Kick clips have no caption limit but their audience is gaming-
    # heavy — long captions there feel less out of place.
    fits.append("kick")
    # Long captions (>250 chars) downrank TikTok + Shorts.
    if char_count > 250:
        fits = [f for f in fits if f not in ("tiktok", "shorts")]
    return fits


def _retention_confidence(
    *,
    word_count: int,
    char_count: int,
    emojis_count: int,
    exclaims: int,
    questions: int,
    caps_words: int,
    has_title: bool,
    tone: Tone,
) -> int:
    """0-100 estimate of how likely this variant keeps viewers past 3s.
    Heuristic — rewards strong hooks, punishes generic / too-long text."""
    score = 30  # baseline
    # Hook length sweet spot: 4-12 words.
    if 4 <= word_count <= 12:
        score += 18
    elif word_count >= 13:
        score += 6
    # Char length sanity.
    if char_count < 200:
        score += 8
    # Mild emoji bump (over-emojified gets capped).
    score += min(8, emojis_count * 2)
    # Punctuation / emphasis.
    if exclaims:
        score += min(8, exclaims * 4)
    if questions:
        score += 4   # open-loop hooks
    if caps_words:
        score += min(8, caps_words * 3)
    if has_title:
        score += 5
    # Strong-tone variants tend to outperform "clean".
    if tone in ("funny", "controversial", "aggressive"):
        score += 8
    elif tone == "clean":
        score -= 4
    return max(0, min(100, score))


def _classify_risk(text: str, blob: str, tone: Tone) -> tuple[RiskLabel, str]:
    """Flag spicy (profanity / inflammatory) and generic (LLM filler) variants."""
    generic_hits = sum(1 for p in _GENERIC_PATTERNS if p.search(text))
    if generic_hits >= 2:
        return "generic", "Caption uses 2+ LLM-filler phrases — try regenerating"
    if generic_hits == 1 and tone in ("clean", "neutral"):
        return "generic", "Caption sounds template-y — try a more specific hook"
    for token in _SPICY_TOKENS:
        if token in blob:
            return "spicy", f"Contains '{token}' — flagged in case you're optimizing for SFW reach"
    return "safe", ""


__all__ = [
    "EMOTIONAL_LABELS",
    "EmotionalLabel",
    "EnrichedVariant",
    "PlatformFit",
    "RiskLabel",
    "Tone",
    "enrich_variant",
    "enrich_variants",
]
