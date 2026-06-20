"""Viral-hook variant generator — voice-markers spec slice F.7-B.

Distinct from `variants.service.generate_variants` (which writes
captions in a persona's voice). This generator's only job is to
produce 5 candidate TITLE LINES — the bold-on-white text that sits
at the top of the rendered clip and drives the swipe rate.

Pipeline:
    persona context + clip transcript snippet + tone preset
        ↓
    LLMRouter.complete(purpose="hook_generation", schema=HookBatch)
        ↓
    list[Hook]  (5 short, punchy, swipe-stopping titles)

The tone preset rotates the prompt voice without burning a separate
LLM purpose for each. Five presets ship in F.7-B:

  * default      — generic creator-tone, the safe default
  * aggressive   — confrontational, no-punches-pulled
  * gen_z        — TikTok-native slang, abbreviations, emoji use
  * corporate    — clean, professional, no slang or emoji
  * curious      — open-loop questions that demand the answer

Stays under the existing budget governor — every call routes
through `LLMRouter.complete` with `purpose="hook_generation"`,
which the router config maps to a default quality + provider chain.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nexoclip.llm import LLMRouter

ToneId = Literal["default", "aggressive", "gen_z", "corporate", "curious"]

# Default count — five gives the operator a real choice without
# blowing the budget. The router caps this; the schema enforces it.
DEFAULT_N = 5
MIN_N = 1
MAX_N = 10


class Hook(BaseModel):
    """One generated title line."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=140)


class HookBatch(BaseModel):
    """Structured response from the hook-generation purpose."""

    model_config = ConfigDict(extra="forbid")
    hooks: list[Hook] = Field(min_length=1, max_length=MAX_N)


# ---- prompt assembly ----


_TONE_PROMPTS: dict[ToneId, str] = {
    "default": (
        "Write in the streamer's natural voice — punchy, short, no fluff. "
        "Avoid clickbait that misrepresents the moment."
    ),
    "aggressive": (
        "Write CONFRONTATIONAL titles. Pick a fight with the viewer's "
        "expectations. Conflict, status, callout. No emoji. ALL-CAPS for "
        "the most charged word, sparingly."
    ),
    "gen_z": (
        "Write TikTok-native Gen Z titles. Lowercase, abbreviated, slang "
        "(fr, ngl, lowkey, no cap, ate, ratio'd). One emoji max. "
        "Concise — under 8 words."
    ),
    "corporate": (
        "Write CLEAN, PROFESSIONAL titles. No slang, no emoji, no "
        "ALL-CAPS. Could appear on a brand's official social channel "
        "without seeming off-brand."
    ),
    "curious": (
        "Write OPEN-LOOP titles that demand resolution — questions, "
        "cliffhangers, partial reveals. The viewer should NEED to watch "
        "to find out the answer. No spoilers."
    ),
}


_SYSTEM_PROMPT = """\
You are a short-form video title writer working with a streamer.
You're given:
  - the streamer's persona / voice
  - context about the clip (the stream title / what the moment is about)
  - a transcript snippet from the clip (may be empty)
  - the language to write in
  - a tone instruction

Return exactly N title candidates as a JSON object matching the
HookBatch schema:
    { "hooks": [ { "text": "..." }, { "text": "..." }, ... ] }

Rules:
  - 6-12 words is the sweet spot. Hard max 14 words.
  - Title-case the first word; otherwise sentence case unless the
    tone instruction explicitly says ALL-CAPS.
  - No quotation marks around the whole title.
  - No hashtags.
  - Every title must be a coherent sentence the viewer can read in
    under 1.5 seconds.
  - Don't repeat the transcript verbatim — the title is a HOOK, not
    a summary. It dramatizes the moment.
  - NEVER write meta-commentary about your own inputs. Do not mention a
    missing/absent transcript, "no captions", that you're guessing, or
    that you bet something happened. Every title must read as if written
    by someone who watched the clip.
  - When the transcript is empty, write the titles from the CONTEXT
    (the stream title / topic). E.g. a stream titled "Mexico 1 - 0 Korea"
    should yield football-goal hooks, never "no transcript but..." filler.

Generate variations that differ in approach (curiosity, conflict,
status, identity, surprise) — the operator picks the best one.
"""


def _user_prompt(
    *,
    persona_voice: str,
    persona_language: str,
    transcript_snippet: str,
    tone: ToneId,
    n: int,
    clip_context: str = "",
) -> str:
    tone_block = _TONE_PROMPTS.get(tone, _TONE_PROMPTS["default"])
    has_snippet = bool(transcript_snippet.strip())
    snippet_block = (
        transcript_snippet.strip()
        if has_snippet
        # No defeatist placeholder — the system prompt forbids writing
        # about a missing transcript, so steer the model to the context.
        else "(transcript unavailable — write the hooks from the context above)"
    )
    persona_block = persona_voice.strip() or "(no persona voice provided)"
    context_block = (
        f"What this clip / stream is about:\n{clip_context.strip()}\n\n"
        if clip_context.strip()
        else ""
    )
    return (
        f"Streamer persona / voice:\n{persona_block}\n\n"
        f"{context_block}"
        f"Language to write in: {persona_language}\n\n"
        f"Tone instruction: {tone_block}\n\n"
        f"Transcript snippet from the clip:\n\"\"\"\n{snippet_block}\n\"\"\"\n\n"
        f"Generate {n} title candidates for this clip."
    )


# ---- public entry ----


async def generate_hooks(
    *,
    tenant_id: str,
    persona_voice: str,
    persona_language: str,
    transcript_snippet: str,
    clip_context: str = "",
    tone: ToneId = "default",
    n: int = DEFAULT_N,
    router: LLMRouter,
) -> list[Hook]:
    """Generate `n` viral-hook title candidates for one clip.

    Args:
        tenant_id: Bound tenant — passed through to the router for
            cost-tracking attribution.
        persona_voice: The streamer's voice prompt (free-form text
            describing how they talk).
        persona_language: ISO 639-1 (`es`, `en`, ...) the hooks
            should be written in.
        transcript_snippet: 1-3 sentences from the clip's transcript,
            used as the hook's context. Empty string is allowed — the
            LLM then writes from `clip_context` instead of inventing
            meta-commentary about the missing transcript.
        clip_context: extra context the model should hook off of when
            the transcript is thin/empty — typically the stream title
            (e.g. "Mexico 1 - 0 Korea") plus the detection reason.
        tone: One of the five ToneId presets. Falls back to "default"
            if the caller passes an unrecognized id.
        n: Number to generate. Clamped to [MIN_N, MAX_N].
        router: Configured LLMRouter — purpose `hook_generation`
            must be in its routing rules.

    Returns:
        list of `Hook` objects, length `n` (or whatever the LLM
        actually returned, validated against the schema).
    """
    n_clamped = max(MIN_N, min(MAX_N, n))
    user = _user_prompt(
        persona_voice=persona_voice,
        persona_language=persona_language,
        transcript_snippet=transcript_snippet,
        clip_context=clip_context,
        tone=tone,
        n=n_clamped,
    )
    batch = await router.complete(
        tenant_id=tenant_id,
        purpose="hook_generation",
        system=_SYSTEM_PROMPT,
        user=user,
        schema=HookBatch,
    )
    return list(batch.hooks)


def tone_choices() -> list[tuple[ToneId, str]]:
    """(id, label) list for the editor's tone-picker dropdown."""
    return [
        ("default", "Default — streamer's natural voice"),
        ("aggressive", "Aggressive — confrontational, no-punches-pulled"),
        ("gen_z", "Gen Z — TikTok-native, lowercase, slang"),
        ("corporate", "Corporate — clean, professional, no slang"),
        ("curious", "Curious — open-loop questions, cliffhangers"),
    ]
