"""Deterministic (no-LLM) hook generation — titles from what we already have.

Hook generation is an LLM call, and LLM calls fail for boring, durable
reasons: the billing cap is hit, the key is revoked, the provider is down.
When that happened the pipeline shipped clips with NO hook, which cascaded
into degenerate one-word captions and dead posts (the "viral" incident:
18 shorts titled literally "viral", 0 views).

This module guarantees a usable, clip-specific title line with zero
external calls, in priority order:

    1. the best transcript sentence — scored for hook-ness (questions,
       exclamations, viral lexicon, second person, numbers)
    2. the cleaned stream title (often the whole story for no-speech
       clips: "Mexico 1 - 0 Korea")
    3. a language-appropriate template, picked stably from the seed so a
       batch of clips never ships N identical titles

Pure functions, no I/O. The LLM path stays primary everywhere — these are
the fallback the publish flow can always count on.
"""

from __future__ import annotations

import re
import zlib

MAX_HOOK_WORDS = 12

# Sentence boundary: ., !, ?, … followed by whitespace/end. Keeps the
# terminal punctuation attached to the sentence.
_SENTENCE_RE = re.compile(r"[^.!?…]+[.!?…]*")

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HASHTAG_RE = re.compile(r"#\w+")
_WS_RE = re.compile(r"\s+")

# Openers that mark a sentence as filler/ramble rather than a hook.
_FILLER_STARTS = frozenset(
    {
        # es
        "eh", "em", "este", "bueno", "entonces", "pues", "osea", "o", "ya",
        "aja", "ajá", "mmm", "vale", "digo",
        # en
        "um", "uh", "so", "like", "okay", "ok", "well", "anyway", "yeah",
    }
)

# Words that signal a charged / curiosity-driving moment (both languages,
# lowercase, accent-insensitive matching is NOT attempted — common forms
# are listed explicitly).
_VIRAL_LEXICON = frozenset(
    {
        # es
        "increible", "increíble", "locura", "loco", "nunca", "jamas", "jamás",
        "nadie", "todos", "secreto", "dinero", "gratis", "error", "verdad",
        "mentira", "peor", "mejor", "ultimo", "último", "final", "gano",
        "ganó", "perdio", "perdió", "explota", "insulto", "record", "récord",
        "prohibido", "viral",
        # en
        "insane", "crazy", "never", "nobody", "everyone", "secret", "money",
        "free", "mistake", "truth", "lie", "worst", "best", "last", "finally",
        "won", "lost", "exposed", "banned", "unhinged", "wild",
    }
)

_SECOND_PERSON = frozenset(
    {"tú", "tu", "te", "usted", "ustedes", "you", "your", "u"}
)

# Last-resort templates. Picked by a stable hash of the seed, so two clips
# from the same batch land on different lines — a wall of identical titles
# reads as spam to both viewers and ranking systems.
_TEMPLATES: dict[str, tuple[str, ...]] = {
    "es": (
        "El momento que nadie vio venir",
        "Esto se salió de control en segundos",
        "Nadie esperaba lo que pasó aquí",
        "Espera al final — no es lo que crees",
        "Esto pasó en vivo y quedó grabado",
        "La reacción que todos están comentando",
    ),
    "en": (
        "The moment nobody saw coming",
        "This went off the rails in seconds",
        "Nobody expected what happened here",
        "Wait for the end — not what you think",
        "This happened live and it's all on tape",
        "The reaction everyone is talking about",
    ),
}


def _lang_key(language: str) -> str:
    return "es" if (language or "").strip().lower().startswith("es") else "en"


def _words(text: str) -> list[str]:
    return [w for w in _WS_RE.split(text.strip()) if w]


def _split_sentences(text: str) -> list[str]:
    return [m.group(0).strip() for m in _SENTENCE_RE.finditer(text) if m.group(0).strip()]


def _score_sentence(sentence: str) -> int:
    """Hook-ness score for one transcript sentence. Higher = better."""
    words = _words(sentence)
    if not words:
        return -10
    n = len(words)
    score = 0
    if 5 <= n <= 14:
        score += 2
    elif 3 <= n <= 18:
        score += 1
    else:
        score -= 1
    if "?" in sentence:
        score += 2
    if "!" in sentence:
        score += 2
    lowered = [w.strip(".,!?…\"'()").lower() for w in words]
    lexicon_hits = sum(1 for w in lowered if w in _VIRAL_LEXICON)
    score += min(lexicon_hits, 2) * 2
    if any(w in _SECOND_PERSON for w in lowered):
        score += 1
    if any(any(ch.isdigit() for ch in w) for w in words):
        score += 1
    if lowered[0] in _FILLER_STARTS:
        score -= 2
    return score


def _trim_to_hook(sentence: str, *, max_words: int = MAX_HOOK_WORDS) -> str:
    """Normalize a sentence into title shape: collapsed whitespace, capped
    word count (truncation keeps an ellipsis as an open loop), no trailing
    period, capitalized first letter."""
    words = _words(sentence)
    truncated = len(words) > max_words
    text = " ".join(words[:max_words]).strip().rstrip(",;:—-")
    if truncated:
        text = text.rstrip(".!?…") + "…"
    text = text.rstrip(".")
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def clean_stream_title(title: str, *, max_words: int = MAX_HOOK_WORDS) -> str:
    """Strip stream-title noise (URLs, hashtags, chatbot commands, pipe
    segments) down to the human part, trimmed to hook length."""
    text = _URL_RE.sub("", title or "")
    text = _HASHTAG_RE.sub("", text)
    # "!discord"-style chatbot commands are noise in a title.
    text = re.sub(r"(?<!\w)![a-z]\w*", "", text, flags=re.IGNORECASE)
    # Streamers chain segments with | or •; the longest one is the story.
    segments = [s.strip() for s in re.split(r"[|•]", text) if s.strip()]
    if not segments:
        return ""
    best = max(segments, key=lambda s: len(_words(s)))
    return _trim_to_hook(best, max_words=max_words)


def deterministic_hook(
    *,
    transcript_snippet: str = "",
    stream_title: str = "",
    language: str = "es",
    seed: str = "",
    max_words: int = MAX_HOOK_WORDS,
) -> str:
    """One usable title line, guaranteed non-empty, with zero LLM calls.

    `seed` should be something stable and clip-specific (the clip id) so
    the template fallback varies across a batch but is idempotent per clip.
    """
    candidates = deterministic_hook_candidates(
        transcript_snippet=transcript_snippet,
        stream_title=stream_title,
        language=language,
        seed=seed,
        n=1,
        max_words=max_words,
    )
    return candidates[0]


def deterministic_hook_candidates(
    *,
    transcript_snippet: str = "",
    stream_title: str = "",
    language: str = "es",
    seed: str = "",
    n: int = 5,
    max_words: int = MAX_HOOK_WORDS,
) -> list[str]:
    """Up to `n` distinct title candidates, best first: scored transcript
    sentences, then the cleaned stream title, then templates. Always
    returns at least one entry (templates never run dry)."""
    n = max(1, n)
    out: list[str] = []
    seen: set[str] = set()

    def push(text: str) -> None:
        text = text.strip()
        key = text.lower()
        # Two-word "sentences" are transcription shrapnel, not hooks.
        if len(_words(text)) >= 3 and key not in seen:
            seen.add(key)
            out.append(text)

    scored = sorted(
        ((s, _score_sentence(s)) for s in _split_sentences(transcript_snippet)),
        key=lambda pair: -pair[1],
    )
    for sentence, score in scored:
        if score >= 2:
            push(_trim_to_hook(sentence, max_words=max_words))
        if len(out) >= n:
            return out[:n]

    push(clean_stream_title(stream_title, max_words=max_words))

    template_pool = _TEMPLATES[_lang_key(language)]
    first = zlib.crc32((seed or "").encode("utf-8")) % len(template_pool)
    for offset in range(len(template_pool)):
        if len(out) >= n:
            break
        push(template_pool[(first + offset) % len(template_pool)])

    return out[:n]
