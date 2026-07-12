"""Deterministic (no-LLM) hook generation — transcript sentence pick,
stream-title fallback, seeded template fallback. Pure functions."""

from __future__ import annotations

from nexoclip.variants.deterministic import (
    clean_stream_title,
    deterministic_hook,
    deterministic_hook_candidates,
)


def test_picks_the_most_hooky_transcript_sentence() -> None:
    snippet = (
        "bueno entonces estábamos hablando de eso. "
        "¡No puedo creer que ganó todo el dinero en la última ronda! "
        "y luego seguimos con el stream."
    )
    hook = deterministic_hook(transcript_snippet=snippet, language="es")
    assert "dinero" in hook.lower()
    first_alpha = next(c for c in hook if c.isalpha())
    assert first_alpha.isupper()


def test_questions_and_exclamations_beat_flat_sentences() -> None:
    snippet = (
        "we played some rounds today. "
        "why does nobody ever talk about this trick?"
    )
    hook = deterministic_hook(transcript_snippet=snippet, language="en")
    assert hook.endswith("?")


def test_long_sentences_are_trimmed_with_open_loop() -> None:
    snippet = (
        "esto es increíble porque nunca en toda mi vida había visto una jugada "
        "tan absurda como la que acaba de pasar en esta partida clasificatoria!"
    )
    hook = deterministic_hook(transcript_snippet=snippet, language="es")
    assert len(hook.split()) <= 13  # 12 words + possible trailing ellipsis merge
    assert hook.endswith("…")


def test_falls_back_to_cleaned_stream_title() -> None:
    hook = deterministic_hook(
        transcript_snippet="",
        stream_title="Mexico 1 - 0 Korea | !discord | https://x.com #futbol",
        language="es",
    )
    assert "Mexico 1 - 0 Korea" in hook
    assert "!discord" not in hook
    assert "#futbol" not in hook
    assert "https" not in hook


def test_template_fallback_is_stable_per_seed_and_varies_across_seeds() -> None:
    a1 = deterministic_hook(language="es", seed="clp_a")
    a2 = deterministic_hook(language="es", seed="clp_a")
    assert a1 == a2  # idempotent per clip
    hooks = {deterministic_hook(language="es", seed=f"clp_{i}") for i in range(12)}
    assert len(hooks) > 1  # a batch never ships one identical wall of titles


def test_hook_is_never_empty() -> None:
    assert deterministic_hook() != ""
    assert deterministic_hook(transcript_snippet="eh", stream_title="", seed="") != ""


def test_language_routes_template_pool() -> None:
    es = deterministic_hook(language="es", seed="x")
    en = deterministic_hook(language="en", seed="x")
    assert es != en


def test_candidates_are_distinct_and_capped() -> None:
    snippet = (
        "¡Esto fue una locura total! "
        "¿Cómo es posible que nadie lo viera venir? "
        "Nunca había pasado algo así en el canal."
    )
    candidates = deterministic_hook_candidates(
        transcript_snippet=snippet, stream_title="Gran final del torneo",
        language="es", seed="clp_x", n=5,
    )
    assert 1 <= len(candidates) <= 5
    assert len({c.lower() for c in candidates}) == len(candidates)


def test_clean_stream_title_prefers_longest_segment() -> None:
    assert clean_stream_title("!prime | GRAN FINAL contra el campeón | !sub") == (
        "GRAN FINAL contra el campeón"
    )
