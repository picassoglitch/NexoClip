"""Variant generator — turns a clip + persona into N caption variants.

Path through the system per CLAUDE.md (rule #3, rule #5):
    persona voice + clip evidence → prompt → LLMRouter.complete(VariantBatch)
                                                       ↓
                                  validated Pydantic Variants
                                                       ↓
                                  saved to `<clip_dir>/variants.json`
"""

from __future__ import annotations

from pathlib import Path

from nexoclip.clip import Clip
from nexoclip.errors import VariantError
from nexoclip.llm import LLMRouter, Variant, VariantBatch
from nexoclip.llm.config import Quality

from .models import VariantsFile
from .personas import Persona

_PURPOSE = "variant_generation"


async def generate_variants(
    tenant_id: str,
    clip: Clip,
    persona: Persona,
    *,
    router: LLMRouter,
    n: int = 5,
    language: str | None = None,
    chat_snippet: str = "",
    quality: Quality | None = None,
    clip_dir: Path | None = None,
    force: bool = False,
) -> list[Variant]:
    """Generate `n` caption variants for `clip` in `persona`'s voice.

    Args:
        tenant_id: Must match `clip.tenant_id`.
        clip: The Clip produced by `cut_clips`.
        persona: Persona to write in (voice prompt + language).
        router: Concrete LLMRouter (cost tracking + retries + JSONL log).
        n: How many variants to ask for. Defaults to 5 (per PHASE_0).
        language: ISO 639-1; falls back to persona's primary language.
        chat_snippet: Phase 1+ chat replay context (empty in Phase 0).
        quality: Override `default_quality` for this purpose.
        clip_dir: Where to save `variants.json`. Defaults to `clip.path.parent`.
        force: Re-generate even if `variants.json` already matches this persona.
    """
    if tenant_id != clip.tenant_id:
        raise VariantError(f"tenant mismatch: caller={tenant_id!r}, clip={clip.tenant_id!r}")
    if n <= 0:
        raise VariantError(f"n must be > 0, got {n}")

    effective_lang = language or persona.primary_language
    out_dir = clip_dir or clip.path.parent
    out_path = Path(out_dir) / "variants.json"

    if not force and out_path.exists():
        cached = VariantsFile.model_validate_json(out_path.read_text("utf-8"))
        if (
            cached.persona_id == persona.id
            and cached.tenant_id == tenant_id
            and cached.language == effective_lang
            and len(cached.variants) >= n
        ):
            return cached.variants[:n]

    system, user = _build_prompts(
        persona=persona,
        clip=clip,
        language=effective_lang,
        n=n,
        chat_snippet=chat_snippet,
    )

    batch: VariantBatch = await router.complete(
        tenant_id=tenant_id,
        purpose=_PURPOSE,
        system=system,
        user=user,
        schema=VariantBatch,
        quality=quality,
    )
    if not batch.variants:
        raise VariantError(f"LLM returned 0 variants for clip={clip.id} persona={persona.id}")

    variants = batch.variants[:n]
    file = VariantsFile(
        clip_id=clip.id,
        tenant_id=tenant_id,
        persona_id=persona.id,
        persona_name=persona.name,
        language=effective_lang,
        variants=variants,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(file.model_dump_json(indent=2), encoding="utf-8")
    return variants


def _build_prompts(
    *,
    persona: Persona,
    clip: Clip,
    language: str,
    n: int,
    chat_snippet: str,
) -> tuple[str, str]:
    """Build (system, user) prompts. Tight + factual — the LLM does the rest."""
    system = (
        f"{persona.voice_prompt.strip()}\n\n"
        f"Write each variant in {language}. Return exactly {n} distinct variants — "
        "different angles/hooks, not paraphrases of each other. "
        "Captions <= 280 chars. Hashtags WITHOUT the leading `#`. "
        "title_card_text is short (<=4 words) overlay text; leave empty if it would feel forced."
    )

    evidence = clip.candidate.evidence
    phrase = evidence.get("phrase", "")
    transcript_snippet = evidence.get("transcript_snippet", "")

    user_lines = [
        f"Generate {n} caption variants for this clip.",
        f"- Clip duration: {clip.duration_s:.1f}s",
        f"- Trigger reason: {clip.candidate.reason}",
        f"- Trigger phrase: {phrase!r}" if phrase else "",
        f"- Transcript at trigger: {transcript_snippet!r}" if transcript_snippet else "",
        f"- Score: {clip.candidate.score:.3f}",
    ]
    if chat_snippet:
        user_lines.append(f"- Chat context: {chat_snippet}")
    user = "\n".join(line for line in user_lines if line)
    return system, user


def find_clip(clip_id: str, output_dir: Path) -> tuple[Clip, Path, Path]:
    """Locate a clip by id under `<output_dir>/<stream>/clips/<clip_id>/`.

    Returns (clip, clip_dir, stream_dir).
    """
    output_dir = Path(output_dir)
    matches = sorted(output_dir.glob(f"*/clips/{clip_id}/metadata.json"))
    if not matches:
        raise VariantError(f"clip not found: {clip_id} (under {output_dir})")
    if len(matches) > 1:
        raise VariantError(f"clip id collision: {clip_id} matched {len(matches)} streams")
    metadata_path = matches[0]
    clip = Clip.model_validate_json(metadata_path.read_text("utf-8"))
    clip_dir = metadata_path.parent
    stream_dir = clip_dir.parent.parent
    return clip, clip_dir, stream_dir


def load_variants(clip_dir: Path) -> VariantsFile:
    """Read `<clip_dir>/variants.json` back from disk."""
    path = Path(clip_dir) / "variants.json"
    if not path.exists():
        raise VariantError(f"variants not found at {path}")
    return VariantsFile.model_validate_json(path.read_text("utf-8"))
