"""Variant generator - turns a clip + persona into N caption variants.

Path through the system per CLAUDE.md (rule #3, rule #5):
    persona voice + clip evidence -> prompt -> LLMRouter.complete(VariantBatch)
                                                       v
                                  validated Pydantic Variants
                                                       v
                                  saved to `<clip_dir>/variants.json`

When `use_vision=True`, the same path runs through `complete_multimodal`
with three frames sampled across the cut clip. The frames are cached by
(stream_id, source_ts) so a second persona pass on the same clip - or a
retry inside the router - reuses the JPEG bytes instead of re-decoding
the video.
"""

from __future__ import annotations

from pathlib import Path

from nexoclip.clip import Clip
from nexoclip.errors import VariantError
from nexoclip.llm import FrameCache, LLMRouter, MultimodalImage, Variant, VariantBatch
from nexoclip.llm.config import Quality

from .models import VariantsFile
from .personas import Persona

_PURPOSE = "variant_generation"
_DEFAULT_VISION_FRAMES = 3


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
    use_vision: bool = False,
    frame_cache: FrameCache | None = None,
    n_vision_frames: int = _DEFAULT_VISION_FRAMES,
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
        use_vision: When True, sample `n_vision_frames` JPEGs from the cut
            clip and route through `complete_multimodal`. The persona has to
            be on a vision-capable model for this to make sense - the router
            doesn't second-guess that decision.
        frame_cache: Optional shared cache. When omitted in vision mode, a
            fresh per-call cache is created (still avoids the second decode
            on a router-side retry).
        n_vision_frames: How many frames to sample. Defaults to 3.
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

    if use_vision:
        images = _gather_vision_frames(
            clip=clip,
            n_frames=n_vision_frames,
            cache=frame_cache if frame_cache is not None else FrameCache(),
        )
        batch: VariantBatch = await router.complete_multimodal(
            tenant_id=tenant_id,
            purpose=_PURPOSE,
            system=system,
            user=user,
            images=images,
            schema=VariantBatch,
            quality=quality,
        )
    else:
        batch = await router.complete(
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


def _gather_vision_frames(
    *, clip: Clip, n_frames: int, cache: FrameCache
) -> list[MultimodalImage]:
    """Sample `n_frames` from the cut clip, caching by source-stream ts.

    Sample offsets are spread evenly across the clip's interior - first
    sample at `duration / (n+1)`, last at `n * duration / (n+1)` - which
    avoids picking the very first/last frame (often a fade or letterbox).
    """
    if n_frames < 1:
        raise VariantError(f"n_vision_frames must be >= 1, got {n_frames}")
    duration = max(0.0, clip.duration_s)
    offsets = [duration * (i + 1) / (n_frames + 1) for i in range(n_frames)]

    blobs: list[bytes] = []
    for offset in offsets:
        source_ts = clip.start_s + offset
        cached = cache.get(clip.stream_id, source_ts)
        if cached is not None:
            blobs.append(cached)
            continue

        from nexoclip.vision.frame_sampler import sample_frames

        sampled = sample_frames(clip.path, ts=offset, n=1, spread_s=0.0)
        if not sampled:
            raise VariantError(
                f"frame_sampler returned 0 frames for clip={clip.id} ts={source_ts}"
            )
        blob = sampled[0]
        cache.put(clip.stream_id, source_ts, blob)
        blobs.append(blob)

    return [MultimodalImage(media_type="image/jpeg", data=blob) for blob in blobs]


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
