"""Variant generator — LLM-driven caption variants per (clip, persona)."""

from .enrichment import (
    EnrichedVariant,
    PlatformFit,
    RiskLabel,
    Tone,
    enrich_variant,
    enrich_variants,
)
from .hooks import (
    Hook,
    HookBatch,
    ToneId,
    generate_hooks,
    tone_choices,
)
from .models import VariantsFile
from .personas import Persona, get_persona, load_personas
from .service import find_clip, generate_variants, load_variants

__all__ = [
    "EnrichedVariant",
    "Hook",
    "HookBatch",
    "Persona",
    "PlatformFit",
    "RiskLabel",
    "Tone",
    "ToneId",
    "VariantsFile",
    "enrich_variant",
    "enrich_variants",
    "find_clip",
    "generate_hooks",
    "generate_variants",
    "get_persona",
    "load_personas",
    "load_variants",
    "tone_choices",
]
