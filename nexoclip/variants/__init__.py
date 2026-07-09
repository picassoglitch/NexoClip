"""Persona loading + viral-hook generation.

The LLM caption-variant generator was removed (clips ship a deterministic
overlay/hook instead). What remains here is the persona model and the
hook generator, both still used by the pipeline and the publish composer.
"""

from .hooks import (
    Hook,
    HookBatch,
    ToneId,
    generate_hooks,
    tone_choices,
)
from .personas import Persona, get_persona, load_personas

__all__ = [
    "Hook",
    "HookBatch",
    "Persona",
    "ToneId",
    "generate_hooks",
    "get_persona",
    "load_personas",
    "tone_choices",
]
