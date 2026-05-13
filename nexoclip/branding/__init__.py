"""Brand kit resolution + asset management (voice-markers spec slice C).

The resolver picks the right brand kit per candidate / clip at render
time. Priority order:

    1. speaker.preferred_brand_kit_id (if the candidate's speaker is
       resolved AND that speaker has an explicit preferred kit)
    2. tenant's default brand kit (`brand_kits.is_default = True`)
    3. None — renderer falls back to system defaults (Pico styles,
       no logo, no custom captions)

A future Storage abstraction (docs/production_deploy.md §3) will swap
asset path resolution from local-filesystem to S3/R2 without touching
this module.
"""

from __future__ import annotations

from .service import (
    merged_trigger_phrases_for_speaker,
    resolve_brand_kit_for_candidate,
    resolve_brand_kit_for_speaker,
)

__all__ = [
    "merged_trigger_phrases_for_speaker",
    "resolve_brand_kit_for_candidate",
    "resolve_brand_kit_for_speaker",
]
