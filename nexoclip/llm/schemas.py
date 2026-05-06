"""Pydantic schemas for structured LLM outputs.

CLAUDE.md hard rule #5: every LLM call goes through the router with a Pydantic
schema. No string parsing, no regex over LLM output. Keep these schemas tight
— the LLM is more reliable when it has fewer optional fields to think about.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Variant(BaseModel):
    """One short-form caption variant for a clip.

    Fields are deliberately minimal — the variant generator emits 3-5 of these
    per (clip, persona) pair and the user picks one to publish.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Stable id within the batch, e.g. `v_1`.")
    language: str = Field(description="ISO 639-1 code, e.g. `es`, `en`.")
    caption: str = Field(min_length=1, description="Body copy, ≤ 280 chars.")
    title_card_text: str = Field(
        default="",
        description="Short overlay text rendered on the first frame; may be empty.",
    )
    hashtags: list[str] = Field(
        default_factory=list,
        description="Hashtags WITHOUT the leading `#`. May be empty.",
    )


class VariantBatch(BaseModel):
    """Wrapper returned by the LLM when generating multiple variants at once."""

    variants: list[Variant] = Field(default_factory=list)
