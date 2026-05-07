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


class CropBoxVerdict(BaseModel):
    """Vision-LLM verdict for the 9:16 smart-crop picker (P2 Task 5).

    The model gets a representative frame at the clip's anchor + the
    source resolution, and returns the *fractional* crop box (0..1 range)
    so we can scale it to whatever frame size the clip uses. The picker
    converts these fractions to integer pixel coords before persisting.
    """

    model_config = ConfigDict(extra="ignore")

    x_frac: float = Field(
        ge=0.0, le=1.0, description="Left edge of the 9:16 crop, fraction of source width."
    )
    width_frac: float = Field(
        gt=0.0, le=1.0, description="Crop width, fraction of source width."
    )
    reason: str = Field(min_length=1, max_length=200)


class ThumbnailPickVerdict(BaseModel):
    """Vision-LLM verdict for the auto-thumbnail picker (P2 Task 5).

    The model sees N candidate frames sampled across the clip; it returns
    the index of the strongest frame and a one-sentence reason.
    """

    model_config = ConfigDict(extra="ignore")

    index: int = Field(ge=0, description="0-based index into the N frames shown.")
    reason: str = Field(min_length=1, max_length=200)


class FaceEmotionVerdict(BaseModel):
    """Vision-LLM verdict for one sampled frame's face emotion (P2 Task 6).

    `has_face=False` clears `emotion` to null. Otherwise emotion takes one
    of the FaceEmotion labels mirrored from `nexoclip.vision.models`.
    """

    model_config = ConfigDict(extra="ignore")

    has_face: bool
    emotion: str | None = Field(
        default=None,
        description="neutral | smile | laugh | shock | anger | sad. Null when no face.",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RescoreVerdict(BaseModel):
    """Vision-LLM verdict on whether a candidate is *actually* clip-worthy.

    Phase 2 Task 3 schema. The LLM sees N frames sampled around the
    candidate timestamp + the audio/chat context the heuristic detector
    fired on, and returns a single number plus a short reason.

    `score` is 0.0-1.0 where:
        * < 0.30  — the moment doesn't look like a real reaction; suppress
        * 0.30-0.65 — ambiguous; keep heuristic ranking
        * > 0.65  — strong on-screen reaction; promote
    """

    model_config = ConfigDict(extra="ignore")

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence the moment is genuinely clip-worthy.",
    )
    face_emotion: str | None = Field(
        default=None,
        description=(
            "One of {neutral, smile, laugh, shock, anger, sad}, or null when "
            "no face is visible / readable. Propagates back into visual_signals."
        ),
    )
    reason: str = Field(
        min_length=1,
        max_length=400,
        description=(
            "One short sentence: what the model saw that justifies the score. "
            "Surfaces in the dashboard's confidence-breakdown panel."
        ),
    )
