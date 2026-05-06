"""Persistence shape for `<clip_dir>/variants.json`."""

from __future__ import annotations

from pydantic import BaseModel, Field

from nexoclip.llm import Variant


class VariantsFile(BaseModel):
    """One file per (clip, persona). Re-running with a different persona overwrites."""

    clip_id: str
    tenant_id: str
    persona_id: str
    persona_name: str
    language: str
    variants: list[Variant] = Field(default_factory=list)
