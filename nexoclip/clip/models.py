"""Pydantic schemas for the clip module."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from nexoclip.detect import Candidate


class Clip(BaseModel):
    """One vertical clip cut + reformatted from a source VOD."""

    id: str = Field(description="ULID with `clp_` prefix.")
    tenant_id: str
    stream_id: str
    candidate: Candidate
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    duration_s: float = Field(ge=0.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    path: Path


class ClipManifest(BaseModel):
    """Saved at `<stream_dir>/clips_manifest.json`."""

    stream_id: str
    tenant_id: str
    clips: list[Clip] = Field(default_factory=list)
