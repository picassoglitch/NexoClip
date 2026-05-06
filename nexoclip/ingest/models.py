"""Pydantic schemas for ingested streams."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Platform = Literal["kick", "twitch", "youtube", "unknown"]


class Stream(BaseModel):
    """A VOD that has been downloaded and audio-extracted."""

    model_config = ConfigDict(frozen=False)

    id: str = Field(description="ULID with `str_` prefix")
    tenant_id: str
    vod_url: str
    platform: Platform
    title: str | None = None
    channel: str | None = None
    duration_s: float = Field(ge=0.0)
    source_video_path: Path
    source_audio_path: Path
