"""Pydantic schemas for Whisper transcripts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Word(BaseModel):
    """One word with start/end times and Whisper-reported probability."""

    ts: float = Field(ge=0.0, description="Word start time, seconds.")
    end_ts: float = Field(ge=0.0, description="Word end time, seconds.")
    text: str
    prob: float = Field(ge=0.0, le=1.0)


class Segment(BaseModel):
    """A Whisper segment — a sentence-ish span containing word-level entries."""

    ts: float = Field(ge=0.0, description="Segment start time, seconds.")
    end_ts: float = Field(ge=0.0, description="Segment end time, seconds.")
    text: str
    words: list[Word] = Field(default_factory=list)


class Transcript(BaseModel):
    """Full transcript for one stream."""

    stream_id: str
    tenant_id: str
    language: str
    duration_s: float = Field(ge=0.0)
    model: str = Field(description="Whisper model size, e.g. `medium`.")
    segments: list[Segment] = Field(default_factory=list)
