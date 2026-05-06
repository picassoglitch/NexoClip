"""Typed exception hierarchy for NexoClip.

Per CLAUDE.md: errors are typed; never catch `Exception` broadly except at
process boundaries. Each step in the pipeline raises one of these.
"""

from __future__ import annotations


class NexoClipError(Exception):
    """Base class for all NexoClip errors."""


class IngestError(NexoClipError):
    """VOD download or audio extraction failed."""


class TranscriptionError(NexoClipError):
    """Whisper transcription failed."""


class DetectionError(NexoClipError):
    """Trigger detection failed."""


class ClipError(NexoClipError):
    """ffmpeg cut / reformat failed."""


class LLMError(NexoClipError):
    """LLM provider call failed (after retries)."""


class VariantError(NexoClipError):
    """Variant generation failed (clip not found, bad persona, etc.)."""


class QuotaExceeded(NexoClipError):  # noqa: N818  # name pinned by CLAUDE.md
    """Tenant quota would be exceeded by this call."""
