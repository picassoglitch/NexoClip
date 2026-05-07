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


class TenancyError(NexoClipError):
    """Tenancy contract violation: no tenant bound, mismatch, unknown token."""


class QuotaExceeded(NexoClipError):  # noqa: N818  # name pinned by CLAUDE.md
    """Tenant quota would be exceeded by this call."""


class BudgetExceeded(NexoClipError):  # noqa: N818
    """Tenant's daily LLM USD budget would be exceeded by this call.

    Raised by the BudgetGovernor (P2 Task 1) before LLMRouter issues a
    request. Higher-up callers catch this, emit `llm.budget_exhausted`,
    and halt the current pipeline run cleanly.
    """


class CooldownActive(NexoClipError):  # noqa: N818
    """Repeated low-confidence rescore verdicts triggered a cooldown.

    The governor refuses new rescore requests until `cooldown_s` after the
    last refusal. Operators can clear it by lowering the threshold or
    waiting it out.
    """
