"""Detection module — voice triggers (Phase 0); chat/audio/visual (Phase 1+)."""

from .models import Candidate, CandidateBatch, CandidateReason
from .service import (
    detect_voice_triggers,
    load_candidates,
    save_candidates,
)

__all__ = [
    "Candidate",
    "CandidateBatch",
    "CandidateReason",
    "detect_voice_triggers",
    "load_candidates",
    "save_candidates",
]
