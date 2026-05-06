"""Detection module — voice + chat heat (Phase 1); audio/visual (Phase 1.4+)."""

from .chat_heat import detect_chat_heat
from .models import Candidate, CandidateBatch, CandidateReason
from .service import (
    detect_candidates,
    detect_voice_triggers,
    load_candidates,
    save_candidates,
)

__all__ = [
    "Candidate",
    "CandidateBatch",
    "CandidateReason",
    "detect_candidates",
    "detect_chat_heat",
    "detect_voice_triggers",
    "load_candidates",
    "save_candidates",
]
