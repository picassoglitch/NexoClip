"""Detection module — voice + chat heat + audio energy (Phase 1); visual (Phase 1.5+)."""

from .audio_energy import detect_audio_energy
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
    "detect_audio_energy",
    "detect_candidates",
    "detect_chat_heat",
    "detect_voice_triggers",
    "load_candidates",
    "save_candidates",
]
