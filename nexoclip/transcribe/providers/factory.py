"""Factory — picks the TranscribeProvider based on Settings."""

from __future__ import annotations

import os

from nexoclip.errors import NexoClipError
from nexoclip.settings import get_settings

from .base import TranscribeProvider
from .cloud_whisper import CloudWhisperProvider
from .local_whisper import LocalWhisperProvider


def get_provider() -> TranscribeProvider:
    """Return the configured transcribe provider.

    Selection lives entirely in `Settings.transcribe_provider`:

      "local"      → LocalWhisperProvider (faster-whisper on this host)
      "assemblyai" → CloudWhisperProvider(vendor="assemblyai")  [stub]
      "deepgram"   → CloudWhisperProvider(vendor="deepgram")    [stub]
      "openai"     → CloudWhisperProvider(vendor="openai")      [stub]

    Defaults to "local" so existing deployments don't change behavior.

    The legacy `NEXOCLIP_TRANSCRIBE_INPROCESS=1` env override is still
    honored — it flips the local provider's subprocess isolation off.
    """
    settings = get_settings()
    choice = (settings.transcribe_provider or "local").strip().lower()

    if choice == "local":
        # Honor the legacy subprocess-isolation toggle. Off (subprocess
        # mode = on) is the default — see local_whisper.py for the
        # crash-isolation rationale.
        use_subprocess = (
            os.environ.get("NEXOCLIP_TRANSCRIBE_INPROCESS", "").strip() != "1"
        )
        return LocalWhisperProvider(
            model_size=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            use_subprocess=use_subprocess,
        )

    if choice in {"assemblyai", "deepgram", "openai"}:
        api_key = getattr(settings, f"{choice}_api_key", None)
        if not api_key:
            raise NexoClipError(
                f"transcribe_provider={choice!r} requires "
                f"NEXOCLIP_{choice.upper()}_API_KEY in .env"
            )
        return CloudWhisperProvider(vendor=choice, api_key=api_key)

    raise NexoClipError(
        f"unknown transcribe_provider {choice!r}; expected one of "
        f"local / assemblyai / deepgram / openai"
    )
