"""Whisper transcription — orchestrator.

Holds the idempotence + tenant-validation + write-cache logic. The
actual audio→Transcript step is delegated to a `TranscribeProvider`
so deployments can pick local Whisper (default) or a cloud STT
vendor without touching pipeline.py.

The transcript is saved alongside the audio at
`<stream_dir>/source/transcript.json`. Idempotent: a second call
returns the cached `Transcript` unless `force=True`.
"""

from __future__ import annotations

from pathlib import Path

from nexoclip.errors import TranscriptionError
from nexoclip.ingest import Stream

from .models import Transcript
from .providers import TranscribeProvider, TranscribeRequest, get_provider


def _transcript_path(stream: Stream) -> Path:
    return stream.source_audio_path.parent / "transcript.json"


async def transcribe(
    tenant_id: str,
    stream: Stream,
    *,
    model_size: str = "medium",
    device: str = "cuda",
    compute_type: str = "float16",
    language: str | None = "es",
    force: bool = False,
    provider: TranscribeProvider | None = None,
) -> Transcript:
    """Run the configured TranscribeProvider on `stream.source_audio_path`
    and save `transcript.json`.

    Args:
        tenant_id: Tenant owning the stream.
        stream: The ingested stream.
        model_size: faster-whisper model name. Only honored when the
            configured provider is local Whisper; cloud providers
            ignore this and read their own knobs from Settings.
        device: `cuda` or `cpu`. Same caveat as `model_size`.
        compute_type: `float16` / `int8` / etc. Same caveat.
        language: ISO 639-1 code; pass `None` to auto-detect.
        force: Re-run even when `transcript.json` already exists.
        provider: Override the configured provider (tests inject
            `FakeTranscribeProvider` here).

    Note: `model_size` / `device` / `compute_type` are kept on the
    signature for backward compatibility with the pipeline call site.
    They override the Settings-driven defaults of the local provider
    when supplied; cloud providers ignore them entirely.
    """
    if tenant_id != stream.tenant_id:
        raise TranscriptionError(
            f"tenant mismatch: caller={tenant_id!r}, stream={stream.tenant_id!r}"
        )
    if not stream.source_audio_path.exists():
        raise TranscriptionError(f"audio file missing: {stream.source_audio_path}")

    out_path = _transcript_path(stream)
    if not force and out_path.exists():
        return Transcript.model_validate_json(out_path.read_text("utf-8"))

    # Pick the provider. Caller can inject (tests do); otherwise we
    # honor the per-call overrides for local Whisper to keep the old
    # pipeline.py call shape working without a config change.
    if provider is None:
        from .providers.factory import get_provider as _get_provider
        from .providers.local_whisper import LocalWhisperProvider

        # If the call site is passing per-call overrides for the local
        # Whisper knobs (which the pipeline does), build a fresh local
        # provider with those values. Otherwise use the configured one.
        configured = _get_provider()
        if isinstance(configured, LocalWhisperProvider):
            provider = LocalWhisperProvider(
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                # Inherit subprocess-isolation default from the factory.
                use_subprocess=configured._use_subprocess,  # noqa: SLF001
            )
        else:
            provider = configured

    transcript = await provider.transcribe(
        TranscribeRequest(
            audio_path=stream.source_audio_path,
            stream_id=stream.id,
            tenant_id=tenant_id,
            language=language,
        )
    )
    out_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    return transcript


def load_transcript(stream_dir: Path) -> Transcript:
    """Load a previously-saved Transcript from `<stream_dir>/source/transcript.json`."""
    path = Path(stream_dir) / "source" / "transcript.json"
    if not path.exists():
        raise TranscriptionError(f"transcript not found at {path}")
    return Transcript.model_validate_json(path.read_text("utf-8"))


# Re-export for callers that imported these from service.py historically.
# (The implementations now live in providers/local_whisper.py; this is the
# minimal shim so we don't break in-flight imports.)
__all__ = ["load_transcript", "transcribe"]


# --- Provide get_provider in this namespace too. -----------------
# Defensive: anyone who imported `get_provider` from
# `nexoclip.transcribe.service` keeps working.
def __getattr__(name: str) -> object:
    if name == "get_provider":
        from .providers import get_provider as _gp
        return _gp
    raise AttributeError(name)
