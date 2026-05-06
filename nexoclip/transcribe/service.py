"""Whisper transcription via faster-whisper.

The transcript is saved alongside the audio at
`<stream_dir>/source/transcript.json`. Idempotent: a second call returns
the cached `Transcript` unless `force=True`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from faster_whisper import WhisperModel

from nexoclip.errors import TranscriptionError
from nexoclip.ingest import Stream

from .models import Segment, Transcript, Word


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
) -> Transcript:
    """Run Whisper on `stream.source_audio_path` and save `transcript.json`.

    Args:
        tenant_id: Tenant owning the stream.
        stream: The ingested stream.
        model_size: faster-whisper model name (`tiny`, `base`, ..., `large-v3`).
        device: `cuda` or `cpu`.
        compute_type: `float16`, `int8_float16`, `int8`, etc.
        language: ISO 639-1 code; pass `None` to auto-detect.
        force: Re-run even when `transcript.json` already exists.
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

    transcript = await asyncio.to_thread(
        _run_whisper,
        audio_path=stream.source_audio_path,
        stream_id=stream.id,
        tenant_id=tenant_id,
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        language=language,
    )
    out_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    return transcript


def _run_whisper(
    *,
    audio_path: Path,
    stream_id: str,
    tenant_id: str,
    model_size: str,
    device: str,
    compute_type: str,
    language: str | None,
) -> Transcript:
    """Blocking faster-whisper call, kept out of the event loop via `to_thread`."""
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
        )
    except Exception as e:
        raise TranscriptionError(
            f"Whisper failed to start ({model_size}/{device}/{compute_type}): {e}"
        ) from e

    segments: list[Segment] = []
    try:
        for seg in segments_iter:
            words = [
                Word(
                    ts=float(w.start),
                    end_ts=float(w.end),
                    text=w.word,
                    prob=float(w.probability),
                )
                for w in (seg.words or [])
            ]
            segments.append(
                Segment(
                    ts=float(seg.start),
                    end_ts=float(seg.end),
                    text=seg.text,
                    words=words,
                )
            )
    except Exception as e:
        raise TranscriptionError(f"Whisper segment iteration failed: {e}") from e

    return Transcript(
        stream_id=stream_id,
        tenant_id=tenant_id,
        language=info.language,
        duration_s=float(info.duration),
        model=model_size,
        segments=segments,
    )


def load_transcript(stream_dir: Path) -> Transcript:
    """Load a previously-saved Transcript from `<stream_dir>/source/transcript.json`."""
    path = Path(stream_dir) / "source" / "transcript.json"
    if not path.exists():
        raise TranscriptionError(f"transcript not found at {path}")
    return Transcript.model_validate_json(path.read_text("utf-8"))
