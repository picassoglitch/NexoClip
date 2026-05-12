"""Whisper transcription via faster-whisper.

The transcript is saved alongside the audio at
`<stream_dir>/source/transcript.json`. Idempotent: a second call returns
the cached `Transcript` unless `force=True`.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel

from nexoclip.errors import TranscriptionError
from nexoclip.ingest import Stream

from .models import Segment, Transcript, Word

# Use subprocess isolation by default. When the worker hard-crashes
# (CUDA OOM, driver SIGSEGV, etc.) only the child dies; the dashboard
# parent process survives and writes a clean failed-step event. Set
# NEXOCLIP_TRANSCRIBE_INPROCESS=1 to fall back to the legacy in-process
# path (faster startup, but the dashboard goes down with a hard crash).
_USE_SUBPROCESS = os.environ.get("NEXOCLIP_TRANSCRIBE_INPROCESS", "").strip() != "1"


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

    runner = _run_whisper_subprocess if _USE_SUBPROCESS else _run_whisper
    transcript = await asyncio.to_thread(
        runner,
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


def _run_whisper_subprocess(
    *,
    audio_path: Path,
    stream_id: str,
    tenant_id: str,
    model_size: str,
    device: str,
    compute_type: str,
    language: str | None,
) -> Transcript:
    """Spawn `nexoclip.transcribe._worker` as a child process and read its JSON.

    Hard crashes (CUDA OOM SIGABRT, driver SIGSEGV) leave the child dead with
    a non-zero exit code and stderr full of context. The dashboard parent
    process stays up, the pipeline emits a clean failed-step event, and the
    user gets a real error message in the progress card instead of the dreaded
    'PS prompt comes back, no error' silent kill.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        out_path = Path(tmp.name)
    try:
        cmd = [
            sys.executable,
            "-m",
            "nexoclip.transcribe._worker",
            "--audio",
            str(audio_path),
            "--stream-id",
            stream_id,
            "--tenant-id",
            tenant_id,
            "--model",
            model_size,
            "--device",
            device,
            "--compute",
            compute_type,
            "--language",
            language or "",
            "--out",
            str(out_path),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            # No timeout — the pipeline-level timeout governs total runtime.
            # The worker will be killed by uvicorn shutdown if the parent dies.
        )
        if result.returncode != 0:
            tail = (result.stderr or "")[-1500:]
            raise TranscriptionError(
                f"Whisper worker exited with code {result.returncode}. "
                f"This usually means CUDA out-of-memory or a driver crash on "
                f"a long video — drop to NEXOCLIP_WHISPER_MODEL=base or set "
                f"NEXOCLIP_WHISPER_DEVICE=cpu in .env and re-run. "
                f"Worker stderr tail: {tail}"
            )
        if not out_path.exists():
            raise TranscriptionError(
                f"Whisper worker exited 0 but produced no output at {out_path}"
            )
        return Transcript.model_validate_json(out_path.read_text("utf-8"))
    finally:
        try:
            out_path.unlink()
        except OSError:
            pass


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
    """Blocking faster-whisper call, kept out of the event loop via `to_thread`.

    Tuned for long-VOD stability on consumer GPUs:
      * `vad_filter=True` — Silero VAD skips silence. On a 2-hour stream with
        natural pauses, this can cut work by 30-60% and dramatically reduces
        peak VRAM since silent regions never enter the decoder.
      * `condition_on_previous_text=False` — stops feeding previous segment
        text as context. Otherwise the prompt grows over the run, accumulating
        VRAM use and eventually OOMs on multi-hour videos.
      * `beam_size=1` (default 5) — greedy decoding. Slightly lower quality
        but ~5x less peak memory per batch.
    These three together are the canonical 'big-VOD survival kit' for
    Whisper on a 6-8GB consumer GPU.
    """
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
            beam_size=1,
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
