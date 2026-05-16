"""LocalWhisperProvider — faster-whisper on the local GPU/CPU.

Wraps the existing `_run_whisper` and `_run_whisper_subprocess` paths
from `nexoclip.transcribe.service` so behavior is unchanged for
operators running NexoClip on their own hardware.

For Stage 1 of the nexo-ai deployment plan this is what runs both
locally (dev laptop with RTX 4060) AND on Modal (when wrapped in a
Modal `@function(gpu="A10G")` — Modal mounts pin the model weights
to a persistent volume so first-run cold-start is bounded).

The subprocess path (`use_subprocess=True`, default) isolates Whisper
in a child process so CUDA OOM / driver crashes don't take down the
dashboard. The in-process path stays available for tests that don't
want the subprocess overhead.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

from nexoclip.errors import TranscriptionError
from nexoclip.transcribe.models import Segment, Transcript, Word

from .base import TranscribeRequest


class LocalWhisperProvider:
    """faster-whisper, called from the local process."""

    def __init__(
        self,
        *,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
        use_subprocess: bool = True,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._use_subprocess = use_subprocess

    @property
    def name(self) -> str:
        return f"local-whisper-{self._model_size}-{self._device}"

    async def transcribe(self, req: TranscribeRequest) -> Transcript:
        runner = self._run_subprocess if self._use_subprocess else self._run_inprocess
        return await asyncio.to_thread(
            runner,
            audio_path=req.audio_path,
            stream_id=req.stream_id,
            tenant_id=req.tenant_id,
            language=req.language,
        )

    # ---- subprocess path ----

    def _run_subprocess(
        self,
        *,
        audio_path: Path,
        stream_id: str,
        tenant_id: str,
        language: str | None,
    ) -> Transcript:
        """Spawn `nexoclip.transcribe._worker` so a CUDA OOM doesn't take
        down the parent. Same protocol as the legacy `_run_whisper_subprocess`.
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
                "--audio", str(audio_path),
                "--stream-id", stream_id,
                "--tenant-id", tenant_id,
                "--model", self._model_size,
                "--device", self._device,
                "--compute", self._compute_type,
                "--language", language or "",
                "--out", str(out_path),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False
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

    # ---- in-process path ----

    def _run_inprocess(
        self,
        *,
        audio_path: Path,
        stream_id: str,
        tenant_id: str,
        language: str | None,
    ) -> Transcript:
        """Blocking faster-whisper call, kept off the event loop via
        `asyncio.to_thread` at the caller.

        Tuned for long-VOD stability on consumer GPUs:
          * `vad_filter=True` — Silero VAD skips silence.
          * `condition_on_previous_text=False` — bounds VRAM growth.
          * `beam_size=1` — greedy decode, ~5× less peak memory.
        """
        from faster_whisper import WhisperModel  # lazy — heavy import

        try:
            model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
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
                f"Whisper failed to start "
                f"({self._model_size}/{self._device}/{self._compute_type}): {e}"
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
            model=self._model_size,
            segments=segments,
        )
