"""AssemblyAI transcribe provider — Task A1.

Replaces the GPU-bound `faster-whisper` + `pyannote` stack with a single
cloud API call. The pipeline can now run on CPU-only hosts (Railway,
fly.io, any Docker image without CUDA).

Flow:
  1. POST /v2/upload      — stream the audio file → AssemblyAI upload URL
  2. POST /v2/transcript  — submit with speaker_labels + language + words
  3. GET  /v2/transcript/{id} — poll until status="completed"
  4. Map words[] / utterances[] → our Transcript model

Diarization arrives free: `speaker_labels=true` flags each utterance
with a per-video label (A / B / C). Cross-video persistent identity is
deferred (see TODO marker in nexoclip/diarize/__init__.py); for the v1
migration we use AssemblyAI's per-video labels only.

Cost model (cached for reference, NOT computed here — Task A1b wires
this into the budget governor):
    base transcription:  $0.15 / hr
    + speaker_labels:    $0.02 / hr
    Total:              ~$0.17 / hr of audio
On a typical 1-hour VOD that's $0.17 vs Modal's ~$0.40-0.60 GPU minute
math, AND no GPU dependency to keep alive.

Caption fidelity guard: the existing karaoke `<div id="pv-caption">`
renderer reads `segments[].words[].ts/end_ts/text` from the saved
transcript. AssemblyAI emits words in milliseconds; we convert to
seconds in the mapper so the burn-in stays bit-for-bit identical
to the libwhisper output shape.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
import structlog

from nexoclip.errors import TranscriptionError
from nexoclip.transcribe.models import Segment, Transcript, Word

from .base import TranscribeRequest

_log = structlog.get_logger(__name__)

# AssemblyAI v2 base. Hardcoded — vendor URL is stable, no need to make
# this configurable (and exposing it as a knob would invite "point this
# at our staging" foot-guns).
_API_BASE = "https://api.assemblyai.com/v2"

# Poll cadence for the transcript-status endpoint. AssemblyAI's batch
# API typically completes a 1-hour VOD in 30-90s; 3s gives us 10-30
# polls per run. Faster cadence wastes API quota for no latency win.
_POLL_INTERVAL_S = 3.0

# Status strings the poll loop recognizes.
_STATUS_QUEUED = "queued"
_STATUS_PROCESSING = "processing"
_STATUS_COMPLETED = "completed"
_STATUS_ERROR = "error"

# Upload chunk size. 5 MB is the sweet spot — large enough that
# Python's HTTPS overhead isn't dominant, small enough that a 4-hour
# 4 GB VOD doesn't pin all of it in memory.
_UPLOAD_CHUNK_BYTES = 5 * 1024 * 1024


class AssemblyAIProvider:
    """Submits audio to AssemblyAI and returns a Transcript.

    Constructor reads its config from explicit kwargs — the factory
    pulls the values from Settings. Keep this class env-agnostic so
    tests can spin one up without monkeypatching Settings.
    """

    def __init__(
        self,
        *,
        api_key: str,
        language_code: str | None = "es",
        language_detection: bool = False,
        speaker_labels: bool = True,
        speech_model: str = "best",
        polling_interval_s: float = _POLL_INTERVAL_S,
        request_timeout_s: float = 3600.0,
    ) -> None:
        if not api_key:
            raise TranscriptionError(
                "AssemblyAIProvider misconfigured: api_key is required."
            )
        self._api_key = api_key
        # `None` means "let AssemblyAI auto-detect" (when language_detection
        # is also True). Otherwise honor the explicit code.
        self._language_code = language_code
        self._language_detection = language_detection
        self._speaker_labels = speaker_labels
        self._speech_model = speech_model
        self._polling_interval_s = polling_interval_s
        self._timeout_s = request_timeout_s

    @property
    def name(self) -> str:
        return f"assemblyai-{self._speech_model}"

    def cost_for_duration_micros(self, duration_s: float) -> int:
        """USD micros for a transcript of `duration_s` seconds at the
        current settings.

        Computed from the full duration rather than a per-second rate
        × seconds so rounding doesn't slowly drift the operator's
        invoice off the published $/hr math. Providers without a
        marginal cost (LocalWhisperProvider) don't define this
        method; the service's hasattr check falls through to a no-op.
        """
        return cost_micros_for(
            duration_s=duration_s, speaker_labels=self._speaker_labels,
        )

    # ---- public entry ----

    async def transcribe(self, req: TranscribeRequest) -> Transcript:
        """Upload audio → submit → poll → map. Raises TranscriptionError
        on any failure with the AssemblyAI-side message attached."""
        if not req.audio_path.exists():
            raise TranscriptionError(
                f"audio file missing: {req.audio_path}"
            )

        async with httpx.AsyncClient(
            timeout=self._timeout_s, headers={"authorization": self._api_key}
        ) as client:
            _log.info(
                "assemblyai.upload.start",
                stream_id=req.stream_id, tenant_id=req.tenant_id,
                audio_path=str(req.audio_path),
                file_bytes=req.audio_path.stat().st_size,
            )
            upload_url = await self._upload_audio(client, req.audio_path)
            _log.info(
                "assemblyai.submit",
                stream_id=req.stream_id, language=self._language_code,
                speaker_labels=self._speaker_labels,
                language_detection=self._language_detection,
            )
            transcript_id = await self._submit_transcript(
                client, upload_url=upload_url, language=req.language,
            )
            raw = await self._poll_until_done(
                client, transcript_id=transcript_id,
                stream_id=req.stream_id,
            )

        return _to_transcript(raw, req=req, model_name=self.name)

    # ---- internals ----

    async def _upload_audio(
        self, client: httpx.AsyncClient, audio_path: Path,
    ) -> str:
        """Stream the audio to /v2/upload. Returns the resulting upload
        URL that the submit step references.

        We iterate file chunks rather than reading the whole thing into
        a buffer — a 4 GB ingested VOD's audio.wav would be ~600 MB at
        16 kHz mono PCM and pinning that in RAM on a 512 MB Railway
        instance gets us OOM-killed.

        The iterator is `async` because httpx's AsyncClient requires it
        — a sync generator triggers a runtime mismatch when MockTransport
        sees an async client + sync content. We offload the blocking
        `file.read()` to a worker thread so the event loop stays
        responsive during the upload.
        """
        async def _iter_chunks():
            # Open in the calling task; the actual read() happens in
            # a worker thread so this stays cooperative.
            handle = await asyncio.to_thread(audio_path.open, "rb")
            try:
                while True:
                    chunk = await asyncio.to_thread(
                        handle.read, _UPLOAD_CHUNK_BYTES,
                    )
                    if not chunk:
                        break
                    yield chunk
            finally:
                handle.close()

        try:
            resp = await client.post(
                f"{_API_BASE}/upload",
                content=_iter_chunks(),
                # AssemblyAI accepts the raw audio bytes; setting
                # content-type=application/octet-stream is required
                # per their docs.
                headers={"content-type": "application/octet-stream"},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise TranscriptionError(
                f"AssemblyAI upload failed ({e.response.status_code}): "
                f"{e.response.text[:300]}"
            ) from e
        except httpx.HTTPError as e:
            raise TranscriptionError(
                f"AssemblyAI upload transport error: {e}"
            ) from e

        body = resp.json()
        upload_url = body.get("upload_url")
        if not upload_url:
            raise TranscriptionError(
                f"AssemblyAI upload returned no upload_url; got: {body!r}"
            )
        return upload_url

    async def _submit_transcript(
        self,
        client: httpx.AsyncClient,
        *,
        upload_url: str,
        language: str | None,
    ) -> str:
        """Submit the transcript request and return its id."""
        payload: dict[str, Any] = {
            "audio_url": upload_url,
            "speaker_labels": self._speaker_labels,
            "speech_model": self._speech_model,
            # Always request word-level timestamps — the karaoke caption
            # renderer needs them. Mid-pipeline cost is rounding error.
            "word_boost": [],
        }
        # Language vs detection are mutually exclusive in the API:
        # `language_detection=true` requires omitting `language_code`.
        # Request-level `language` overrides the provider default so
        # per-stream overrides work.
        effective_lang = language if language is not None else self._language_code
        if self._language_detection and effective_lang is None:
            payload["language_detection"] = True
        else:
            payload["language_code"] = effective_lang or "es"

        try:
            resp = await client.post(f"{_API_BASE}/transcript", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise TranscriptionError(
                f"AssemblyAI submit failed ({e.response.status_code}): "
                f"{e.response.text[:300]}"
            ) from e
        except httpx.HTTPError as e:
            raise TranscriptionError(
                f"AssemblyAI submit transport error: {e}"
            ) from e

        body = resp.json()
        transcript_id = body.get("id")
        if not transcript_id:
            raise TranscriptionError(
                f"AssemblyAI submit returned no id; got: {body!r}"
            )
        return transcript_id

    async def _poll_until_done(
        self,
        client: httpx.AsyncClient,
        *,
        transcript_id: str,
        stream_id: str,
    ) -> dict[str, Any]:
        """GET /v2/transcript/{id} every poll_interval_s until status
        is `completed` or `error`. Raises TranscriptionError on the
        error path or on deadline expiry."""
        deadline = time.monotonic() + max(0.0, self._timeout_s)
        url = f"{_API_BASE}/transcript/{transcript_id}"
        polls = 0
        while True:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise TranscriptionError(
                    f"AssemblyAI poll failed ({e.response.status_code}): "
                    f"{e.response.text[:300]}"
                ) from e
            except httpx.HTTPError as e:
                raise TranscriptionError(
                    f"AssemblyAI poll transport error: {e}"
                ) from e

            body = resp.json()
            status = body.get("status")
            if status == _STATUS_COMPLETED:
                _log.info(
                    "assemblyai.poll.complete",
                    stream_id=stream_id, transcript_id=transcript_id,
                    polls=polls,
                )
                return body
            if status == _STATUS_ERROR:
                err = body.get("error") or "unknown"
                raise TranscriptionError(
                    f"AssemblyAI returned status=error for transcript "
                    f"{transcript_id}: {err}"
                )
            if status not in (_STATUS_QUEUED, _STATUS_PROCESSING):
                raise TranscriptionError(
                    f"AssemblyAI returned unexpected status "
                    f"{status!r} for transcript {transcript_id}"
                )

            polls += 1
            if polls == 1 or polls % 10 == 0:
                _log.info(
                    "assemblyai.poll",
                    stream_id=stream_id, transcript_id=transcript_id,
                    poll_n=polls, status=status,
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TranscriptionError(
                    f"AssemblyAI poll deadline reached after {polls} polls "
                    f"(deadline={self._timeout_s:.0f}s) for transcript "
                    f"{transcript_id}"
                )
            await asyncio.sleep(min(self._polling_interval_s, remaining))


# ---- mapping ----


def _to_transcript(
    raw: dict[str, Any],
    *,
    req: TranscribeRequest,
    model_name: str,
) -> Transcript:
    """Convert the AssemblyAI completed-transcript JSON into our
    Transcript model.

    AssemblyAI ts are in milliseconds; we convert to seconds. When
    `utterances[]` is present (speaker_labels=true) each utterance
    becomes one Segment with speaker tagged. When absent, we fall back
    to one Segment per logical chunk derived from words.
    """
    try:
        duration_ms = float(raw.get("audio_duration") or 0.0)
        # AssemblyAI's `audio_duration` is in SECONDS (despite words
        # being in ms) — yes, their API is inconsistent. The schema
        # says "Total duration of audio file in seconds".
        duration_s = duration_ms  # already seconds

        language = str(
            raw.get("language_code") or req.language or "es"
        )

        utterances = raw.get("utterances") or []
        words_top = raw.get("words") or []

        segments: list[Segment] = []
        speakers_seen: list[str] = []

        if utterances:
            for utt in utterances:
                speaker_label = _normalize_speaker(utt.get("speaker"))
                if speaker_label and speaker_label not in speakers_seen:
                    speakers_seen.append(speaker_label)
                seg_words = [
                    _word_from_assemblyai(w) for w in utt.get("words") or []
                ]
                segments.append(
                    Segment(
                        ts=_ms_to_s(utt.get("start", 0)),
                        end_ts=_ms_to_s(utt.get("end", 0)),
                        text=str(utt.get("text") or "").strip(),
                        words=seg_words,
                        speaker=speaker_label,
                    )
                )
        elif words_top:
            # No utterances — degrade to one Segment per sentence
            # boundary inferred from word.text ending in . ? !. Keeps
            # the downstream karaoke chunker happy without forcing
            # speaker labels.
            current: list[Word] = []
            for w in words_top:
                current.append(_word_from_assemblyai(w))
                if _word_ends_sentence(w):
                    segments.append(_segment_from_words(current))
                    current = []
            if current:
                segments.append(_segment_from_words(current))

        return Transcript(
            stream_id=req.stream_id,
            tenant_id=req.tenant_id,
            language=language,
            duration_s=duration_s,
            model=model_name,
            segments=segments,
            speakers=speakers_seen,
        )
    except TranscriptionError:
        raise
    except Exception as e:  # noqa: BLE001
        raise TranscriptionError(
            f"AssemblyAI response unparseable: {e}; "
            f"raw keys: {sorted(raw.keys())[:8]}"
        ) from e


def _ms_to_s(value: Any) -> float:
    """AssemblyAI returns word start/end in milliseconds. Defensive
    cast — None / missing → 0.0, never raise."""
    try:
        return max(0.0, float(value) / 1000.0)
    except (TypeError, ValueError):
        return 0.0


def _normalize_speaker(speaker: Any) -> str | None:
    """AssemblyAI emits speakers as 'A', 'B', etc. Some test fixtures
    pass None. Pass through verbatim — we don't try to translate to
    SPEAKER_00 style here because downstream code is already speaker-
    label-agnostic (it just uses the value as a dict key)."""
    if speaker is None:
        return None
    s = str(speaker).strip()
    return s or None


def _word_from_assemblyai(w: dict[str, Any]) -> Word:
    """Map one AssemblyAI word record to our Word.

    Confidence is already a 0-1 float on AssemblyAI's side; clamp
    defensively in case a vendor quirk pushes it slightly off."""
    confidence = float(w.get("confidence", 0.0) or 0.0)
    return Word(
        ts=_ms_to_s(w.get("start")),
        end_ts=_ms_to_s(w.get("end")),
        text=str(w.get("text") or ""),
        prob=max(0.0, min(1.0, confidence)),
    )


# ---- cost ----
#
# AssemblyAI pricing — Task A1b. Numbers from the public pricing page
# at https://www.assemblyai.com/pricing as of 2026-06. If they change,
# operators can override via the per-tenant LLM cap (budget governor)
# or by setting NEXOCLIP_ASSEMBLYAI_SPEECH_MODEL=nano which is ~5×
# cheaper on the base rate.
_COST_BASE_USD_PER_SECOND = 0.15 / 3600.0
_COST_DIARIZATION_USD_PER_SECOND = 0.02 / 3600.0


def cost_micros_for(*, duration_s: float, speaker_labels: bool) -> int:
    """USD micros for a transcript of `duration_s` seconds, accounting
    for the +$0.02/hr diarization upcharge when speaker_labels is on.

    Reused by the transcribe service's post-call accounting and by any
    pre-call estimator that wants to show the operator a quote before
    they spend credits.
    """
    rate = _COST_BASE_USD_PER_SECOND
    if speaker_labels:
        rate += _COST_DIARIZATION_USD_PER_SECOND
    return int(round(max(0.0, duration_s) * rate * 1_000_000))


_SENTENCE_END = {".", "?", "!"}


def _word_ends_sentence(w: dict[str, Any]) -> bool:
    """Crude sentence-boundary check for the no-utterances fallback."""
    text = str(w.get("text") or "").strip()
    return bool(text) and text[-1] in _SENTENCE_END


def _segment_from_words(words: list[Word]) -> Segment:
    """Bundle a list of Words into one Segment. Used by the fallback
    when AssemblyAI doesn't return utterances."""
    if not words:
        return Segment(ts=0.0, end_ts=0.0, text="", words=[])
    return Segment(
        ts=words[0].ts,
        end_ts=words[-1].end_ts,
        text=" ".join(w.text for w in words).strip(),
        words=words,
    )


__all__ = ["AssemblyAIProvider", "cost_micros_for"]
