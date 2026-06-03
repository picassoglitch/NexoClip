"""AssemblyAI transcribe provider — Task A1.

We pin the request/response shape against mocked HTTP exchanges so the
karaoke-caption JSON stays bit-for-bit compatible with the existing
clip_render.html consumer. The mapper (`_to_transcript`) is the part
that matters most — the upload/submit/poll wiring is plumbing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from nexoclip.errors import TranscriptionError
from nexoclip.transcribe.models import Transcript
from nexoclip.transcribe.providers import assemblyai
from nexoclip.transcribe.providers.base import TranscribeRequest

# Capture before any monkeypatching so the deadline test (which needs a
# real micro-sleep to advance time.monotonic) doesn't recurse into the
# autouse no-op patch.
_REAL_SLEEP = asyncio.sleep
_REAL_ASYNC_CLIENT = httpx.AsyncClient


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop(_s: float) -> None:
        return None
    monkeypatch.setattr(assemblyai.asyncio, "sleep", _noop)


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    def _factory(*args, **kwargs):
        # Forward the provider's headers/timeout so the default
        # Authorization header reaches the MockTransport.
        kwargs.pop("transport", None)
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr(assemblyai.httpx, "AsyncClient", _factory)


def _make_provider(**overrides) -> assemblyai.AssemblyAIProvider:
    defaults = dict(
        api_key="aai_test_key",
        language_code="es",
        language_detection=False,
        speaker_labels=True,
        speech_model="best",
        polling_interval_s=0.01,
        request_timeout_s=2.0,
    )
    defaults.update(overrides)
    return assemblyai.AssemblyAIProvider(**defaults)


def _req(tmp_path: Path, *, language: str | None = None) -> TranscribeRequest:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFFFAKEWAVE")
    return TranscribeRequest(
        audio_path=audio,
        stream_id="str_test",
        tenant_id="ten_test",
        language=language,
    )


_COMPLETED_BODY = {
    "id": "tr_abc",
    "status": "completed",
    "language_code": "es",
    "audio_duration": 12.5,
    "text": "Hola mundo. Adiós.",
    "utterances": [
        {
            "start": 0,
            "end": 1500,
            "text": "Hola mundo.",
            "speaker": "A",
            "confidence": 0.95,
            "words": [
                {"text": "Hola", "start": 0, "end": 500, "confidence": 0.9, "speaker": "A"},
                {"text": "mundo.", "start": 500, "end": 1500, "confidence": 0.92, "speaker": "A"},
            ],
        },
        {
            "start": 2000,
            "end": 3500,
            "text": "Adiós.",
            "speaker": "B",
            "confidence": 0.91,
            "words": [
                {"text": "Adiós.", "start": 2000, "end": 3500, "confidence": 0.93, "speaker": "B"},
            ],
        },
    ],
}


def test_constructor_requires_api_key() -> None:
    with pytest.raises(TranscriptionError, match="api_key is required"):
        assemblyai.AssemblyAIProvider(api_key="")


def test_provider_name_reflects_model() -> None:
    p = _make_provider(speech_model="nano")
    assert p.name == "assemblyai-nano"


@pytest.mark.asyncio
async def test_happy_path_returns_transcript_with_speakers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full flow: upload → submit → poll once → completed body.
    Asserts the karaoke-relevant fields (segments[].words[].ts/end_ts/text)
    are populated correctly and speakers list dedupes."""
    state = {"phase": "upload"}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if state["phase"] == "upload":
            assert request.method == "POST"
            assert "/v2/upload" in url
            assert request.headers["authorization"] == "aai_test_key"
            state["phase"] = "submit"
            return httpx.Response(200, json={"upload_url": "https://cdn/audio.bin"})
        if state["phase"] == "submit":
            assert request.method == "POST"
            assert "/v2/transcript" in url
            body = request.read()
            import json as _json
            payload = _json.loads(body)
            assert payload["audio_url"] == "https://cdn/audio.bin"
            assert payload["speaker_labels"] is True
            assert payload["language_code"] == "es"
            assert "language_detection" not in payload
            state["phase"] = "poll"
            return httpx.Response(200, json={"id": "tr_abc", "status": "queued"})
        # poll
        assert request.method == "GET"
        assert "/v2/transcript/tr_abc" in url
        return httpx.Response(200, json=_COMPLETED_BODY)

    _patch_client(monkeypatch, handler)
    transcript = await _make_provider().transcribe(_req(tmp_path))
    assert isinstance(transcript, Transcript)
    assert transcript.language == "es"
    assert transcript.duration_s == pytest.approx(12.5)
    assert transcript.model == "assemblyai-best"
    assert transcript.speakers == ["A", "B"]

    assert len(transcript.segments) == 2
    s0 = transcript.segments[0]
    assert s0.ts == pytest.approx(0.0)
    assert s0.end_ts == pytest.approx(1.5)
    assert s0.text == "Hola mundo."
    assert s0.speaker == "A"
    assert [w.text for w in s0.words] == ["Hola", "mundo."]
    # Word timestamps in seconds (AssemblyAI returns ms — mapper converts).
    assert s0.words[1].ts == pytest.approx(0.5)
    assert s0.words[1].end_ts == pytest.approx(1.5)
    # Confidence preserved + clamped to [0,1].
    assert 0.0 <= s0.words[0].prob <= 1.0


@pytest.mark.asyncio
async def test_language_detection_omits_language_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When language_detection=True AND the request has no explicit
    language, AssemblyAI's payload must use language_detection instead
    of language_code (they're mutually exclusive in the API)."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if "/v2/upload" in str(request.url):
            return httpx.Response(200, json={"upload_url": "u"})
        if request.method == "POST":
            import json as _json
            captured.update(_json.loads(request.read()))
            return httpx.Response(200, json={"id": "tr_x", "status": "queued"})
        return httpx.Response(200, json=_COMPLETED_BODY)

    _patch_client(monkeypatch, handler)
    provider = _make_provider(language_code=None, language_detection=True)
    await provider.transcribe(_req(tmp_path))
    assert captured.get("language_detection") is True
    assert "language_code" not in captured


@pytest.mark.asyncio
async def test_request_language_overrides_provider_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The TranscribeRequest's language wins over the provider's
    constructor default — per-stream overrides work even when the
    provider was built with language=es."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if "/v2/upload" in str(request.url):
            return httpx.Response(200, json={"upload_url": "u"})
        if request.method == "POST":
            import json as _json
            captured.update(_json.loads(request.read()))
            return httpx.Response(200, json={"id": "tr_x", "status": "queued"})
        return httpx.Response(200, json=_COMPLETED_BODY)

    _patch_client(monkeypatch, handler)
    provider = _make_provider(language_code="es")
    await provider.transcribe(_req(tmp_path, language="en"))
    assert captured["language_code"] == "en"


@pytest.mark.asyncio
async def test_upload_failure_surfaces_as_transcription_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid key")

    _patch_client(monkeypatch, handler)
    with pytest.raises(TranscriptionError, match="upload failed.*401"):
        await _make_provider().transcribe(_req(tmp_path))


@pytest.mark.asyncio
async def test_error_status_surfaces_with_error_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AssemblyAI returns status=error with an error field when
    transcription fails server-side. Provider surfaces the message."""
    error_body = {"id": "tr_x", "status": "error", "error": "Audio too short"}

    def handler(request: httpx.Request) -> httpx.Response:
        if "/v2/upload" in str(request.url):
            return httpx.Response(200, json={"upload_url": "u"})
        if request.method == "POST":
            return httpx.Response(200, json={"id": "tr_x", "status": "queued"})
        return httpx.Response(200, json=error_body)

    _patch_client(monkeypatch, handler)
    with pytest.raises(TranscriptionError, match="Audio too short"):
        await _make_provider().transcribe(_req(tmp_path))


@pytest.mark.asyncio
async def test_missing_audio_file_raises_before_any_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Don't waste an API call when the file is missing locally —
    surface the FS error first."""
    p = _make_provider()
    req = TranscribeRequest(
        audio_path=tmp_path / "nope.wav",
        stream_id="str_x", tenant_id="ten_x", language=None,
    )
    with pytest.raises(TranscriptionError, match="audio file missing"):
        await p.transcribe(req)


@pytest.mark.asyncio
async def test_no_utterances_falls_back_to_sentence_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When speaker_labels is off (or AssemblyAI doesn't emit utterances
    for whatever reason), the mapper builds segments from words by
    sentence-ending punctuation. Caption pipeline still works."""
    body = {
        "id": "tr_x", "status": "completed",
        "language_code": "es", "audio_duration": 3.0,
        "utterances": [],  # explicitly empty
        "words": [
            {"text": "Hola", "start": 0, "end": 500, "confidence": 0.9},
            {"text": "mundo.", "start": 500, "end": 1500, "confidence": 0.9},
            {"text": "Adiós.", "start": 2000, "end": 3000, "confidence": 0.9},
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if "/v2/upload" in str(request.url):
            return httpx.Response(200, json={"upload_url": "u"})
        if request.method == "POST":
            return httpx.Response(200, json={"id": "tr_x", "status": "queued"})
        return httpx.Response(200, json=body)

    _patch_client(monkeypatch, handler)
    transcript = await _make_provider(speaker_labels=False).transcribe(_req(tmp_path))
    assert len(transcript.segments) == 2
    assert transcript.segments[0].text == "Hola mundo."
    assert transcript.segments[1].text == "Adiós."
    # No speaker info → field is None, speakers list empty.
    assert all(s.speaker is None for s in transcript.segments)
    assert transcript.speakers == []


@pytest.mark.asyncio
async def test_poll_respects_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stuck AssemblyAI job (queued forever) bails after deadline_s."""
    real_calls = {"n": 0}

    async def _short_sleep(_s: float) -> None:
        real_calls["n"] += 1
        await _REAL_SLEEP(0.001)
    monkeypatch.setattr(assemblyai.asyncio, "sleep", _short_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        if "/v2/upload" in str(request.url):
            return httpx.Response(200, json={"upload_url": "u"})
        if request.method == "POST":
            return httpx.Response(200, json={"id": "tr_stuck", "status": "queued"})
        return httpx.Response(200, json={"id": "tr_stuck", "status": "queued"})

    _patch_client(monkeypatch, handler)
    provider = _make_provider(request_timeout_s=0.05, polling_interval_s=0.01)
    with pytest.raises(TranscriptionError, match="poll deadline"):
        await provider.transcribe(_req(tmp_path))


def test_mapper_handles_missing_fields_gracefully() -> None:
    """Direct unit-test of the mapper — a malformed AAI response (e.g.
    missing duration, garbage timestamps) shouldn't crash; we
    coerce to safe defaults."""
    minimal = {
        "id": "tr_x", "status": "completed",
        "language_code": "es",
        "utterances": [{"start": 0, "end": 1000, "text": "hi", "speaker": "A", "words": []}],
    }
    req = TranscribeRequest(
        audio_path=Path("/dev/null"), stream_id="s", tenant_id="t",
    )
    out = assemblyai._to_transcript(minimal, req=req, model_name="assemblyai-best")
    assert out.duration_s == 0.0  # missing audio_duration → 0
    assert len(out.segments) == 1
    assert out.segments[0].speaker == "A"
