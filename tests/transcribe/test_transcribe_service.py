"""Tests for `transcribe()` with WhisperModel stubbed out."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nexoclip.errors import TranscriptionError
from nexoclip.ingest import Stream
from nexoclip.transcribe import Transcript, transcribe
from nexoclip.transcribe import service as transcribe_service

from ._fakes import FakeInfo, FakeSegment, FakeWhisperModel, FakeWord


def _make_stream(tmp_path: Path, *, tenant_id: str = "default") -> Stream:
    """Build a Stream pointing at fake source files inside `tmp_path`."""
    stream_id = "str_01TEST"
    stream_dir = tmp_path / stream_id
    source_dir = stream_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    video_path = source_dir / "video.mp4"
    audio_path = source_dir / "audio.wav"
    video_path.write_bytes(b"v")
    audio_path.write_bytes(b"a")
    return Stream(
        id=stream_id,
        tenant_id=tenant_id,
        vod_url="https://kick.com/c/videos/1",
        platform="kick",
        title="t",
        channel="c",
        duration_s=120.0,
        source_video_path=video_path,
        source_audio_path=audio_path,
    )


@pytest.fixture(autouse=True)
def _patch_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeWhisperModel.reset()
    monkeypatch.setattr(transcribe_service, "WhisperModel", FakeWhisperModel)
    FakeWhisperModel.canned_info = FakeInfo(language="es", duration=10.5)
    FakeWhisperModel.canned_segments = [
        FakeSegment(
            start=0.0,
            end=2.5,
            text="hola mundo",
            words=[
                FakeWord(start=0.0, end=0.5, word="hola", probability=0.95),
                FakeWord(start=0.6, end=1.2, word="mundo", probability=0.91),
            ],
        ),
        FakeSegment(
            start=3.0,
            end=4.0,
            text="clipéalo",
            words=[FakeWord(start=3.0, end=4.0, word="clipéalo", probability=0.88)],
        ),
    ]


def test_transcribe_runs_and_writes_json(tmp_path: Path) -> None:
    stream = _make_stream(tmp_path)
    transcript = asyncio.run(
        transcribe(
            tenant_id="default",
            stream=stream,
            model_size="medium",
            device="cuda",
            compute_type="float16",
            language="es",
        )
    )

    assert isinstance(transcript, Transcript)
    assert transcript.stream_id == stream.id
    assert transcript.tenant_id == "default"
    assert transcript.language == "es"
    assert transcript.duration_s == pytest.approx(10.5)
    assert transcript.model == "medium"
    assert len(transcript.segments) == 2
    assert transcript.segments[0].words[0].text == "hola"
    assert transcript.segments[0].words[0].prob == pytest.approx(0.95)

    json_path = stream.source_audio_path.parent / "transcript.json"
    assert json_path.exists()
    reloaded = Transcript.model_validate_json(json_path.read_text("utf-8"))
    assert reloaded == transcript

    # Verify whisper was called with the parameters we passed.
    assert len(FakeWhisperModel.constructor_calls) == 1
    ctor_args, ctor_kwargs = FakeWhisperModel.constructor_calls[0]
    assert ctor_args == ("medium",)
    assert ctor_kwargs == {"device": "cuda", "compute_type": "float16"}
    transcribe_args, transcribe_kwargs = FakeWhisperModel.transcribe_calls[0]
    assert transcribe_args == (str(stream.source_audio_path),)
    assert transcribe_kwargs == {"language": "es", "word_timestamps": True}


def test_transcribe_is_idempotent(tmp_path: Path) -> None:
    stream = _make_stream(tmp_path)
    first = asyncio.run(transcribe(tenant_id="default", stream=stream))
    second = asyncio.run(transcribe(tenant_id="default", stream=stream))

    assert second == first
    # Whisper should have been instantiated only on the first call.
    assert len(FakeWhisperModel.constructor_calls) == 1
    assert len(FakeWhisperModel.transcribe_calls) == 1


def test_transcribe_force_reruns(tmp_path: Path) -> None:
    stream = _make_stream(tmp_path)
    asyncio.run(transcribe(tenant_id="default", stream=stream))
    asyncio.run(transcribe(tenant_id="default", stream=stream, force=True))

    assert len(FakeWhisperModel.constructor_calls) == 2
    assert len(FakeWhisperModel.transcribe_calls) == 2


def test_transcribe_rejects_tenant_mismatch(tmp_path: Path) -> None:
    stream = _make_stream(tmp_path, tenant_id="alice")
    with pytest.raises(TranscriptionError, match="tenant mismatch"):
        asyncio.run(transcribe(tenant_id="bob", stream=stream))


def test_transcribe_rejects_missing_audio(tmp_path: Path) -> None:
    stream = _make_stream(tmp_path)
    stream.source_audio_path.unlink()
    with pytest.raises(TranscriptionError, match="audio file missing"):
        asyncio.run(transcribe(tenant_id="default", stream=stream))


def test_transcribe_auto_language_passes_none(tmp_path: Path) -> None:
    stream = _make_stream(tmp_path)
    asyncio.run(transcribe(tenant_id="default", stream=stream, language=None))
    _, kwargs = FakeWhisperModel.transcribe_calls[0]
    assert kwargs["language"] is None


def test_transcribe_wraps_whisper_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = _make_stream(tmp_path)

    class BoomModel:
        def __init__(self, *_a: object, **_k: object) -> None:
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(transcribe_service, "WhisperModel", BoomModel)
    with pytest.raises(TranscriptionError, match="Whisper failed to start"):
        asyncio.run(transcribe(tenant_id="default", stream=stream))
