"""Unit tests for the F.8 TranscribeProvider factory + base contract."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nexoclip.errors import NexoClipError, TranscriptionError
from nexoclip.settings import get_settings
from nexoclip.transcribe.providers import (
    CloudWhisperProvider,
    LocalWhisperProvider,
    TranscribeRequest,
    get_provider,
)


def _req() -> TranscribeRequest:
    return TranscribeRequest(
        audio_path=Path("/tmp/audio.wav"),  # noqa: S108 — never read in these tests
        stream_id="str_TEST",
        tenant_id="ten_TEST",
        language="es",
    )


# ---- get_provider factory ----


def test_factory_defaults_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEXOCLIP_TRANSCRIBE_PROVIDER", raising=False)
    get_settings.cache_clear()
    p = get_provider()
    assert isinstance(p, LocalWhisperProvider)
    assert "local-whisper" in p.name


def test_factory_picks_local_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEXOCLIP_TRANSCRIBE_PROVIDER", "local")
    get_settings.cache_clear()
    assert isinstance(get_provider(), LocalWhisperProvider)


def test_factory_threads_whisper_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local provider must honor NEXOCLIP_WHISPER_* env vars so operators
    can flip model size / device without code changes."""
    monkeypatch.setenv("NEXOCLIP_TRANSCRIBE_PROVIDER", "local")
    monkeypatch.setenv("NEXOCLIP_WHISPER_MODEL", "tiny")
    monkeypatch.setenv("NEXOCLIP_WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("NEXOCLIP_WHISPER_COMPUTE_TYPE", "int8")
    get_settings.cache_clear()
    p = get_provider()
    assert isinstance(p, LocalWhisperProvider)
    assert "tiny" in p.name and "cpu" in p.name


def test_factory_picks_assemblyai_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEXOCLIP_TRANSCRIBE_PROVIDER", "assemblyai")
    monkeypatch.setenv("NEXOCLIP_ASSEMBLYAI_API_KEY", "fake-key")
    get_settings.cache_clear()
    p = get_provider()
    assert isinstance(p, CloudWhisperProvider)
    assert p.name == "cloud-whisper-assemblyai"


def test_factory_rejects_cloud_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEXOCLIP_TRANSCRIBE_PROVIDER", "deepgram")
    monkeypatch.delenv("NEXOCLIP_DEEPGRAM_API_KEY", raising=False)
    get_settings.cache_clear()
    with pytest.raises(NexoClipError, match="requires NEXOCLIP_DEEPGRAM_API_KEY"):
        get_provider()


def test_factory_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEXOCLIP_TRANSCRIBE_PROVIDER", "whispersaurus")
    get_settings.cache_clear()
    with pytest.raises(NexoClipError, match="unknown transcribe_provider"):
        get_provider()


# ---- CloudWhisperProvider stub ----


def test_cloud_stub_raises_on_transcribe() -> None:
    """The stub must fail loud + actionable, never silently swallow a job."""
    p = CloudWhisperProvider(vendor="assemblyai", api_key="fake")
    with pytest.raises(TranscriptionError, match="is a stub"):
        asyncio.run(p.transcribe(_req()))
