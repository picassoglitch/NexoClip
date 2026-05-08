"""Tests for `ingest_vod` with the network/ffmpeg layers stubbed."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from nexoclip.ingest import Stream, ingest_vod
from nexoclip.ingest import service as ingest_service


def _stub_download_writes_file(monkeypatch: pytest.MonkeyPatch, *, info: dict[str, Any]) -> list[Path]:
    """Replace `_download_vod` with a no-network stub that creates the target file."""
    calls: list[Path] = []

    def fake_download(
        *,
        vod_url: str,
        target_path: Path,
        cookies_from_browser: str | None = None,
        platform: str = "unknown",
    ) -> dict[str, Any]:
        calls.append(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"\x00\x00fakevideo")
        return info

    monkeypatch.setattr(ingest_service, "_download_vod", fake_download)
    return calls


def _stub_audio_extract(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, Path]]:
    calls: list[tuple[Path, Path]] = []

    def fake_extract(video_path: Path, audio_path: Path) -> None:
        calls.append((video_path, audio_path))
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"RIFFfakewav")

    monkeypatch.setattr(ingest_service, "_extract_audio", fake_extract)
    return calls


def _stub_ffprobe(monkeypatch: pytest.MonkeyPatch, *, duration_s: float) -> None:
    monkeypatch.setattr(ingest_service, "_ffprobe_duration", lambda _p: duration_s)


def test_ingest_creates_layout_and_returns_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = {"duration": 612.5, "title": "test stream", "uploader": "aldovillanueva"}
    download_calls = _stub_download_writes_file(monkeypatch, info=info)
    extract_calls = _stub_audio_extract(monkeypatch)

    stream = asyncio.run(
        ingest_vod(
            tenant_id="default",
            vod_url="https://kick.com/aldovillanueva/videos/abc",
            output_dir=tmp_path,
        )
    )

    assert isinstance(stream, Stream)
    assert stream.id.startswith("str_")
    assert stream.tenant_id == "default"
    assert stream.platform == "kick"
    assert stream.title == "test stream"
    assert stream.channel == "aldovillanueva"
    assert stream.duration_s == pytest.approx(612.5)

    stream_dir = tmp_path / stream.id
    assert (stream_dir / "stream.json").exists()
    assert (stream_dir / "source" / "video.mp4").exists()
    assert (stream_dir / "source" / "audio.wav").exists()

    assert len(download_calls) == 1
    assert download_calls[0].name == "video.mp4"
    assert len(extract_calls) == 1


def test_ingest_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = {"duration": 30.0, "title": "t", "uploader": "u"}
    download_calls = _stub_download_writes_file(monkeypatch, info=info)
    extract_calls = _stub_audio_extract(monkeypatch)

    first = asyncio.run(
        ingest_vod(
            tenant_id="default",
            vod_url="https://kick.com/c/videos/1",
            output_dir=tmp_path,
        )
    )
    second = asyncio.run(
        ingest_vod(
            tenant_id="default",
            vod_url="https://kick.com/c/videos/1",
            output_dir=tmp_path,
            stream_id=first.id,
        )
    )

    assert second == first
    assert len(download_calls) == 1
    assert len(extract_calls) == 1


def test_ingest_force_redownloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = {"duration": 30.0, "title": "t", "uploader": "u"}
    download_calls = _stub_download_writes_file(monkeypatch, info=info)
    extract_calls = _stub_audio_extract(monkeypatch)

    first = asyncio.run(
        ingest_vod(
            tenant_id="default",
            vod_url="https://kick.com/c/videos/1",
            output_dir=tmp_path,
        )
    )
    asyncio.run(
        ingest_vod(
            tenant_id="default",
            vod_url="https://kick.com/c/videos/1",
            output_dir=tmp_path,
            stream_id=first.id,
            force=True,
        )
    )

    assert len(download_calls) == 2
    assert len(extract_calls) == 2


def test_ingest_falls_back_to_ffprobe_for_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = {"duration": 0, "title": "no duration in info", "uploader": "u"}
    _stub_download_writes_file(monkeypatch, info=info)
    _stub_audio_extract(monkeypatch)
    _stub_ffprobe(monkeypatch, duration_s=42.0)

    stream = asyncio.run(
        ingest_vod(
            tenant_id="default",
            vod_url="https://twitch.tv/videos/1",
            output_dir=tmp_path,
        )
    )

    assert stream.duration_s == pytest.approx(42.0)
    assert stream.platform == "twitch"
