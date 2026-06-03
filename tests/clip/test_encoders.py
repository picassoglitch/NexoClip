"""Encoder selection — libx264 vs h264_nvenc (Task 1b)."""

from __future__ import annotations

import pytest

from nexoclip.clip import encoders
from nexoclip.config import ClipConfig


@pytest.fixture(autouse=True)
def _clear_nvenc_cache() -> None:
    """Reset the cached probe so each test starts from a clean slate."""
    encoders.reset_nvenc_cache()
    yield
    encoders.reset_nvenc_cache()


def test_libx264_args_when_nvenc_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(encoders, "has_nvenc", lambda: False)
    cfg = ClipConfig()
    args = encoders.pick_video_encoder_args(cfg)
    # Matches the O.49 quality bump (preset=fast, crf=19).
    assert args == ["-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf)]


def test_nvenc_args_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default config uses the libx264 CRF baseline mirrored on the
    NVENC CQ knob so the two encoders look the same to the viewer."""
    monkeypatch.setattr(encoders, "has_nvenc", lambda: True)
    cfg = ClipConfig()
    args = encoders.pick_video_encoder_args(cfg)
    assert args == [
        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-rc", "vbr",
        "-cq", str(cfg.nvenc_cq),
        "-b:v", "0",
    ]


def test_nvenc_disabled_via_config_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`prefer_nvenc=False` is the operator escape hatch — restore the
    libx264 path even when NVENC is present on the host."""
    monkeypatch.setattr(encoders, "has_nvenc", lambda: True)
    cfg = ClipConfig(prefer_nvenc=False)
    args = encoders.pick_video_encoder_args(cfg)
    assert args[0:2] == ["-c:v", "libx264"]


def test_explicit_encoder_override_wins_over_nvenc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the operator picks libx265 (or any non-default codec)
    explicitly, NVENC is ignored — they asked for that codec by name."""
    monkeypatch.setattr(encoders, "has_nvenc", lambda: True)
    cfg = ClipConfig(encoder="libx265", preset="medium", crf=20)
    args = encoders.pick_video_encoder_args(cfg)
    assert args == ["-c:v", "libx265", "-preset", "medium", "-crf", "20"]


def test_nvenc_cq_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """`nvenc_cq` is the quality knob for A/B-ing without changing
    anything else in the chain."""
    monkeypatch.setattr(encoders, "has_nvenc", lambda: True)
    cfg = ClipConfig(nvenc_cq=19)
    args = encoders.pick_video_encoder_args(cfg)
    cq_idx = args.index("-cq")
    assert args[cq_idx + 1] == "19"


def test_probe_returns_false_when_ffmpeg_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ffmpeg on PATH → probe returns False without raising."""
    monkeypatch.setattr(encoders.shutil, "which", lambda _: None)
    encoders.reset_nvenc_cache()
    assert encoders.has_nvenc() is False
