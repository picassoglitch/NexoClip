"""Unit tests for the preview_recorder's validation + manifest helpers.

These tests cover the pure-Python pieces that don't need a real browser:

  - `_validate_export` correctly classifies duration / fps / codec drift
  - `_write_manifest` writes a JSON file alongside the export
  - duration tolerance enforcement matches the documented contract

The end-to-end recorder (Playwright + ffmpeg) is exercised via a
separate manual test plan documented in the slice O.54 commit
message; CI runs the unit layer here.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from nexoclip.clip.preview_recorder import (
    _DURATION_TOLERANCE_MS,
    _validate_export,
    _write_manifest,
)


def _make_ffprobe_response(
    duration_s: float,
    *,
    fps: float = 30.0,
    video_codec: str = "h264",
    pix_fmt: str = "yuv420p",
    audio_codec: str | None = "aac",
    audio_sample_rate: str | None = "48000",
) -> str:
    """Build a JSON payload that mimics ffprobe -show_format -show_streams."""
    streams: list[dict[str, object]] = [
        {
            "codec_type": "video",
            "codec_name": video_codec,
            "pix_fmt": pix_fmt,
            "avg_frame_rate": f"{int(fps)}/1",
        }
    ]
    if audio_codec is not None:
        streams.append({
            "codec_type": "audio",
            "codec_name": audio_codec,
            "sample_rate": audio_sample_rate,
        })
    return json.dumps({
        "format": {"duration": str(duration_s)},
        "streams": streams,
    })


def _fake_ffprobe(stdout: str, returncode: int = 0):
    """Construct a CompletedProcess that subprocess.run would return."""
    return subprocess.CompletedProcess(
        args=["ffprobe"], returncode=returncode, stdout=stdout, stderr=""
    )


# ---- _validate_export -------------------------------------------------------


def test_validate_export_happy_path(tmp_path: Path) -> None:
    """A perfectly-encoded clip should validate ok=True with zero drift."""
    out = tmp_path / "clip_render_1080.mp4"
    out.write_bytes(b"fake mp4")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _fake_ffprobe(_make_ffprobe_response(40.8))
        result = _validate_export(
            output_path=out, expected_duration_s=40.8, expected_fps=30
        )

    assert result["ok"] is True
    checks = result["checks"]
    assert checks["duration_drift_ms"] == 0
    assert checks["actual_fps"] == 30.0
    assert checks["video_codec"] == "h264"
    assert checks["audio_codec"] == "aac"
    assert checks["audio_sample_rate"] == "48000"


def test_validate_export_duration_within_tolerance(tmp_path: Path) -> None:
    """Up to 150ms drift is acceptable (within _DURATION_TOLERANCE_MS)."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")

    with patch("subprocess.run") as mock_run:
        # 40.8 expected, 40.85 actual = 50ms drift
        mock_run.return_value = _fake_ffprobe(_make_ffprobe_response(40.85))
        result = _validate_export(
            output_path=out, expected_duration_s=40.8, expected_fps=30
        )

    assert result["ok"] is True
    assert result["checks"]["duration_drift_ms"] == 50


def test_validate_export_duration_drift_fails(tmp_path: Path) -> None:
    """Drift > 150ms must fail validation."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")

    with patch("subprocess.run") as mock_run:
        # 40.8 expected, 41.5 actual = 700ms drift
        mock_run.return_value = _fake_ffprobe(_make_ffprobe_response(41.5))
        result = _validate_export(
            output_path=out, expected_duration_s=40.8, expected_fps=30
        )

    assert result["ok"] is False
    assert result["checks"]["duration_drift_ms"] == 700
    assert any("drift" in e for e in result["errors"])


def test_validate_export_fps_mismatch_fails(tmp_path: Path) -> None:
    """A 60fps output when we asked for 30 must fail validation."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _fake_ffprobe(
            _make_ffprobe_response(40.8, fps=60.0)
        )
        result = _validate_export(
            output_path=out, expected_duration_s=40.8, expected_fps=30
        )

    assert result["ok"] is False
    assert result["checks"]["actual_fps"] == 60.0
    assert any("fps" in e for e in result["errors"])


def test_validate_export_wrong_video_codec_fails(tmp_path: Path) -> None:
    """If somehow we emit something other than H.264, validation must fail."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _fake_ffprobe(
            _make_ffprobe_response(40.8, video_codec="hevc")
        )
        result = _validate_export(
            output_path=out, expected_duration_s=40.8, expected_fps=30
        )

    assert result["ok"] is False
    assert result["checks"]["video_codec"] == "hevc"


def test_validate_export_missing_audio_is_informational(tmp_path: Path) -> None:
    """Clips without audio still validate ok=True; codec is None in checks."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _fake_ffprobe(
            _make_ffprobe_response(40.8, audio_codec=None)
        )
        result = _validate_export(
            output_path=out, expected_duration_s=40.8, expected_fps=30
        )

    assert result["ok"] is True
    assert result["checks"]["audio_codec"] is None


def test_validate_export_ffprobe_failure(tmp_path: Path) -> None:
    """A non-zero ffprobe exit must mark the export not-ok."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = _fake_ffprobe("", returncode=1)
        result = _validate_export(
            output_path=out, expected_duration_s=40.8, expected_fps=30
        )

    assert result["ok"] is False
    assert any("ffprobe" in e for e in result["errors"])


def test_validate_export_ffprobe_exception(tmp_path: Path) -> None:
    """If subprocess.run raises (binary missing, permission), surface it."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")

    with patch("subprocess.run", side_effect=FileNotFoundError("no ffprobe")):
        result = _validate_export(
            output_path=out, expected_duration_s=40.8, expected_fps=30
        )

    assert result["ok"] is False
    assert any("exception" in e for e in result["errors"])


# ---- _write_manifest --------------------------------------------------------


def test_write_manifest_writes_sibling_json(tmp_path: Path) -> None:
    """Manifest lands at <output>.manifest.json with the full payload."""
    out = tmp_path / "clip_render_1080.mp4"
    out.write_bytes(b"x")
    payload: dict[str, object] = {
        "clip_id": "clp_abc",
        "input": {"duration_s": 40.8},
        "validation": {"ok": True, "errors": []},
    }
    _write_manifest(output_path=out, manifest=payload)

    manifest_path = out.with_suffix(".manifest.json")
    assert manifest_path.exists()
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded == payload


def test_write_manifest_survives_oserror(tmp_path: Path) -> None:
    """Manifest write failures must not raise — they only log."""
    out = tmp_path / "out.mp4"
    out.write_bytes(b"x")
    with patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
        # Must not raise.
        _write_manifest(output_path=out, manifest={"a": 1})


# ---- contract -------------------------------------------------------------


def test_duration_tolerance_documented() -> None:
    """The tolerance is part of the public contract — guard it from
    accidental loosening. If you change this, update the docstring +
    operator docs too."""
    assert _DURATION_TOLERANCE_MS == 150
