"""Tests for the face-emotion detector.

We can't synthesize a face that MediaPipe will actually detect, so the
real `detect_face_emotions` function is exercised end-to-end only in
manual smoke tests. These unit tests pin down:
    * the smile-vs-neutral landmark heuristic
    * the FaceFrame shape returned when MediaPipe finds vs. doesn't find a face
"""

from __future__ import annotations

from types import SimpleNamespace

from nexoclip.vision.face_emotion import (
    _emotion_from_landmarks,
    _face_frame_from_results,
)


def _landmark(y: float) -> SimpleNamespace:
    return SimpleNamespace(x=0.5, y=y, z=0.0)


def _build_landmarks(*, upper_y: float, lower_y: float, corner_y: float) -> dict:
    """Index-addressable container with the four landmarks the heuristic touches."""
    pts = {
        13: _landmark(upper_y),
        14: _landmark(lower_y),
        61: _landmark(corner_y),
        291: _landmark(corner_y),
    }

    class IndexableLandmarks:
        def __getitem__(self, idx: int) -> SimpleNamespace:
            return pts[idx]

    return IndexableLandmarks()


def test_neutral_when_corners_below_lip_midline() -> None:
    # Mouth midpoint at y=0.6; corners below at y=0.62 (no smile).
    lm = _build_landmarks(upper_y=0.55, lower_y=0.65, corner_y=0.62)
    assert _emotion_from_landmarks(lm) == "neutral"


def test_smile_when_corners_above_lip_midline() -> None:
    # Mouth midpoint at y=0.6; corners well above at y=0.55.
    lm = _build_landmarks(upper_y=0.55, lower_y=0.65, corner_y=0.55)
    assert _emotion_from_landmarks(lm) == "smile"


def test_neutral_when_only_one_corner_lifts() -> None:
    """Asymmetric mouth — only one corner up — stays neutral."""
    pts = {
        13: _landmark(0.55),
        14: _landmark(0.65),
        61: _landmark(0.55),  # left up
        291: _landmark(0.62),  # right not up
    }

    class IndexableLandmarks:
        def __getitem__(self, idx: int) -> SimpleNamespace:
            return pts[idx]

    assert _emotion_from_landmarks(IndexableLandmarks()) == "neutral"


def test_face_frame_no_face() -> None:
    results = SimpleNamespace(multi_face_landmarks=None)
    f = _face_frame_from_results(ts=2.0, results=results)
    assert f.has_face is False
    assert f.emotion is None
    assert f.ts == 2.0


def test_face_frame_with_face_returns_smile() -> None:
    landmarks = _build_landmarks(upper_y=0.55, lower_y=0.65, corner_y=0.50)
    results = SimpleNamespace(
        multi_face_landmarks=[SimpleNamespace(landmark=landmarks)]
    )
    f = _face_frame_from_results(ts=5.0, results=results)
    assert f.has_face is True
    assert f.emotion == "smile"
    assert 0.0 < f.confidence <= 1.0
