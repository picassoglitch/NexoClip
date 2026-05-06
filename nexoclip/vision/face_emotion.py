"""MediaPipe Face Mesh face detection + lightweight smile heuristic.

Phase 1 ships a coarse `neutral` vs `smile` label using mouth-corner
landmark geometry: if both corners (idx 61 and 291) are above the lip
midpoint (idx 13/14) by at least `_SMILE_THRESHOLD` of normalized image
height, we call it a smile. Otherwise neutral.

Real emotion classification — laugh / shock / anger / sad — is Phase 2,
when the multimodal LLM can interpret the whole face. Hand-tuning
landmark heuristics for the harder labels would burn a lot of calibration
time on top of a heuristic that the LLM does better anyway.

Sampling: 2 fps by default per PHASE_1.md §5. The MediaPipe model is
loaded once per call (cached `FaceMesh` instance) and released on exit.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from nexoclip.errors import DetectionError

from .models import FaceEmotion, FaceFrame

# Mouth-corner landmark indices in MediaPipe's 468-point Face Mesh.
_UPPER_LIP_CENTER = 13
_LOWER_LIP_CENTER = 14
_LEFT_MOUTH_CORNER = 61
_RIGHT_MOUTH_CORNER = 291
# How much higher the corners must be (in normalized image-y coords) than
# the lip midpoint before we call it a smile. ~1% of image height.
_SMILE_THRESHOLD = 0.01


def detect_face_emotions(video_path: Path, *, sample_rate_hz: float = 2.0) -> list[FaceFrame]:
    """Return one FaceFrame per sampled frame (default = 2 fps)."""
    import cv2
    import mediapipe as mp

    if not video_path.exists():
        raise DetectionError(f"video file missing: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise DetectionError(f"cv2 could not open {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            raise DetectionError(f"unknown fps for {video_path}")
        frame_skip = max(1, round(fps / sample_rate_hz))

        frames: list[FaceFrame] = []
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_skip == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb)
                    ts = frame_idx / fps
                    frames.append(_face_frame_from_results(ts, results))
                frame_idx += 1
        finally:
            face_mesh.close()
    finally:
        cap.release()

    return frames


def _face_frame_from_results(ts: float, results: Any) -> FaceFrame:
    landmarks_list = getattr(results, "multi_face_landmarks", None)
    if not landmarks_list:
        return FaceFrame(ts=ts, has_face=False)
    landmarks = landmarks_list[0].landmark
    emotion = _emotion_from_landmarks(landmarks)
    return FaceFrame(ts=ts, has_face=True, emotion=emotion, confidence=0.7)


def _emotion_from_landmarks(landmarks: Any) -> FaceEmotion:
    """Heuristic: smile when both mouth corners are above the lip midpoint."""
    try:
        upper = landmarks[_UPPER_LIP_CENTER]
        lower = landmarks[_LOWER_LIP_CENTER]
        left = landmarks[_LEFT_MOUTH_CORNER]
        right = landmarks[_RIGHT_MOUTH_CORNER]
    except (IndexError, KeyError):
        return "neutral"
    mid_y = (upper.y + lower.y) / 2.0
    # In image-y, smaller y == higher on screen.
    corners_above_midline = left.y < mid_y - _SMILE_THRESHOLD and right.y < mid_y - _SMILE_THRESHOLD
    return "smile" if corners_above_midline else "neutral"


# Exposed so tests can stub the heuristic without instantiating MediaPipe.
emotion_classifier: Callable[[Any], FaceEmotion] = _emotion_from_landmarks
