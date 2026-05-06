"""Face-aware 9:16 crop selection.

Samples N frames spread across the clip window, runs OpenCV's bundled
Haar Cascade frontal-face detector on each, and chooses a 9:16 crop
window centered on the average face center. Falls back to a centered
crop when no faces are detected anywhere in the sampled frames (the
streamer is off-camera, or the source isn't a face-driven scene).

Phase 1 ships this heuristic. Phase 2 swaps in a vision-LLM pass that
considers compositional cues (foreground objects, on-screen text, gaze
direction) - the smart_crop_box column on `clips` is already provisioned
to carry whatever the upgraded picker decides.
"""

from __future__ import annotations

from pathlib import Path

from nexoclip.errors import ClipError

from .models import SmartCropBox

# Aspect ratio is fixed at 9:16 for Phase 1 (matches PHASE_0/1 spec).
_ASPECT_W = 9
_ASPECT_H = 16


def compute_smart_crop_box(
    video_path: Path,
    *,
    start_s: float,
    end_s: float,
    sample_n: int = 5,
) -> SmartCropBox:
    """Return a 9:16 crop window over the source frame.

    `start_s` / `end_s` define the clip's window on the source timeline.
    Samples `sample_n` frames evenly across that window for face detection.
    Falls back to a centered crop if no face is found.
    """
    if end_s <= start_s:
        raise ClipError(f"end_s must be > start_s ({start_s} -> {end_s})")
    if sample_n < 1:
        raise ClipError(f"sample_n must be >= 1, got {sample_n}")
    if not video_path.exists():
        raise ClipError(f"video file missing: {video_path}")

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ClipError(f"cv2 could not open {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or total_frames <= 0 or source_w <= 0 or source_h <= 0:
            raise ClipError(f"unusable video metadata for {video_path}")

        # 9:16 crop sized to the source height; if the source is portrait
        # already, just take the full frame.
        crop_w = min(source_w, max(1, (source_h * _ASPECT_W) // _ASPECT_H))
        crop_h = source_h
        center_default = source_w / 2.0

        if crop_w >= source_w:
            return SmartCropBox(x=0, y=0, w=source_w, h=crop_h)

        # Sample frames at evenly-spaced timestamps across the window.
        if sample_n == 1:
            sample_times = [(start_s + end_s) / 2.0]
        else:
            step = (end_s - start_s) / (sample_n - 1)
            sample_times = [start_s + i * step for i in range(sample_n)]

        face_centers_x = _detect_face_centers(cap, sample_times, fps, total_frames)

        target_x = (
            sum(face_centers_x) / len(face_centers_x) if face_centers_x else center_default
        )

        x_offset = round(target_x - crop_w / 2.0)
        x_offset = max(0, min(source_w - crop_w, x_offset))
        return SmartCropBox(x=x_offset, y=0, w=crop_w, h=crop_h)
    finally:
        cap.release()


def _detect_face_centers(
    cap: object,
    sample_times: list[float],
    fps: float,
    total_frames: int,
) -> list[float]:
    """Return the x-pixel center of the largest detected face per sample.

    Uses OpenCV's bundled Haar Cascade frontal-face detector. Frames
    without a detected face are silently dropped (the caller falls back
    to a centered crop when the list is empty).
    """
    import cv2

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
    )
    if detector.empty():
        # Bundle file is missing or unreadable - skip detection rather than
        # raise; the caller falls back to a centered crop.
        return []

    centers_x: list[float] = []
    for ts in sample_times:
        frame_idx = round(ts * fps)
        frame_idx = max(0, min(total_frames - 1, frame_idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # type: ignore[attr-defined]
        ret, frame = cap.read()  # type: ignore[attr-defined]
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            continue
        # detectMultiScale returns Nx4 (x, y, w, h). Pick the largest by area.
        x, _y, w, _h = max(faces, key=lambda r: int(r[2]) * int(r[3]))
        centers_x.append(float(int(x) + int(w) / 2.0))
    return centers_x


def crop_box_to_ffmpeg_filter(
    box: SmartCropBox, *, output_w: int, output_h: int
) -> str:
    """Render the crop+scale filter chain for ffmpeg's `-vf` argument."""
    return f"crop={box.w}:{box.h}:{box.x}:{box.y},scale={output_w}:{output_h}"
