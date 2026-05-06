"""Synthetic video generation for vision tests.

We programmatically build short MP4s with known properties (motion bursts,
scene cuts) so the tests don't need committed fixture videos. MediaPipe
won't detect real faces in these — face/emotion gets stubbed at the
module level by the relevant test.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFAULT_FPS = 30.0
FRAME_W = 320
FRAME_H = 240


def write_video(path: Path, frames: list[np.ndarray], *, fps: float = DEFAULT_FPS) -> None:
    """Write `frames` (BGR uint8) to `path` as an mp4."""
    if not frames:
        raise ValueError("write_video needs at least one frame")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def solid_frame(color_bgr: tuple[int, int, int]) -> np.ndarray:
    """A FRAME_HxFRAME_W BGR frame filled with a single color."""
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:] = color_bgr
    return frame


def noise_frame(seed: int = 0, scale: float = 1.0) -> np.ndarray:
    """A noisy BGR frame; `scale` ∈ [0, 1] modulates intensity."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(0, 255 * scale, (FRAME_H, FRAME_W, 3))).astype(np.uint8)


def build_scene_cut_clip(
    path: Path, *, before_color: tuple[int, int, int] = (0, 0, 200),
    after_color: tuple[int, int, int] = (200, 200, 0),
    seconds_each: float = 1.0,
    fps: float = DEFAULT_FPS,
) -> None:
    """Two contiguous solid-color blocks back-to-back. PySceneDetect should
    register a cut at the boundary."""
    n = int(round(seconds_each * fps))
    frames = [solid_frame(before_color) for _ in range(n)] + [
        solid_frame(after_color) for _ in range(n)
    ]
    write_video(path, frames, fps=fps)


def build_motion_clip(
    path: Path, *, quiet_seconds: float = 1.0, busy_seconds: float = 1.0,
    fps: float = DEFAULT_FPS,
) -> None:
    """Stationary frames followed by per-frame noise. Motion energy should
    be near 0 in the quiet half and high in the busy half."""
    n_quiet = int(round(quiet_seconds * fps))
    n_busy = int(round(busy_seconds * fps))
    static = solid_frame((50, 50, 50))
    frames = [static.copy() for _ in range(n_quiet)] + [
        noise_frame(seed=i) for i in range(n_busy)
    ]
    write_video(path, frames, fps=fps)


def build_indexed_clip(
    path: Path, *, n_seconds: float = 2.0, fps: float = DEFAULT_FPS
) -> None:
    """Each frame's content encodes its index — used by the frame_sampler test."""
    n = int(round(n_seconds * fps))
    frames: list[np.ndarray] = []
    for i in range(n):
        frame = solid_frame((30, 30, 30))
        # White rectangle whose width = i px, so the sampler's "which
        # frame did I pick" is recoverable from the encoded JPEG.
        cv2.rectangle(frame, (0, 0), (max(1, i), 10), (255, 255, 255), -1)
        frames.append(frame)
    write_video(path, frames, fps=fps)
