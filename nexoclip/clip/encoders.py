"""Encoder selection — libx264 vs h264_nvenc (Task 1b).

Probes ffmpeg's encoder list once per process and caches the result.
Selection rules (in order):

  1. If `cfg.encoder` is set to anything other than `libx264`, the operator
     has explicitly chosen a codec — honor it with `cfg.preset`/`cfg.crf`.
  2. Else if `cfg.prefer_nvenc` is True AND `ffmpeg -encoders` lists
     `h264_nvenc`, use NVENC tuned to be visually equal to the libx264
     veryfast / CRF 23 baseline.
  3. Else use libx264 with the configured preset + CRF.

The NVENC arg set (`-preset p5 -rc vbr -cq 23 -b:v 0`) is the operator-
agreed calibration target. `cfg.nvenc_cq` overrides the constant-quality
target without changing the rest of the chain — useful for A/B-ing
quality without re-engineering this module.

Visual identity guard: NVENC's quality at p5/CQ 23 is close to libx264
veryfast CRF 23 on the social-video targets we ship (1080×1920, ≤60s),
but the platforms re-encode aggressively anyway. The escape hatch is
`prefer_nvenc=False` in config — restores the old path bit-for-bit.

Thread-safety: `has_nvenc()` is called from worker threads inside the
parallel cut pipeline; the cached probe is a single bool assignment so
torn reads are harmless.
"""

from __future__ import annotations

import shutil
import subprocess

from nexoclip.config import ClipConfig
from nexoclip.logging import get_logger

_log = get_logger("nexoclip.clip.encoders")

# Sentinel: None = not yet probed; True/False = probed result.
_NVENC_CACHE: bool | None = None


def has_nvenc() -> bool:
    """Returns True iff `ffmpeg -encoders` lists `h264_nvenc`.

    Cached per-process. Any failure mode (no ffmpeg on PATH, probe
    timeout, exception) resolves to False — we never fail the cut
    pipeline because the probe blew up.
    """
    global _NVENC_CACHE
    if _NVENC_CACHE is None:
        _NVENC_CACHE = _probe_nvenc()
    return _NVENC_CACHE


def reset_nvenc_cache() -> None:
    """Test hook — discard the cached probe so the next call re-probes."""
    global _NVENC_CACHE
    _NVENC_CACHE = None


def pick_video_encoder_args(cfg: ClipConfig) -> list[str]:
    """Return the ffmpeg `-c:v ...` argument sequence for the reformat
    step. Applies the selection rules in the module docstring.

    Output is a flat argv slice — splice it directly into the ffmpeg
    command between the filter and the audio codec args.
    """
    # Rule 1 — operator override.
    encoder = (cfg.encoder or "").strip() or "libx264"
    if encoder != "libx264":
        return ["-c:v", encoder, "-preset", cfg.preset, "-crf", str(cfg.crf)]

    # Rule 2 — NVENC if available + preferred.
    if getattr(cfg, "prefer_nvenc", True) and has_nvenc():
        cq = int(getattr(cfg, "nvenc_cq", cfg.crf))
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p5",
            "-rc", "vbr",
            "-cq", str(cq),
            "-b:v", "0",
        ]

    # Rule 3 — libx264 fallback.
    return ["-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf)]


# ---- internals ----


def _probe_nvenc() -> bool:
    """Run `ffmpeg -encoders` and look for `h264_nvenc` in the listing."""
    if shutil.which("ffmpeg") is None:
        _log.info("nvenc.probe", available=False, reason="ffmpeg_not_on_path")
        return False
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, check=False, timeout=5.0, text=True,
        )
    except Exception as e:  # noqa: BLE001 — probe failure must not break cut
        _log.warning("nvenc.probe.failed", error=str(e))
        return False
    blob = (proc.stdout or "") + (proc.stderr or "")
    found = "h264_nvenc" in blob
    _log.info("nvenc.probe", available=found)
    return found


__all__ = ["has_nvenc", "pick_video_encoder_args", "reset_nvenc_cache"]
