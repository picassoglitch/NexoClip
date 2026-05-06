"""YAML config loader.

Per CLAUDE.md the layered config order is: defaults → YAML → environment.
The YAML side lives here; environment variables live in `nexoclip.settings`.

Phase 0 only models the sections that have functional code consuming them
(detection). Everything else maps with `extra="allow"` so the same file
serves later phases without a schema version bump.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from nexoclip.errors import NexoClipError


class VoiceDetectorConfig(BaseModel):
    """Voice trigger phrase list + fuzzy match params."""

    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0)
    fuzzy_distance: int = Field(default=2, ge=0)
    phrases: dict[str, list[str]] = Field(
        default_factory=dict,
        description="ISO 639-1 → list of trigger phrases.",
    )


class DetectionConfig(BaseModel):
    """All detector configuration."""

    voice: VoiceDetectorConfig = Field(default_factory=VoiceDetectorConfig)
    merge_window_s: float = Field(default=30.0, ge=0.0)


class ClipConfig(BaseModel):
    """Cut + reformat parameters for the clip step."""

    pre_roll_s: float = Field(default=30.0, ge=0.0)
    post_roll_s: float = Field(default=15.0, ge=0.0)
    output_aspect: str = Field(default="9:16", description="Phase 0 only supports 9:16.")
    output_width: int = Field(default=1080, gt=0)
    output_height: int = Field(default=1920, gt=0)
    encoder: str = "libx264"
    preset: str = "fast"
    crf: int = Field(default=23, ge=0, le=51)
    burn_captions: bool = False


class NexoClipConfig(BaseModel):
    """Root config object loaded from `config/nexoclip.yaml`."""

    model_config = ConfigDict(extra="allow")

    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    clip: ClipConfig = Field(default_factory=ClipConfig)


_DEFAULT_CONFIG_PATH = Path("config/nexoclip.yaml")
_DEFAULT_EXAMPLE_PATH = Path("config/nexoclip.example.yaml")


def load_config(path: Path | None = None) -> NexoClipConfig:
    """Load `config/nexoclip.yaml`, falling back to the example file.

    If neither file exists, return defaults — a fresh checkout still
    has a working detection config.
    """
    candidates: list[Path] = (
        [Path(path)] if path is not None else [_DEFAULT_CONFIG_PATH, _DEFAULT_EXAMPLE_PATH]
    )

    for candidate in candidates:
        if candidate.exists():
            try:
                with candidate.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise NexoClipError(f"failed to parse {candidate}: {e}") from e
            return NexoClipConfig.model_validate(data)

    if path is not None:
        raise NexoClipError(f"config file not found: {path}")
    return NexoClipConfig()


@lru_cache(maxsize=1)
def get_config() -> NexoClipConfig:
    """Cached default loader; tests can call `get_config.cache_clear()`."""
    return load_config()
