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
    """Voice trigger phrase list + fuzzy match params.

    Two phrase families with different clip-window semantics:

    * `phrases` — forward triggers. The streamer says the phrase BEFORE the
      moment ('watch this — clipea esto'). The cut step extends forward
      from the trigger timestamp by `clip.pre_roll_s + post_roll_s`.

    * `retroactive_phrases` — retroactive triggers. The streamer says the
      phrase AFTER the moment ('that was insane — clipeaste eso'). The
      cut step uses `retroactive_lookback_s` of audio BEFORE the trigger
      timestamp and ignores pre/post roll. This is the more natural case
      in live streaming.
    """

    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0)
    fuzzy_distance: int = Field(default=2, ge=0)
    # Slice O.41 — bake in the spec defaults instead of `{}`. Railway
    # boots without a nexoclip.yaml file (the YAML is operator-local
    # config), and the empty default meant the voice detector returned
    # zero candidates on every Spanish stream until the operator
    # discovered the YAML knob. Same list as the example YAML so
    # behavior is identical with-or-without a config file.
    phrases: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "es": [
                "clipea esto",
                "clipéalo",
                "saca un clip",
                "guarda esto",
                "momento clip",
                "este momento",
            ],
            "en": [
                "clip this",
                "clip that",
                "someone clip this",
                "did you clip that",
            ],
        },
        description="ISO 639-1 → list of forward trigger phrases.",
    )
    retroactive_phrases: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "es": [
                "clipeaste eso",
                "clipea eso",
                "clipearon eso",
                "eso fue épico",
            ],
            "en": [
                "did you clip that",
                "clip that",
                "tell me you clipped that",
                "please tell me you got that",
            ],
        },
        description="ISO 639-1 → list of retroactive trigger phrases.",
    )
    retroactive_lookback_s: float = Field(
        default=60.0,
        gt=0.0,
        description="When a retroactive phrase fires, the clip covers this "
        "many seconds BEFORE the timestamp.",
    )
    cooldown_s: float = Field(
        default=10.0,
        ge=0.0,
        description="Minimum gap between two triggers of the same kind to "
        "prevent dupes from rapid speech.",
    )


class ChatHeatConfig(BaseModel):
    """Chat heat detector — spike when msg/sec >> rolling baseline."""

    enabled: bool = False
    weight: float = Field(default=0.7, ge=0.0)
    baseline_window_s: float = Field(default=300.0, gt=0.0)
    spike_ratio: float = Field(default=3.0, gt=0.0)
    absolute_floor_msg_per_s: float = Field(default=5.0, ge=0.0)


class AudioEnergyConfig(BaseModel):
    """Audio energy detector — spike when frame RMS >> rolling baseline,
    sustained for at least `sustain_s` to suppress one-frame pops.
    """

    enabled: bool = False
    weight: float = Field(default=0.5, ge=0.0)
    frame_s: float = Field(default=0.5, gt=0.0, description="RMS bin size in seconds.")
    baseline_window_s: float = Field(default=300.0, gt=0.0)
    spike_ratio: float = Field(default=2.5, gt=0.0)
    sustain_s: float = Field(default=1.5, ge=0.0)


class VisualConfig(BaseModel):
    """Visual detector — fuses scene cuts, emotion transitions, and motion
    spikes (from `analyze-video`'s VisualSignalTrack) into Candidates.
    """

    # Slice F.7-H — flipped to True by default. The detector adds
    # scene-cut + motion + face-emotion signals to the candidate set,
    # which is essential for clips that are visually dramatic but
    # don't have a loud voice / chat moment (silent dunks, jumpscares,
    # reaction shots). It's a one-time-per-stream CPU cost capped by
    # `timeout_s`, and the operator can set this False in nexoclip.yaml
    # if they're running on an underpowered host.
    enabled: bool = True
    weight: float = Field(default=0.6, ge=0.0, description="Outer multiplier on the visual score.")
    cut_weight: float = Field(default=1.0, ge=0.0)
    emotion_weight: float = Field(default=0.7, ge=0.0)
    motion_weight: float = Field(default=0.5, ge=0.0)
    motion_baseline_window_s: float = Field(default=30.0, gt=0.0)
    motion_spike_ratio: float = Field(default=2.0, gt=0.0)
    emotion_labels: list[str] = Field(
        default_factory=lambda: ["smile", "laugh", "shock"],
        description="Which face_emotion values count as 'strong' (worth firing on).",
    )
    timeout_s: float = Field(
        default=600.0,
        gt=0.0,
        description=(
            "Hard cap on analyze_video runtime in seconds (floor). The pipeline "
            "actually uses max(timeout_s, duration_s * analyze_video_timeout_multiplier) "
            "so this is just the minimum even for very short videos. Slice O.27 "
            "bumped the floor 120 → 600 because PySceneDetect on prod CPU runs "
            "near 1× realtime and the 120 floor was getting hit on 60-90s clips."
        ),
    )


class DiarizationConfig(BaseModel):
    """Speaker diarization (pyannote-3.1) settings.

    Slice O.29 — flipped to True by default. The pipeline already
    short-circuits gracefully when (a) pyannote.audio isn't installed
    or (b) HF_TOKEN isn't set — the step logs `diarize.skipped` with
    the reason and downstream code reads `diarization.skipped` to fall
    back to un-attributed candidates. So flipping the default is safe:
    setups that DO have pyannote+HF_TOKEN get the feature for free;
    setups that don't see the same skip-with-reason they saw before.

    `match_threshold` is the cosine-sim cutoff for matching a new VOD's
    speaker embedding against the tenant's persistent `speakers` table.
    Lower values are more permissive (more likely to merge two recordings
    of the same person into one identity); higher values demand cleaner
    matches.
    """

    enabled: bool = True
    model: str = Field(default="pyannote/speaker-diarization-3.1")
    device: str = Field(default="cuda")
    match_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    min_speech_for_id_s: float = Field(
        default=30.0,
        gt=0.0,
        description="Speakers with less than this much total speech in a VOD "
        "are not auto-matched against the persistent speakers table — too "
        "little signal to risk a wrong merge. Still recorded as VOD-scoped "
        "labels so the user can label them manually.",
    )


class ViralConfig(BaseModel):
    """Viral-moment detector — feeds the transcript to an LLM and asks it to
    identify the 5-15 most clip-worthy moments based on controversy, emotion,
    quotability, and shock value.

    Slice O.27 — enabled by default. The voice-trigger-only detector
    returns 0 candidates the moment a streamer doesn't yell "clip this"
    (i.e. essentially every real-world VOD). Without the viral detector
    on, the pipeline finishes with 0 candidates → 0 clips → the user
    sees "Pipeline complete" and nothing to do, with no hint why.
    Costs one LLM call per stream — acceptable for paid-tier users who
    pay for clips and not for LLM-call budgets.
    """

    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0)
    quality: str = Field(
        default="standard",
        description="LLM quality tier (standard | premium).",
    )
    max_moments: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Hard cap on how many moments the LLM can return.",
    )
    min_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Drop moments below this LLM score before fusing.",
    )


class FusionConfigModel(BaseModel):
    """Weighted multi-signal fusion — slice G.1.

    The dataclass-based `FusionConfig` in `nexoclip.detect.fusion` is
    the runtime contract; this Pydantic mirror exists so operators can
    tune weights / bonuses from `nexoclip.yaml` without touching code.
    The two are kept in sync via `to_runtime()` below.
    """

    voice: float = Field(default=0.35, ge=0.0, le=1.0)
    visual: float = Field(default=0.20, ge=0.0, le=1.0)
    audio: float = Field(default=0.15, ge=0.0, le=1.0)
    chat: float = Field(default=0.15, ge=0.0, le=1.0)
    viral: float = Field(default=0.10, ge=0.0, le=1.0)
    transcript_hook: float = Field(default=0.05, ge=0.0, le=1.0)

    two_detector_bonus: float = Field(default=0.05, ge=0.0, le=1.0)
    three_plus_detector_bonus: float = Field(default=0.10, ge=0.0, le=1.0)
    face_visible_bonus: float = Field(default=0.05, ge=0.0, le=1.0)
    strong_signal_bonus: float = Field(default=0.05, ge=0.0, le=1.0)

    overlap_window_s: float = Field(default=10.0, gt=0.0)


class DetectionConfig(BaseModel):
    """All detector configuration."""

    voice: VoiceDetectorConfig = Field(default_factory=VoiceDetectorConfig)
    chat_heat: ChatHeatConfig = Field(default_factory=ChatHeatConfig)
    audio_energy: AudioEnergyConfig = Field(default_factory=AudioEnergyConfig)
    visual: VisualConfig = Field(default_factory=VisualConfig)
    viral: ViralConfig = Field(default_factory=ViralConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    fusion: FusionConfigModel = Field(default_factory=FusionConfigModel)
    merge_window_s: float = Field(default=30.0, ge=0.0)


class ClipConfig(BaseModel):
    """Cut + reformat parameters for the clip step."""

    pre_roll_s: float = Field(default=30.0, ge=0.0)
    post_roll_s: float = Field(default=15.0, ge=0.0)
    output_aspect: str = Field(default="9:16", description="Phase 0 only supports 9:16.")
    output_width: int = Field(default=1080, gt=0)
    output_height: int = Field(default=1920, gt=0)
    encoder: str = "libx264"
    # Slice O.32 — `fast` → `veryfast`. On Railway CPU (no GPU encoder)
    # the `fast` preset was burning ~30-60s per ~12s clip; for 2 clips
    # that's nearly 2 minutes just on cut, blocking the pipeline.
    # `veryfast` is ~2.5× faster at the same CRF 23 quality target;
    # the visible quality difference on social-video output is
    # imperceptible (TikTok / Reels re-encode aggressively anyway).
    preset: str = "veryfast"
    crf: int = Field(default=23, ge=0, le=51)
    burn_captions: bool = False
    # Slice G.1 — dynamic per-candidate clip windowing. When True
    # (the default) and a transcript is available, each clip's
    # start/end snaps to sentence boundaries inside a kind-aware
    # band (reaction 10-22s / quote 12-25s / story 35-60s / etc).
    # Set False to fall back to the legacy pre_roll/post_roll
    # geometry on every clip (useful when transcripts are unreliable
    # or for debugging the static cut path).
    dynamic_windowing: bool = True


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
