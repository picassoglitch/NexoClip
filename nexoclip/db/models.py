"""Pydantic models that mirror the DB tables 1:1.

These are distinct from the service-level models (e.g. `nexoclip.ingest.Stream`,
`nexoclip.detect.Candidate`) because the DB row carries persistence-only
fields (`created_at`, JSON-encoded blobs) that the service models
shouldn't have to know about.

Service models are converted to/from DB models inside the repo methods
(Task 1 wires that through). For Task 0 the DB models stand alone.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------- Tenancy ----------


class Tenant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    created_at: str
    # Phase 2 budget governor knobs. NULL == unlimited.
    daily_llm_budget_usd_micros: int | None = None
    daily_publish_limit: int | None = None
    rescore_concurrency_cap: int = 4
    # Retention windows (voice-markers spec slice E.1 / §9 locked defaults).
    # NULL means the tenant inherits the system default:
    #   - retention_vod_days        → 30
    #   - retention_clip_days       → 90
    #   - retention_transcript_days → 365
    # The sweeper resolves NULL → default at scan time; storing NULL keeps
    # "factory default" distinguishable from "tenant explicitly chose this
    # number" for ops debugging.
    retention_vod_days: int | None = None
    retention_clip_days: int | None = None
    retention_transcript_days: int | None = None


class User(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    email: str
    role: str = "owner"
    created_at: str


class ApiTokenRow(BaseModel):
    """Persisted shape — never carries the raw token, only the hash."""

    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    hash: str
    scope: Literal["full", "read"] = "full"
    created_at: str
    last_used_at: str | None = None


# ---------- Personas + connected accounts ----------


class PersonaRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    name: str
    primary_language: str
    target_languages: list[str] = Field(default_factory=list)
    voice_prompt: str
    routing_tags: list[str] = Field(default_factory=list)
    created_at: str


class ConnectedAccount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    platform: str
    external_id: str
    display_name: str | None = None
    oauth_blob: dict[str, object] | None = None
    created_at: str
    # Phase 2: OAuth refresh + lifecycle tracking.
    refresh_token: str | None = None
    expires_at: str | None = None
    scopes: list[str] = Field(default_factory=list)
    status: str = "active"  # active | auth_failed | disabled


# ---------- Pipeline ----------


class StreamRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    vod_url: str
    platform: str
    title: str | None = None
    channel: str | None = None
    duration_s: float
    source_video_path: str
    source_audio_path: str
    status: str = "ingested"
    created_at: str


class TranscriptRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stream_id: str
    tenant_id: str
    language: str
    duration_s: float
    model: str
    segments_json: str
    created_at: str


class CandidateRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    stream_id: str
    tenant_id: str
    ts: float
    score: float
    reason: str
    evidence: dict[str, object] = Field(default_factory=dict)
    created_at: str
    # Phase 2: vision-LLM rescore verdict (NULL until --vision-rescore runs).
    rescore_score: float | None = None
    rescore_reason: str | None = None
    rescore_model: str | None = None


class ClipRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    stream_id: str
    tenant_id: str
    candidate_id: str | None = None
    start_s: float
    end_s: float
    duration_s: float
    width: int
    height: int
    path: str
    smart_crop_box: dict[str, float] | None = None
    thumbnail_frame_path: str | None = None
    status: str = "cut"
    created_at: str
    # Per-clip overlay customization set in the clip editor screen.
    # Empty / None means "fall back to the brand-kit defaults end-to-end".
    # Schema is intentionally loose (dict[str, object]) — the renderer
    # picks the keys it knows; adding a new overlay knob doesn't need a
    # schema migration.
    overlay_config: dict[str, object] | None = None


class VariantRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    clip_id: str
    tenant_id: str
    persona_id: str
    language: str
    caption: str
    title_card_text: str = ""
    hashtags: list[str] = Field(default_factory=list)
    model: str | None = None
    created_at: str


class LLMCallRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    purpose: str
    provider: str
    model: str
    quality: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd_micros: int = 0
    status: str = "ok"
    error: str | None = None
    attempts: int = 1
    ts: str


class PublishJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    clip_id: str
    variant_id: str
    account_id: str
    platform: str
    status: str = "pending"
    attempts: int = 0
    last_error: str | None = None
    scheduled_for: str | None = None
    external_id: str | None = None
    created_at: str
    # Phase 2: native-publisher metadata (TikTok/YT shareable URL, raw API response bits).
    external_url: str | None = None
    platform_metadata: dict[str, object] | None = None


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    type: str
    payload: dict[str, object] = Field(default_factory=dict)
    ts: str


class VisualSignalRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stream_id: str
    tenant_id: str
    ts_offset_s: float
    scene_cut: bool = False
    face_emotion: str | None = None
    motion_energy: float | None = None
    text_changed: bool = False


class WebhookSubscription(BaseModel):
    """Phase 2: outbound HMAC-signed event delivery to subscriber URLs."""

    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    url: str
    types: list[str] = Field(default_factory=list)
    secret: str
    status: str = "active"  # active | disabled
    created_at: str
    last_dispatch_ts: str | None = None
    failure_count: int = 0


class WebhookSecretVersion(BaseModel):
    """Phase 3: a prior webhook secret kept alive for a rotation grace window.

    Subscribers may verify HMAC signatures against any unexpired secret in
    `webhook_secret_versions` for the subscription, in addition to the
    current secret on the `webhook_subscriptions` row.
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    subscription_id: str
    tenant_id: str
    secret: str
    expires_at: str
    created_at: str


class PublishMetric(BaseModel):
    """Phase 3: one engagement-stats snapshot for a published clip.

    Each `(publish_job_id, fetched_at)` pair is one row; the dashboard's
    outcome card reads the latest row per job, and the calibration loop
    reads the time series. Some platforms won't expose every column - we
    leave the missing ones NULL rather than fabricating zeros.
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    publish_job_id: str
    platform: str
    fetched_at: str
    views: int | None = None
    likes: int | None = None
    comments: int | None = None
    shares: int | None = None
    retention_pct: float | None = None  # 0.0-1.0
    ctr: float | None = None  # 0.0-1.0
    raw_metadata: dict[str, object] | None = None
    created_at: str


# ---------- Speakers (voice-markers spec slice B.2) ----------


class SpeakerRow(BaseModel):
    """Persistent voice identity for one tenant.

    Embedded vector is a 192-dim float list from pyannote/embedding,
    averaged across all VODs we've matched this identity in (weighted
    by total_speech_s). Stored as a JSON list in the embedding_json
    column for portability across SQLite / Postgres without a vector
    extension.
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    display_name: str
    is_self: bool = False
    preferred_brand_kit_id: str | None = None
    embedding: list[float] | None = None
    embedding_dim: int | None = None
    total_speech_s: float = 0.0
    sample_audio_path: str | None = None
    created_at: str
    updated_at: str


class VodSpeakerRow(BaseModel):
    """Per-VOD resolution of one within-VOD speaker label to a persistent identity.

    `resolved_speaker_id=None` means the speaker is unknown — the user
    can label them via the dashboard's /speakers page (slice E).
    `confidence` is the cosine similarity to the matched speaker; below
    `DiarizationConfig.match_threshold` we leave resolved_speaker_id=None.
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    stream_id: str
    tenant_id: str
    speaker_label: str
    resolved_speaker_id: str | None = None
    confidence: float | None = None
    total_speech_s: float = 0.0
    embedding: list[float] | None = None
    created_at: str


# ---------- Brand kits (voice-markers spec slice C.1) ----------


class CustomTriggerPhrases(BaseModel):
    """Per-kit additions to the tenant base trigger list.

    Spec section 9 hard rule: these are ADDITIVE (set() dedup at scan
    time), not overrides. Empty default = inherit tenant base unchanged.
    """

    model_config = ConfigDict(extra="forbid")
    forward: list[str] = Field(default_factory=list)
    retroactive: list[str] = Field(default_factory=list)


class BrandKitRow(BaseModel):
    """One visual identity (colors, fonts, logo, captions, auto-publish opt-in).

    Each tenant has zero or more kits; exactly one CAN be flagged
    `is_default=True` (enforced via a partial unique index in migration
    006). Kits are assigned per-speaker via `speakers.preferred_brand_kit_id`.

    Asset paths are storage-agnostic strings — local paths in dev,
    eventually S3/R2 keys in production via the Storage abstraction
    (see docs/production_deploy.md §3).
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    tenant_id: str
    name: str
    is_default: bool = False

    # Colors
    primary_color: str
    accent_color: str
    text_color: str = "#FFFFFF"

    # Typography
    font_family: str = "Inter"
    font_weight: int = 800

    # Assets
    logo_url: str | None = None
    logo_dark_url: str | None = None
    watermark_url: str | None = None
    intro_sting_url: str | None = None
    outro_sting_url: str | None = None

    # Caption style — opaque JSON; the renderer reads keys it knows.
    caption_style: dict[str, object] | None = None

    # Layout
    default_layout: str = "pip"  # pip | split_stack | blurred_bg

    # Social handles
    handle_tiktok: str | None = None
    handle_youtube: str | None = None
    handle_instagram: str | None = None
    handle_kick: str | None = None

    # AI generation metadata
    ai_generated: bool = False
    ai_prompt: str | None = None
    ai_provider: str | None = None

    # Per-kit auto-publish (default OFF — review-first per spec §9)
    auto_publish_enabled: bool = False
    auto_publish_platforms: list[str] = Field(default_factory=list)
    auto_publish_delay_min: int = 60

    # Per-kit custom trigger phrases — additively merged with the
    # tenant base at scan time.
    custom_trigger_phrases: CustomTriggerPhrases = Field(
        default_factory=CustomTriggerPhrases
    )

    # Slice H.1 — user-level editor preferences. The clip editor's
    # right panel auto-saves to these so re-opening any clip prefills
    # the operator's last choices instead of resetting to "off" /
    # platform defaults / placeholder URL.
    default_platform: str | None = None
    banner_enabled_default: bool = False
    banner_show_context_default: bool = False
    banner_show_safezones_default: bool = False

    created_at: str
    updated_at: str


# ---------- Drive watches (voice-markers spec slice E.4) ----------


class DriveWatchRow(BaseModel):
    """One row in `drive_watches` — a folder NexoClip polls for new VODs."""

    model_config = ConfigDict(extra="forbid")

    id: str
    tenant_id: str
    folder_id: str
    folder_name: str | None = None
    refresh_token: str
    access_token: str | None = None
    access_token_expires_at: str | None = None
    last_polled_at: str | None = None
    # Drive file IDs we've already ingested — the dedup key.
    seen_file_ids: list[str] = Field(default_factory=list)
    enabled: bool = True
    created_at: str
    updated_at: str
