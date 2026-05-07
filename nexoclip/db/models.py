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
