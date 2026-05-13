"""Retention sweeper — voice-markers spec slice E.1 / §9 locked defaults.

Daily cron deletes VOD, clip, and transcript artifacts past their
per-tenant retention windows. Hard-delete only — no soft-delete /
trash bin per spec.

System defaults (when a tenant's column is NULL):
    retention_vod_days        = 30
    retention_clip_days       = 90
    retention_transcript_days = 365

Tenants override via `TenantsRepo.set_retention(...)`.

Run via:
    nexoclip retention sweep              # all tenants
    nexoclip retention sweep --tenant <id>
    nexoclip retention sweep --dry-run    # report only, no deletes

The sweeper is idempotent — re-running it after a partial failure picks
up where it left off (DB rows beyond the window get re-considered;
already-deleted files are skipped via `missing_ok=True`).
"""

from __future__ import annotations

from .service import (
    DEFAULT_RETENTION_CLIP_DAYS,
    DEFAULT_RETENTION_TRANSCRIPT_DAYS,
    DEFAULT_RETENTION_VOD_DAYS,
    RetentionPolicy,
    RetentionReport,
    sweep_retention,
)

__all__ = [
    "DEFAULT_RETENTION_CLIP_DAYS",
    "DEFAULT_RETENTION_TRANSCRIPT_DAYS",
    "DEFAULT_RETENTION_VOD_DAYS",
    "RetentionPolicy",
    "RetentionReport",
    "sweep_retention",
]
