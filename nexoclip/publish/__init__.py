"""Publish-job worker.

Drains pending `publish_jobs` rows, posts them to the connected platform
(Phase 1: Buffer), records the result, retries with backoff on transient
failures, and gives up after `max_attempts`. The same `run_publish_jobs`
entry point is called from:

    1. The `nexoclip publish --tenant <id>` CLI (one-shot drain).
    2. A FastAPI lifespan background task (every 60s while the API is up).

Phase 2 adds native TikTok / YT Shorts publishers behind the same
`Publisher` protocol; the orchestration here doesn't change.
"""

from .buffer import BufferClient, BufferError
from .instagram import InstagramClient, InstagramError
from .oauth import RefreshedToken, refresh_if_expiring
from .protocol import PublisherError
from .service import PublishOutcome, run_publish_jobs
from .tiktok import TikTokClient, TikTokError
from .youtube import YouTubeClient, YouTubeError

__all__ = [
    "BufferClient",
    "BufferError",
    "InstagramClient",
    "InstagramError",
    "PublishOutcome",
    "PublisherError",
    "RefreshedToken",
    "TikTokClient",
    "TikTokError",
    "YouTubeClient",
    "YouTubeError",
    "refresh_if_expiring",
    "run_publish_jobs",
]
