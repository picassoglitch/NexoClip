"""Zernio integration — multi-platform publishing layer.

NexoClip-the-company holds ONE Zernio API key (a `sk_...` bearer
secret). Each NexoClip tenant maps to ONE Zernio `profileId` (their
multi-tenant primitive). Tenants connect their TikTok / IG / YT / X /
LinkedIn etc. accounts via Zernio's hosted OAuth (we mint a per-
platform `authUrl`); Zernio stores tokens; we never see them, never
refresh them. To publish, we POST a video URL + target accounts to
Zernio's /posts endpoint, scoped by the tenant's `profileId`.

  client.py    — async httpx wrapper around the Zernio REST API
  profiles.py  — ensure_profile_for_tenant() — derive + persist the
                 tenant's profileId on the tenant row
  webhooks.py  — inbound webhook signature verification + parsing

Replaces the upload-post.com integration (commit history under
`integrations/upload_post/`).
"""

from .client import (
    ZernioAccount,
    ZernioClient,
    ZernioConnectLink,
    ZernioError,
    ZernioPostResult,
    ZernioPostStatus,
)
from .profiles import ensure_profile_for_tenant
from .webhooks import parse_post_event, verify_zernio_signature

__all__ = [
    "ZernioAccount",
    "ZernioClient",
    "ZernioConnectLink",
    "ZernioError",
    "ZernioPostResult",
    "ZernioPostStatus",
    "ensure_profile_for_tenant",
    "parse_post_event",
    "verify_zernio_signature",
]
