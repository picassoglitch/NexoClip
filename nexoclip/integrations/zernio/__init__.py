"""Zernio integration — multi-platform publishing layer.

NexoClip-the-company holds ONE Zernio API key (a `sk_...` bearer
secret). Each NexoClip tenant maps to ONE Zernio `profileId` (their
multi-tenant primitive). Tenants connect their TikTok / IG / YT / X /
LinkedIn etc. accounts via Zernio's hosted OAuth (we mint a per-
platform `authUrl`); Zernio stores tokens; we never see them, never
refresh them. To publish, we POST a video URL + target accounts to
Zernio's /posts endpoint, scoped by the tenant's `profileId`.

  client.py    — async httpx wrapper around the Zernio REST API
  profiles.py  — create_profile_for_tenant() — POST /profiles + persist
                 the returned profileId + name on the tenant row
  webhooks.py  — inbound webhook signature verification + parsing

Replaces the upload-post.com integration (commit history under
`integrations/upload_post/`).
"""

from .client import (
    ZernioAccount,
    ZernioClient,
    ZernioConnectLink,
    ZernioError,
    ZernioFacebookPage,
    ZernioPostResult,
    ZernioPostStatus,
    ZernioProfile,
)
from .events import process_pending, process_zernio_event
from .profiles import create_profile_for_tenant
from .webhooks import (
    parse_post_event,
    register_zernio_webhook,
    verify_zernio_signature,
)

__all__ = [
    "ZernioAccount",
    "ZernioClient",
    "ZernioConnectLink",
    "ZernioError",
    "ZernioFacebookPage",
    "ZernioPostResult",
    "ZernioPostStatus",
    "ZernioProfile",
    "create_profile_for_tenant",
    "parse_post_event",
    "process_pending",
    "process_zernio_event",
    "register_zernio_webhook",
    "verify_zernio_signature",
]
