"""upload-post.com integration — multi-platform publishing layer.

NexoClip-the-company holds ONE upload-post API key. Each NexoClip
tenant maps to ONE upload-post "user profile" (their multi-tenant
primitive). Tenants connect their TikTok / IG / YT / X / LinkedIn
etc. accounts via upload-post's hosted UI (we mint a JWT magic
link); upload-post stores tokens; we never see them, never refresh
them. To publish, we POST a video URL + target platforms to their
/api/upload endpoint with the tenant's profile name as `user`.

  client.py    — async httpx wrapper around the upload-post REST API
  profiles.py  — ensure_profile_for_tenant() — create-or-fetch +
                 persist on tenant row
"""

from .client import (
    UploadPostClient,
    UploadPostError,
    UploadPostProfile,
    UploadJwtLink,
    UploadResult,
    UploadStatus,
)
from .profiles import ensure_profile_for_tenant

__all__ = [
    "UploadPostClient",
    "UploadPostError",
    "UploadPostProfile",
    "UploadJwtLink",
    "UploadResult",
    "UploadStatus",
    "ensure_profile_for_tenant",
]
