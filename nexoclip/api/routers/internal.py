"""Internal endpoints — slice O.44 + publisher (Zernio) fetch target.

Both routes here share an HMAC-signed-URL auth model that's separate
from the rest of the API:

  - No tenant cookie / bearer required.
  - URL is signed with NEXOCLIP_INTERNAL_SIGNING_SECRET.
  - Signature binds (resource_id, tenant_id, expiry_unix_ts).
  - Expiry is bounded; anything past TTL 403s.

Why: external callers (Modal Whisper, Zernio) can't carry our
operator session cookie. Short-lived signed URLs are bounded — if
one leaks, the worst case is one media file is exposed for the
remaining TTL window. The signing secret never crosses the wire.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from nexoclip.db import ClipsRepo, Database, StreamsRepo
from nexoclip.settings import get_settings
from nexoclip.tenancy import bound_tenant

router = APIRouter(prefix="/api/internal", tags=["internal"], include_in_schema=False)


def _verify_signed_params(
    *,
    resource_id: str,
    tenant: str,
    exp: int,
    sig: str,
    max_ttl_s: int,
) -> None:
    """Shared HMAC + expiry check. Raises HTTPException on any failure.

    `max_ttl_s` upper-bounds the future expiry an attacker can claim
    — defense in depth against a leaked secret minting essentially-
    permanent URLs.
    """
    settings = get_settings()
    secret = (settings.internal_signing_secret or "").strip()
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="server not configured for signed-URL access",
        )
    if not tenant or not exp or not sig:
        raise HTTPException(status_code=400, detail="missing signature params")

    now = int(time.time())
    if int(exp) < now:
        raise HTTPException(status_code=403, detail="signed URL expired")
    if int(exp) > now + max_ttl_s:
        raise HTTPException(status_code=403, detail="signed URL expiry implausible")

    msg = f"{resource_id}|{tenant}|{int(exp)}".encode()
    expected = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        raise HTTPException(status_code=403, detail="signature mismatch")


def mint_signed_clip_url(
    *,
    clip_id: str,
    tenant_id: str,
    base_url: str,
    ttl_seconds: int = 3600,
) -> str:
    """Helper used by the publish router to build the URL we hand to
    the publishing vendor (Zernio). The signature binds (clip_id,
    tenant_id, exp) so a leaked URL only exposes one clip for at most
    `ttl_seconds`."""
    settings = get_settings()
    secret = (settings.internal_signing_secret or "").strip()
    if not secret:
        raise RuntimeError(
            "NEXOCLIP_INTERNAL_SIGNING_SECRET is not configured; "
            "cannot mint signed clip URL for the publisher"
        )
    exp = int(time.time()) + int(ttl_seconds)
    msg = f"{clip_id}|{tenant_id}|{exp}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    base = base_url.rstrip("/")
    return (
        f"{base}/api/internal/clip/{clip_id}"
        f"?tenant={tenant_id}&exp={exp}&sig={sig}"
    )


def sign_render_query(
    *, clip_id: str, tenant_id: str, ttl_seconds: int = 600
) -> str:
    """Just the signed query string (`tenant=..&exp=..&sig=..`) for the
    `/render` page — the `auth_query` the background recorder appends to
    authenticate without a cookie. Same HMAC scheme as the signed clip URL."""
    settings = get_settings()
    secret = (settings.internal_signing_secret or "").strip()
    if not secret:
        raise RuntimeError(
            "NEXOCLIP_INTERNAL_SIGNING_SECRET is not configured; cannot sign "
            "a render URL for background auto-publish"
        )
    exp = int(time.time()) + int(ttl_seconds)
    msg = f"{clip_id}|{tenant_id}|{exp}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    return f"tenant={tenant_id}&exp={exp}&sig={sig}"


def mint_signed_render_url(
    *,
    clip_id: str,
    tenant_id: str,
    base_url: str,
    ttl_seconds: int = 600,
) -> str:
    """Build a signed `/dashboard/clips/{id}/render` URL the headless
    recorder can open when there is NO operator cookie — i.e. auto-publish
    hands-free, which renders from the background pipeline.

    Same HMAC scheme as `mint_signed_clip_url` (binds clip_id, tenant_id,
    exp). Short TTL by default — the render starts within seconds. The
    `/render` handler verifies it via `_verify_signed_params` and binds
    that tenant, an alternative to the cookie path."""
    settings = get_settings()
    secret = (settings.internal_signing_secret or "").strip()
    if not secret:
        raise RuntimeError(
            "NEXOCLIP_INTERNAL_SIGNING_SECRET is not configured; cannot mint "
            "a signed render URL for background auto-publish"
        )
    exp = int(time.time()) + int(ttl_seconds)
    msg = f"{clip_id}|{tenant_id}|{exp}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    base = base_url.rstrip("/")
    return (
        f"{base}/dashboard/clips/{clip_id}/render"
        f"?capture=1&tenant={tenant_id}&exp={exp}&sig={sig}"
    )


@router.get("/audio/{stream_id}")
async def fetch_audio_for_transcribe(
    stream_id: str,
    request: Request,
    tenant: str = "",
    exp: int = 0,
    sig: str = "",
) -> FileResponse:
    """Serve the stream's source audio if (stream_id, tenant, exp) HMAC checks."""
    _verify_signed_params(
        resource_id=stream_id,
        tenant=tenant,
        exp=exp,
        sig=sig,
        max_ttl_s=24 * 3600,
    )

    # The dashboard binds tenants via middleware — for this admin-less
    # path we bind manually to the tenant claim in the signed URL.
    db: Database = request.app.state.db
    with bound_tenant(tenant):
        stream = await StreamsRepo(db).get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="stream not found")

    audio_path = Path(stream.source_audio_path)
    if not audio_path.exists():
        raise HTTPException(
            status_code=410,
            detail=f"audio extract missing from disk: {audio_path}",
        )

    return FileResponse(
        path=audio_path,
        media_type="audio/wav",
        filename=f"nexoclip_{stream_id}.wav",
    )


# GET + HEAD: Zernio probes the media URL with HEAD before downloading
# it — a GET-only route 405s the probe. FileResponse handles HEAD
# natively (headers only, no body).
@router.api_route("/clip/{clip_id}", methods=["GET", "HEAD"])
async def fetch_clip_for_publisher(
    clip_id: str,
    request: Request,
    tenant: str = "",
    exp: int = 0,
    sig: str = "",
) -> FileResponse:
    """Serve the rendered clip MP4 to the publishing vendor (Zernio) or
    any caller with a valid signed URL.

    Used by the publish router: when we call Zernio's `POST /posts`,
    each `mediaItems[].url` points here. Zernio downloads the file,
    re-hosts it, then publishes to each target platform. TTL is wider
    than the audio path (1h ceiling + 24h cap) because the per-platform
    pipeline can take several minutes for a 4K clip across 5 platforms.

    Serves ONLY the FINAL rendered MP4 (overlays + libass captions
    burned in) from the 1080-cache path. It deliberately does NOT fall
    back to the raw pre-render source: that fallback was the bug where
    a published clip went out missing its hooks + captions while the
    operator's download had them. The publish path
    (`zernio._publish_clip` → `ensure_clip_rendered`) renders this file
    before it hands us the URL, so by the time Zernio fetches it's on
    disk; if it somehow isn't, we 409 so the post errors loudly rather
    than shipping the unedited clip.
    """
    _verify_signed_params(
        resource_id=clip_id,
        tenant=tenant,
        exp=exp,
        sig=sig,
        max_ttl_s=24 * 3600,
    )

    db: Database = request.app.state.db
    with bound_tenant(tenant):
        clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="clip not found")

    from nexoclip.api._render_validation import is_servable_cached_mp4

    rendered = Path(clip.path).parent / "clip_render_1080.mp4"
    # Gate on is_servable_cached_mp4 (size floor + ISO BMFF magic) so a
    # partial/0-byte file from an aborted encode is treated as "not
    # ready" rather than served as a corrupt download.
    if not is_servable_cached_mp4(rendered):
        raise HTTPException(
            status_code=409,
            detail=(
                "clip render not ready (hooks + captions not burned in "
                "yet). Re-publish from the dashboard — the render runs "
                "before the post is sent."
            ),
        )

    return FileResponse(
        path=rendered,
        media_type="video/mp4",
        filename=f"nexoclip_{clip_id}.mp4",
    )
