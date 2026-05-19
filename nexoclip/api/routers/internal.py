"""Internal endpoints — slice O.44.

The only route here today is the HMAC-signed audio-fetch endpoint
the Modal Whisper provider pulls from. The auth model is intentionally
different from the rest of the API:

  - No tenant cookie / bearer required.
  - URL is signed with NEXOCLIP_INTERNAL_SIGNING_SECRET.
  - Signature binds (stream_id, tenant_id, expiry_unix_ts).
  - Expiry is 30 min after the URL is minted; anything older 403s.

Why a separate auth scheme: Modal can't carry the operator's session
cookie. We could store a long-lived service token, but a short-lived
signed URL is bounded — if it leaks, the worst case is one audio
extract is exposed for at most 30 min. The signing secret never
crosses the wire.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from nexoclip.db import Database, StreamsRepo
from nexoclip.settings import get_settings
from nexoclip.tenancy import bound_tenant

router = APIRouter(prefix="/api/internal", tags=["internal"], include_in_schema=False)


@router.get("/audio/{stream_id}")
async def fetch_audio_for_transcribe(
    stream_id: str,
    request: Request,
    tenant: str = "",
    exp: int = 0,
    sig: str = "",
) -> FileResponse:
    """Serve the stream's source audio if (stream_id, tenant, exp) HMAC checks."""
    settings = get_settings()
    secret = (settings.internal_signing_secret or "").strip()
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="server not configured for internal audio fetch",
        )
    if not tenant or not exp or not sig:
        raise HTTPException(status_code=400, detail="missing signature params")

    now = int(time.time())
    if int(exp) < now:
        raise HTTPException(status_code=403, detail="signed URL expired")
    # Reject far-future expiries too — defense in depth against a
    # leaked secret minting permanent URLs.
    if int(exp) > now + 24 * 3600:
        raise HTTPException(status_code=403, detail="signed URL expiry implausible")

    msg = f"{stream_id}|{tenant}|{int(exp)}".encode()
    expected = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        raise HTTPException(status_code=403, detail="signature mismatch")

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
