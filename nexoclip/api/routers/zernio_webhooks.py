"""Inbound Zernio webhook receiver.

Zernio POSTs post-status events here (published / failed / etc.) so we
don't have to poll. There's no tenant cookie on a server-to-server
callback — the transport auth IS the HMAC signature, verified before
we trust the body. This router is exempt from the bearer-cookie
middleware (see `/api/webhooks/` in auth._PUBLIC_PREFIXES); the handler
enforces its own signature check.

v1 scope: verify + log + 200. Recording an `Event` for the outbound
webhook fanout is a follow-up — it needs a profileId → tenant_id
reverse lookup, and the exact Zernio payload contract is still being
confirmed against real traffic.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from nexoclip.integrations.zernio.webhooks import (
    SIGNATURE_HEADER,
    parse_post_event,
    verify_zernio_signature,
)
from nexoclip.settings import get_settings

_log = logging.getLogger("nexoclip.api.zernio_webhooks")

router = APIRouter(prefix="/api/webhooks", tags=["zernio"], include_in_schema=False)


@router.post("/zernio")
async def zernio_webhook(request: Request) -> JSONResponse:
    """Receive + verify a Zernio webhook event.

    503 when the verification secret isn't configured (we refuse to
    trust an event we can't verify), 403 on signature mismatch. On a
    valid event we log it and ack 200 so Zernio doesn't retry.
    """
    settings = get_settings()
    secret = (settings.zernio_webhook_secret or "").strip()
    if not secret:
        raise HTTPException(
            status_code=503,
            detail=(
                "NEXOCLIP_ZERNIO_WEBHOOK_SECRET is not configured; "
                "cannot verify inbound Zernio webhooks."
            ),
        )

    body = await request.body()
    provided = request.headers.get(SIGNATURE_HEADER)
    if not verify_zernio_signature(
        secret=secret, body=body, signature_header=provided,
    ):
        # Log the received header (NOT the secret) so we can confirm
        # Zernio's real signature scheme during rollout, then enforce.
        _log.warning(
            "zernio.webhook.bad_signature header=%r len=%d",
            provided, len(body),
        )
        raise HTTPException(status_code=403, detail="invalid signature")

    try:
        payload = json.loads(body)
    except ValueError:
        raise HTTPException(status_code=400, detail="body is not JSON") from None
    event = parse_post_event(payload) if isinstance(payload, dict) else {}

    _log.info(
        "zernio.webhook.received event=%s post_id=%s status=%s profile_id=%s",
        event.get("event"), event.get("post_id"),
        event.get("status"), event.get("profile_id"),
    )
    # TODO: map event.profile_id → tenant_id and record an Event via
    # EventsRepo so the outbound webhook fanout can relay to subscribers.
    return JSONResponse({"ok": True})


__all__ = ["router"]
