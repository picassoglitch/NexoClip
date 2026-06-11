"""Zernio webhook event processing — the hub's event backbone.

The receiver (routers/zernio_webhooks.py) verifies + dedup-inserts the
raw event and ACKs fast; THIS module does the actual work in a
background task:

  1. Resolve the tenant: profileId → tenants.zernio_profile_id, or the
     post id → zernio_publishes (posts fired through the hub).
  2. Update local state: post.* events feed zernio_publishes.status so
     the dashboard/internal API read live status without polling.
     account.* needs no local write — connection chips read Zernio
     live (more accurate than a mirror). post.external.*, comment.*,
     message.* stay in zernio_events verbatim; the calendar (phase 8)
     and inbox (phase 9) stores consume them from there.
  3. Fan out: record a `zernio.<type>` row in the events table and
     drain the tenant's webhook subscriptions (existing HMAC-signed
     dispatcher) so NexoOBS / Nexo AI engines get the relay.

Everything here is idempotent: a redelivery never reaches processing
(dedup at insert), and a re-run of process_event on an already-
processed row is a no-op.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from nexoclip.db import (
    Database,
    EventsRepo,
    TenantsRepo,
    ZernioEventsRepo,
    ZernioPublishesRepo,
)
from nexoclip.tenancy import bound_tenant

_log = logging.getLogger("nexoclip.integrations.zernio.events")

# Post-level events → the status we persist on zernio_publishes.
# The payload's own post.status wins when present; this is the
# fallback derived from the event name.
POST_EVENT_STATUS: dict[str, str] = {
    "post.scheduled": "scheduled",
    "post.published": "published",
    "post.failed": "failed",
    "post.partial": "partial",
    "post.cancelled": "cancelled",
    "post.recycled": "published",
}

# Events the hub subscribes to on Zernio (register_zernio_webhook).
# Only enum-valid names from the OpenAPI spec — conversation.started
# is documented in prose but absent from the subscription enum, so we
# can't subscribe to it (we still process it if it ever arrives).
SUBSCRIBED_EVENTS: tuple[str, ...] = (
    "post.scheduled",
    "post.published",
    "post.failed",
    "post.partial",
    "post.cancelled",
    "post.external.created",
    "post.external.updated",
    "post.external.deleted",
    "account.connected",
    "account.disconnected",
    "comment.received",
    "message.received",
    "message.sent",
)


def extract_profile_id(payload: dict[str, Any]) -> str | None:
    """Pull the Zernio profileId out of any documented payload shape.

    account.* carries it at account.profileId; other shapes have used a
    top-level or post-nested profileId. Tolerates the nested-object
    form ({_id: ...}) — same regression the accounts list had."""
    candidates: list[Any] = [payload.get("profileId")]
    for key in ("account", "post", "data"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            candidates.append(nested.get("profileId"))
    for c in candidates:
        if isinstance(c, dict):
            c = c.get("_id") or c.get("id")
        if isinstance(c, str) and c:
            return c
    return None


def _extract_post(payload: dict[str, Any]) -> dict[str, Any] | None:
    post = payload.get("post")
    return post if isinstance(post, dict) else None


def _event_summary(type_: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Small primitive-typed payload for the fan-out event row.

    Ids + statuses only — never message/comment text (subscribers that
    need the full body fetch it via the API; our events table and the
    relay bodies stay free of user content)."""
    out: dict[str, Any] = {"zernio_event": type_}
    post = _extract_post(payload)
    if post:
        out["post_id"] = post.get("id") or post.get("_id")
        out["status"] = post.get("status")
    account = payload.get("account")
    if isinstance(account, dict):
        out["account_id"] = account.get("accountId")
        out["platform"] = account.get("platform")
    profile_id = extract_profile_id(payload)
    if profile_id:
        out["profile_id"] = profile_id
    return {k: v for k, v in out.items() if v is not None}


async def process_zernio_event(db: Database, event_id: str) -> None:
    """Process ONE stored webhook event. Safe to re-run; never raises
    (a webhook background task must not take the worker down)."""
    try:
        await _process(db, event_id)
    except Exception as e:
        _log.warning(
            "zernio.event.process_failed event_id=%s err=%s", event_id, e,
        )


async def _process(db: Database, event_id: str) -> None:
    events_repo = ZernioEventsRepo(db)
    row = await events_repo.get(event_id)
    if row is None or row.processed:
        return

    try:
        payload = json.loads(row.payload)
    except ValueError:
        payload = None
    if not isinstance(payload, dict):
        # Malformed but stored — mark processed so the sweep doesn't
        # spin on it; the raw body stays for diagnosis.
        await events_repo.mark_processed(event_id)
        return

    # --- tenant resolution ---
    tenant_id = row.tenant_id
    profile_id = row.profile_id or extract_profile_id(payload)
    if tenant_id is None and profile_id:
        tenant = await TenantsRepo(db).find_by_zernio_profile(profile_id)
        tenant_id = tenant.id if tenant else None

    # --- local state: post.* feeds the publish record ---
    post = _extract_post(payload)
    post_id = None
    if post:
        pid = post.get("id") or post.get("_id")
        post_id = pid if isinstance(pid, str) and pid else None
    if post_id and (
        row.type in POST_EVENT_STATUS or row.type.startswith("post.platform.")
    ):
        pub = await ZernioPublishesRepo(db).get_by_post_id(post_id)
        if pub is not None:
            tenant_id = tenant_id or pub.tenant_id
            status = post.get("status") if post else None
            if not isinstance(status, str) or not status:
                status = POST_EVENT_STATUS.get(row.type)
            platforms = post.get("platforms") if post else None
            platforms_json = (
                json.dumps(platforms) if isinstance(platforms, list) else None
            )
            if status:
                await ZernioPublishesRepo(db).set_status(
                    post_id, status=status, platforms_json=platforms_json,
                )

    # --- fan-out: event row + drain the tenant's webhook subscriptions ---
    if tenant_id:
        try:
            with bound_tenant(tenant_id):
                await EventsRepo(db).emit(
                    type=f"zernio.{row.type}",
                    payload=_event_summary(row.type, payload),
                )
            from nexoclip.webhooks import run_webhook_dispatch

            await run_webhook_dispatch(tenant_id, db)
        except Exception as e:
            _log.warning(
                "zernio.event.fanout_failed event_id=%s tenant=%s err=%s",
                event_id, tenant_id, e,
            )
    else:
        _log.info(
            "zernio.event.unresolved event_id=%s type=%s profile_id=%s",
            event_id, row.type, profile_id,
        )

    await events_repo.mark_processed(event_id, tenant_id=tenant_id)


async def process_pending(db: Database, *, limit: int = 100) -> int:
    """Sweep events whose background task died before mark_processed.

    Returns the number of events processed. Idempotent — safe to run
    on a schedule or at startup."""
    rows = await ZernioEventsRepo(db).list_unprocessed(limit=limit)
    for row in rows:
        await process_zernio_event(db, row.event_id)
    return len(rows)


__all__ = [
    "POST_EVENT_STATUS",
    "SUBSCRIBED_EVENTS",
    "extract_profile_id",
    "process_pending",
    "process_zernio_event",
]
