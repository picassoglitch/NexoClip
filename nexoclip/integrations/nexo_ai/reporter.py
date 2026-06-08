"""Outbound usage reporter — pushes LLM consumption back to Nexo AI.

Called from LLMRouter._log after each successful Claude call. The flow:

  1. Resolve the tenant's external_user_id (the Nexo AI user_id we stored
     during provisioning).
  2. POST {NEXO_AI_BASE_URL}/api/engines/nexoclip/usage with:
       Authorization: Bearer {NEXO_AI_ADMIN_TOKEN}
       { external_user_id, events: [{kind, amount, source_id, occurred_at}] }
  3. Parse the response's `balance` block and persist it on the tenant row
     so the dashboard nav chip can read it without making its own call.
  4. Best-effort: timeout 3s, swallow all errors. The LLM call already
     happened — we never want to break a successful response on a network
     hiccup with the platform.

Idempotency: Nexo AI's usage_events table has UNIQUE (engine_id, source_id),
so re-runs are no-ops. We use the llm_calls.id ULID as the source_id.

When NEXO_AI_BASE_URL is unset (NexoClip running standalone), the reporter
is a complete no-op (logged at INFO level once per call so the operator
knows why nothing reaches Nexo AI). Same when the tenant has no
external_user_id (CLI-created, never linked).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from nexoclip.db import Database, TenantsRepo
from nexoclip.settings import get_settings

# Module-level logger — show INFO and above by default so the operator can
# see the reporter doing its job without flipping debug on. Errors get a
# loud .exception() so they're impossible to miss.
_log = logging.getLogger("nexoclip.nexo_ai.reporter")

# Hard cap on the outbound call. If Nexo AI is down or slow we don't want
# the LLM hot path to wait — 3s is plenty for a 1-event POST over the
# public internet under normal load.
_REPORT_TIMEOUT_S = 3.0


async def report_usage(
    db: Database,
    *,
    tenant_id: str,
    kind: str,
    amount: int,
    cost_usd_micros: int,
    source_id: str,
    occurred_at_iso: str,
    operation: str | None = None,
) -> None:
    """Push ONE usage event from ANY provider to Nexo AI + persist the
    returned balance. Never raises — all errors are logged and swallowed.

    Token T4 — this is the unified, provider-agnostic reporter. Every
    event carries BOTH the native `amount` + `kind` (e.g. llm.tokens /
    transcription.seconds) AND the real `cost_usd_micros`, so Nexo AI
    can price off real cost while still seeing what was consumed. The
    economics platform pulls true provider cost from this stream — not
    just Claude tokens.

      kind    — "llm.tokens", "transcription.seconds", ... (provider's
                native unit, namespaced)
      amount  — the count in that native unit (tokens, seconds, …)
      cost_usd_micros — the real USD cost of this usage, the universal
                unit Nexo AI deducts/prices against
      source_id — the originating row id (llm_calls.id) for idempotency
                (Nexo AI's usage_events has UNIQUE(engine, source_id))
    """
    settings = get_settings()
    base = settings.nexo_ai_base_url
    token = settings.nexo_ai_admin_token

    amount = max(0, int(amount))
    cost_usd_micros = max(0, int(cost_usd_micros))

    if not base:
        _log.info(
            "report skipped: NEXO_AI_BASE_URL unset · tenant=%s call=%s "
            "kind=%s amount=%d cost_um=%d",
            tenant_id, source_id, kind, amount, cost_usd_micros,
        )
        return
    if not token:
        _log.warning(
            "report skipped: NEXO_AI_ADMIN_TOKEN unset · tenant=%s call=%s",
            tenant_id, source_id,
        )
        await _record_status(
            db, tenant_id, ok=False, at_iso=occurred_at_iso,
            error="NEXO_AI_ADMIN_TOKEN not configured on this instance",
        )
        return

    # Resolve the tenant's Nexo AI user id. CLI-created tenants don't have one.
    try:
        tenant = await TenantsRepo(db).get(tenant_id)
    except Exception:
        _log.exception("report failed: tenant lookup error · tenant=%s", tenant_id)
        return
    if tenant is None:
        _log.warning("report skipped: tenant not found · tenant=%s", tenant_id)
        return
    if not tenant.external_user_id:
        _log.info(
            "report skipped: tenant has no external_user_id (not linked to Nexo AI) · tenant=%s",
            tenant_id,
        )
        return

    # Nothing consumed (zero count AND zero cost) → no event to send.
    if amount <= 0 and cost_usd_micros <= 0:
        return

    url = f"{base.rstrip('/')}/api/engines/nexoclip/usage"
    # Build a single-event POST. `operation` is forwarded so Nexo AI's
    # /app/usage page can collapse a multi-call pipeline run into one
    # row (it groups events by user+engine+operation+date). Optional —
    # omitted when the caller didn't pass a purpose, in which case
    # Nexo AI renders each event individually as before.
    event: dict[str, Any] = {
        "kind": kind,
        "amount": amount,
        "cost_usd_micros": cost_usd_micros,
        "source_id": source_id,
        "occurred_at": occurred_at_iso,
    }
    if operation:
        event["operation"] = operation
    body: dict[str, Any] = {
        "external_user_id": tenant.external_user_id,
        "events": [event],
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=_REPORT_TIMEOUT_S) as client:
            response = await client.post(url, json=body, headers=headers)
    except httpx.TimeoutException:
        _log.warning(
            "report failed: timeout · tenant=%s url=%s amount=%d",
            tenant_id, url, amount,
        )
        await _record_status(
            db, tenant_id, ok=False, at_iso=occurred_at_iso,
            error=f"timeout after {_REPORT_TIMEOUT_S:.0f}s contacting Nexo AI",
        )
        return
    except Exception as e:  # noqa: BLE001 — best-effort, never propagate
        _log.exception(
            "report failed: unexpected error · tenant=%s url=%s amount=%d",
            tenant_id, url, amount,
        )
        await _record_status(
            db, tenant_id, ok=False, at_iso=occurred_at_iso,
            error=f"network error: {type(e).__name__}",
        )
        return

    if response.status_code >= 400:
        _log.warning(
            "report rejected by Nexo AI: %d · tenant=%s body=%s",
            response.status_code, tenant_id, response.text[:300],
        )
        await _record_status(
            db, tenant_id, ok=False, at_iso=occurred_at_iso,
            error=(
                f"Nexo AI rejected the report: HTTP {response.status_code}"
                + (" (check NEXO_AI_ADMIN_TOKEN)" if response.status_code in (401, 403) else "")
            ),
        )
        return

    # Parse the balance returned by Nexo AI and stash it on the tenant row
    # so the dashboard nav chip can render without making its own call.
    try:
        data = response.json()
    except Exception:
        _log.warning("report ok but response JSON parse failed · tenant=%s", tenant_id)
        return

    balance = data.get("balance")
    if isinstance(balance, dict):
        try:
            await TenantsRepo(db).set_balance_cache(
                tenant_id,
                remaining=int(balance.get("remaining", 0)),
                unlimited=bool(balance.get("unlimited", False)),
                monthly_used=int(balance.get("monthlyUsed", 0)),
                at_iso=occurred_at_iso,
            )
        except Exception:
            _log.exception("balance cache update failed · tenant=%s", tenant_id)
            # Don't return — the report itself succeeded.

    # Success — clear any prior failure so the chip stops warning.
    await _record_status(db, tenant_id, ok=True, at_iso=occurred_at_iso)

    _log.info(
        "reported %s=%d (cost_um=%d) · tenant=%s call=%s · remaining=%s (unlimited=%s)",
        kind,
        amount,
        cost_usd_micros,
        tenant_id,
        source_id,
        balance.get("remaining") if isinstance(balance, dict) else "?",
        balance.get("unlimited") if isinstance(balance, dict) else "?",
    )


async def report_llm_usage(
    db: Database,
    *,
    tenant_id: str,
    llm_call_id: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd_micros: int = 0,
    occurred_at_iso: str,
    operation: str | None = None,
) -> None:
    """LLM (Claude) wrapper over report_usage — kind=llm.tokens, amount =
    input + output tokens. Kept as a named entry point for the router
    and its tests. cost_usd_micros is the router's computed cost so
    Nexo AI sees real USD, not just token count."""
    await report_usage(
        db,
        tenant_id=tenant_id,
        kind="llm.tokens",
        amount=max(0, int(input_tokens) + int(output_tokens)),
        cost_usd_micros=cost_usd_micros,
        source_id=llm_call_id,
        occurred_at_iso=occurred_at_iso,
        operation=operation,
    )


async def _record_status(
    db: Database,
    tenant_id: str,
    *,
    ok: bool,
    at_iso: str,
    error: str | None = None,
) -> None:
    """Persist the report outcome, best-effort. Never raises — this is
    observability for a fire-and-forget path; failing to record a
    failure must not itself blow up the background task."""
    try:
        await TenantsRepo(db).set_usage_report_status(
            tenant_id, ok=ok, at_iso=at_iso, error=error,
        )
    except Exception:  # noqa: BLE001 — observability never blocks
        _log.warning(
            "usage report status write failed · tenant=%s ok=%s", tenant_id, ok,
        )


def schedule_usage(
    db: Database,
    *,
    tenant_id: str,
    kind: str,
    amount: int,
    cost_usd_micros: int,
    source_id: str,
    occurred_at_iso: str,
    operation: str | None = None,
) -> None:
    """Fire-and-forget wrapper over report_usage. Spawns a background
    task so the caller (the LLM router's hot path, the transcribe
    service) doesn't pay the network round-trip. The task runs to
    completion even after the calling coroutine returns.

    Token T4 — the generic entry point for ANY provider. Tracks the
    task on a module-level set so the asyncio GC doesn't cancel it
    prematurely (https://docs.python.org/3/library/asyncio-task.html)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _log.warning(
            "schedule_usage: no running event loop — skipping. tenant=%s call=%s",
            tenant_id, source_id,
        )
        return

    task = loop.create_task(
        report_usage(
            db,
            tenant_id=tenant_id,
            kind=kind,
            amount=amount,
            cost_usd_micros=cost_usd_micros,
            source_id=source_id,
            occurred_at_iso=occurred_at_iso,
            operation=operation,
        )
    )
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


def schedule_report(
    db: Database,
    *,
    tenant_id: str,
    llm_call_id: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd_micros: int = 0,
    occurred_at_iso: str,
    operation: str | None = None,
) -> None:
    """LLM (Claude) fire-and-forget wrapper — kept as the router's entry
    point. Delegates to schedule_usage with kind=llm.tokens."""
    schedule_usage(
        db,
        tenant_id=tenant_id,
        kind="llm.tokens",
        amount=max(0, int(input_tokens) + int(output_tokens)),
        cost_usd_micros=cost_usd_micros,
        source_id=llm_call_id,
        occurred_at_iso=occurred_at_iso,
        operation=operation,
    )


# Strong references to in-flight tasks so they don't get GC'd before
# completion. The done callback above clears each entry once finished.
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()
