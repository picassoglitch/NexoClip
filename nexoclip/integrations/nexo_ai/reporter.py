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


async def report_llm_usage(
    db: Database,
    *,
    tenant_id: str,
    llm_call_id: str,
    input_tokens: int,
    output_tokens: int,
    occurred_at_iso: str,
    operation: str | None = None,
) -> None:
    """Push one LLM call's token consumption to Nexo AI + persist the
    returned balance on the tenant row. Never raises — all errors are
    logged and swallowed."""
    settings = get_settings()
    base = settings.nexo_ai_base_url
    token = settings.nexo_ai_admin_token

    if not base:
        _log.info(
            "report skipped: NEXO_AI_BASE_URL unset · tenant=%s call=%s (%d+%d tokens)",
            tenant_id, llm_call_id, input_tokens, output_tokens,
        )
        return
    if not token:
        _log.warning(
            "report skipped: NEXO_AI_ADMIN_TOKEN unset · tenant=%s call=%s",
            tenant_id, llm_call_id,
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

    amount = max(0, int(input_tokens) + int(output_tokens))
    if amount == 0:
        return

    url = f"{base.rstrip('/')}/api/engines/nexoclip/usage"
    # Build a single-event POST. `operation` is forwarded so Nexo AI's
    # /app/usage page can collapse a multi-call pipeline run into one
    # row (it groups events by user+engine+operation+date). Optional —
    # omitted when the caller didn't pass a purpose, in which case
    # Nexo AI renders each event individually as before.
    event: dict[str, Any] = {
        "kind": "llm.tokens",
        "amount": amount,
        "source_id": llm_call_id,
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
        return
    except Exception:  # noqa: BLE001 — best-effort, never propagate
        _log.exception(
            "report failed: unexpected error · tenant=%s url=%s amount=%d",
            tenant_id, url, amount,
        )
        return

    if response.status_code >= 400:
        _log.warning(
            "report rejected by Nexo AI: %d · tenant=%s body=%s",
            response.status_code, tenant_id, response.text[:300],
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

    _log.info(
        "reported %d tokens · tenant=%s call=%s · remaining=%s (unlimited=%s)",
        amount,
        tenant_id,
        llm_call_id,
        balance.get("remaining") if isinstance(balance, dict) else "?",
        balance.get("unlimited") if isinstance(balance, dict) else "?",
    )


def schedule_report(
    db: Database,
    *,
    tenant_id: str,
    llm_call_id: str,
    input_tokens: int,
    output_tokens: int,
    occurred_at_iso: str,
    operation: str | None = None,
) -> None:
    """Fire-and-forget wrapper. Spawns a task so the caller (LLMRouter._log)
    doesn't pay the network round-trip on the hot path. The task runs to
    completion in the background even after the calling coroutine returns.

    We track the task on a module-level set to keep the asyncio garbage
    collector from cancelling it prematurely — see
    https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _log.warning(
            "schedule_report: no running event loop — skipping. tenant=%s call=%s",
            tenant_id, llm_call_id,
        )
        return

    task = loop.create_task(
        report_llm_usage(
            db,
            tenant_id=tenant_id,
            llm_call_id=llm_call_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            occurred_at_iso=occurred_at_iso,
            operation=operation,
        )
    )
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


# Strong references to in-flight tasks so they don't get GC'd before
# completion. The done callback above clears each entry once finished.
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()
