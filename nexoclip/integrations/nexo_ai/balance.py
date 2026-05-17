"""On-demand balance fetcher.

The reporter (reporter.py) updates the balance cache as a side-effect of
LLM calls — but a user who hasn't run any AI work yet has no cache, so the
nav chip shows the muted "— tokens" placeholder. This module exposes an
explicit fetch so the chip can be clicked to populate the cache without
waiting for the next LLM call.

GET {NEXO_AI_BASE_URL}/api/engines/nexoclip/usage/balance?external_user_id=<id>
  Auth: Bearer NEXO_AI_ADMIN_TOKEN
  Returns: { balance: { remaining, unlimited, monthlyUsed, periodStart, ... } }

Always best-effort: timeout 5s, swallow on error. The chip can show stale
data rather than break the dashboard render.
"""

from __future__ import annotations

import datetime as _dt
import logging

import httpx

from nexoclip.db import Database, TenantsRepo
from nexoclip.settings import get_settings

_log = logging.getLogger("nexoclip.nexo_ai.balance")

_FETCH_TIMEOUT_S = 5.0


async def fetch_balance_now(db: Database, *, tenant_id: str) -> bool:
    """Force-fetch the current Nexo AI balance for this tenant + update cache.

    Returns True if cache was updated, False otherwise. Reasons for False
    show up in the log so the operator can debug the silent path.
    """
    settings = get_settings()
    base = settings.nexo_ai_base_url
    token = settings.nexo_ai_admin_token

    if not base:
        _log.info("fetch skipped: NEXO_AI_BASE_URL unset · tenant=%s", tenant_id)
        return False
    if not token:
        _log.warning("fetch skipped: NEXO_AI_ADMIN_TOKEN unset · tenant=%s", tenant_id)
        return False

    try:
        tenant = await TenantsRepo(db).get(tenant_id)
    except Exception:
        _log.exception("fetch failed: tenant lookup error · tenant=%s", tenant_id)
        return False
    if tenant is None or not tenant.external_user_id:
        _log.info(
            "fetch skipped: tenant has no external_user_id · tenant=%s", tenant_id
        )
        return False

    url = (
        f"{base.rstrip('/')}/api/engines/nexoclip/usage/balance"
        f"?external_user_id={tenant.external_user_id}"
    )
    headers = {"Authorization": f"Bearer {token}"}

    try:
        async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT_S) as client:
            response = await client.get(url, headers=headers)
    except httpx.TimeoutException:
        _log.warning("fetch failed: timeout · tenant=%s url=%s", tenant_id, url)
        return False
    except Exception:  # noqa: BLE001
        _log.exception("fetch failed: unexpected error · tenant=%s", tenant_id)
        return False

    if response.status_code >= 400:
        _log.warning(
            "fetch rejected by Nexo AI: %d · tenant=%s body=%s",
            response.status_code, tenant_id, response.text[:300],
        )
        return False

    try:
        balance = response.json().get("balance") or {}
    except Exception:
        _log.warning("fetch ok but JSON parse failed · tenant=%s", tenant_id)
        return False

    now_iso = _dt.datetime.now(_dt.UTC).isoformat()
    try:
        await TenantsRepo(db).set_balance_cache(
            tenant_id,
            remaining=int(balance.get("remaining", 0)),
            unlimited=bool(balance.get("unlimited", False)),
            monthly_used=int(balance.get("monthlyUsed", 0)),
            at_iso=now_iso,
        )
    except Exception:
        _log.exception("fetch ok but cache update failed · tenant=%s", tenant_id)
        return False

    _log.info(
        "balance fetched · tenant=%s remaining=%s unlimited=%s",
        tenant_id,
        balance.get("remaining"),
        balance.get("unlimited"),
    )
    return True
