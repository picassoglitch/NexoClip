"""Dashboard surfaces the cost projection cards on /llm-calls."""

from __future__ import annotations

import datetime as _dt

import httpx

from nexoclip.db import Database, LLMCallsRepo
from nexoclip.db.models import LLMCallRow
from nexoclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def test_dashboard_llm_calls_renders_projection_cards(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    tenant_id = tenants["alice"]["id"]
    with bound_tenant(tenant_id):
        await LLMCallsRepo(db).record(
            LLMCallRow(
                id="llm_one",
                tenant_id=tenant_id,
                purpose="variant_generation",
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                quality="standard",
                input_tokens=100,
                output_tokens=50,
                cost_usd_micros=2_500_000,
                status="ok",
                error=None,
                attempts=1,
                ts=_now(),
            )
        )

    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/llm-calls")
    assert r.status_code == 200
    assert "Today" in r.text
    assert "Month-to-date" in r.text
    assert "Projected EOM" in r.text
    assert "By purpose (MTD)" in r.text
    assert "By model (MTD)" in r.text


async def test_dashboard_llm_calls_renders_budget_bar_when_set(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """When the tenant has a budget cap, the headroom bar shows."""
    from nexoclip.db import TenantsRepo

    tenant_id = tenants["alice"]["id"]
    await TenantsRepo(db).set_budget(
        tenant_id, daily_llm_budget_usd_micros=10_000_000
    )
    with bound_tenant(tenant_id):
        await LLMCallsRepo(db).record(
            LLMCallRow(
                id="llm_b",
                tenant_id=tenant_id,
                purpose="variant_generation",
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                quality="standard",
                input_tokens=100,
                output_tokens=50,
                cost_usd_micros=2_500_000,
                status="ok",
                error=None,
                attempts=1,
                ts=_now(),
            )
        )
    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/llm-calls")
    assert r.status_code == 200
    assert "daily ceiling" in r.text
    assert "consumed" in r.text
