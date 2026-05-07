"""run_publish_jobs respects the BudgetGovernor's publish quota.

Reuses the publish fixtures (conftest seeds a tenant + 1 pending Buffer
job). When the tenant's `daily_publish_limit` is at the cap, the drain
emits `publish.quota_exhausted` and skips remaining jobs without firing
the Buffer client.
"""

from __future__ import annotations

from typing import Any

import httpx

from nexoclip.db import Database, EventsRepo, PublishJobsRepo, TenantsRepo
from nexoclip.governance import BudgetGovernor
from nexoclip.publish import run_publish_jobs
from nexoclip.publish.buffer import BufferClient
from nexoclip.tenancy import bound_tenant


class _RecordingBufferClient(BufferClient):
    """Buffer client that succeeds without hitting the network."""

    def __init__(self, access_token: str = "x") -> None:
        super().__init__(access_token=access_token)
        self.posts: list[dict[str, Any]] = []

    async def __aenter__(self) -> _RecordingBufferClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def create_update(
        self, *, profile_external_id: str, text: str
    ) -> dict[str, Any]:
        self.posts.append(
            {"profile_external_id": profile_external_id, "text": text}
        )
        return {"updates": [{"id": f"buf_{len(self.posts)}"}]}


async def _noop_sleep(_s: float) -> None:
    return None


async def test_quota_exhaustion_skips_dispatch_and_emits_event(
    db: Database, seeded: dict[str, str]
) -> None:
    """daily_publish_limit=0 -> no quota -> first call refused."""
    tenant_id = seeded["tenant_id"]
    await TenantsRepo(db).set_budget(tenant_id, daily_publish_limit=0)
    gov = BudgetGovernor(db)
    client = _RecordingBufferClient()

    outcome = await run_publish_jobs(
        tenant_id,
        db,
        buffer_factory=lambda _t: client,
        sleep=_noop_sleep,
        governor=gov,
    )
    assert outcome.sent == 0
    assert outcome.skipped == 1
    # Buffer was never asked.
    assert client.posts == []

    # `publish.quota_exhausted` event lands.
    with bound_tenant(tenant_id):
        events = await EventsRepo(db).list_for_tenant(type="publish.quota_exhausted")
    assert len(events) == 1
    assert events[0].payload.get("platform") == "buffer"


async def test_drain_under_quota_dispatches_normally(
    db: Database, seeded: dict[str, str]
) -> None:
    tenant_id = seeded["tenant_id"]
    await TenantsRepo(db).set_budget(tenant_id, daily_publish_limit=10)
    gov = BudgetGovernor(db)
    client = _RecordingBufferClient()

    outcome = await run_publish_jobs(
        tenant_id,
        db,
        buffer_factory=lambda _t: client,
        sleep=_noop_sleep,
        governor=gov,
    )
    assert outcome.sent == 1
    assert outcome.skipped == 0
    assert len(client.posts) == 1


async def test_drain_without_governor_unrestricted(
    db: Database, seeded: dict[str, str]
) -> None:
    """Backward compat: callers that don't pass a governor see no enforcement."""
    tenant_id = seeded["tenant_id"]
    await TenantsRepo(db).set_budget(tenant_id, daily_publish_limit=0)
    client = _RecordingBufferClient()

    outcome = await run_publish_jobs(
        tenant_id,
        db,
        buffer_factory=lambda _t: client,
        sleep=_noop_sleep,
        # no governor
    )
    assert outcome.sent == 1


# Silence unused-import warnings while keeping the import set tidy.
_ = (httpx, PublishJobsRepo)
