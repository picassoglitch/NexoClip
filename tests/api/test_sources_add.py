"""Connected channels (Conexiones): adding a channel watch.

Regression for the 500 when re-adding a channel you already watch — the
UNIQUE(tenant_id, channel_url) constraint raised an unhandled
IntegrityError. A duplicate must bounce back with a friendly banner, not
crash.
"""

from __future__ import annotations

import httpx
import pytest

from .conftest import auth


@pytest.mark.asyncio
async def test_add_channel_then_duplicate_does_not_500(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    headers = auth(tenants["alice"]["token"])
    form = {
        "channel_url": "https://kick.com/n3on/videos",
        "persona_id": "aldo",
        "platform": "kick",
        "max_per_poll": "3",
        "polls_per_day": "1",
    }

    # First add succeeds.
    r1 = await client.post(
        "/dashboard/sources", data=form, headers=headers, follow_redirects=False,
    )
    assert r1.status_code == 303
    assert r1.headers["location"] == "/dashboard/sources?added=1"

    # Re-adding the SAME channel must NOT 500 — friendly redirect instead.
    r2 = await client.post(
        "/dashboard/sources", data=form, headers=headers, follow_redirects=False,
    )
    assert r2.status_code == 303
    assert r2.headers["location"] == "/dashboard/sources?error=already_watching"


@pytest.mark.asyncio
async def test_duplicate_is_per_tenant(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    # The same channel URL under a DIFFERENT tenant is not a duplicate —
    # the unique constraint is scoped to (tenant_id, channel_url).
    form = {
        "channel_url": "https://www.youtube.com/@creator/videos",
        "persona_id": "aldo",
        "platform": "youtube",
        "max_per_poll": "3",
        "polls_per_day": "1",
    }
    r_alice = await client.post(
        "/dashboard/sources", data=form,
        headers=auth(tenants["alice"]["token"]), follow_redirects=False,
    )
    r_bob = await client.post(
        "/dashboard/sources", data=form,
        headers=auth(tenants["bob"]["token"]), follow_redirects=False,
    )
    assert r_alice.headers["location"] == "/dashboard/sources?added=1"
    assert r_bob.headers["location"] == "/dashboard/sources?added=1"
