"""Multistream M1 — restream destinations + at-rest secret encryption.

Covers the secret round-trip, platform templating, validation, the
encrypted-at-rest guarantee, enabled-only resolution, and tenant isolation.
No HTTP — exercises the service + repo directly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from nexoclip.db import (
    Database,
    StreamDestinationsRepo,
    TenantsRepo,
    apply_migrations,
)
from nexoclip.destinations import (
    DestinationError,
    add_destination,
    list_destinations,
    resolve_targets,
    supported_platforms,
)
from nexoclip.secrets import SecretError, decrypt_secret, encrypt_secret
from nexoclip.tenancy import bound_tenant


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "dest.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexoclip.settings import get_settings

    monkeypatch.setenv("NEXOCLIP_SECRET_KEY", "unit-test-key-material")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---- secrets --------------------------------------------------------------


def test_encrypt_decrypt_round_trip() -> None:
    tok = encrypt_secret("live_secret_xyz", key_material="k1")
    assert tok != "live_secret_xyz"
    assert "live_secret_xyz" not in tok
    assert decrypt_secret(tok, key_material="k1") == "live_secret_xyz"


def test_decrypt_with_wrong_key_fails() -> None:
    tok = encrypt_secret("x", key_material="k1")
    with pytest.raises(SecretError):
        decrypt_secret(tok, key_material="k2")


# ---- destinations service -------------------------------------------------


async def test_twitch_templates_url_and_encrypts(db: Database) -> None:
    t = await TenantsRepo(db).create(name="A")
    with bound_tenant(t.id):
        d = await add_destination(db, platform="twitch", stream_key="live_abc")
    assert d.ingest_url == "rtmp://live.twitch.tv/app/"
    assert d.platform == "twitch"
    # Key encrypted at rest — plaintext must NOT appear in the stored blob.
    assert "live_abc" not in d.stream_key_enc


async def test_kick_requires_explicit_url(db: Database) -> None:
    t = await TenantsRepo(db).create(name="A")
    with bound_tenant(t.id):
        with pytest.raises(DestinationError):
            await add_destination(db, platform="kick", stream_key="k")
        # With a URL it works.
        d = await add_destination(
            db, platform="kick", stream_key="k",
            ingest_url="rtmps://x.live-video.net/app/",
        )
    assert d.ingest_url == "rtmps://x.live-video.net/app/"


async def test_unknown_platform_and_empty_key_rejected(db: Database) -> None:
    t = await TenantsRepo(db).create(name="A")
    with bound_tenant(t.id):
        with pytest.raises(DestinationError):
            await add_destination(db, platform="facebook", stream_key="k")
        with pytest.raises(DestinationError):
            await add_destination(db, platform="twitch", stream_key="   ")
    assert set(supported_platforms()) == {"twitch", "youtube", "kick", "custom"}


async def test_resolve_targets_enabled_only_with_keys(db: Database) -> None:
    t = await TenantsRepo(db).create(name="A")
    with bound_tenant(t.id):
        await add_destination(db, platform="twitch", stream_key="key_tw")
        yt = await add_destination(db, platform="youtube", stream_key="key_yt")
        await StreamDestinationsRepo(db).set_enabled(yt.id, False)  # disable YT
        targets = await resolve_targets(db)
    assert len(targets) == 1
    assert targets[0].platform == "twitch"
    # The decrypted push URL is ingest + key.
    assert targets[0].push_url == "rtmp://live.twitch.tv/app/key_tw"


async def test_tenant_isolation(db: Database) -> None:
    a = await TenantsRepo(db).create(name="A")
    b = await TenantsRepo(db).create(name="B")
    with bound_tenant(a.id):
        await add_destination(db, platform="twitch", stream_key="key_a")
    with bound_tenant(b.id):
        assert await list_destinations(db) == []
        assert await resolve_targets(db) == []
