"""mint_signed_clip_url tests.

The URL we hand to Zernio (as `mediaItems[].url`) must:
  - bind clip_id + tenant_id + exp in the signature
  - reject expired URLs at the verify side
  - reject tampered signatures
  - reject far-future expiries (defense in depth)

The signing mechanism is shared + unchanged from the upload-post era —
only the consumer (Zernio) changed. Verification is exercised via the
actual /api/internal/clip endpoint contract Zernio will hit.
"""
from __future__ import annotations

import time

import pytest

from nexoclip.api.routers.internal import (
    _IMMEDIATE_PUBLISH_TTL_S,
    _SIGNED_CLIP_MAX_TTL_S,
    _verify_signed_params,
    mint_signed_clip_url,
    signed_clip_ttl_for_schedule,
)
from nexoclip.settings import get_settings


@pytest.fixture(autouse=True)
def _set_secret(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        get_settings(),
        "internal_signing_secret",
        "test-internal-secret",
        raising=False,
    )


def test_mint_then_parse_url_roundtrip() -> None:
    url = mint_signed_clip_url(
        clip_id="clp_xyz",
        tenant_id="ten_alice",
        base_url="https://nexoclip.test",
        ttl_seconds=600,
    )
    assert url.startswith(
        "https://nexoclip.test/api/internal/clip/clp_xyz?tenant=ten_alice&exp="
    )
    assert "&sig=" in url


def test_verify_accepts_signed_url() -> None:
    import re
    url = mint_signed_clip_url(
        clip_id="clp_xyz",
        tenant_id="ten_alice",
        base_url="https://nexoclip.test",
        ttl_seconds=600,
    )
    exp = int(re.search(r"exp=(\d+)", url).group(1))
    sig = re.search(r"sig=([a-f0-9]+)", url).group(1)
    _verify_signed_params(
        resource_id="clp_xyz",
        tenant="ten_alice",
        exp=exp,
        sig=sig,
        max_ttl_s=24 * 3600,
    )  # raises HTTPException on failure


def test_verify_rejects_tampered_sig() -> None:
    import re

    from fastapi import HTTPException
    url = mint_signed_clip_url(
        clip_id="clp_xyz",
        tenant_id="ten_alice",
        base_url="https://nexoclip.test",
        ttl_seconds=600,
    )
    exp = int(re.search(r"exp=(\d+)", url).group(1))
    sig = re.search(r"sig=([a-f0-9]+)", url).group(1)
    bad_sig = sig[:-1] + ("0" if sig[-1] != "0" else "1")
    with pytest.raises(HTTPException) as ei:
        _verify_signed_params(
            resource_id="clp_xyz",
            tenant="ten_alice",
            exp=exp,
            sig=bad_sig,
            max_ttl_s=24 * 3600,
        )
    assert ei.value.status_code == 403


def test_verify_rejects_expired() -> None:
    import hashlib
    import hmac

    from fastapi import HTTPException
    secret = get_settings().internal_signing_secret
    past = int(time.time()) - 60
    msg = f"clp_xyz|ten_alice|{past}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    with pytest.raises(HTTPException) as ei:
        _verify_signed_params(
            resource_id="clp_xyz",
            tenant="ten_alice",
            exp=past,
            sig=sig,
            max_ttl_s=24 * 3600,
        )
    assert ei.value.status_code == 403
    assert "expired" in ei.value.detail.lower()


def test_verify_rejects_implausibly_far_expiry() -> None:
    import hashlib
    import hmac

    from fastapi import HTTPException
    secret = get_settings().internal_signing_secret
    far_future = int(time.time()) + (30 * 24 * 3600)  # 30 days
    msg = f"clp_xyz|ten_alice|{far_future}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    with pytest.raises(HTTPException) as ei:
        _verify_signed_params(
            resource_id="clp_xyz",
            tenant="ten_alice",
            exp=far_future,
            sig=sig,
            max_ttl_s=24 * 3600,
        )
    assert ei.value.status_code == 403
    assert "implausible" in ei.value.detail.lower()


def test_ttl_for_immediate_post_is_default() -> None:
    # No schedule -> the 24h default floor covers the vendor's retry tail.
    assert signed_clip_ttl_for_schedule(None) == _IMMEDIATE_PUBLISH_TTL_S


def test_ttl_for_past_or_unparseable_schedule_is_default() -> None:
    import datetime as _dt
    # Unparseable, and a deep-past schedule whose gap+margin underflows,
    # both degrade to the default TTL.
    assert signed_clip_ttl_for_schedule("not-a-timestamp") == _IMMEDIATE_PUBLISH_TTL_S
    deep_past = _dt.datetime.now(_dt.UTC) - _dt.timedelta(days=1)
    assert signed_clip_ttl_for_schedule(deep_past.isoformat()) == _IMMEDIATE_PUBLISH_TTL_S


def test_ttl_covers_a_multi_day_schedule() -> None:
    import datetime as _dt
    now = int(time.time())
    when = _dt.datetime.now(_dt.UTC) + _dt.timedelta(days=3)
    ttl = signed_clip_ttl_for_schedule(when, now=now)
    # Must outlive the gap to the scheduled time so the platform can still
    # download at posting time — the bug was a flat 3600s here.
    assert ttl > 3 * 24 * 3600
    assert ttl <= _SIGNED_CLIP_MAX_TTL_S


def test_ttl_is_clamped_to_ceiling() -> None:
    import datetime as _dt
    when = _dt.datetime.now(_dt.UTC) + _dt.timedelta(days=400)
    assert signed_clip_ttl_for_schedule(when) == _SIGNED_CLIP_MAX_TTL_S


def test_clip_route_accepts_a_scheduled_url_days_out() -> None:
    # Regression: a clip scheduled days out used to 403 as "implausible"
    # under the old 24h clip-route ceiling. It must verify now.
    import datetime as _dt
    when = _dt.datetime.now(_dt.UTC) + _dt.timedelta(days=3)
    url = mint_signed_clip_url(
        clip_id="clp_sched",
        tenant_id="ten_alice",
        base_url="https://nexoclip.test",
        ttl_seconds=signed_clip_ttl_for_schedule(when),
    )
    import re
    exp = int(re.search(r"exp=(\d+)", url).group(1))
    sig = re.search(r"sig=([a-f0-9]+)", url).group(1)
    _verify_signed_params(
        resource_id="clp_sched",
        tenant="ten_alice",
        exp=exp,
        sig=sig,
        max_ttl_s=_SIGNED_CLIP_MAX_TTL_S,
    )  # raises on failure


def test_mint_refuses_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        get_settings(), "internal_signing_secret", "", raising=False,
    )
    with pytest.raises(RuntimeError, match="NEXOCLIP_INTERNAL_SIGNING_SECRET"):
        mint_signed_clip_url(
            clip_id="x", tenant_id="y", base_url="https://x.test",
        )
