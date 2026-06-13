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

from nexoclip.api.routers.internal import _verify_signed_params, mint_signed_clip_url
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


def test_mint_refuses_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        get_settings(), "internal_signing_secret", "", raising=False,
    )
    with pytest.raises(RuntimeError, match="NEXOCLIP_INTERNAL_SIGNING_SECRET"):
        mint_signed_clip_url(
            clip_id="x", tenant_id="y", base_url="https://x.test",
        )
