"""Signed-state CSRF token tests.

OAuth callbacks are public — the platform's redirect lands on us
with no cookie. The state token is the only thing binding the
callback to a real tenant. These tests pin:
  - happy-path sign → verify recovers the claims
  - tampered signature is rejected
  - past TTL is rejected as expired
  - iat in the future is rejected (clock skew + replay)
  - malformed payload is rejected
  - unknown platform is rejected at sign AND verify
"""
from __future__ import annotations

import pytest

from nexoclip.integrations.oauth.state import (
    InvalidState,
    sign_state,
    verify_state,
)
from nexoclip.settings import get_settings


def _setup_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most tests need the HMAC secret set. Done via monkeypatch on
    settings rather than env so we don't trip the Pydantic cache."""
    monkeypatch.setattr(
        get_settings(),
        "nexo_ai_sso_secret",
        "test-secret-for-state-tests",
        raising=False,
    )


def test_sign_then_verify_recovers_claims(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_secret(monkeypatch)
    token = sign_state(platform="tiktok", tenant_id="ten_alice", now=1_700_000_000)
    claims = verify_state(token, now=1_700_000_100)
    assert claims.platform == "tiktok"
    assert claims.tenant_id == "ten_alice"
    assert claims.iat == 1_700_000_000
    assert claims.nonce  # 32 hex chars from 16 random bytes
    assert len(claims.nonce) == 32


def test_tampered_signature_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_secret(monkeypatch)
    token = sign_state(platform="tiktok", tenant_id="ten_alice", now=1_700_000_000)
    # Flip the last char — the sig is in the last segment, base64 of
    # the payload. Any flip in there breaks the HMAC.
    bad = token[:-1] + ("A" if token[-1] != "A" else "B")
    with pytest.raises(InvalidState, match="HMAC|base64|fields"):
        verify_state(bad, now=1_700_000_100)


def test_expired_token_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_secret(monkeypatch)
    token = sign_state(platform="tiktok", tenant_id="ten_alice", now=1_700_000_000)
    # TTL is 3600s; verify 4000s later.
    with pytest.raises(InvalidState, match="expired"):
        verify_state(token, now=1_700_004_000)


def test_iat_in_the_future_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clock skew between dyno + an attacker's machine, or simple
    replay forgery. Either way, a future iat should never validate."""
    _setup_secret(monkeypatch)
    token = sign_state(platform="tiktok", tenant_id="ten_alice", now=1_700_000_500)
    with pytest.raises(InvalidState, match="future"):
        verify_state(token, now=1_700_000_000)


def test_unknown_platform_rejected_at_sign(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_secret(monkeypatch)
    with pytest.raises(InvalidState, match="unknown platform"):
        sign_state(platform="myspace", tenant_id="ten_alice")


def test_empty_tenant_rejected_at_sign(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_secret(monkeypatch)
    with pytest.raises(InvalidState, match="tenant_id"):
        sign_state(platform="tiktok", tenant_id="")


def test_malformed_state_token_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_secret(monkeypatch)
    with pytest.raises(InvalidState):
        verify_state("not-base64!!!", now=1_700_000_000)
    with pytest.raises(InvalidState, match="empty"):
        verify_state("", now=1_700_000_000)


def test_missing_signing_secret_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    """If NEXO_AI_SSO_SECRET is unset we can't sign OR verify."""
    monkeypatch.setattr(
        get_settings(), "nexo_ai_sso_secret", None, raising=False,
    )
    with pytest.raises(InvalidState, match="NEXO_AI_SSO_SECRET"):
        sign_state(platform="tiktok", tenant_id="ten_alice")
