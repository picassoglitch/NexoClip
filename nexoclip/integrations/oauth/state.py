"""HMAC-signed OAuth state token.

OAuth callbacks land on a public endpoint (no auth cookie — the
platform redirects the user's browser back to us with no session).
We carry the originating tenant_id through the round-trip in the
`state` query param, signed so a hostile redirect can't forge one
or replay another tenant's state.

Payload layout (post-base64-decode):

    <platform>|<tenant_id>|<nonce>|<iat>|<sig>

  - platform   — "tiktok" | "instagram" | "youtube". Lets the same
                 callback URL discriminate without an extra path
                 segment, and lets us reject mid-flight platform
                 switches (start TikTok, finish IG with hijacked
                 state).
  - tenant_id  — the dashboard-side tenant that clicked Connect.
                 Becomes the bound tenant on the callback's DB write.
  - nonce      — 128 bits of randomness so two simultaneous Connect
                 clicks from the same tenant produce distinct states.
                 Doubles as a single-use guard if we ever bolt on a
                 server-side seen-nonces table (not Wave 1).
  - iat        — issued-at UNIX seconds. Used for the TTL window.
  - sig        — HMAC-SHA256 over the first four fields with the
                 NEXO_AI_SSO_SECRET (reused — same trust boundary,
                 same rotation cadence as the existing in-app HMAC).

TTL: 60 minutes. Long enough to cover an operator who opens the
TikTok consent dialog, gets pulled away, and finishes the click
later — but short enough that a leaked state can't be replayed
hours after the fact.
"""
from __future__ import annotations

import base64
import hmac
import secrets
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Final

from nexoclip.errors import NexoClipError


class InvalidState(NexoClipError):
    """Raised when a callback's state param fails verification.

    Caught by the callback handler and surfaced to the operator as
    "Connect attempt expired or tampered — please click Connect
    again." We do NOT distinguish forged from expired in the user-
    facing message (no value in telling an attacker which knob to
    turn) but the structured log captures the reason.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# Each `state` lives at most this many seconds after issue. Cookie-
# style 10 min would frustrate operators who tab away mid-consent;
# 60 min covers the realistic abandon-and-resume window without
# extending the replay surface meaningfully.
_STATE_TTL_S: Final = 3600

# Field separator. Pipe instead of `.` so JWT-shaped lookalikes don't
# confuse a future operator scanning logs.
_SEP: Final = "|"

_VALID_PLATFORMS: Final = frozenset({"tiktok", "instagram", "youtube"})


@dataclass(frozen=True, slots=True)
class StateClaims:
    """Verified payload returned from `verify_state`. Immutable."""

    platform: str
    tenant_id: str
    nonce: str
    iat: int


def _signing_secret() -> bytes:
    """Pull the HMAC secret from settings, raising if unset."""
    from nexoclip.settings import get_settings
    secret = get_settings().nexo_ai_sso_secret
    if not secret:
        raise InvalidState(
            "NEXO_AI_SSO_SECRET is unset; cannot sign OAuth state"
        )
    return secret.encode("utf-8")


def _b64encode(payload: str) -> str:
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")


def _b64decode(token: str) -> str:
    pad = "=" * (-len(token) % 4)
    return base64.urlsafe_b64decode(token + pad).decode("utf-8")


def sign_state(*, platform: str, tenant_id: str, now: int | None = None) -> str:
    """Build a signed state token for the OAuth authorize URL.

    `now` is injectable so tests can pin time without monkey-patching
    `time.time()` globally.
    """
    if platform not in _VALID_PLATFORMS:
        raise InvalidState(
            f"unknown platform {platform!r}; expected one of "
            f"{sorted(_VALID_PLATFORMS)}"
        )
    if not tenant_id:
        raise InvalidState("tenant_id is empty")

    now_s = int(now if now is not None else time.time())
    nonce = secrets.token_hex(16)
    body = f"{platform}{_SEP}{tenant_id}{_SEP}{nonce}{_SEP}{now_s}"
    sig = hmac.new(_signing_secret(), body.encode("utf-8"), sha256).hexdigest()
    return _b64encode(f"{body}{_SEP}{sig}")


def verify_state(token: str, *, now: int | None = None) -> StateClaims:
    """Verify a state token from a callback URL.

    Raises `InvalidState` on any failure: malformed encoding, bad
    signature, expired, unknown platform. The reason string is
    structured-logged on the callback path but the user-facing
    response is intentionally uniform.
    """
    if not token:
        raise InvalidState("empty state token")
    try:
        decoded = _b64decode(token)
    except (ValueError, UnicodeDecodeError) as e:
        raise InvalidState(f"state base64 decode failed: {e}") from e

    parts = decoded.split(_SEP)
    if len(parts) != 5:
        raise InvalidState(
            f"state has {len(parts)} fields; expected 5 "
            "(platform|tenant_id|nonce|iat|sig)"
        )

    platform, tenant_id, nonce, iat_str, sig = parts
    if platform not in _VALID_PLATFORMS:
        raise InvalidState(f"unknown platform {platform!r}")

    body = f"{platform}{_SEP}{tenant_id}{_SEP}{nonce}{_SEP}{iat_str}"
    expected = hmac.new(_signing_secret(), body.encode("utf-8"), sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        raise InvalidState("HMAC signature mismatch")

    try:
        iat = int(iat_str)
    except ValueError as e:
        raise InvalidState(f"iat is not an integer: {iat_str!r}") from e

    now_s = int(now if now is not None else time.time())
    age = now_s - iat
    if age < 0:
        raise InvalidState(f"state iat is in the future ({age}s)")
    if age > _STATE_TTL_S:
        raise InvalidState(
            f"state expired {age - _STATE_TTL_S}s past TTL "
            f"(TTL={_STATE_TTL_S}s)"
        )

    return StateClaims(platform=platform, tenant_id=tenant_id, nonce=nonce, iat=iat)


__all__ = [
    "StateClaims",
    "InvalidState",
    "sign_state",
    "verify_state",
]
