"""Fernet wrapper round-trip + failure-mode pins.

Wave 1's at-rest encryption layer for OAuth tokens. The tests prove:
  - happy-path encrypt → decrypt round-trip recovers the input
  - None inputs stay None (so NULL refresh tokens for Meta-style
    providers don't get encrypted-with-no-content)
  - missing / malformed NEXOCLIP_CREDS_KEY raises EncryptionKeyMissing
    at constructor time (fail at boot, not at first connect)
  - decrypt with a different key raises DecryptionFailed (typed —
    the publisher branches on this to flip status=auth_failed)
"""
from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from nexoclip.integrations.oauth.encryption import (
    DecryptionFailed,
    EncryptionKeyMissing,
    TokenEncryptor,
)


def _fresh_key() -> str:
    return Fernet.generate_key().decode("ascii")


def test_encrypt_decrypt_round_trip_recovers_input() -> None:
    enc = TokenEncryptor(_fresh_key())
    plaintext = "tt-1234abcd...this-is-an-access-token"
    ct = enc.encrypt(plaintext)
    assert isinstance(ct, bytes)
    assert ct != plaintext.encode()  # must actually be encrypted
    assert enc.decrypt(ct) == plaintext


def test_encrypt_none_returns_none() -> None:
    """Meta's long-lived token model doesn't ship a classic refresh
    token, so we store None. Encrypting None must stay None, not
    produce a non-empty ciphertext."""
    enc = TokenEncryptor(_fresh_key())
    assert enc.encrypt(None) is None


def test_decrypt_none_returns_none() -> None:
    enc = TokenEncryptor(_fresh_key())
    assert enc.decrypt(None) is None


def test_missing_key_raises_at_constructor() -> None:
    """Boot-time fail-loud, not first-connect surprise."""
    with pytest.raises(EncryptionKeyMissing):
        TokenEncryptor(None)
    with pytest.raises(EncryptionKeyMissing):
        TokenEncryptor("")


def test_malformed_key_length_raises() -> None:
    """44 chars is the only valid Fernet key length (base64 of 32
    raw bytes)."""
    with pytest.raises(EncryptionKeyMissing):
        TokenEncryptor("too-short")


def test_decrypt_with_wrong_key_raises_typed_error() -> None:
    """Key rotation without re-encrypting must produce DecryptionFailed
    so the publisher can flip status to auth_failed."""
    enc_a = TokenEncryptor(_fresh_key())
    ct = enc_a.encrypt("secret-token")

    enc_b = TokenEncryptor(_fresh_key())
    with pytest.raises(DecryptionFailed):
        enc_b.decrypt(ct)


def test_round_trip_is_non_deterministic() -> None:
    """Fernet's nonce is per-call, so the same plaintext yields
    different ciphertext each time. Tests against an accidental
    'optimization' that swaps to a deterministic AES mode."""
    enc = TokenEncryptor(_fresh_key())
    ct1 = enc.encrypt("a")
    ct2 = enc.encrypt("a")
    assert ct1 != ct2
    assert enc.decrypt(ct1) == enc.decrypt(ct2) == "a"
