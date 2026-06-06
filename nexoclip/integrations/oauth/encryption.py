"""Fernet-based at-rest encryption for OAuth tokens.

The connected_accounts table stores access_token / refresh_token in
encrypted BLOB columns. Plaintext tokens never sit on disk past the
brief window between the platform's HTTP response and the
INSERT/UPDATE writing the encrypted bytes. Reads decrypt on demand
right before the publish step uses them.

The key is a base64-encoded 32-byte Fernet symmetric key, read from
the NEXOCLIP_CREDS_KEY env var. Generate one with:

    python -c "from cryptography.fernet import Fernet; \\
               print(Fernet.generate_key().decode())"

Init-time validation: the connect router refuses to mount, and the
publisher refuses to dispatch, if NEXOCLIP_CREDS_KEY is unset. We
fail at boot rather than at the first connect click, so a missing
key never silently corrupts production data.

Rotation: TODO Wave 2 — Fernet supports MultiFernet for key rotation;
add a NEXOCLIP_CREDS_KEY_PREV env var read alongside the current one
when the time comes. For Wave 1 we keep it single-key and the
operator rotates with a one-shot re-encrypt migration if needed.
"""
from __future__ import annotations

from typing import Final

from cryptography.fernet import Fernet, InvalidToken

from nexoclip.errors import NexoClipError


class EncryptionKeyMissing(NexoClipError):
    """Raised at boot when NEXOCLIP_CREDS_KEY is unset.

    Surfaced separately from the generic decryption-failed error so
    operations can tell "key is gone" (operator error, fixable in
    Railway settings) apart from "ciphertext won't decode" (key
    rotated or DB row corrupt).
    """


class DecryptionFailed(NexoClipError):
    """Raised when a stored ciphertext can't be decrypted.

    Usually means the key was rotated without re-encrypting the row,
    or the row was corrupted. Caller treats this the same way as a
    revoked token: flip the account to auth_failed and prompt the
    operator to re-connect.
    """


# Sentinel length for the encoded key. Fernet keys are base64-encoded
# 32 raw bytes → exactly 44 chars including the trailing `=` pad.
_FERNET_KEY_LEN: Final = 44


def _validate_key(key: str | None) -> bytes:
    """Coerce + sanity-check the env var value before handing to Fernet."""
    if not key:
        raise EncryptionKeyMissing(
            "NEXOCLIP_CREDS_KEY is not set. Generate a key with "
            "`python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\"` and add it to "
            "Railway env vars before mounting the connect router."
        )
    encoded = key.encode("ascii", errors="strict")
    if len(encoded) != _FERNET_KEY_LEN:
        raise EncryptionKeyMissing(
            f"NEXOCLIP_CREDS_KEY is the wrong length "
            f"({len(encoded)} chars; expected {_FERNET_KEY_LEN}). "
            "Fernet keys are base64-encoded 32 raw bytes."
        )
    return encoded


class TokenEncryptor:
    """Thin wrapper around `cryptography.Fernet` for token I/O.

    Constructor validates the key shape so the singleton accessor
    `get_encryptor()` can fail fast at import / first-call time
    instead of silently storing nothing.
    """

    def __init__(self, key: str | None) -> None:
        self._fernet = Fernet(_validate_key(key))

    def encrypt(self, plaintext: str | None) -> bytes | None:
        """Encrypt a UTF-8 token string for at-rest storage.

        Returns None on a None input so caller logic stays clean:
        a row with no refresh_token (e.g. Meta long-lived tokens
        that don't ship a classic refresh) stores NULL ciphertext,
        not the encryption of an empty string.
        """
        if plaintext is None:
            return None
        return self._fernet.encrypt(plaintext.encode("utf-8"))

    def decrypt(self, ciphertext: bytes | None) -> str | None:
        """Decrypt + UTF-8 decode a stored ciphertext.

        Returns None on a None input (matches `encrypt`'s symmetry).
        Wraps `InvalidToken` into the project's typed
        `DecryptionFailed` so the publisher can branch on it.
        """
        if ciphertext is None:
            return None
        try:
            return self._fernet.decrypt(ciphertext).decode("utf-8")
        except InvalidToken as e:
            raise DecryptionFailed(
                "stored OAuth token failed to decrypt — key rotated "
                "without re-encrypting, or row was corrupted"
            ) from e


_singleton: TokenEncryptor | None = None


def get_encryptor() -> TokenEncryptor:
    """Process-wide TokenEncryptor cached after the first build.

    Reads NEXOCLIP_CREDS_KEY at first call via `get_settings()`,
    raising EncryptionKeyMissing if unset. Subsequent calls return
    the cached instance without re-reading settings.

    Tests can reset by passing the env var through Pydantic's
    settings cache_clear before constructing, or by manually
    overriding `_singleton`.
    """
    global _singleton
    if _singleton is None:
        from nexoclip.settings import get_settings
        _singleton = TokenEncryptor(get_settings().nexoclip_creds_key)
    return _singleton


def reset_encryptor_for_tests() -> None:
    """Drop the cached singleton so the next get_encryptor() rebuilds.

    Pytest helper — production code never calls this. Used by the
    test that exercises EncryptionKeyMissing (which needs the
    singleton to NOT exist when settings.nexoclip_creds_key is unset).
    """
    global _singleton
    _singleton = None


__all__ = [
    "TokenEncryptor",
    "EncryptionKeyMissing",
    "DecryptionFailed",
    "get_encryptor",
    "reset_encryptor_for_tests",
]
