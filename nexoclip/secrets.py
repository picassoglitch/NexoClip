"""Symmetric encryption for at-rest secrets (multistream stream keys, …).

Fernet (AES-128-CBC + HMAC) keyed by material from settings. We derive a
valid 32-byte Fernet key from arbitrary key material via SHA-256, so the
operator can set any string for `NEXOCLIP_SECRET_KEY` (it does NOT have to
be a pre-generated Fernet key).

Rules:
  * Plaintext secrets NEVER get logged.
  * Ciphertext is what lands in the DB; decryption happens only where the
    plaintext is genuinely needed (e.g. handing a stream key to the relay
    over the internal-bearer channel).
"""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from nexoclip.errors import NexoClipError


class SecretError(NexoClipError):
    """Encryption/decryption failed (bad key material or corrupt token)."""


def _fernet(key_material: str) -> Fernet:
    if not key_material:
        raise SecretError(
            "no secret key material configured — set NEXOCLIP_SECRET_KEY "
            "(or NEXOCLIP_INTERNAL_SIGNING_SECRET) before storing secrets"
        )
    # Derive a urlsafe-base64 32-byte key from arbitrary material.
    digest = hashlib.sha256(key_material.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_secret(plaintext: str, *, key_material: str) -> str:
    """Encrypt `plaintext` → opaque ascii token for storage."""
    return _fernet(key_material).encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_secret(token: str, *, key_material: str) -> str:
    """Decrypt a token from `encrypt_secret`. Raises SecretError on a bad
    key or tampered/corrupt token."""
    try:
        return _fernet(key_material).decrypt(token.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError) as e:
        raise SecretError("could not decrypt secret (bad key or corrupt token)") from e


def resolve_key_material(settings: object) -> str:
    """The key material the app encrypts secrets with: the dedicated
    NEXOCLIP_SECRET_KEY, falling back to the internal signing secret so a
    single-secret deployment still works."""
    key = (getattr(settings, "secret_key", None) or "").strip()
    if key:
        return key
    return (getattr(settings, "internal_signing_secret", None) or "").strip()


__all__ = [
    "SecretError",
    "decrypt_secret",
    "encrypt_secret",
    "resolve_key_material",
]
