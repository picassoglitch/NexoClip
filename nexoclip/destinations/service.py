"""Multistream M1 — restream destinations service.

Owns the secret handling: encrypts the stream key on write, decrypts ONLY
when resolving relay push targets. The dashboard never sees a plaintext key
(write-only), and a key is never logged (CLAUDE rule #10).

The full RTMP push target is `ingest_url + key`. Known platforms template
the ingest URL (Twitch/YouTube); Kick (per-user IVS) and custom carry a
user-supplied URL.
"""

from __future__ import annotations

from dataclasses import dataclass

from nexoclip.db import Database, StreamDestinationsRepo
from nexoclip.db.models import StreamDestinationRow
from nexoclip.errors import NexoClipError
from nexoclip.secrets import decrypt_secret, encrypt_secret, resolve_key_material
from nexoclip.settings import get_settings

# Known platforms → default ingest-URL base (push target = base + key).
# Empty string ⇒ the user must supply an explicit URL (Kick is per-user
# IVS; custom is arbitrary).
PLATFORM_TEMPLATES: dict[str, str] = {
    "twitch": "rtmp://live.twitch.tv/app/",
    "youtube": "rtmp://a.rtmp.youtube.com/live2/",
    "kick": "",
    "custom": "",
}


class DestinationError(NexoClipError):
    """Invalid destination input (unknown platform, missing URL/key, …)."""


def supported_platforms() -> list[str]:
    return list(PLATFORM_TEMPLATES.keys())


@dataclass(frozen=True)
class DestinationTarget:
    """A resolved push target for the relay — carries the PLAINTEXT push URL
    (ingest + key). Never persisted, never logged; produced only by
    `resolve_targets` and handed to the relay over the internal bearer."""

    id: str
    platform: str
    label: str | None
    push_url: str


def _join_url(base: str, key: str) -> str:
    base = base.rstrip()
    return base + key if base.endswith("/") else base + "/" + key


async def add_destination(
    db: Database,
    *,
    platform: str,
    stream_key: str,
    ingest_url: str | None = None,
    label: str | None = None,
) -> StreamDestinationRow:
    """Validate + encrypt + persist a new restream destination."""
    platform = (platform or "").strip().lower()
    if platform not in PLATFORM_TEMPLATES:
        raise DestinationError(
            f"unsupported platform {platform!r}; one of {supported_platforms()}"
        )
    url = (ingest_url or "").strip() or PLATFORM_TEMPLATES[platform]
    if not url:
        raise DestinationError(
            f"{platform} needs an explicit RTMP ingest URL (copy it from the "
            "platform's stream settings)"
        )
    if not (url.startswith("rtmp://") or url.startswith("rtmps://")):
        raise DestinationError("ingest URL must start with rtmp:// or rtmps://")
    key = (stream_key or "").strip()
    if not key:
        raise DestinationError("stream key is required")

    enc = encrypt_secret(key, key_material=resolve_key_material(get_settings()))
    return await StreamDestinationsRepo(db).create(
        platform=platform,
        ingest_url=url,
        stream_key_enc=enc,
        label=(label or "").strip() or None,
    )


async def list_destinations(db: Database) -> list[StreamDestinationRow]:
    """All destinations for the bound tenant (encrypted keys; the UI shows
    platform/label/status, never the key)."""
    return await StreamDestinationsRepo(db).list_for_tenant()


async def resolve_targets(db: Database) -> list[DestinationTarget]:
    """Decrypt the ENABLED destinations into plaintext push URLs for the
    relay. The only code path that ever produces plaintext keys.

    A destination whose key was encrypted under a now-rotated secret can't
    be decrypted — we skip it (and leave it for the operator to re-enter)
    rather than fail the whole fan-out. Must be called within a bound
    tenant.
    """
    key_material = resolve_key_material(get_settings())
    out: list[DestinationTarget] = []
    for d in await StreamDestinationsRepo(db).list_for_tenant():
        if not d.enabled:
            continue
        try:
            key = decrypt_secret(d.stream_key_enc, key_material=key_material)
        except Exception:  # skip undecryptable (rotated key), don't crash
            continue
        out.append(
            DestinationTarget(
                id=d.id,
                platform=d.platform,
                label=d.label,
                push_url=_join_url(d.ingest_url, key),
            )
        )
    return out
