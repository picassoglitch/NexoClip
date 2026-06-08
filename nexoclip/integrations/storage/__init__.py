"""Recording object-storage handoff for live ingest (Phase L.2, Path B).

The live-ingest service (the separate `nexoclip-live` MediaMTX deployment)
uploads each finished recording to an S3-compatible bucket under
`<prefix>/<stream_id>/<file>.mp4`. NexoClip pulls it from there to run the
clip pipeline — so the two services share object storage instead of a
Railway volume, and scale independently.

Vendor-neutral: anything S3-compatible works via the `endpoint` setting —
Supabase Storage (reuse the Supabase you already run), Cloudflare R2,
MinIO, Backblaze B2, AWS S3, …

`build_recording_store(settings)` returns a store only when a bucket is
configured; otherwise None, and live ingest falls back to reading a shared
`/data` volume (Path A).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from nexoclip.settings import Settings

# Skip a fetched object smaller than this (bytes) — a 0-byte placeholder or
# a multipart upload that hasn't completed. R2 only lists an object once its
# upload finishes, but this guards against truncated/odd states too.
_MIN_RECORDING_BYTES = 1024


@runtime_checkable
class RecordingStore(Protocol):
    """Fetches a live recording for a stream onto local disk."""

    async def fetch_latest(self, *, stream_id: str, dest_dir: Path) -> Path | None:
        """Download the newest recording object for `stream_id` into
        `dest_dir` and return its local path, or None if none exists yet."""
        ...


class S3RecordingStore:
    """Pulls live recordings from any S3-compatible bucket (Supabase
    Storage, Cloudflare R2, MinIO, S3, …).

    Lists `<prefix>/<stream_id>/` and downloads the newest non-trivial
    object. `client` is an S3 client (boto3 or a stub in tests); all calls
    run in a worker thread since boto3 is synchronous.
    """

    def __init__(self, *, client: Any, bucket: str, prefix: str = "live") -> None:
        self._client = client
        self._bucket = bucket
        self._prefix = prefix.strip("/")

    async def fetch_latest(self, *, stream_id: str, dest_dir: Path) -> Path | None:
        return await asyncio.to_thread(self._fetch_sync, stream_id, Path(dest_dir))

    def _fetch_sync(self, stream_id: str, dest_dir: Path) -> Path | None:
        key_prefix = f"{self._prefix}/{stream_id}/"
        resp: Any = self._client.list_objects_v2(
            Bucket=self._bucket, Prefix=key_prefix
        )
        contents: list[Any] = list(resp.get("Contents") or [])
        candidates = [c for c in contents if int(c.get("Size") or 0) > _MIN_RECORDING_BYTES]
        if not candidates:
            return None
        # Newest by upload time. If a stream produced multiple segments, the
        # last one wins; single-segment streams (the norm) have exactly one.
        newest = max(candidates, key=lambda c: c["LastModified"])
        key = str(newest["Key"])
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / Path(key).name
        self._client.download_file(self._bucket, key, str(dest))
        return dest


def build_recording_store(settings: Settings) -> RecordingStore | None:
    """Construct the object store from settings, or None when not configured.

    None means "no object storage" → live ingest reads the shared volume
    (Path A). boto3 is imported lazily so it's only a hard dependency when
    object storage is actually wired up.
    """
    bucket = (getattr(settings, "live_recording_storage_bucket", None) or "").strip()
    if not bucket:
        return None

    import boto3  # lazy: only needed when object storage is configured

    endpoint = (
        getattr(settings, "live_recording_storage_endpoint", None) or ""
    ).strip()
    client = boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        aws_access_key_id=getattr(
            settings, "live_recording_storage_access_key_id", None
        ),
        aws_secret_access_key=getattr(
            settings, "live_recording_storage_secret_access_key", None
        ),
        region_name=(
            getattr(settings, "live_recording_storage_region", None) or "auto"
        ),
    )
    return S3RecordingStore(
        client=client,
        bucket=bucket,
        prefix=(getattr(settings, "live_recording_storage_prefix", None) or "live"),
    )


__all__ = ["RecordingStore", "S3RecordingStore", "build_recording_store"]
