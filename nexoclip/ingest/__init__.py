"""VOD ingest module — public surface."""

from .models import Platform, Stream
from .service import detect_platform, ingest_vod, load_stream

__all__ = ["Platform", "Stream", "detect_platform", "ingest_vod", "load_stream"]
