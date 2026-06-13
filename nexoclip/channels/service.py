"""Channel-poll loop — auto-ingest new VODs from creator channels.

The auto-ingest counterpart to `nexoclip.drive.service`. Per-watch:

  1. List the channel's recent uploads via yt-dlp (flat — metadata only,
     no media download).
  2. For each video whose id is NOT in `seen_video_ids`, call
     `ingest_callback(tenant_id, vod_url, video_id, persona_id, language)`
     — which ingests the VOD and kicks off the pipeline.
  3. Append ingested ids to `seen_video_ids`; advance `last_polled_at`
     only on a clean pass (mirrors the drive watcher's failure handling).

First-poll guard: a brand-new watch only ingests the newest
`max_per_poll` videos and marks the rest of the listed window as seen, so
connecting a channel with a deep back-catalog doesn't flood the pipeline.
Later polls ingest up to `max_per_poll` fresh uploads per pass.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from collections.abc import Awaitable, Callable
from typing import Protocol

import structlog

from nexoclip.db import ChannelWatchesRepo, Database, TenantsRepo
from nexoclip.db.models import ChannelWatchRow
from nexoclip.tenancy import bound_tenant

from .models import ChannelPollReport, ChannelVOD

_log = structlog.get_logger(__name__)

# How many entries to pull off the channel each poll. We list a window a
# bit wider than the per-poll ingest cap so a burst of uploads between
# polls is still caught (deduped against the seen-set).
_DEFAULT_LIST_WINDOW = 10


ChannelIngestCallback = Callable[[str, str, str, str, str | None], Awaitable[None]]
"""(tenant_id, vod_url, video_id, persona_id, language) -> None.

The poller hands you a freshly-detected VOD; you ingest it and kick off
the pipeline. In production this is `make_channel_ingest_callback`."""


class ChannelVODLister(Protocol):
    """Lists a channel's recent videos. Injected so tests can fake it."""

    async def __call__(
        self, platform: str, channel_url: str, *, limit: int
    ) -> list[ChannelVOD]: ...


# ---- VOD listing (yt-dlp, flat) ----


def _list_channel_vods_sync(channel_url: str, limit: int) -> list[dict[str, object]]:
    # Imported here so yt-dlp stays out of module-load time.
    import yt_dlp

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        # Flat: list playlist/channel entries without resolving each video.
        "extract_flat": "in_playlist",
        "playlistend": limit,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    entries = (info or {}).get("entries") or []
    return [e for e in entries if isinstance(e, dict)]


def _canonical_url(platform: str, entry: dict[str, object]) -> str | None:
    """Resolve a flat entry to a single-video URL ingest_vod can download."""
    url = entry.get("url")
    if isinstance(url, str) and url.startswith("http"):
        return url
    vid = entry.get("id")
    if not isinstance(vid, str) or not vid:
        return url if isinstance(url, str) and url else None
    if platform == "youtube":
        return f"https://www.youtube.com/watch?v={vid}"
    if platform == "twitch":
        return f"https://www.twitch.tv/videos/{vid}"
    # Kick / unknown — fall back to whatever flat gave us.
    return url if isinstance(url, str) and url else None


async def list_channel_vods(
    platform: str, channel_url: str, *, limit: int
) -> list[ChannelVOD]:
    """List a channel's newest `limit` videos (newest first), no download."""
    entries = await asyncio.to_thread(_list_channel_vods_sync, channel_url, limit)
    vods: list[ChannelVOD] = []
    for e in entries[:limit]:
        vid = e.get("id")
        url = _canonical_url(platform, e)
        if not isinstance(vid, str) or not vid or not url:
            continue
        title = e.get("title")
        vods.append(
            ChannelVOD(
                video_id=vid,
                url=url,
                title=title if isinstance(title, str) else None,
            )
        )
    return vods


# ---- public entry ----


async def poll_channel_watches(
    db: Database,
    *,
    ingest_callback: ChannelIngestCallback,
    list_vods: ChannelVODLister = list_channel_vods,
    tenant_id: str | None = None,
) -> list[ChannelPollReport]:
    """Poll every enabled channel watch for new VODs.

    Args:
        db: Database handle. Migrations must already be applied.
        ingest_callback: Called per newly-detected VOD with
            `(tenant_id, vod_url, video_id, persona_id, language)`.
        list_vods: Channel lister (defaults to the yt-dlp one). Injected
            in tests.
        tenant_id: When set, restricts to that tenant; otherwise polls
            every tenant.

    Returns:
        One report per watch row scanned.
    """
    tenants_repo = TenantsRepo(db)
    if tenant_id is not None:
        t = await tenants_repo.get(tenant_id)
        tenants = [t] if t is not None else []
    else:
        tenants = await tenants_repo.list_all()

    reports: list[ChannelPollReport] = []
    for t in tenants:
        with bound_tenant(t.id):
            watches = await ChannelWatchesRepo(db).list_for_tenant()
            for watch in watches:
                report = await _poll_one_watch(
                    db=db,
                    watch=watch,
                    ingest_callback=ingest_callback,
                    list_vods=list_vods,
                )
                reports.append(report)
                _log.info(
                    "channel.poll_done",
                    watch_id=watch.id,
                    tenant_id=watch.tenant_id,
                    platform=watch.platform,
                    videos_seen=report.videos_seen,
                    videos_ingested=report.videos_ingested,
                    videos_failed=report.videos_failed,
                    skipped_disabled=report.skipped_disabled,
                )
    return reports


async def _poll_one_watch(
    *,
    db: Database,
    watch: ChannelWatchRow,
    ingest_callback: ChannelIngestCallback,
    list_vods: ChannelVODLister,
) -> ChannelPollReport:
    """Per-watch body. Tenant is already bound."""
    if not watch.enabled:
        return ChannelPollReport(
            watch_id=watch.id,
            tenant_id=watch.tenant_id,
            channel_url=watch.channel_url,
            videos_seen=0,
            videos_ingested=0,
            videos_failed=0,
            skipped_disabled=True,
        )

    window = max(watch.max_per_poll, _DEFAULT_LIST_WINDOW)
    try:
        vods = await list_vods(watch.platform, watch.channel_url, limit=window)
    except Exception as e:
        _log.warning(
            "channel.list_failed",
            watch_id=watch.id,
            platform=watch.platform,
            channel_url=watch.channel_url,
            error=str(e),
        )
        return ChannelPollReport(
            watch_id=watch.id,
            tenant_id=watch.tenant_id,
            channel_url=watch.channel_url,
            videos_seen=0,
            videos_ingested=0,
            videos_failed=1,
            skipped_disabled=False,
        )

    seen = set(watch.seen_video_ids)
    new_vods = [v for v in vods if v.video_id not in seen]

    if watch.last_polled_at is None:
        # First sight of this channel: ingest only the newest few; treat the
        # rest of the listed window as back-catalog and mark it seen so it
        # never reprocesses.
        to_ingest = new_vods[: watch.max_per_poll]
        for v in new_vods[watch.max_per_poll :]:
            seen.add(v.video_id)
    else:
        # Steady state: ingest fresh uploads, capped per pass as a flood
        # guard. Any overflow stays unseen and lands on the next poll.
        to_ingest = new_vods[: watch.max_per_poll]

    ingested = 0
    failed = 0
    for v in to_ingest:
        try:
            await ingest_callback(
                watch.tenant_id, v.url, v.video_id, watch.persona_id, watch.language
            )
            seen.add(v.video_id)
            ingested += 1
        except Exception as e:
            _log.warning(
                "channel.ingest_failed",
                watch_id=watch.id,
                video_id=v.video_id,
                vod_url=v.url,
                error=str(e),
            )
            failed += 1

    # Advance last_polled_at only on a clean pass, same as the drive watcher.
    new_polled_at = _now_iso() if failed == 0 else watch.last_polled_at
    await ChannelWatchesRepo(db).mark_polled(
        watch.id,
        seen_video_ids=sorted(seen),
        last_polled_at=new_polled_at,
    )

    return ChannelPollReport(
        watch_id=watch.id,
        tenant_id=watch.tenant_id,
        channel_url=watch.channel_url,
        videos_seen=len(vods),
        videos_ingested=ingested,
        videos_failed=failed,
        skipped_disabled=False,
    )


# ---- helpers ----


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()
