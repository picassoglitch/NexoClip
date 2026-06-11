"""Hub publish service unit tests (phase 3) — the deterministic parts.

The API surface is covered end-to-end in tests/api/test_internal_publish.py;
these pin the pure logic: batch time planning (anti-spam cap), best-time
resolution, media-URL validation, and service-token parsing.
"""

from __future__ import annotations

import datetime as _dt

import httpx
import pytest
import respx

from nexoclip.publish.hub import (
    HubPublishError,
    next_best_time,
    plan_batch_times,
    validate_media_url,
)
from nexoclip.settings import Settings

_NOW = _dt.datetime(2026, 6, 10, 8, 0, tzinfo=_dt.UTC)  # Wednesday 08:00 UTC


# ---- plan_batch_times ----


def test_batch_spreads_within_daily_cap() -> None:
    times = plan_batch_times(6, now=_NOW, cap_per_day=4, existing_today=0)
    assert len(times) == 6
    by_day: dict[str, int] = {}
    for t in times:
        by_day[t.date().isoformat()] = by_day.get(t.date().isoformat(), 0) + 1
    # Never more than the cap on any single day; overflow rolls forward.
    assert all(n <= 4 for n in by_day.values())
    assert by_day["2026-06-10"] == 4
    assert by_day["2026-06-11"] == 2
    # Strictly future (> now + 30 min) and chronological per day.
    assert all(t > _NOW + _dt.timedelta(minutes=30) for t in times)
    assert times == sorted(times)


def test_batch_counts_existing_posts_against_today() -> None:
    times = plan_batch_times(3, now=_NOW, cap_per_day=4, existing_today=3)
    # Only one slot left today; the rest roll to tomorrow.
    assert sum(1 for t in times if t.date() == _NOW.date()) == 1
    assert sum(1 for t in times if t.date() == _NOW.date() + _dt.timedelta(days=1)) == 2


def test_batch_full_day_rolls_everything_forward() -> None:
    times = plan_batch_times(2, now=_NOW, cap_per_day=4, existing_today=4)
    assert all(t.date() > _NOW.date() for t in times)


def test_batch_late_day_skips_past_hours() -> None:
    late = _NOW.replace(hour=23)
    times = plan_batch_times(3, now=late, cap_per_day=4, existing_today=0)
    assert len(times) == 3
    assert all(t > late + _dt.timedelta(minutes=30) for t in times)


def test_batch_prefers_best_time_hours_for_matching_weekday() -> None:
    # Wednesday = weekday 2; high-engagement slot at 18:00 UTC.
    slots = [
        {"day_of_week": 2, "hour": 18, "avg_engagement": 500.0, "post_count": 9},
        {"day_of_week": 0, "hour": 9, "avg_engagement": 300.0, "post_count": 5},
    ]
    times = plan_batch_times(
        1, now=_NOW, cap_per_day=1, existing_today=0, best_slots=slots,
    )
    assert times[0].hour == 18
    assert times[0].date() == _NOW.date()


# ---- next_best_time ----


def test_next_best_time_picks_top_slot_next_occurrence() -> None:
    slots = [
        {"day_of_week": 2, "hour": 18, "avg_engagement": 510.3, "post_count": 15},
        {"day_of_week": 0, "hour": 9, "avg_engagement": 342.5, "post_count": 12},
    ]
    best = next_best_time(slots, now=_NOW)
    assert best is not None
    # Wednesday 18:00 is later today — the top slot wins.
    assert (best.weekday(), best.hour) == (2, 18)
    assert best.date() == _NOW.date()


def test_next_best_time_rolls_a_week_when_slot_already_passed() -> None:
    slots = [{"day_of_week": 2, "hour": 6, "avg_engagement": 100.0}]
    best = next_best_time(slots, now=_NOW)  # 08:00 Wed > 06:00 Wed
    assert best is not None
    assert best - _NOW == _dt.timedelta(days=7) - _dt.timedelta(hours=2)


def test_next_best_time_none_without_usable_slots() -> None:
    assert next_best_time([], now=_NOW) is None
    assert next_best_time([{"day_of_week": "x", "hour": "y"}], now=_NOW) is None


# ---- validate_media_url ----


@pytest.mark.asyncio
async def test_media_url_ok_video() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.head("https://cdn.test/clip.mp4").mock(
                return_value=httpx.Response(
                    200, headers={"content-type": "video/mp4"}
                )
            )
            await validate_media_url("https://cdn.test/clip.mp4", http=http)


@pytest.mark.asyncio
async def test_media_url_head_405_falls_back_to_ranged_get() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock(assert_all_called=True) as mock:
            mock.head("https://cdn.test/clip.mp4").mock(
                return_value=httpx.Response(405)
            )
            get = mock.get("https://cdn.test/clip.mp4").mock(
                return_value=httpx.Response(
                    206, headers={"content-type": "video/mp4"}
                )
            )
            await validate_media_url("https://cdn.test/clip.mp4", http=http)
    assert get.calls.last.request.headers["Range"] == "bytes=0-0"


@pytest.mark.asyncio
async def test_media_url_404_is_unreachable() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.head("https://cdn.test/gone.mp4").mock(
                return_value=httpx.Response(404)
            )
            with pytest.raises(HubPublishError) as ei:
                await validate_media_url("https://cdn.test/gone.mp4", http=http)
    assert ei.value.code == "media_url_unreachable"


@pytest.mark.asyncio
async def test_media_url_html_is_not_video() -> None:
    async with httpx.AsyncClient() as http:
        with respx.mock() as mock:
            mock.head("https://cdn.test/page").mock(
                return_value=httpx.Response(
                    200, headers={"content-type": "text/html; charset=utf-8"}
                )
            )
            with pytest.raises(HubPublishError) as ei:
                await validate_media_url("https://cdn.test/page", http=http)
    assert ei.value.code == "media_url_not_video"


@pytest.mark.asyncio
async def test_media_url_non_http_rejected() -> None:
    with pytest.raises(HubPublishError) as ei:
        await validate_media_url("file:///etc/passwd")
    assert ei.value.code == "media_url_invalid"


# ---- service-token parsing ----


def test_hub_service_token_map_parses_pairs() -> None:
    s = Settings(hub_service_tokens="nexoobs:tok_a, nexoai:tok_b,broken,:empty")
    assert s.hub_service_token_map() == {"tok_a": "nexoobs", "tok_b": "nexoai"}


def test_hub_service_token_map_empty_when_unset() -> None:
    assert Settings(hub_service_tokens=None).hub_service_token_map() == {}
