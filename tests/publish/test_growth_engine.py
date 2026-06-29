"""Growth Engine planner integration test (Phases 1-6 wired together)."""

from __future__ import annotations

import datetime as _dt
import random

from nexoclip.llm import GrowthScoreCard, PlatformGrowthScore
from nexoclip.publish.growth_engine import ClipContent, plan_growth_publish
from nexoclip.publish.pacing import DEFAULT_PLATFORM_RULES

_NOW = _dt.datetime(2026, 6, 10, 8, 0, tzinfo=_dt.UTC)


def _card(overall: int, platforms: dict[str, tuple[int, str]], tags: list[str]) -> GrowthScoreCard:
    return GrowthScoreCard(
        overall_score=overall,
        decision="publish_select",
        platforms=[
            PlatformGrowthScore(platform=p, score=s, verdict=v, label="x", reason="r")
            for p, (s, v) in platforms.items()
        ],
        content_tags=tags,
    )


def _clip(cid: str, overall: int, platforms: dict, tags: list[str]) -> ClipContent:
    return ClipContent(
        clip_id=cid, caption=f"caption for {cid}", hook="hook",
        hashtags=["gaming", "viral", "clip"], card=_card(overall, platforms, tags),
    )


def test_skip_verdict_and_floor_drop_weak_platforms() -> None:
    clips = [
        _clip("c1", 92, {"tiktok": (95, "publish"), "linkedin": (12, "skip")}, ["a"]),
    ]
    plan = plan_growth_publish(
        clips, connected=["tiktok", "linkedin"], rules=DEFAULT_PLATFORM_RULES,
        now=_NOW, min_score=40, rng=random.Random(1),
    )
    platforms = {p.platform for p in plan.posts}
    assert platforms == {"tiktok"}  # linkedin skipped (verdict + floor)


def test_per_platform_cap_holds_the_rest() -> None:
    # 8 clips all great on tiktok (cap 5) → exactly 5 scheduled today, the rest
    # held for the next sweep (Phase 2: "the rest wait in the queue").
    clips = [
        _clip(f"c{i}", 90 - i, {"tiktok": (90 - i, "publish")}, [f"theme{i}"])
        for i in range(8)
    ]
    plan = plan_growth_publish(
        clips, connected=["tiktok"], rules=DEFAULT_PLATFORM_RULES, now=_NOW,
        rng=random.Random(2),
    )
    tiktok_posts = [p for p in plan.posts if p.platform == "tiktok"]
    cap = DEFAULT_PLATFORM_RULES["tiktok"].max_per_day
    assert len(tiktok_posts) == cap  # 5 best-fit clips today
    assert {p.clip_id for p in tiktok_posts} == {f"c{i}" for i in range(cap)}
    assert set(plan.held_allocation) == {f"c{i}" for i in range(cap, 8)}  # 3 held


def test_budget_limits_pool() -> None:
    clips = [_clip(f"c{i}", 90 - i * 5, {"tiktok": (90 - i * 5, "publish")}, [f"t{i}"])
             for i in range(5)]
    plan = plan_growth_publish(
        clips, connected=["tiktok"], rules=DEFAULT_PLATFORM_RULES, now=_NOW,
        budget=2, rng=random.Random(3),
    )
    scheduled_clips = {p.clip_id for p in plan.posts}
    assert scheduled_clips == {"c0", "c1"}
    assert set(plan.queued) == {"c2", "c3", "c4"}


def test_fatigue_holds_similar_clips() -> None:
    clips = [
        _clip("c1", 95, {"tiktok": (95, "publish")}, ["warzone", "verdansk", "ak47"]),
        _clip("c2", 90, {"tiktok": (90, "publish")}, ["warzone", "verdansk", "ak47"]),
        _clip("c3", 85, {"tiktok": (85, "publish")}, ["warzone", "verdansk", "ak47"]),
    ]
    plan = plan_growth_publish(
        clips, connected=["tiktok"], rules=DEFAULT_PLATFORM_RULES, now=_NOW,
        fatigue_window=2, fatigue_threshold=0.5, rng=random.Random(4),
    )
    assert "c3" in plan.held_fatigue
    assert {p.clip_id for p in plan.posts} == {"c1", "c2"}


def test_schedule_honours_min_gap_and_assets_conformed() -> None:
    clips = [
        _clip("c1", 95, {"tiktok": (95, "publish")}, ["a"]),
        _clip("c2", 90, {"tiktok": (90, "publish")}, ["b"]),
    ]
    plan = plan_growth_publish(
        clips, connected=["tiktok"], rules=DEFAULT_PLATFORM_RULES, now=_NOW,
        rng=random.Random(5),
    )
    posts = sorted(plan.posts, key=lambda p: p.when)
    gap = posts[1].when - posts[0].when
    # tiktok min_gap 180 minus jitter window (±20*2) → still clearly spaced.
    assert gap >= _dt.timedelta(minutes=180 - 40)
    # Per-platform asset: caption clamped to tiktok 150, hashtags to 5.
    for p in posts:
        assert len(p.asset.caption) <= 150
        assert len(p.asset.hashtags) <= 5
        assert p.asset.filename.endswith("_tiktok.mp4")
