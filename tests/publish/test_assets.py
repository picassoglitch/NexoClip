"""Per-platform asset matrix unit tests (Phase 3)."""

from __future__ import annotations

from nexoclip.publish.assets import build_platform_assets


def test_distinct_filenames_per_platform() -> None:
    assets = build_platform_assets(
        clip_id="clp_123", base_caption="hi", base_hashtags=[], base_title=None,
        hook="", platforms=["tiktok", "instagram", "youtube"],
    )
    names = {a.filename for a in assets.values()}
    assert names == {"clp_123_tiktok.mp4", "clp_123_instagram.mp4", "clp_123_youtube.mp4"}


def test_caption_clamped_per_platform() -> None:
    long = "x" * 500
    assets = build_platform_assets(
        clip_id="c", base_caption=long, base_hashtags=[], base_title=None,
        hook="", platforms=["tiktok", "linkedin"],
    )
    # TikTok caps at 150, LinkedIn at 1200 → tiktok must be shorter.
    assert len(assets["tiktok"].caption) <= 150
    assert len(assets["linkedin"].caption) == 500


def test_hashtags_clamped_per_platform() -> None:
    tags = [f"t{i}" for i in range(10)]
    assets = build_platform_assets(
        clip_id="c", base_caption="hi", base_hashtags=tags, base_title=None,
        hook="", platforms=["tiktok", "linkedin", "pinterest"],
    )
    assert len(assets["tiktok"].hashtags) == 5
    assert assets["linkedin"].hashtags == []  # hashtag_max=0
    assert len(assets["pinterest"].hashtags) == 5


def test_youtube_gets_seo_title_from_hook() -> None:
    assets = build_platform_assets(
        clip_id="c", base_caption="body text here", base_hashtags=[], base_title=None,
        hook="INSANE 1v5 clutch", platforms=["youtube", "tiktok"],
    )
    assert assets["youtube"].title == "INSANE 1v5 clutch"
    # TikTok carries no separate title.
    assert assets["tiktok"].title is None


def test_first_comment_only_where_supported() -> None:
    assets = build_platform_assets(
        clip_id="c", base_caption="hi", base_hashtags=[], base_title=None, hook="",
        platforms=["instagram", "tiktok"], first_comment="follow for more",
    )
    assert assets["instagram"].first_comment == "follow for more"
    assert assets["tiktok"].first_comment is None  # TikTok has no firstComment


def test_x_alias_keys_canonically() -> None:
    assets = build_platform_assets(
        clip_id="c", base_caption="hi", base_hashtags=["a", "b", "c"], base_title=None,
        hook="", platforms=["x"],
    )
    assert "twitter" in assets
    assert len(assets["twitter"].hashtags) == 2  # twitter hashtag_max=2


def test_caption_with_tags_appends() -> None:
    assets = build_platform_assets(
        clip_id="c", base_caption="great clip", base_hashtags=["gaming", "viral"],
        base_title=None, hook="", platforms=["tiktok"],
    )
    out = assets["tiktok"].caption_with_tags()
    assert "#gaming" in out and "#viral" in out
