"""Growth Score service tests (Phase 5/6) — fallback + LLM path with a fake."""

from __future__ import annotations

import pytest

from nexoclip.llm import GrowthScoreCard, PlatformGrowthScore
from nexoclip.score.growth import GrowthInput, compute_growth_score, fallback_card


def test_fallback_uses_publishability() -> None:
    inp = GrowthInput(
        clip_id="c1", duration_s=12, caption="hi", platforms=["tiktok", "x"],
        publishability_score=82,
    )
    card = fallback_card(inp)
    assert card.overall_score == 82
    assert {p.platform for p in card.platforms} == {"tiktok", "twitter"}
    assert all(p.verdict == "publish" for p in card.platforms)
    assert all(p.label == "Very Good" for p in card.platforms)


def test_fallback_low_score_archives() -> None:
    inp = GrowthInput(clip_id="c1", duration_s=12, caption="hi", platforms=["tiktok"],
                      publishability_score=20)
    card = fallback_card(inp)
    assert card.decision == "archive"
    assert card.platforms[0].verdict == "skip"


class _FakeRouter:
    def __init__(self, card: GrowthScoreCard) -> None:
        self._card = card
        self.calls: list[dict] = []

    async def complete(self, **kwargs):  # mimics LLMRouter.complete
        self.calls.append(kwargs)
        return self._card


@pytest.mark.asyncio
async def test_compute_normalizes_platform_aliases() -> None:
    card = GrowthScoreCard(
        overall_score=90, decision="publish_all",
        platforms=[PlatformGrowthScore(platform="X", score=88, verdict="publish",
                                       label="Very Good", reason="fits")],
        content_tags=["valorant", "ace"],
    )
    router = _FakeRouter(card)
    inp = GrowthInput(clip_id="c1", duration_s=12, caption="hi", platforms=["x"])
    out = await compute_growth_score(inp, tenant_id="t1", router=router)  # type: ignore[arg-type]
    assert out.platforms[0].platform == "twitter"  # alias normalized
    assert router.calls[0]["purpose"] == "growth_score"


@pytest.mark.asyncio
async def test_empty_platforms_skips_llm() -> None:
    router = _FakeRouter(GrowthScoreCard(overall_score=0, decision="skip"))
    inp = GrowthInput(clip_id="c1", duration_s=12, caption="hi", platforms=[])
    out = await compute_growth_score(inp, tenant_id="t1", router=router)  # type: ignore[arg-type]
    assert router.calls == []  # no LLM call when there are no targets
    assert out.decision in {"archive", "publish_select", "skip"}
