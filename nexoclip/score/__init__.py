"""Multimodal scoring — the pre-publish Growth Score.

`score.growth.compute_growth_score` turns a clip's signals + caption + target
platforms into a `GrowthScoreCard`: an overall growth score, a per-platform
fit + publish/skip verdict, optimization recommendations, and the content
tags that drive fatigue spacing. Computed BEFORE publishing so the engine
optimizes instead of blasting.
"""

from .growth import GrowthInput, compute_growth_score, fallback_card

__all__ = ["GrowthInput", "compute_growth_score", "fallback_card"]
