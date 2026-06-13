"""Built-in per-platform safety defaults + brand-kit policy assembly.

The numbers here are deliberately conservative rules-of-thumb for avoiding
platform spam-detection / shadowbanning — frequent, evenly-spaced,
round-the-clock posting from one account is exactly the fingerprint the
platforms throttle. Operators override per brand kit; this module is the
single place the defaults live.
"""

from __future__ import annotations

from typing import Any

from .models import PlatformSafetyRule, SafetyPolicy

# Per-platform anti-ghost envelopes. Quiet hours are local (policy TZ).
PLATFORM_DEFAULTS: dict[str, PlatformSafetyRule] = {
    # TikTok is the strictest about burst posting from one account.
    "tiktok": PlatformSafetyRule(
        min_spacing_min=150,
        daily_cap=4,
        quiet_hours_start=1,
        quiet_hours_end=7,
        jitter_min=12,
    ),
    # YouTube Shorts tolerates a higher cadence.
    "youtube": PlatformSafetyRule(
        min_spacing_min=90,
        daily_cap=8,
        quiet_hours_start=2,
        quiet_hours_end=6,
        jitter_min=10,
    ),
    # Instagram Reels punishes frequent posting hardest.
    "instagram": PlatformSafetyRule(
        min_spacing_min=180,
        daily_cap=3,
        quiet_hours_start=1,
        quiet_hours_end=7,
        jitter_min=15,
    ),
    # Buffer is a scheduler, not a platform — it does its own pacing, so we
    # only keep a light spacing nudge and no cap.
    "buffer": PlatformSafetyRule(
        min_spacing_min=30,
        daily_cap=0,
        quiet_hours_start=None,
        quiet_hours_end=None,
        jitter_min=5,
    ),
}

# Anything we don't have a tuned default for gets a moderate envelope.
_FALLBACK = PlatformSafetyRule(
    min_spacing_min=120,
    daily_cap=5,
    quiet_hours_start=1,
    quiet_hours_end=7,
    jitter_min=10,
)


def default_rule_for(platform: str) -> PlatformSafetyRule:
    """The built-in rule for a platform (a moderate fallback if unknown)."""
    return PLATFORM_DEFAULTS.get(platform, _FALLBACK)


def policy_for_kit(kit: Any) -> SafetyPolicy:
    """Build a `SafetyPolicy` from a brand kit.

    Reads `kit.content_timezone` and `kit.safety_policy` — a
    `{platform: {field: value, ...}}` dict of partial overrides. Each
    override is overlaid field-by-field on the built-in default so an
    operator can tweak just the daily cap without restating quiet hours.
    """
    tz = getattr(kit, "content_timezone", None) or "UTC"
    overrides = getattr(kit, "safety_policy", None) or {}

    rules: dict[str, PlatformSafetyRule] = {}
    if isinstance(overrides, dict):
        for platform, override in overrides.items():
            if not isinstance(override, dict):
                continue
            merged = default_rule_for(platform).model_dump()
            merged.update({k: v for k, v in override.items() if v is not None})
            rules[platform] = PlatformSafetyRule.model_validate(merged)

    return SafetyPolicy(timezone=tz, rules=rules)
