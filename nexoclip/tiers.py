"""Canonical subscription tiers — single source of truth.

NexoClip recognizes exactly three internal tiers, lowest → highest:

    free        — ingest + clip + DOWNLOAD (watermarked). No publishing.
    pro         — "mid" tier. Adds the paid perks below the top tier
                  (e.g. Drive export — see task #31). NOT publishing.
    all_access  — "top" tier. Adds external publishing (Zernio /
                  TikTok / YouTube / IG auto-posting).

Nexo AI (the IdP) sometimes labels the top tier differently on its
side — e.g. it sends `partner` in the SSO token for partner accounts.
Those are ALIASES of our canonical tiers. Before this module existed
the alias was silently dropped (the provisioning validator only
accepted the three canonical names), so a `partner` user landed as
`free` in NexoClip — losing every paid perk including the upload-post
profile limit. Centralizing the alias map here means a new label from
Nexo AI is a one-line addition, applied everywhere tier is ingested
or read.

Rule: normalize at every boundary where a tier ENTERS NexoClip
(provisioning, SSO sync) and at the request-scoped READ choke point
(auth middleware). Downstream gates then compare against the
canonical sets without re-normalizing.
"""

from __future__ import annotations

from typing import Final

# The only tier strings the rest of the codebase should ever compare
# against. Anything stored / read goes through normalize_tier first.
FREE: Final = "free"
PRO: Final = "pro"
ALL_ACCESS: Final = "all_access"

CANONICAL_TIERS: Final[frozenset[str]] = frozenset({FREE, PRO, ALL_ACCESS})

# Tiers that may publish to external platforms (TikTok / YouTube / IG).
# Top tier only — `pro` is the mid tier (Drive export), `free` downloads
# locally with a watermark.
TOP_TIERS: Final[frozenset[str]] = frozenset({ALL_ACCESS})

# Tiers above free — any paid perk that isn't specifically top-tier.
PAID_TIERS: Final[frozenset[str]] = frozenset({PRO, ALL_ACCESS})

# Aliases Nexo AI (or future billing sources) may use for a canonical
# tier. Keyed lowercase. Extend this — NOT the comparison sites — when
# a new label shows up. `partner` == our top tier `all_access`.
_ALIASES: Final[dict[str, str]] = {
    "partner": ALL_ACCESS,
    "partners": ALL_ACCESS,
    "enterprise": ALL_ACCESS,
    "allaccess": ALL_ACCESS,
    "all-access": ALL_ACCESS,
}


def resolve_tier_alias(raw: str | None) -> str | None:
    """Map a raw tier label to its canonical name, or None if it isn't a
    tier we recognize.

    Used where "unrecognized → don't touch the existing value" is the
    right call (the provisioning validator, the SSO sync write): a typo
    or a brand-new label we haven't mapped must NOT silently downgrade
    an existing paid tenant to free. Returns None in that case so the
    caller can skip the override.
    """
    if not raw:
        return None
    v = raw.lower().strip()
    v = _ALIASES.get(v, v)
    return v if v in CANONICAL_TIERS else None


def normalize_tier(raw: str | None) -> str:
    """Map a raw tier label to its canonical name, defaulting to `free`.

    Used at READ time where we always need a concrete tier to gate on
    (the auth middleware that sets request.state.tenant_tier). An
    unrecognized / missing value resolves to `free` — the safe default
    (least privilege), since a read-time fallback only ever GRANTS
    fewer perks, never more.
    """
    return resolve_tier_alias(raw) or FREE


__all__ = [
    "FREE",
    "PRO",
    "ALL_ACCESS",
    "CANONICAL_TIERS",
    "TOP_TIERS",
    "PAID_TIERS",
    "resolve_tier_alias",
    "normalize_tier",
]
