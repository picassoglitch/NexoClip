"""ULID-based entity ID generation.

Per CLAUDE.md hard rule #7: IDs are ULIDs with a per-entity prefix.
ULIDs sort by creation time, which makes pagination and debugging nicer.
"""

from __future__ import annotations

from typing import Literal

from ulid import ULID

EntityKind = Literal[
    "str",
    "clp",
    "var",
    "job",
    "ten",
    "usr",
    "tok",
    "evt",
    "cnd",
    "llm",
    "acc",  # connected accounts (Buffer / future platforms)
    "pjb",  # publish jobs
    "whk",  # webhook subscriptions (Phase 2)
    "met",  # publish_metrics (Phase 3)
    "spk",  # speakers — persistent voice identities (voice-markers spec slice B.2)
    "vsp",  # vod_speakers — per-VOD speaker resolution
]


def new_id(kind: EntityKind) -> str:
    """Mint a new prefixed ULID for the given entity kind."""
    return f"{kind}_{ULID()}"
