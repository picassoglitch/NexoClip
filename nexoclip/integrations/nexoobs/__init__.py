"""NexoOBS integration — the bidirectional live connection switch.

NexoOBS is the streaming front-door; NexoClip is the clipper. The
"connection" between them is a single per-tenant flag whose source of
truth lives in NexoOBS (nexoobs_sessions.clips_enabled). This module
lets NexoClip read + flip that flag so the switch works from the
NexoClip Live page too.

Both sides present the same shared internal bearer
(NEXOCLIP_INTERNAL_SIGNING_SECRET == NexoOBS's NEXOOBS_RELAY_SECRET).
Keyed by the tenant's Nexo AI user id (external_user_id), which NexoOBS
uses as its tenant_id.
"""

from .connection import get_connection, set_connection

__all__ = ["get_connection", "set_connection"]
