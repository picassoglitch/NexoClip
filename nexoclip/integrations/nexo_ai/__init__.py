"""Nexo AI ↔ NexoClip bridge.

Two surfaces:
  * `service.provision_tenant_for_nexo_ai` — idempotent: creates a tenant + an
    api_token bound to the Nexo AI user_id, OR returns the existing pair if
    that user_id already maps to one. Called from POST /api/admin/tenants.
  * `sso.verify_token` + `sso.sign_token` — HMAC-SHA256 over the JSON payload
    Nexo AI signs when the user clicks "Abrir NexoClip ↗". Called from
    GET /auth/sso.

The wire contract is documented in docs/nexo_ai_integration.md.
"""

from .service import (
    NexoAiProvisionResult,
    NexoAiServiceError,
    provision_tenant_for_nexo_ai,
    sync_tenant_tier,
)
from .sso import SsoTokenError, SsoTokenPayload, verify_sso_token

__all__ = [
    "NexoAiProvisionResult",
    "NexoAiServiceError",
    "SsoTokenError",
    "SsoTokenPayload",
    "provision_tenant_for_nexo_ai",
    "sync_tenant_tier",
    "verify_sso_token",
]
