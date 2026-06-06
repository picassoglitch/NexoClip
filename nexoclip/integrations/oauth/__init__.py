"""OAuth plumbing shared across platform integrations.

  encryption.py — Fernet wrapper around access_token / refresh_token
                  storage on connected_accounts. NEXOCLIP_CREDS_KEY
                  env var holds the symmetric key.

  state.py      — HMAC-signed CSRF state token. Carries the tenant_id
                  through the OAuth round-trip so the callback knows
                  which tenant just authorized. Public-callback safe.

Per-platform connect flows live in `nexoclip.integrations.<platform>.auth`
and re-use these helpers. Pattern A: one app per platform owned by
NexoClip; tenants never see a client_secret.
"""
