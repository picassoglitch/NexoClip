-- Nexo AI integration — slice NX.2.
--
-- Adds `tenants.status` so Nexo AI can pause a tenant remotely. When a PRO
-- subscriber on Nexo AI switches their live engine to something else (e.g.
-- NexoStream), Nexo AI calls POST /api/admin/tenants/{id}/status with
-- {status: 'paused'} so we stop running jobs / spending tokens / publishing
-- for that tenant.
--
-- 'active' is the only state that lets the pipeline + publish workers run.
-- 'paused' means jobs queued before the pause still finish their current
-- step, but no NEW jobs start, no LLM calls fire, no clip publishes happen.
-- 'cancelled' is reserved for future hard-stop (delete data, refuse logins)
-- — not used yet.
--
-- Existing tenants default to 'active' so nothing changes for users who
-- aren't tied to Nexo AI's live-engine selection.

ALTER TABLE tenants ADD COLUMN status TEXT NOT NULL DEFAULT 'active'
  CHECK (status IN ('active', 'paused', 'cancelled'));

CREATE INDEX IF NOT EXISTS idx_tenants_status ON tenants(status);
