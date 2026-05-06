"""Repository layer — one class per table, tenancy-enforced at the SQL boundary.

Every read and write of tenant-scoped data calls `current_tenant_id()` and
includes `tenant_id = ?` in the WHERE / VALUES clause. Defense in depth:
even if a handler forgot to bind the right tenant, the repo refuses to
return another tenant's rows.

Phase 0 service models (Stream, Candidate, Clip, ...) are converted
to/from these DB rows inside Task 1's persistence migration. For Task 0
the repos accept and return the DB models in `nexoclip.db.models`.

Auth path note: `ApiTokensRepo.lookup_by_hash` is the ONE tenant-unscoped
read in this module — it has to be, because we don't know the tenant
until the lookup succeeds. Every other read enforces tenancy.
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any, TypeVar

import aiosqlite
from pydantic import BaseModel

from nexoclip.errors import NexoClipError, TenancyError
from nexoclip.ids import new_id
from nexoclip.tenancy import current_tenant_id

from .connection import Database
from .models import (
    ApiTokenRow,
    CandidateRow,
    ClipRow,
    ConnectedAccount,
    Event,
    LLMCallRow,
    PersonaRow,
    PublishJob,
    StreamRow,
    Tenant,
    TranscriptRow,
    User,
    VariantRow,
)

_M = TypeVar("_M", bound=BaseModel)


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


class TenantsRepo:
    """Tenants table — the one place where we don't filter by tenant_id."""

    def __init__(self, db: Database):
        self._db = db

    async def create(self, *, tenant_id: str | None = None, name: str) -> Tenant:
        tid = tenant_id or new_id("ten")
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO tenants (id, name, created_at) VALUES (?, ?, ?)",
            (tid, name, _now()),
        )
        await conn.commit()
        return await self.get_or_raise(tid)

    async def get(self, tenant_id: str) -> Tenant | None:
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT id, name, created_at FROM tenants WHERE id = ?",
            (tenant_id,),
        )
        return _model(Tenant, await cur.fetchone())

    async def get_or_raise(self, tenant_id: str) -> Tenant:
        t = await self.get(tenant_id)
        if t is None:
            raise NexoClipError(f"tenant not found: {tenant_id}")
        return t

    async def list_all(self) -> list[Tenant]:
        conn = await self._db.connect()
        cur = await conn.execute("SELECT id, name, created_at FROM tenants ORDER BY created_at")
        return [Tenant.model_validate(dict(r)) for r in await cur.fetchall()]


class UsersRepo:
    """Users live under a tenant; reads filter on `current_tenant_id()`."""

    def __init__(self, db: Database):
        self._db = db

    async def create(self, *, email: str, role: str = "owner") -> User:
        tenant_id = current_tenant_id()
        user_id = new_id("usr")
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO users (id, tenant_id, email, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, tenant_id, email, role, _now()),
        )
        await conn.commit()
        u = await self.get(user_id)
        assert u is not None  # just inserted
        return u

    async def get(self, user_id: str) -> User | None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT id, tenant_id, email, role, created_at FROM users "
            "WHERE id = ? AND tenant_id = ?",
            (user_id, tenant_id),
        )
        return _model(User, await cur.fetchone())

    async def list_for_tenant(self) -> list[User]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT id, tenant_id, email, role, created_at FROM users "
            "WHERE tenant_id = ? ORDER BY created_at",
            (tenant_id,),
        )
        return [User.model_validate(dict(r)) for r in await cur.fetchall()]


class ApiTokensRepo:
    """API tokens. Stores only the sha256 hash, never the raw token."""

    def __init__(self, db: Database):
        self._db = db

    async def create(self, *, hash_: str, scope: str = "full") -> ApiTokenRow:
        tenant_id = current_tenant_id()
        token_id = new_id("tok")
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO api_tokens (id, tenant_id, hash, scope, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (token_id, tenant_id, hash_, scope, _now()),
        )
        await conn.commit()
        cur = await conn.execute(
            "SELECT id, tenant_id, hash, scope, created_at, last_used_at "
            "FROM api_tokens WHERE id = ? AND tenant_id = ?",
            (token_id, tenant_id),
        )
        row = await cur.fetchone()
        assert row is not None
        return ApiTokenRow.model_validate(dict(row))

    async def lookup_by_hash(self, hash_: str) -> ApiTokenRow | None:
        """Auth path: tenant-unscoped lookup. Returns the row including tenant_id.

        This is the ONLY read in the repo layer that doesn't filter by
        tenant — by definition we don't know the tenant until this lookup
        succeeds. After it returns, the caller is expected to bind that
        tenant via `bound_tenant(...)` before issuing any further query.
        """
        if not hash_:
            raise TenancyError("empty token hash")
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT id, tenant_id, hash, scope, created_at, last_used_at "
            "FROM api_tokens WHERE hash = ?",
            (hash_,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        # Touch last_used_at so we can audit token usage.
        await conn.execute(
            "UPDATE api_tokens SET last_used_at = ? WHERE id = ?",
            (_now(), row["id"]),
        )
        await conn.commit()
        return ApiTokenRow.model_validate(dict(row))

    async def list_for_tenant(self) -> list[ApiTokenRow]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT id, tenant_id, hash, scope, created_at, last_used_at "
            "FROM api_tokens WHERE tenant_id = ? ORDER BY created_at",
            (tenant_id,),
        )
        return [ApiTokenRow.model_validate(dict(r)) for r in await cur.fetchall()]


class PersonasRepo:
    def __init__(self, db: Database):
        self._db = db

    async def create(
        self,
        *,
        persona_id: str,
        name: str,
        primary_language: str,
        target_languages: list[str],
        voice_prompt: str,
        routing_tags: list[str] | None = None,
    ) -> PersonaRow:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO personas (id, tenant_id, name, primary_language, "
            "target_languages_json, voice_prompt, routing_tags_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                persona_id,
                tenant_id,
                name,
                primary_language,
                json.dumps(target_languages),
                voice_prompt,
                json.dumps(routing_tags or []),
                _now(),
            ),
        )
        await conn.commit()
        p = await self.get(persona_id)
        assert p is not None
        return p

    async def upsert(
        self,
        *,
        persona_id: str,
        name: str,
        primary_language: str,
        target_languages: list[str],
        voice_prompt: str,
        routing_tags: list[str] | None = None,
    ) -> PersonaRow:
        """Insert or replace. Used by the YAML→DB transition in Task 1."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO personas (id, tenant_id, name, primary_language, "
            "target_languages_json, voice_prompt, routing_tags_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "name = excluded.name, "
            "primary_language = excluded.primary_language, "
            "target_languages_json = excluded.target_languages_json, "
            "voice_prompt = excluded.voice_prompt, "
            "routing_tags_json = excluded.routing_tags_json",
            (
                persona_id,
                tenant_id,
                name,
                primary_language,
                json.dumps(target_languages),
                voice_prompt,
                json.dumps(routing_tags or []),
                _now(),
            ),
        )
        await conn.commit()
        p = await self.get(persona_id)
        assert p is not None
        return p

    async def get(self, persona_id: str) -> PersonaRow | None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM personas WHERE id = ? AND tenant_id = ?",
            (persona_id, tenant_id),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return _persona_from_row(row)

    async def list_for_tenant(self) -> list[PersonaRow]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM personas WHERE tenant_id = ? ORDER BY created_at",
            (tenant_id,),
        )
        return [_persona_from_row(r) for r in await cur.fetchall()]


def _persona_from_row(row: aiosqlite.Row) -> PersonaRow:
    d = dict(row)
    d["target_languages"] = json.loads(d.pop("target_languages_json"))
    d["routing_tags"] = json.loads(d.pop("routing_tags_json"))
    return PersonaRow.model_validate(d)


class StreamsRepo:
    def __init__(self, db: Database):
        self._db = db

    async def upsert(self, stream: StreamRow) -> StreamRow:
        """Insert if new; otherwise leave the row untouched (idempotent resume)."""
        if stream.tenant_id != current_tenant_id():
            raise TenancyError(
                f"stream tenant {stream.tenant_id!r} != bound {current_tenant_id()!r}"
            )
        conn = await self._db.connect()
        await conn.execute(
            "INSERT OR IGNORE INTO streams "
            "(id, tenant_id, vod_url, platform, title, channel, duration_s, "
            "source_video_path, source_audio_path, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                stream.id,
                stream.tenant_id,
                stream.vod_url,
                stream.platform,
                stream.title,
                stream.channel,
                stream.duration_s,
                stream.source_video_path,
                stream.source_audio_path,
                stream.status,
                stream.created_at,
            ),
        )
        await conn.commit()
        existing = await self.get(stream.id)
        assert existing is not None
        return existing

    async def get(self, stream_id: str) -> StreamRow | None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM streams WHERE id = ? AND tenant_id = ?",
            (stream_id, tenant_id),
        )
        return _model(StreamRow, await cur.fetchone())

    async def list_for_tenant(self) -> list[StreamRow]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM streams WHERE tenant_id = ? ORDER BY created_at DESC",
            (tenant_id,),
        )
        return [StreamRow.model_validate(dict(r)) for r in await cur.fetchall()]


class LLMCallsRepo:
    """Append-only log of every LLM call for billing + audit."""

    def __init__(self, db: Database):
        self._db = db

    async def record(self, row: LLMCallRow) -> None:
        if row.tenant_id != current_tenant_id():
            raise TenancyError(
                f"llm_call tenant {row.tenant_id!r} != bound {current_tenant_id()!r}"
            )
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO llm_calls (id, tenant_id, purpose, provider, model, quality, "
            "input_tokens, output_tokens, cost_usd_micros, status, error, attempts, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.id,
                row.tenant_id,
                row.purpose,
                row.provider,
                row.model,
                row.quality,
                row.input_tokens,
                row.output_tokens,
                row.cost_usd_micros,
                row.status,
                row.error,
                row.attempts,
                row.ts,
            ),
        )
        await conn.commit()

    async def list_for_tenant(self, *, limit: int = 100) -> list[LLMCallRow]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM llm_calls WHERE tenant_id = ? ORDER BY ts DESC LIMIT ?",
            (tenant_id, limit),
        )
        return [LLMCallRow.model_validate(dict(r)) for r in await cur.fetchall()]


class EventsRepo:
    """Append-only event log. Every state transition lands here."""

    def __init__(self, db: Database):
        self._db = db

    async def emit(
        self,
        *,
        type: str,  # matches the column name
        payload: dict[str, Any] | None = None,
    ) -> Event:
        tenant_id = current_tenant_id()
        event_id = new_id("evt")
        ts = _now()
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO events (id, tenant_id, type, payload_json, ts) VALUES (?, ?, ?, ?, ?)",
            (event_id, tenant_id, type, json.dumps(payload or {}), ts),
        )
        await conn.commit()
        return Event(
            id=event_id,
            tenant_id=tenant_id,
            type=type,
            payload=payload or {},
            ts=ts,
        )

    async def list_for_tenant(
        self,
        *,
        type: str | None = None,
        limit: int = 100,
    ) -> list[Event]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        if type is None:
            cur = await conn.execute(
                "SELECT id, tenant_id, type, payload_json, ts FROM events "
                "WHERE tenant_id = ? ORDER BY ts DESC LIMIT ?",
                (tenant_id, limit),
            )
        else:
            cur = await conn.execute(
                "SELECT id, tenant_id, type, payload_json, ts FROM events "
                "WHERE tenant_id = ? AND type = ? ORDER BY ts DESC LIMIT ?",
                (tenant_id, type, limit),
            )
        return [_event_from_row(r) for r in await cur.fetchall()]


def _event_from_row(row: aiosqlite.Row) -> Event:
    d = dict(row)
    d["payload"] = json.loads(d.pop("payload_json") or "{}")
    return Event.model_validate(d)


def _model(cls: type[_M], row: aiosqlite.Row | None) -> _M | None:
    """Round-trip an aiosqlite Row to a Pydantic model, or return None."""
    if row is None:
        return None
    return cls.model_validate(dict(row))


# ---------------------------------------------------------------------------
# Pipeline repos: transcripts, candidates, clips, variants.
#
# All four follow the same pattern as StreamsRepo: every read filters by
# `current_tenant_id()`; every write asserts the row's tenant_id matches.
# Bulk methods are idempotent on insert conflicts so resuming a partially-
# committed pipeline doesn't duplicate rows.
# ---------------------------------------------------------------------------


class TranscriptsRepo:
    """One transcript per stream (PK is stream_id)."""

    def __init__(self, db: Database):
        self._db = db

    async def upsert(self, row: TranscriptRow) -> TranscriptRow:
        if row.tenant_id != current_tenant_id():
            raise TenancyError(
                f"transcript tenant {row.tenant_id!r} != bound {current_tenant_id()!r}"
            )
        conn = await self._db.connect()
        # Replace any existing transcript for this stream — Whisper output
        # for the same stream is deterministic enough that a re-run is fine.
        await conn.execute(
            "INSERT INTO transcripts "
            "(stream_id, tenant_id, language, duration_s, model, segments_json, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(stream_id) DO UPDATE SET "
            "language = excluded.language, "
            "duration_s = excluded.duration_s, "
            "model = excluded.model, "
            "segments_json = excluded.segments_json",
            (
                row.stream_id,
                row.tenant_id,
                row.language,
                row.duration_s,
                row.model,
                row.segments_json,
                row.created_at,
            ),
        )
        await conn.commit()
        existing = await self.get(row.stream_id)
        assert existing is not None
        return existing

    async def get(self, stream_id: str) -> TranscriptRow | None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM transcripts WHERE stream_id = ? AND tenant_id = ?",
            (stream_id, tenant_id),
        )
        return _model(TranscriptRow, await cur.fetchone())


class CandidatesRepo:
    """Detected candidates. Insert is idempotent on (id) so resumes work."""

    def __init__(self, db: Database):
        self._db = db

    async def upsert_many(self, rows: list[CandidateRow]) -> int:
        """Insert candidates, ignoring conflicts. Returns count inserted (or 0)."""
        if not rows:
            return 0
        bound = current_tenant_id()
        for row in rows:
            if row.tenant_id != bound:
                raise TenancyError(f"candidate tenant {row.tenant_id!r} != bound {bound!r}")
        conn = await self._db.connect()
        await conn.executemany(
            "INSERT OR IGNORE INTO candidates "
            "(id, stream_id, tenant_id, ts, score, reason, evidence_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r.id,
                    r.stream_id,
                    r.tenant_id,
                    r.ts,
                    r.score,
                    r.reason,
                    json.dumps(r.evidence),
                    r.created_at,
                )
                for r in rows
            ],
        )
        await conn.commit()
        return len(rows)

    async def list_for_stream(self, stream_id: str) -> list[CandidateRow]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM candidates WHERE tenant_id = ? AND stream_id = ? ORDER BY ts",
            (tenant_id, stream_id),
        )
        return [_candidate_from_row(r) for r in await cur.fetchall()]


def _candidate_from_row(row: aiosqlite.Row) -> CandidateRow:
    d = dict(row)
    d["evidence"] = json.loads(d.pop("evidence_json") or "{}")
    return CandidateRow.model_validate(d)


class ClipsRepo:
    """Clips. Inserts are INSERT OR IGNORE so the same (id) on resume is a no-op."""

    def __init__(self, db: Database):
        self._db = db

    async def upsert_many(self, rows: list[ClipRow]) -> int:
        if not rows:
            return 0
        bound = current_tenant_id()
        for row in rows:
            if row.tenant_id != bound:
                raise TenancyError(f"clip tenant {row.tenant_id!r} != bound {bound!r}")
        conn = await self._db.connect()
        await conn.executemany(
            "INSERT OR IGNORE INTO clips "
            "(id, stream_id, tenant_id, candidate_id, start_s, end_s, duration_s, "
            "width, height, path, smart_crop_box_json, thumbnail_frame_path, status, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r.id,
                    r.stream_id,
                    r.tenant_id,
                    r.candidate_id,
                    r.start_s,
                    r.end_s,
                    r.duration_s,
                    r.width,
                    r.height,
                    r.path,
                    json.dumps(r.smart_crop_box) if r.smart_crop_box else None,
                    r.thumbnail_frame_path,
                    r.status,
                    r.created_at,
                )
                for r in rows
            ],
        )
        await conn.commit()
        return len(rows)

    async def get(self, clip_id: str) -> ClipRow | None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM clips WHERE id = ? AND tenant_id = ?",
            (clip_id, tenant_id),
        )
        row = await cur.fetchone()
        return _clip_from_row(row) if row else None

    async def list_for_stream(self, stream_id: str) -> list[ClipRow]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM clips WHERE tenant_id = ? AND stream_id = ? ORDER BY created_at",
            (tenant_id, stream_id),
        )
        return [_clip_from_row(r) for r in await cur.fetchall()]


def _clip_from_row(row: aiosqlite.Row) -> ClipRow:
    d = dict(row)
    box = d.pop("smart_crop_box_json")
    d["smart_crop_box"] = json.loads(box) if box else None
    return ClipRow.model_validate(d)


class VariantsRepo:
    """Variants per (clip, persona). Re-running with a new persona adds rows."""

    def __init__(self, db: Database):
        self._db = db

    async def replace_for_clip_persona(
        self, clip_id: str, persona_id: str, rows: list[VariantRow]
    ) -> int:
        """Delete the existing batch for (clip, persona), then insert the new rows."""
        bound = current_tenant_id()
        for row in rows:
            if row.tenant_id != bound:
                raise TenancyError(f"variant tenant {row.tenant_id!r} != bound {bound!r}")
            if row.clip_id != clip_id or row.persona_id != persona_id:
                raise TenancyError("variant clip_id/persona_id must match the replace target")
        conn = await self._db.connect()
        await conn.execute(
            "DELETE FROM variants WHERE tenant_id = ? AND clip_id = ? AND persona_id = ?",
            (bound, clip_id, persona_id),
        )
        if rows:
            await conn.executemany(
                "INSERT INTO variants "
                "(id, clip_id, tenant_id, persona_id, language, caption, "
                "title_card_text, hashtags_json, model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        r.id,
                        r.clip_id,
                        r.tenant_id,
                        r.persona_id,
                        r.language,
                        r.caption,
                        r.title_card_text,
                        json.dumps(r.hashtags),
                        r.model,
                        r.created_at,
                    )
                    for r in rows
                ],
            )
        await conn.commit()
        return len(rows)

    async def list_for_clip(
        self, clip_id: str, *, persona_id: str | None = None
    ) -> list[VariantRow]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        if persona_id is None:
            cur = await conn.execute(
                "SELECT * FROM variants WHERE tenant_id = ? AND clip_id = ? ORDER BY created_at",
                (tenant_id, clip_id),
            )
        else:
            cur = await conn.execute(
                "SELECT * FROM variants WHERE tenant_id = ? AND clip_id = ? "
                "AND persona_id = ? ORDER BY created_at",
                (tenant_id, clip_id, persona_id),
            )
        return [_variant_from_row(r) for r in await cur.fetchall()]


def _variant_from_row(row: aiosqlite.Row) -> VariantRow:
    d = dict(row)
    d["hashtags"] = json.loads(d.pop("hashtags_json") or "[]")
    return VariantRow.model_validate(d)


class ConnectedAccountsRepo:
    """Per-tenant social-account connections (Buffer, etc.).

    Phase 1 only inserts + lists; Phase 3's OAuth flow will refresh tokens
    via a separate path. The `oauth_blob` column carries provider-specific
    credentials as opaque JSON - the publisher (Task 11) reads it.
    """

    def __init__(self, db: Database):
        self._db = db

    async def create(
        self,
        *,
        platform: str,
        external_id: str,
        display_name: str | None = None,
        oauth_blob: dict[str, object] | None = None,
    ) -> ConnectedAccount:
        tenant_id = current_tenant_id()
        account_id = new_id("acc")
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO connected_accounts "
            "(id, tenant_id, platform, external_id, display_name, oauth_blob_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                account_id,
                tenant_id,
                platform,
                external_id,
                display_name,
                json.dumps(oauth_blob) if oauth_blob is not None else None,
                _now(),
            ),
        )
        await conn.commit()
        acct = await self.get(account_id)
        assert acct is not None
        return acct

    async def get(self, account_id: str) -> ConnectedAccount | None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT id, tenant_id, platform, external_id, display_name, "
            "oauth_blob_json, created_at FROM connected_accounts "
            "WHERE id = ? AND tenant_id = ?",
            (account_id, tenant_id),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return _connected_account_from_row(row)

    async def list_for_tenant(self) -> list[ConnectedAccount]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT id, tenant_id, platform, external_id, display_name, "
            "oauth_blob_json, created_at FROM connected_accounts "
            "WHERE tenant_id = ? ORDER BY created_at",
            (tenant_id,),
        )
        return [_connected_account_from_row(r) for r in await cur.fetchall()]


def _connected_account_from_row(row: aiosqlite.Row) -> ConnectedAccount:
    d = dict(row)
    blob = d.pop("oauth_blob_json")
    d["oauth_blob"] = json.loads(blob) if blob else None
    return ConnectedAccount.model_validate(d)


class PublishJobsRepo:
    """Pending / running / sent publish jobs.

    Task 9 uses `enqueue` to write rows; Task 11's worker reads pending
    jobs via `list_pending`, transitions them through the status column,
    and records final state.
    """

    def __init__(self, db: Database):
        self._db = db

    async def enqueue(
        self,
        *,
        clip_id: str,
        variant_id: str,
        account_id: str,
        platform: str,
        scheduled_for: str | None = None,
    ) -> PublishJob:
        tenant_id = current_tenant_id()
        job_id = new_id("pjb")
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO publish_jobs "
            "(id, tenant_id, clip_id, variant_id, account_id, platform, "
            "status, attempts, last_error, scheduled_for, external_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, NULL, ?, NULL, ?)",
            (
                job_id,
                tenant_id,
                clip_id,
                variant_id,
                account_id,
                platform,
                scheduled_for,
                _now(),
            ),
        )
        await conn.commit()
        cur = await conn.execute(
            "SELECT * FROM publish_jobs WHERE id = ? AND tenant_id = ?",
            (job_id, tenant_id),
        )
        row = await cur.fetchone()
        assert row is not None
        return PublishJob.model_validate(dict(row))

    async def list_for_clip(self, clip_id: str) -> list[PublishJob]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM publish_jobs WHERE tenant_id = ? AND clip_id = ? "
            "ORDER BY created_at",
            (tenant_id, clip_id),
        )
        return [PublishJob.model_validate(dict(r)) for r in await cur.fetchall()]

    async def list_pending(self, *, limit: int = 50) -> list[PublishJob]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM publish_jobs WHERE tenant_id = ? AND status = 'pending' "
            "ORDER BY created_at LIMIT ?",
            (tenant_id, limit),
        )
        return [PublishJob.model_validate(dict(r)) for r in await cur.fetchall()]


class VisualSignalsRepo:
    """One row per second of visual signals per stream.

    `replace_for_stream` is the canonical write — re-running analyze_video
    wipes the prior batch and inserts the new one. Composite PK is
    `(stream_id, ts_offset_s)`.
    """

    def __init__(self, db: Database):
        self._db = db

    async def replace_for_stream(
        self,
        stream_id: str,
        track: object,  # nexoclip.vision.VisualSignalTrack — avoid import cycle
    ) -> int:
        bound = current_tenant_id()
        # Late-bound to avoid a hard import cycle with nexoclip.vision.
        from nexoclip.vision import VisualSignalTrack

        if not isinstance(track, VisualSignalTrack):
            raise TenancyError("replace_for_stream requires a VisualSignalTrack")
        if track.tenant_id != bound:
            raise TenancyError(f"visual_signals tenant {track.tenant_id!r} != bound {bound!r}")
        if track.stream_id != stream_id:
            raise TenancyError(f"track.stream_id {track.stream_id!r} != target {stream_id!r}")
        conn = await self._db.connect()
        await conn.execute(
            "DELETE FROM visual_signals WHERE tenant_id = ? AND stream_id = ?",
            (bound, stream_id),
        )
        if track.signals:
            await conn.executemany(
                "INSERT INTO visual_signals "
                "(stream_id, tenant_id, ts_offset_s, scene_cut, face_emotion, "
                "motion_energy, text_changed) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        stream_id,
                        bound,
                        s.ts_offset_s,
                        1 if s.scene_cut else 0,
                        s.face_emotion,
                        s.motion_energy,
                        1 if s.text_changed else 0,
                    )
                    for s in track.signals
                ],
            )
        await conn.commit()
        return len(track.signals)

    async def list_for_stream(self, stream_id: str) -> list[dict[str, Any]]:
        """Return raw rows so callers (or tests) can compare directly."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT stream_id, tenant_id, ts_offset_s, scene_cut, face_emotion, "
            "motion_energy, text_changed FROM visual_signals "
            "WHERE tenant_id = ? AND stream_id = ? ORDER BY ts_offset_s",
            (tenant_id, stream_id),
        )
        rows = await cur.fetchall()
        return [
            {
                "stream_id": r["stream_id"],
                "tenant_id": r["tenant_id"],
                "ts_offset_s": float(r["ts_offset_s"]),
                "scene_cut": bool(r["scene_cut"]),
                "face_emotion": r["face_emotion"],
                "motion_energy": r["motion_energy"],
                "text_changed": bool(r["text_changed"]),
            }
            for r in rows
        ]
