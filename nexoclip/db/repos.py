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
    Event,
    LLMCallRow,
    PersonaRow,
    StreamRow,
    Tenant,
    User,
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
