"""Publish-job worker orchestration.

`run_publish_jobs(tenant_id, ...)` pulls up to `max_jobs` pending rows for
the given tenant, fetches the matching variant + connected account, posts
the update to the platform's API (Buffer in Phase 1), and writes the
final row state. Transient failures bump `attempts` and stay `pending`
until they exceed `max_attempts`, at which point they flip to `failed`.

Two callers:
    * `nexoclip publish --tenant <id>` runs one drain pass.
    * The FastAPI lifespan task wakes every 60s and runs the same drain.

Both share the same code path so behaviour is identical between manual
and automatic invocations.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from nexoclip.db import (
    ConnectedAccountsRepo,
    Database,
    EventsRepo,
    PublishJobsRepo,
    VariantsRepo,
)
from nexoclip.db.models import ConnectedAccount, PublishJob, VariantRow
from nexoclip.errors import QuotaExceeded
from nexoclip.tenancy import bound_tenant

from .buffer import BufferClient, BufferError

if TYPE_CHECKING:
    from nexoclip.governance import BudgetGovernor

_log = structlog.get_logger(__name__)


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


# Default backoff schedule for transient retries. The orchestrator stops
# retrying inside one drain pass after a single failure - the next pass
# picks the row up because it's still `pending`. The schedule below is
# only applied within one Buffer call's own retry loop.
_DEFAULT_MAX_ATTEMPTS = 4
_DEFAULT_INITIAL_BACKOFF_S = 1.0
_DEFAULT_BACKOFF_MULTIPLIER = 2.0


@dataclass(frozen=True)
class PublishOutcome:
    """Roll-up of one drain pass."""

    sent: int = 0
    transient_failures: int = 0
    permanent_failures: int = 0
    skipped: int = 0

    @property
    def total(self) -> int:
        return self.sent + self.transient_failures + self.permanent_failures + self.skipped


# Type for the optional buffer-client factory (test seam).
BufferClientFactory = Callable[[str], BufferClient]


def _default_buffer_factory(access_token: str) -> BufferClient:
    return BufferClient(access_token=access_token)


async def run_publish_jobs(
    tenant_id: str,
    db: Database,
    *,
    max_jobs: int = 50,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    initial_backoff_s: float = _DEFAULT_INITIAL_BACKOFF_S,
    backoff_multiplier: float = _DEFAULT_BACKOFF_MULTIPLIER,
    buffer_factory: BufferClientFactory | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    governor: BudgetGovernor | None = None,
) -> PublishOutcome:
    """Drain pending publish_jobs for `tenant_id`. Returns a roll-up.

    `buffer_factory` and `sleep` are test seams - tests pass a fake client
    factory and an async no-op sleep so the retry-with-backoff path runs
    deterministically.

    `governor` is the Phase 2 budget governor. When supplied,
    `check_publish_quota` is consulted before each dispatch; once the daily
    quota is exhausted we stop dispatching (remaining `pending` jobs roll
    over to tomorrow's drain pass).
    """
    factory = buffer_factory or _default_buffer_factory
    sleeper = sleep or asyncio.sleep

    sent = 0
    transient = 0
    permanent = 0
    skipped = 0
    quota_exhausted = False

    with bound_tenant(tenant_id):
        jobs_repo = PublishJobsRepo(db)
        accounts_repo = ConnectedAccountsRepo(db)
        variants_repo = VariantsRepo(db)
        events_repo = EventsRepo(db)

        pending = await jobs_repo.list_pending(limit=max_jobs)
        if not pending:
            return PublishOutcome()

        # Index account + variant lookups by id so we don't hit the DB once
        # per job for the same account.
        accounts_index: dict[str, ConnectedAccount] = {
            a.id: a for a in await accounts_repo.list_for_tenant()
        }
        variant_cache: dict[tuple[str, str], VariantRow] = {}

        for job in pending:
            # Phase 2: budget governor gate. Once today's publish quota is
            # exhausted, stop the drain - leave remaining `pending` rows for
            # tomorrow rather than burning retries against a hard limit.
            if governor is not None and not quota_exhausted:
                try:
                    await governor.check_publish_quota(
                        tenant_id, platform=job.platform
                    )
                except QuotaExceeded as e:
                    quota_exhausted = True
                    await events_repo.emit(
                        type="publish.quota_exhausted",
                        payload={"platform": job.platform, "error": str(e)},
                    )
            if quota_exhausted:
                skipped += 1
                continue

            account = accounts_index.get(job.account_id)
            if account is None:
                await _mark_failed(
                    db, job, last_error=f"unknown account_id {job.account_id!r}"
                )
                permanent += 1
                continue

            cache_key = (job.clip_id, job.variant_id)
            variant = variant_cache.get(cache_key)
            if variant is None:
                clip_variants = await variants_repo.list_for_clip(job.clip_id)
                for v in clip_variants:
                    variant_cache[(job.clip_id, v.id)] = v
                variant = variant_cache.get(cache_key)
            if variant is None:
                await _mark_failed(
                    db, job, last_error=f"variant {job.variant_id!r} missing for clip"
                )
                permanent += 1
                continue

            access_token = _access_token_for(account)
            if access_token is None:
                await _mark_failed(
                    db, job, last_error="connected account has no access_token"
                )
                permanent += 1
                continue

            outcome = await _post_with_retries(
                client_factory=factory,
                access_token=access_token,
                profile_external_id=account.external_id,
                text=_compose_caption(variant),
                max_attempts=max_attempts,
                initial_backoff_s=initial_backoff_s,
                backoff_multiplier=backoff_multiplier,
                sleep=sleeper,
            )

            if outcome["status"] == "ok":
                await _mark_sent(
                    db,
                    job,
                    external_id=str(outcome.get("external_id") or ""),
                    attempts=outcome["attempts"],
                )
                sent += 1
                await events_repo.emit(
                    type="clip.published",
                    payload={"clip_id": job.clip_id, "variant_id": job.variant_id},
                )
            elif outcome["status"] == "transient":
                await _bump_attempts(
                    db,
                    job,
                    last_error=outcome["error"],
                    attempts=outcome["attempts"],
                    fail=outcome["attempts"] >= max_attempts,
                )
                if outcome["attempts"] >= max_attempts:
                    permanent += 1
                    await events_repo.emit(
                        type="publish_job.failed",
                        payload={"job_id": job.id, "reason": outcome["error"]},
                    )
                else:
                    transient += 1
            else:
                await _mark_failed(db, job, last_error=outcome["error"])
                permanent += 1
                await events_repo.emit(
                    type="publish_job.failed",
                    payload={"job_id": job.id, "reason": outcome["error"]},
                )

    if sent or permanent:
        _log.info(
            "publish_drain",
            tenant_id=tenant_id,
            sent=sent,
            transient_failures=transient,
            permanent_failures=permanent,
            skipped=skipped,
        )

    return PublishOutcome(
        sent=sent,
        transient_failures=transient,
        permanent_failures=permanent,
        skipped=skipped,
    )


def _access_token_for(account: ConnectedAccount) -> str | None:
    blob = account.oauth_blob or {}
    raw = blob.get("access_token")
    return raw if isinstance(raw, str) and raw else None


def _compose_caption(variant: VariantRow) -> str:
    """Buffer text body = caption + " " + space-joined hashtags."""
    parts = [variant.caption.strip()]
    if variant.hashtags:
        parts.append(" ".join(f"#{h.lstrip('#')}" for h in variant.hashtags))
    return " ".join(p for p in parts if p)


async def _post_with_retries(
    *,
    client_factory: BufferClientFactory,
    access_token: str,
    profile_external_id: str,
    text: str,
    max_attempts: int,
    initial_backoff_s: float,
    backoff_multiplier: float,
    sleep: Callable[[float], Awaitable[None]],
) -> dict[str, Any]:
    """Try posting `max_attempts` times. Returns one of:

        {"status": "ok",        "attempts": int, "external_id": str}
        {"status": "transient", "attempts": int, "error": str}
        {"status": "fatal",     "attempts": int, "error": str}
    """
    last_error = "no attempts made"
    attempt = 0
    async with client_factory(access_token) as client:
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await client.create_update(
                    profile_external_id=profile_external_id, text=text
                )
            except BufferError as e:
                last_error = f"{type(e).__name__}: {e}"
                if not e.transient:
                    return {"status": "fatal", "attempts": attempt, "error": last_error}
                if attempt < max_attempts:
                    await sleep(initial_backoff_s * (backoff_multiplier ** (attempt - 1)))
                    continue
                return {"status": "transient", "attempts": attempt, "error": last_error}
            return {
                "status": "ok",
                "attempts": attempt,
                "external_id": _extract_external_id(resp),
            }
    return {"status": "transient", "attempts": attempt, "error": last_error}


def _extract_external_id(payload: dict[str, Any]) -> str:
    """Buffer returns `{"updates": [{"id": "..."}], ...}`.

    Phase 1 records the first update id; if Buffer changes shape we'll
    just record an empty string and the row remains queryable.
    """
    updates = payload.get("updates")
    if isinstance(updates, list) and updates:
        first = updates[0]
        if isinstance(first, dict):
            first_id = first.get("id")
            if isinstance(first_id, str):
                return first_id
    top_id = payload.get("id")
    if isinstance(top_id, str):
        return top_id
    return ""


async def _mark_sent(
    db: Database, job: PublishJob, *, external_id: str, attempts: int
) -> None:
    conn = await db.connect()
    await conn.execute(
        "UPDATE publish_jobs SET status = 'sent', external_id = ?, attempts = ?, "
        "last_error = NULL WHERE id = ? AND tenant_id = ?",
        (external_id, attempts, job.id, job.tenant_id),
    )
    await conn.commit()


async def _mark_failed(db: Database, job: PublishJob, *, last_error: str) -> None:
    conn = await db.connect()
    await conn.execute(
        "UPDATE publish_jobs SET status = 'failed', last_error = ?, "
        "attempts = attempts + 1 WHERE id = ? AND tenant_id = ?",
        (last_error[:1000], job.id, job.tenant_id),
    )
    await conn.commit()


async def _bump_attempts(
    db: Database,
    job: PublishJob,
    *,
    last_error: str,
    attempts: int,
    fail: bool,
) -> None:
    """After a transient failure: bump attempts; flip to failed if cap hit."""
    new_status = "failed" if fail else "pending"
    conn = await db.connect()
    await conn.execute(
        "UPDATE publish_jobs SET status = ?, last_error = ?, attempts = ? "
        "WHERE id = ? AND tenant_id = ?",
        (new_status, last_error[:1000], attempts, job.id, job.tenant_id),
    )
    await conn.commit()
