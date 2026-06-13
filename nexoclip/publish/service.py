"""Publish-job worker orchestration.

`run_publish_jobs(tenant_id, ...)` pulls up to `max_jobs` pending rows for
the given tenant, dispatches each through a platform-specific Publisher,
records final state, and retries transient failures with exponential
backoff. Two callers:

  * `nexoclip publish --tenant <id>` runs one drain pass.
  * The FastAPI lifespan task wakes every 60s and runs the same drain.

Phase 2 changes:

  * Dispatcher routes by `account.platform` via a Publisher registry so
    Buffer / TikTok / YouTube / future platforms share one code path.
  * Pre-dispatch consults the BudgetGovernor's publish quota and halts
    cleanly when exhausted.
  * Per-account OAuth refresh runs once per drain pass (TikTok + YT);
    on refresh failure the account flips to `auth_failed` and is skipped.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import httpx
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
from nexoclip.safety import SafetyPolicy, evaluate_post_window, next_safe_slot
from nexoclip.tenancy import bound_tenant

from .buffer import BufferClient, BufferPublisher
from .protocol import PublisherError

if TYPE_CHECKING:
    from nexoclip.governance import BudgetGovernor

_log = structlog.get_logger(__name__)


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


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


# ---------------------------------------------------------------------------
# Publisher protocol surface used inside this module.
# ---------------------------------------------------------------------------


class _PublishResult(Protocol):
    external_id: str
    external_url: str | None
    metadata: dict[str, Any]


class _Publisher(Protocol):
    async def __aenter__(self) -> _Publisher: ...

    async def __aexit__(self, *exc: object) -> None: ...

    async def publish(
        self,
        *,
        clip_path: str,
        variant: VariantRow,
        account: ConnectedAccount,
    ) -> _PublishResult: ...


# Factories return `Any` rather than the structural _Publisher protocol so
# mypy doesn't trip on the per-platform concrete result types (each returns
# its own *PublishResult dataclass; the protocol's structural duck-typing
# isn't enforced through method return-type matching).
PublisherFactory = Callable[[ConnectedAccount, httpx.AsyncClient], Any]
BufferClientFactory = Callable[[str], BufferClient]


def _default_buffer_factory(access_token: str) -> BufferClient:
    return BufferClient(access_token=access_token)


def _default_buffer_publisher(
    account: ConnectedAccount, _http: httpx.AsyncClient
) -> Any:
    """Default Buffer publisher: pull token from oauth_blob.access_token."""
    token = _access_token_for(account)
    if not token:
        raise PublisherError(
            f"buffer account {account.id} has no access_token", transient=False
        )
    return BufferPublisher(BufferClient(access_token=token))


def _default_tiktok_publisher(
    account: ConnectedAccount, http: httpx.AsyncClient
) -> Any:
    """Default TikTok publisher: shared httpx client, token from oauth_blob."""
    from .tiktok import TikTokClient

    token = _access_token_for(account)
    if not token:
        raise PublisherError(
            f"tiktok account {account.id} has no access_token", transient=False
        )
    return TikTokClient(access_token=token, client=http)


def _default_youtube_publisher(
    account: ConnectedAccount, http: httpx.AsyncClient
) -> Any:
    from .youtube import YouTubeClient

    token = _access_token_for(account)
    if not token:
        raise PublisherError(
            f"youtube account {account.id} has no access_token", transient=False
        )
    return YouTubeClient(access_token=token, client=http)


def _default_instagram_publisher(
    account: ConnectedAccount, http: httpx.AsyncClient
) -> Any:
    from .instagram import InstagramClient

    token = _access_token_for(account)
    if not token:
        raise PublisherError(
            f"instagram account {account.id} has no access_token", transient=False
        )
    return InstagramClient(access_token=token, client=http)


_DEFAULT_PUBLISHERS: dict[str, PublisherFactory] = {
    "buffer": _default_buffer_publisher,
    "tiktok": _default_tiktok_publisher,
    "youtube": _default_youtube_publisher,
    "instagram": _default_instagram_publisher,
}


# ---------------------------------------------------------------------------
# Drain entry point
# ---------------------------------------------------------------------------


async def run_publish_jobs(
    tenant_id: str,
    db: Database,
    *,
    max_jobs: int = 50,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    initial_backoff_s: float = _DEFAULT_INITIAL_BACKOFF_S,
    backoff_multiplier: float = _DEFAULT_BACKOFF_MULTIPLIER,
    buffer_factory: BufferClientFactory | None = None,
    publishers: dict[str, PublisherFactory] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    governor: BudgetGovernor | None = None,
    http: httpx.AsyncClient | None = None,
) -> PublishOutcome:
    """Drain pending publish_jobs for `tenant_id`.

    `buffer_factory` is the Phase 1 test seam (still honored - tests pass a
    recording fake to short-circuit the network). `publishers` is the
    Phase 2 generalization: a `{platform: factory}` map that overrides the
    default registry.

    `governor` enforces the daily publish quota before each dispatch.
    On exhaustion we halt the drain - remaining `pending` rows roll over.

    Per-account OAuth refresh runs once at the top of the drain. Refresh
    failure flips the account to `auth_failed` and skips its jobs.
    """
    sleeper = sleep or asyncio.sleep
    factories = {**_DEFAULT_PUBLISHERS, **(publishers or {})}
    if buffer_factory is not None:
        # Back-compat: if a Buffer-specific factory is passed, wrap whatever
        # it returns through the BufferPublisher adapter so the dispatcher
        # talks to it like any other Publisher.
        original_buffer = buffer_factory

        def _wrapped_buffer(
            account: ConnectedAccount, _http: httpx.AsyncClient
        ) -> Any:
            token = _access_token_for(account) or ""
            return BufferPublisher(original_buffer(token))

        factories["buffer"] = _wrapped_buffer

    sent = 0
    transient = 0
    permanent = 0
    skipped = 0
    quota_exhausted = False
    own_http = http is None
    http_client = http or httpx.AsyncClient(timeout=30.0)

    try:
        with bound_tenant(tenant_id):
            jobs_repo = PublishJobsRepo(db)
            accounts_repo = ConnectedAccountsRepo(db)
            variants_repo = VariantsRepo(db)
            events_repo = EventsRepo(db)

            pending = await jobs_repo.list_pending(limit=max_jobs)
            if not pending:
                return PublishOutcome()

            accounts_index = await _refresh_and_index_accounts(
                accounts_repo,
                http=http_client,
                accounts=await accounts_repo.list_for_tenant(),
                db=db,
            )
            variant_cache: dict[tuple[str, str], VariantRow] = {}

            for job in pending:
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

                # Safe trap — hard gate. Jobs enqueued in auto-schedule mode
                # carry their resolved policy under `safe_gate`. Re-check the
                # window at post time: if it's now blocked (cadence shifted,
                # quiet hours, daily cap hit), defer to the next safe slot
                # instead of risking a shadowban. Advisory-only jobs have no
                # `safe_gate` and fall straight through.
                if await _defer_if_unsafe(db, jobs_repo, events_repo, job):
                    skipped += 1
                    continue

                account = accounts_index.get(job.account_id)
                if account is None:
                    await _mark_failed(
                        db, job, last_error=f"unknown account_id {job.account_id!r}"
                    )
                    permanent += 1
                    continue
                if account.status != "active":
                    skipped += 1
                    continue
                if _access_token_for(account) is None:
                    await _mark_failed(
                        db, job,
                        last_error="connected account has no access_token",
                    )
                    permanent += 1
                    continue

                variant = await _fetch_variant(
                    variants_repo, variant_cache, job.clip_id, job.variant_id
                )
                if variant is None:
                    await _mark_failed(
                        db, job, last_error=f"variant {job.variant_id!r} missing for clip"
                    )
                    permanent += 1
                    continue

                factory = factories.get(account.platform)
                if factory is None:
                    await _mark_failed(
                        db, job,
                        last_error=f"no publisher registered for platform {account.platform!r}",
                    )
                    permanent += 1
                    continue

                outcome = await _post_with_retries(
                    factory=factory,
                    account=account,
                    http=http_client,
                    clip_path=variant.clip_id,  # placeholder; clip_path resolved below
                    variant=variant,
                    max_attempts=max_attempts,
                    initial_backoff_s=initial_backoff_s,
                    backoff_multiplier=backoff_multiplier,
                    sleep=sleeper,
                    db=db,
                    job=job,
                )

                if outcome["status"] == "ok":
                    await _mark_sent(
                        db,
                        job,
                        external_id=str(outcome.get("external_id") or ""),
                        external_url=outcome.get("external_url"),
                        metadata=outcome.get("metadata"),
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
    finally:
        if own_http:
            await http_client.aclose()

    if sent or permanent or skipped:
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _defer_if_unsafe(
    db: Database,
    jobs_repo: PublishJobsRepo,
    events_repo: EventsRepo,
    job: PublishJob,
) -> bool:
    """Re-check a safe-gated job's posting window; defer it if blocked.

    Returns True iff the job was deferred (rescheduled to the next safe
    slot) and the drain should skip it this pass. Jobs without a
    `safe_gate` stamp — i.e. advisory-only or manual — always return False.
    """
    meta = job.platform_metadata or {}
    if not meta.get("safe_gate"):
        return False
    raw_policy = meta.get("safety_policy")
    if not isinstance(raw_policy, dict):
        return False

    policy = SafetyPolicy.model_validate(raw_policy)
    verdict = await evaluate_post_window(
        jobs_repo, platform=job.platform, policy=policy, exclude_job_id=job.id
    )
    if verdict.risk != "blocked":
        return False

    next_slot = await next_safe_slot(
        jobs_repo,
        platform=job.platform,
        policy=policy,
        earliest=_now(),
        exclude_job_id=job.id,
    )
    moved = await jobs_repo.reschedule(job.id, scheduled_for=next_slot)
    await events_repo.emit(
        type="publish.safe_window_deferred",
        payload={
            "job_id": job.id,
            "platform": job.platform,
            "reason": verdict.reason,
            "rescheduled_for": next_slot,
        },
    )
    _log.info(
        "publish.safe_window_deferred",
        job_id=job.id,
        platform=job.platform,
        reason=verdict.reason,
        rescheduled_for=next_slot,
        moved=moved,
    )
    return True


async def _refresh_and_index_accounts(
    accounts_repo: ConnectedAccountsRepo,
    *,
    http: httpx.AsyncClient,
    accounts: list[ConnectedAccount],
    db: Database,
) -> dict[str, ConnectedAccount]:
    """Refresh each TikTok / YT / IG account whose token is near expiry; index by id."""
    from .instagram import refresh_instagram_token
    from .oauth import refresh_if_expiring
    from .tiktok import refresh_tiktok_token
    from .youtube import refresh_youtube_token

    out: dict[str, ConnectedAccount] = {}
    for acct in accounts:
        if acct.platform == "tiktok":
            try:
                acct = await refresh_if_expiring(
                    acct,
                    db=db,
                    http=http,
                    refresh_impl=lambda rt, http_, acct=acct: refresh_tiktok_token(  # type: ignore[misc]
                        rt, http_,
                        client_key=str((acct.oauth_blob or {}).get("client_key", "")),
                        client_secret=str(
                            (acct.oauth_blob or {}).get("client_secret", "")
                        ),
                    ),
                )
            except Exception as e:
                _log.warning("tiktok_refresh_failed", account_id=acct.id, error=str(e))
        elif acct.platform == "youtube":
            try:
                acct = await refresh_if_expiring(
                    acct,
                    db=db,
                    http=http,
                    refresh_impl=lambda rt, http_, acct=acct: refresh_youtube_token(  # type: ignore[misc]
                        rt, http_,
                        client_id=str((acct.oauth_blob or {}).get("client_id", "")),
                        client_secret=str(
                            (acct.oauth_blob or {}).get("client_secret", "")
                        ),
                    ),
                )
            except Exception as e:
                _log.warning("youtube_refresh_failed", account_id=acct.id, error=str(e))
        elif acct.platform == "instagram":
            try:
                acct = await refresh_if_expiring(
                    acct,
                    db=db,
                    http=http,
                    refresh_impl=lambda rt, http_, acct=acct: refresh_instagram_token(  # type: ignore[misc]
                        rt, http_,
                        app_id=str((acct.oauth_blob or {}).get("app_id", "")),
                        app_secret=str((acct.oauth_blob or {}).get("app_secret", "")),
                    ),
                )
            except Exception as e:
                _log.warning(
                    "instagram_refresh_failed", account_id=acct.id, error=str(e)
                )
        out[acct.id] = acct
    return out


async def _fetch_variant(
    variants_repo: VariantsRepo,
    cache: dict[tuple[str, str], VariantRow],
    clip_id: str,
    variant_id: str,
) -> VariantRow | None:
    cache_key = (clip_id, variant_id)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    rows = await variants_repo.list_for_clip(clip_id)
    for v in rows:
        cache[(clip_id, v.id)] = v
    return cache.get(cache_key)


def _access_token_for(account: ConnectedAccount) -> str | None:
    blob = account.oauth_blob or {}
    raw = blob.get("access_token")
    return raw if isinstance(raw, str) and raw else None


async def _post_with_retries(
    *,
    factory: PublisherFactory,
    account: ConnectedAccount,
    http: httpx.AsyncClient,
    clip_path: str,
    variant: VariantRow,
    max_attempts: int,
    initial_backoff_s: float,
    backoff_multiplier: float,
    sleep: Callable[[float], Awaitable[None]],
    db: Database,
    job: PublishJob,
) -> dict[str, Any]:
    """Try `max_attempts` times. Returns `{"status": ok|transient|fatal, ...}`."""
    last_error = "no attempts made"
    attempt = 0
    # Resolve the actual clip path - we need to fetch the ClipRow because
    # only it knows where the .mp4 lives. (Buffer ignores this, but TikTok+YT
    # need it.)
    clip_path_resolved = await _resolve_clip_path(db, job.clip_id)

    try:
        publisher = factory(account, http)
    except PublisherError as e:
        return {"status": "fatal" if not e.transient else "transient",
                "attempts": 0, "error": f"factory: {e}"}

    async with publisher:
        for attempt in range(1, max_attempts + 1):
            try:
                result = await publisher.publish(
                    clip_path=clip_path_resolved,
                    variant=variant,
                    account=account,
                )
            except PublisherError as e:
                last_error = f"{type(e).__name__}: {e}"
                if not e.transient:
                    return {
                        "status": "fatal",
                        "attempts": attempt,
                        "error": last_error,
                    }
                if attempt < max_attempts:
                    await sleep(initial_backoff_s * (backoff_multiplier ** (attempt - 1)))
                    continue
                return {
                    "status": "transient",
                    "attempts": attempt,
                    "error": last_error,
                }
            return {
                "status": "ok",
                "attempts": attempt,
                "external_id": result.external_id,
                "external_url": result.external_url,
                "metadata": result.metadata,
            }
    return {"status": "transient", "attempts": attempt, "error": last_error}


async def _resolve_clip_path(db: Database, clip_id: str) -> str:
    """Fetch the on-disk clip path. Buffer ignores it; TikTok/YT need it.

    Prefers `clip_final.mp4` when present — that's the burned-in
    version produced by the overlay finalize endpoint (slice F.7-E,
    with title / KICK banner / captions composited into the pixels).
    Falls back to the original `clip.mp4` when the operator never
    finalized OR the burn failed (in which case we still want
    publishing to work, just without the overlay layer).
    """
    from pathlib import Path

    from nexoclip.db import ClipsRepo

    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        return ""
    original = Path(clip.path)
    final = original.parent / "clip_final.mp4"
    return str(final) if final.exists() else clip.path


async def _mark_sent(
    db: Database,
    job: PublishJob,
    *,
    external_id: str,
    external_url: str | None,
    metadata: dict[str, Any] | None,
    attempts: int,
) -> None:
    conn = await db.connect()
    await conn.execute(
        "UPDATE publish_jobs SET status = 'sent', external_id = ?, "
        "external_url = ?, platform_metadata_json = ?, attempts = ?, "
        "last_error = NULL WHERE id = ? AND tenant_id = ?",
        (
            external_id,
            external_url,
            json.dumps(metadata) if metadata is not None else None,
            attempts,
            job.id,
            job.tenant_id,
        ),
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
    new_status = "failed" if fail else "pending"
    conn = await db.connect()
    await conn.execute(
        "UPDATE publish_jobs SET status = ?, last_error = ?, attempts = ? "
        "WHERE id = ? AND tenant_id = ?",
        (new_status, last_error[:1000], attempts, job.id, job.tenant_id),
    )
    await conn.commit()
