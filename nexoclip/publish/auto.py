"""Auto-publish dispatcher — voice-markers spec slice E.2.

For brand kits with `auto_publish_enabled = True`, the dispatcher walks
each kit's clips and enqueues publish_jobs with `scheduled_for =
clip.created_at + delay_min`. The existing publish worker already
respects scheduled_for (see PublishJobsRepo.list_pending), so jobs sit
in the undo window invisibly until the timer fires.

Cancellation: `PublishJobsRepo.cancel(job_id)` flips a pending job to
'canceled'. The dashboard exposes that as the "Undo" button on the
inbox (slice E.3).

Idempotence — we skip a (clip, platform) pair when there's ANY existing
job (pending / sent / failed / canceled). Re-running the dispatcher
doesn't double-enqueue.

What we DON'T do here:
  - Talk to TikTok / YouTube directly — that's the worker's job.
  - Pick the variant intelligently — we use the first variant the
    VariantsRepo returns for the clip (Phase 0's persona-default).
  - Check the clip's `status` — voice-markers spec explicitly bypasses
    manual review when auto-publish is on for the resolved kit.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping
from typing import NamedTuple

import structlog

from nexoclip.branding.service import resolve_brand_kit_for_candidate
from nexoclip.db import (
    BrandKitsRepo,
    CandidatesRepo,
    ClipsRepo,
    ConnectedAccountsRepo,
    Database,
    PublishJobsRepo,
    StreamsRepo,
    TenantsRepo,
    VariantsRepo,
)
from nexoclip.db.models import BrandKitRow, ConnectedAccount
from nexoclip.tenancy import bound_tenant

_log = structlog.get_logger(__name__)


class AutoPublishReport(NamedTuple):
    """Per-tenant outcome from one dispatch pass."""

    tenant_id: str
    jobs_enqueued: int
    clips_considered: int
    clips_skipped_no_kit: int
    clips_skipped_no_variant: int
    clips_skipped_already_queued: int
    clips_skipped_no_account: int
    dry_run: bool


# ---- public entry ----


async def dispatch_auto_publish(
    db: Database,
    *,
    tenant_id: str | None = None,
    dry_run: bool = False,
) -> list[AutoPublishReport]:
    """Walk every brand kit with `auto_publish_enabled=True` and enqueue
    publish_jobs for fresh clips routed to that kit.

    Args:
        db: Database handle. Migrations must already be applied.
        tenant_id: When set, restricts to that tenant. None sweeps every
            tenant.
        dry_run: When True, the report records the would-be enqueues
            without writing any DB rows.

    Returns:
        One report per tenant scanned. Empty list iff no tenants matched.
    """
    tenants_repo = TenantsRepo(db)
    if tenant_id is not None:
        t = await tenants_repo.get(tenant_id)
        tenants = [t] if t is not None else []
    else:
        tenants = await tenants_repo.list_all()

    reports: list[AutoPublishReport] = []
    for t in tenants:
        with bound_tenant(t.id):
            report = await _dispatch_one_tenant(
                db=db, tenant_id=t.id, dry_run=dry_run
            )
        reports.append(report)
        _log.info(
            "auto_publish.dispatch_done",
            tenant_id=t.id,
            enqueued=report.jobs_enqueued,
            skipped_no_kit=report.clips_skipped_no_kit,
            skipped_no_variant=report.clips_skipped_no_variant,
            skipped_already_queued=report.clips_skipped_already_queued,
            skipped_no_account=report.clips_skipped_no_account,
            dry_run=dry_run,
        )
    return reports


async def _dispatch_one_tenant(
    *,
    db: Database,
    tenant_id: str,
    dry_run: bool,
) -> AutoPublishReport:
    """Per-tenant body of `dispatch_auto_publish`. Tenant is already bound."""
    kits_repo = BrandKitsRepo(db)
    clips_repo = ClipsRepo(db)
    candidates_repo = CandidatesRepo(db)
    variants_repo = VariantsRepo(db)
    accounts_repo = ConnectedAccountsRepo(db)
    jobs_repo = PublishJobsRepo(db)
    streams_repo = StreamsRepo(db)

    auto_kits_exist = any(
        k.auto_publish_enabled for k in await kits_repo.list_for_tenant()
    )
    if not auto_kits_exist:
        return _empty_report(tenant_id, dry_run)

    # Look up connected accounts once per tenant — keyed by platform.
    accounts = await accounts_repo.list_for_tenant()
    accounts_by_platform: dict[str, ConnectedAccount] = {
        acc.platform: acc for acc in accounts
    }

    jobs_enqueued = 0
    clips_considered = 0
    clips_skipped_no_kit = 0
    clips_skipped_no_variant = 0
    clips_skipped_already_queued = 0
    clips_skipped_no_account = 0

    streams = await streams_repo.list_for_tenant()
    for stream in streams:
        # Cache candidates for this stream so we can do speaker-label
        # lookup once per clip without a per-clip query.
        candidates_by_id = {
            c.id: c for c in await candidates_repo.list_for_stream(stream.id)
        }
        clips = await clips_repo.list_for_stream(stream.id)
        for clip in clips:
            clips_considered += 1
            speaker_label = _speaker_label_for_clip(clip, candidates_by_id)
            kit = await resolve_brand_kit_for_candidate(
                db,
                stream_id=stream.id,
                speaker_label=speaker_label,
            )
            if kit is None or not kit.auto_publish_enabled:
                clips_skipped_no_kit += 1
                continue

            clip_variants = await variants_repo.list_for_clip(clip.id)
            if not clip_variants:
                clips_skipped_no_variant += 1
                continue
            variant = clip_variants[0]

            existing_jobs = await jobs_repo.list_for_clip(clip.id)
            existing_platforms = {j.platform for j in existing_jobs}

            for platform in kit.auto_publish_platforms or []:
                account = accounts_by_platform.get(platform)
                if account is None:
                    clips_skipped_no_account += 1
                    continue
                if platform in existing_platforms:
                    clips_skipped_already_queued += 1
                    continue
                scheduled_for = _undo_window_iso(
                    clip_created_at=clip.created_at,
                    delay_min=kit.auto_publish_delay_min,
                )
                if not dry_run:
                    await jobs_repo.enqueue(
                        clip_id=clip.id,
                        variant_id=variant.id,
                        account_id=account.id,
                        platform=platform,
                        scheduled_for=scheduled_for,
                    )
                jobs_enqueued += 1

    return AutoPublishReport(
        tenant_id=tenant_id,
        jobs_enqueued=jobs_enqueued,
        clips_considered=clips_considered,
        clips_skipped_no_kit=clips_skipped_no_kit,
        clips_skipped_no_variant=clips_skipped_no_variant,
        clips_skipped_already_queued=clips_skipped_already_queued,
        clips_skipped_no_account=clips_skipped_no_account,
        dry_run=dry_run,
    )


# ---- helpers ----


def _empty_report(tenant_id: str, dry_run: bool) -> AutoPublishReport:
    return AutoPublishReport(
        tenant_id=tenant_id,
        jobs_enqueued=0,
        clips_considered=0,
        clips_skipped_no_kit=0,
        clips_skipped_no_variant=0,
        clips_skipped_already_queued=0,
        clips_skipped_no_account=0,
        dry_run=dry_run,
    )


def _speaker_label_for_clip(clip: object, candidates_by_id: Mapping[str, object]) -> str | None:
    """Peel the speaker_label off the clip's source candidate.

    Returns None when:
      - the clip has no candidate_id (came from an audio/visual trigger)
      - the candidate doesn't carry a speaker_label (diarization skipped)
      - the lookup fails for any reason

    All of which are fine — the resolver falls through to the tenant
    default kit in those cases.
    """
    candidate_id = getattr(clip, "candidate_id", None)
    if not candidate_id:
        return None
    cand = candidates_by_id.get(candidate_id)
    if cand is None:
        return None
    evidence = getattr(cand, "evidence", None) or {}
    label = evidence.get("speaker_label") if isinstance(evidence, dict) else None
    return label if isinstance(label, str) else None


def _undo_window_iso(*, clip_created_at: str, delay_min: int) -> str:
    """ISO timestamp = clip.created_at + delay_min, in UTC.

    Offsetting from `clip.created_at` (not `now()`) keeps the dispatch
    cron robust to lag: if the cron fires 5 minutes after the clip is
    cut and the kit's delay is 60min, the worker picks up the job 55
    minutes from now, not 60. That matches operator intuition — the
    undo window starts when the clip exists, not when the dispatcher
    notices it."""
    try:
        base = _dt.datetime.fromisoformat(clip_created_at)
    except ValueError:
        # Defensive: fall back to now() rather than skipping the clip.
        base = _dt.datetime.now(_dt.UTC)
    if base.tzinfo is None:
        base = base.replace(tzinfo=_dt.UTC)
    return (base + _dt.timedelta(minutes=delay_min)).isoformat()


def auto_publish_enabled_kits(kits: list[BrandKitRow]) -> list[BrandKitRow]:
    """Filter helper for the dashboard's settings page."""
    return [k for k in kits if k.auto_publish_enabled]


# ---- per-clip trigger (slice G.5) ----
#
# `dispatch_auto_publish` is the periodic sweep. G.5 is the
# operator-driven counterpart: the moment the operator clicks
# "Complete / Approve" we want the publish_job in the queue so the
# undo timer starts visibly, not silently on the next cron tick.
#
# Same idempotence + skip rules as `_dispatch_one_tenant` — calling
# this from the approve handler AND letting the sweep also run is
# safe; the sweep just skips a (clip, platform) pair when a job
# already exists.


async def dispatch_for_clip(
    db: Database,
    *,
    clip_id: str,
    dry_run: bool = False,
) -> AutoPublishReport:
    """Enqueue auto-publish jobs for a single approved clip.

    Returns a single-clip report. Caller is responsible for binding the
    tenant via `bound_tenant` before invoking — same contract as the
    periodic dispatcher's per-tenant body.
    """
    clips_repo = ClipsRepo(db)
    candidates_repo = CandidatesRepo(db)
    variants_repo = VariantsRepo(db)
    accounts_repo = ConnectedAccountsRepo(db)
    jobs_repo = PublishJobsRepo(db)

    clip = await clips_repo.get(clip_id)
    if clip is None:
        return _empty_report(tenant_id="", dry_run=dry_run)

    # Resolve the kit via the same path as the sweep so per-speaker
    # routing is honored even on the per-clip trigger. CandidatesRepo
    # only exposes list_for_stream — cheap enough for the per-clip
    # case (one stream's worth of candidates) so we just filter.
    candidates_by_id: dict[str, object] = {}
    if clip.candidate_id:
        for cand in await candidates_repo.list_for_stream(clip.stream_id):
            if cand.id == clip.candidate_id:
                candidates_by_id[cand.id] = cand
                break
    speaker_label = _speaker_label_for_clip(clip, candidates_by_id)
    kit = await resolve_brand_kit_for_candidate(
        db,
        stream_id=clip.stream_id,
        speaker_label=speaker_label,
    )

    report = _empty_report(tenant_id=clip.tenant_id, dry_run=dry_run)
    if kit is None or not kit.auto_publish_enabled:
        return report._replace(clips_considered=1, clips_skipped_no_kit=1)

    clip_variants = await variants_repo.list_for_clip(clip.id)
    if not clip_variants:
        return report._replace(clips_considered=1, clips_skipped_no_variant=1)
    variant = clip_variants[0]

    accounts = await accounts_repo.list_for_tenant()
    accounts_by_platform: dict[str, ConnectedAccount] = {
        acc.platform: acc for acc in accounts
    }

    existing_jobs = await jobs_repo.list_for_clip(clip.id)
    existing_platforms = {j.platform for j in existing_jobs}

    enqueued = 0
    skipped_no_account = 0
    skipped_already_queued = 0
    for platform in kit.auto_publish_platforms or []:
        account = accounts_by_platform.get(platform)
        if account is None:
            skipped_no_account += 1
            continue
        if platform in existing_platforms:
            skipped_already_queued += 1
            continue
        scheduled_for = _undo_window_iso(
            clip_created_at=clip.created_at,
            delay_min=kit.auto_publish_delay_min,
        )
        if not dry_run:
            await jobs_repo.enqueue(
                clip_id=clip.id,
                variant_id=variant.id,
                account_id=account.id,
                platform=platform,
                scheduled_for=scheduled_for,
            )
        enqueued += 1

    _log.info(
        "auto_publish.dispatched_for_clip",
        tenant_id=clip.tenant_id,
        clip_id=clip.id,
        enqueued=enqueued,
        skipped_no_account=skipped_no_account,
        skipped_already_queued=skipped_already_queued,
    )
    return AutoPublishReport(
        tenant_id=clip.tenant_id,
        jobs_enqueued=enqueued,
        clips_considered=1,
        clips_skipped_no_kit=0,
        clips_skipped_no_variant=0,
        clips_skipped_already_queued=skipped_already_queued,
        clips_skipped_no_account=skipped_no_account,
        dry_run=dry_run,
    )
