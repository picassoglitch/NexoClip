"""Auto-publish dispatcher tests — slice E.2.

Walks through:
  * Kits with auto_publish=False are ignored.
  * Kits with auto_publish=True enqueue one job per platform per clip.
  * scheduled_for is clip.created_at + delay_min, not now().
  * Re-running is idempotent (existing job blocks re-enqueue).
  * Missing variant / missing account / unknown platform each have their
    own skip counter so ops can debug.
  * Tenant isolation: dispatching for Alice doesn't queue Bob's clips.
  * The worker (`list_pending`) hides jobs while they're in the window
    and surfaces them once it elapses.
  * `PublishJobsRepo.cancel` flips pending -> canceled, blocking re-enqueue.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio

from nexoclip.db import (
    BrandKitsRepo,
    CandidatesRepo,
    ClipsRepo,
    ConnectedAccountsRepo,
    Database,
    PersonasRepo,
    PublishJobsRepo,
    StreamsRepo,
    TenantsRepo,
    VariantsRepo,
    apply_migrations,
)
from nexoclip.db.models import (
    BrandKitRow,
    CandidateRow,
    ClipRow,
    StreamRow,
    VariantRow,
)
from nexoclip.publish import dispatch_auto_publish
from nexoclip.publish.auto import _undo_window_iso
from nexoclip.tenancy import bound_tenant


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "auto.db")
    try:
        await apply_migrations(d)
        yield d
    finally:
        await d.close()


def _iso(*, days_ago: int = 0, minutes_ago: int = 0) -> str:
    now = _dt.datetime.now(_dt.UTC)
    delta = _dt.timedelta(days=days_ago, minutes=minutes_ago)
    return (now - delta).isoformat()


async def _seed_tenant_with_kit(
    db: Database,
    *,
    tenant_id: str,
    auto_enabled: bool = True,
    platforms: list[str] | None = None,
    delay_min: int = 60,
) -> tuple[str, BrandKitRow]:
    """Create a tenant + a brand kit (default) + a connected account
    per platform. Returns (tenant_id, kit)."""
    if platforms is None:
        platforms = ["tiktok"]
    await TenantsRepo(db).create(tenant_id=tenant_id, name=f"T-{tenant_id}")
    with bound_tenant(tenant_id):
        kit = await BrandKitsRepo(db).create(
            name="K",
            primary_color="#000000",
            accent_color="#FFFFFF",
            is_default=True,
            auto_publish_enabled=auto_enabled,
            auto_publish_platforms=platforms,
            auto_publish_delay_min=delay_min,
        )
        for p in platforms:
            await ConnectedAccountsRepo(db).create(
                platform=p, external_id=f"ext-{p}-{tenant_id}"
            )
    return tenant_id, kit


async def _seed_clip_with_variant(
    db: Database,
    *,
    tenant_id: str,
    stream_id: str = "str_x",
    clip_id: str = "clp_x",
    speaker_label: str | None = None,
    has_variant: bool = True,
    clip_created_at: str | None = None,
) -> None:
    """Seed a stream, candidate (optionally carrying a speaker_label),
    clip, and one variant per persona."""
    created = clip_created_at or _iso()
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id,
                tenant_id=tenant_id,
                vod_url="x",
                platform="kick",
                title=None,
                channel=None,
                duration_s=60.0,
                source_video_path="/tmp/v",
                source_audio_path="/tmp/a",
                status="ingested",
                created_at=created,
            )
        )
        cand_evidence: dict[str, object] = {}
        if speaker_label is not None:
            cand_evidence["speaker_label"] = speaker_label
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id=f"cnd_{clip_id[4:]}",
                    stream_id=stream_id,
                    tenant_id=tenant_id,
                    ts=10.0,
                    score=0.9,
                    reason="voice",
                    evidence=cand_evidence,
                    created_at=created,
                )
            ]
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id=clip_id,
                    stream_id=stream_id,
                    tenant_id=tenant_id,
                    candidate_id=f"cnd_{clip_id[4:]}",
                    start_s=0.0,
                    end_s=10.0,
                    duration_s=10.0,
                    width=1080,
                    height=1920,
                    path="/tmp/c.mp4",
                    status="cut",
                    created_at=created,
                )
            ]
        )
        if has_variant:
            # Personas are globally-unique-id keyed; suffix with tenant
            # so multi-tenant tests can each seed their own persona row
            # without colliding on the PK.
            persona_id = f"p_{tenant_id}"
            existing = await PersonasRepo(db).get(persona_id)
            if existing is None:
                await PersonasRepo(db).create(
                    persona_id=persona_id,
                    name="Persona",
                    primary_language="es",
                    target_languages=["es"],
                    voice_prompt="v",
                )
            await VariantsRepo(db).replace_for_clip_persona(
                clip_id,
                persona_id,
                [
                    VariantRow(
                        id=f"var_{clip_id[4:]}",
                        clip_id=clip_id,
                        tenant_id=tenant_id,
                        persona_id=persona_id,
                        language="es",
                        caption="c",
                        title_card_text="",
                        hashtags=[],
                        model=None,
                        created_at=created,
                    )
                ],
            )


# ---- _undo_window_iso ----


def test_undo_window_adds_delay_to_clip_timestamp() -> None:
    base = "2026-05-13T12:00:00+00:00"
    out = _undo_window_iso(clip_created_at=base, delay_min=60)
    assert out == "2026-05-13T13:00:00+00:00"


def test_undo_window_falls_back_to_now_on_bad_iso() -> None:
    """A junk timestamp doesn't crash the dispatcher; we add the delay
    to wall-clock now()."""
    out = _undo_window_iso(clip_created_at="not-an-iso", delay_min=10)
    parsed = _dt.datetime.fromisoformat(out)
    # Should be 10 minutes from now (give or take a few seconds).
    delta = (parsed - _dt.datetime.now(_dt.UTC)).total_seconds()
    assert 8 * 60 < delta < 12 * 60


def test_undo_window_assumes_utc_for_naive_input() -> None:
    """A naive (no-tz) timestamp shouldn't shift hours when we add the
    delay — we treat it as already-UTC."""
    out = _undo_window_iso(clip_created_at="2026-05-13T12:00:00", delay_min=30)
    assert out == "2026-05-13T12:30:00+00:00"


# ---- dispatcher ----


async def test_dispatch_no_tenants_returns_empty(db: Database) -> None:
    assert await dispatch_auto_publish(db) == []


async def test_dispatch_skips_kits_with_auto_disabled(db: Database) -> None:
    tid, _kit = await _seed_tenant_with_kit(
        db, tenant_id="aldo", auto_enabled=False
    )
    await _seed_clip_with_variant(db, tenant_id=tid)
    reports = await dispatch_auto_publish(db)
    assert reports[0].jobs_enqueued == 0
    assert reports[0].clips_considered == 0
    # No auto-kits at all means we skip the whole tenant.


async def test_dispatch_enqueues_one_job_per_platform(db: Database) -> None:
    tid, _kit = await _seed_tenant_with_kit(
        db, tenant_id="aldo", platforms=["tiktok", "youtube"]
    )
    await _seed_clip_with_variant(db, tenant_id=tid)
    reports = await dispatch_auto_publish(db)
    assert reports[0].jobs_enqueued == 2
    with bound_tenant(tid):
        jobs = await PublishJobsRepo(db).list_for_clip("clp_x")
    assert {j.platform for j in jobs} == {"tiktok", "youtube"}
    assert all(j.status == "pending" for j in jobs)
    assert all(j.scheduled_for is not None for j in jobs)


async def test_dispatch_is_idempotent_across_runs(db: Database) -> None:
    tid, _kit = await _seed_tenant_with_kit(db, tenant_id="aldo")
    await _seed_clip_with_variant(db, tenant_id=tid)
    r1 = await dispatch_auto_publish(db)
    r2 = await dispatch_auto_publish(db)
    assert r1[0].jobs_enqueued == 1
    assert r2[0].jobs_enqueued == 0
    assert r2[0].clips_skipped_already_queued == 1


async def test_dispatch_skips_clips_without_variants(db: Database) -> None:
    tid, _kit = await _seed_tenant_with_kit(db, tenant_id="aldo")
    await _seed_clip_with_variant(db, tenant_id=tid, has_variant=False)
    reports = await dispatch_auto_publish(db)
    assert reports[0].jobs_enqueued == 0
    assert reports[0].clips_skipped_no_variant == 1


async def test_dispatch_skips_when_no_connected_account(db: Database) -> None:
    """Kit lists 'instagram' but there's no instagram account — skip."""
    # Seed manually so we DON'T create the connected account.
    tid = "aldo"
    await TenantsRepo(db).create(tenant_id=tid, name="A")
    with bound_tenant(tid):
        await BrandKitsRepo(db).create(
            name="K",
            primary_color="#000",
            accent_color="#FFF",
            is_default=True,
            auto_publish_enabled=True,
            auto_publish_platforms=["instagram"],
        )
    await _seed_clip_with_variant(db, tenant_id=tid)
    reports = await dispatch_auto_publish(db)
    assert reports[0].jobs_enqueued == 0
    assert reports[0].clips_skipped_no_account == 1


async def test_dispatch_uses_clip_created_at_for_scheduled_for(db: Database) -> None:
    """scheduled_for offsets from the CLIP, not the dispatcher's `now`,
    so a late cron doesn't slide the undo window."""
    tid, _kit = await _seed_tenant_with_kit(db, tenant_id="aldo", delay_min=60)
    clip_created = _iso(minutes_ago=30)  # clip is 30 min old
    await _seed_clip_with_variant(db, tenant_id=tid, clip_created_at=clip_created)
    await dispatch_auto_publish(db)
    with bound_tenant(tid):
        job = (await PublishJobsRepo(db).list_for_clip("clp_x"))[0]
    expected = (
        _dt.datetime.fromisoformat(clip_created)
        + _dt.timedelta(minutes=60)
    ).isoformat()
    assert job.scheduled_for == expected


async def test_dispatch_dry_run_does_not_write_jobs(db: Database) -> None:
    tid, _kit = await _seed_tenant_with_kit(db, tenant_id="aldo")
    await _seed_clip_with_variant(db, tenant_id=tid)
    reports = await dispatch_auto_publish(db, dry_run=True)
    assert reports[0].dry_run is True
    assert reports[0].jobs_enqueued == 1  # reported
    with bound_tenant(tid):
        jobs = await PublishJobsRepo(db).list_for_clip("clp_x")
    assert jobs == []  # but nothing in the DB


async def test_dispatch_is_per_tenant_isolated(db: Database) -> None:
    """Sweeping Alice doesn't enqueue Bob's clips."""
    await _seed_tenant_with_kit(db, tenant_id="alice")
    await _seed_clip_with_variant(
        db, tenant_id="alice", stream_id="str_a", clip_id="clp_a"
    )
    await _seed_tenant_with_kit(db, tenant_id="bob")
    await _seed_clip_with_variant(
        db, tenant_id="bob", stream_id="str_b", clip_id="clp_b"
    )
    reports = await dispatch_auto_publish(db, tenant_id="alice")
    assert len(reports) == 1
    assert reports[0].tenant_id == "alice"
    assert reports[0].jobs_enqueued == 1
    # Bob's clip didn't get a job.
    with bound_tenant("bob"):
        assert await PublishJobsRepo(db).list_for_clip("clp_b") == []


# ---- list_pending / scheduled_for window ----


async def test_list_pending_hides_jobs_inside_undo_window(db: Database) -> None:
    """A job with scheduled_for in the future is invisible to the worker."""
    tid, _kit = await _seed_tenant_with_kit(db, tenant_id="aldo", delay_min=60)
    # Clip created NOW → scheduled_for ~60min in the future.
    await _seed_clip_with_variant(db, tenant_id=tid)
    await dispatch_auto_publish(db)
    with bound_tenant(tid):
        pending = await PublishJobsRepo(db).list_pending()
        scheduled = await PublishJobsRepo(db).list_scheduled()
    assert pending == []
    assert len(scheduled) == 1


async def test_list_pending_surfaces_jobs_past_undo_window(db: Database) -> None:
    """A job whose scheduled_for already passed shows up in list_pending."""
    tid, _kit = await _seed_tenant_with_kit(db, tenant_id="aldo", delay_min=10)
    # Clip is 30 min old, delay is 10 min → scheduled_for was 20 min ago.
    await _seed_clip_with_variant(
        db, tenant_id=tid, clip_created_at=_iso(minutes_ago=30)
    )
    await dispatch_auto_publish(db)
    with bound_tenant(tid):
        pending = await PublishJobsRepo(db).list_pending()
        scheduled = await PublishJobsRepo(db).list_scheduled()
    assert len(pending) == 1
    assert scheduled == []


# ---- cancel ----


async def test_cancel_flips_pending_to_canceled(db: Database) -> None:
    tid, _kit = await _seed_tenant_with_kit(db, tenant_id="aldo")
    await _seed_clip_with_variant(db, tenant_id=tid)
    await dispatch_auto_publish(db)
    with bound_tenant(tid):
        job = (await PublishJobsRepo(db).list_for_clip("clp_x"))[0]
        ok = await PublishJobsRepo(db).cancel(job.id)
        after = (await PublishJobsRepo(db).list_for_clip("clp_x"))[0]
    assert ok is True
    assert after.status == "canceled"


async def test_cancel_no_op_on_unknown_job(db: Database) -> None:
    await TenantsRepo(db).create(tenant_id="aldo", name="A")
    with bound_tenant("aldo"):
        ok = await PublishJobsRepo(db).cancel("pjb_nope")
    assert ok is False


async def test_cancel_blocks_redispatch(db: Database) -> None:
    """Once an operator cancels, the next dispatch run doesn't re-enqueue.

    `list_for_clip` returns ALL jobs (including canceled), and the
    dispatcher's `existing_platforms` check is by platform — so the
    canceled job blocks the slot."""
    tid, _kit = await _seed_tenant_with_kit(db, tenant_id="aldo")
    await _seed_clip_with_variant(db, tenant_id=tid)
    await dispatch_auto_publish(db)
    with bound_tenant(tid):
        job = (await PublishJobsRepo(db).list_for_clip("clp_x"))[0]
        await PublishJobsRepo(db).cancel(job.id)
    r2 = await dispatch_auto_publish(db)
    assert r2[0].jobs_enqueued == 0
    assert r2[0].clips_skipped_already_queued == 1
