"""Creator-facing stream_detail redesign — render-through tests.

Pins the new page structure: outcome hero, AI summary, big clip cards
with viral score + humanized AI signals, visual timeline, and the
collapsed Technical details. Exercises the real handler (which builds
the `overview` view model) + the real template + base.html.
"""

from __future__ import annotations

import datetime as _dt

import httpx
import pytest

from nexoclip.db import (
    CandidatesRepo,
    ClipsRepo,
    Database,
    StreamsRepo,
)
from nexoclip.db.models import CandidateRow, ClipRow, StreamRow
from nexoclip.tenancy import bound_tenant

from .conftest import auth


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed_stream(
    db: Database,
    tenant_id: str,
    *,
    stream_status: str = "done",
    with_clips: bool = True,
) -> str:
    stream_id = "str_sd"
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=stream_id, tenant_id=tenant_id, vod_url="https://kick.com/x",
                platform="kick", title="vertical_test.mp4", channel="aldo",
                duration_s=122.0, source_video_path="/tmp/v.mp4",
                source_audio_path="/tmp/a.wav", status=stream_status,
                created_at=_now(),
            )
        )
        if with_clips:
            await CandidatesRepo(db).upsert_many([
                CandidateRow(
                    id="cnd_sd", stream_id=stream_id, tenant_id=tenant_id,
                    ts=20.0, score=0.82, reason="voice",
                    evidence={"phrase": "clipea esto", "speaker_label": "Aldo"},
                    created_at=_now(),
                ),
            ])
            await ClipsRepo(db).upsert_many([
                ClipRow(
                    id="clp_sd_top", stream_id=stream_id, tenant_id=tenant_id,
                    candidate_id="cnd_sd", start_s=20.0, end_s=55.0,
                    duration_s=35.0, width=1080, height=1920,
                    path="/tmp/c1.mp4", status="approved", created_at=_now(),
                    publishability_score=85, publishability_status="publish_ready",
                ),
                ClipRow(
                    id="clp_sd_2", stream_id=stream_id, tenant_id=tenant_id,
                    candidate_id=None, start_s=98.0, end_s=113.0,
                    duration_s=15.0, width=1080, height=1920,
                    path="/tmp/c2.mp4", status="cut", created_at=_now(),
                    publishability_score=61,
                ),
            ])
    return stream_id


async def test_complete_stream_shows_outcome_hero(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """A finished stream leads with the outcome, not the pipeline:
    'Processing complete', clip counts, and a Download-approved CTA."""
    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id)

    r = await client.get(
        "/dashboard/streams/str_sd", headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 200, r.text
    html = r.text
    # Outcome-first hero.
    assert "Processing complete" in html
    assert "clips ready" in html or "clip ready" in html
    # Download-approved CTA (1 approved clip seeded).
    assert "Download 1 approved" in html


async def test_complete_stream_shows_ai_summary_and_scores(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id)
    r = await client.get(
        "/dashboard/streams/str_sd", headers=auth(tenants["alice"]["token"]),
    )
    html = r.text
    # AI summary card.
    assert "AI summary" in html
    assert "Suggested platforms" in html
    assert "TikTok" in html
    # Viral score badge + the top-pick highlight on the 85-score clip.
    assert "VIRAL SCORE" in html
    assert "TOP PICK" in html
    assert "85" in html


async def test_complete_stream_shows_humanized_ai_signals(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """The clip card explains the AI in plain language, not raw numbers."""
    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id)
    r = await client.get(
        "/dashboard/streams/str_sd", headers=auth(tenants["alice"]["token"]),
    )
    html = r.text
    assert "Voice trigger" in html
    assert "clipea esto" in html  # the detected phrase, quoted
    assert "Speaker: Aldo" in html


async def test_complete_stream_shows_timeline(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id)
    r = await client.get(
        "/dashboard/streams/str_sd", headers=auth(tenants["alice"]["token"]),
    )
    html = r.text
    assert "nc-timeline__track" in html
    assert "nc-timeline__seg" in html


async def test_technical_details_collapsed_but_present(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Pipeline / re-run / candidates table move into a <details> block —
    present for power users, out of the creator's way."""
    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id)
    r = await client.get(
        "/dashboard/streams/str_sd", headers=auth(tenants["alice"]["token"]),
    )
    html = r.text
    assert "Technical details" in html
    assert "<details" in html
    # The re-run form + candidates table still exist (just nested).
    assert "Run pipeline" in html
    assert "Candidates" in html
    # Source video moved into technical too.
    assert "/dashboard/streams/str_sd/source" in html


async def test_finished_run_with_stale_upload_status_shows_complete(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """A stream whose status column is still 'upload' but which HAS
    clips is DONE — the hero must show the outcome, not 'Analyzing your
    video…'. (The status update is best-effort; clips are the truth.)"""
    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id, stream_status="upload", with_clips=True)
    r = await client.get(
        "/dashboard/streams/str_sd", headers=auth(tenants["alice"]["token"]),
    )
    html = r.text
    assert "Processing complete" in html
    assert "Analyzing your video" not in html


async def test_running_stream_shows_progress_not_outcome(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """While processing, the hero says 'Analyzing…' and the live HTMX
    progress poller renders; the outcome metrics don't."""
    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id, stream_status="running", with_clips=False)
    r = await client.get(
        "/dashboard/streams/str_sd", headers=auth(tenants["alice"]["token"]),
    )
    html = r.text
    assert "Analyzing your video" in html
    # Live progress poller present while running.
    assert 'id="stream-progress"' in html
    assert "Processing complete" not in html


async def test_download_machinery_preserved(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """The fetch+blob download JS + its hooks survive the redesign — the
    approved clip still carries js-download-btn + data-duration-s."""
    tenant_id = tenants["alice"]["id"]
    await _seed_stream(db, tenant_id)
    r = await client.get(
        "/dashboard/streams/str_sd", headers=auth(tenants["alice"]["token"]),
    )
    html = r.text
    assert "js-download-btn" in html
    assert "data-duration-s" in html
    assert "pollUntilReady" in html
