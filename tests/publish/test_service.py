"""run_publish_jobs orchestration - retries, transient/fatal split, row state."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from nexoclip.db import (
    ConnectedAccountsRepo,
    Database,
    EventsRepo,
    PublishJobsRepo,
)
from nexoclip.publish import BufferClient, BufferError, run_publish_jobs
from nexoclip.tenancy import bound_tenant


class _FakeBufferClient(BufferClient):
    """Replays a queued list of (response_dict | exception) responses."""

    def __init__(self, access_token: str, responses: list[Any]):
        # Skip BufferClient.__init__ since we never make HTTP calls here.
        self._access_token = access_token
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def aclose(self) -> None:  # type: ignore[override]
        return None

    async def __aenter__(self) -> _FakeBufferClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def create_update(
        self, *, profile_external_id: str, text: str, media_url: str | None = None
    ) -> dict[str, Any]:
        self.calls.append({"profile_external_id": profile_external_id, "text": text})
        if not self._responses:
            raise AssertionError("no queued buffer response")
        nxt = self._responses.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


def _factory(responses: list[Any]) -> AsyncIterator[_FakeBufferClient]:
    """Returns a factory that hands the same fake to every call (one drain
    pass keeps reusing the same `_FakeBufferClient`)."""
    fake = _FakeBufferClient("btok_abc", responses)

    def make(_token: str) -> _FakeBufferClient:
        return fake

    return make, fake  # type: ignore[return-value]


async def _no_sleep(_s: float) -> None:
    return None


async def test_successful_post_marks_sent(
    db: Database, seeded: dict[str, str]
) -> None:
    factory, fake = _factory([{"updates": [{"id": "buf_update_1"}]}])
    outcome = await run_publish_jobs(
        seeded["tenant_id"], db, buffer_factory=factory, sleep=_no_sleep
    )
    assert outcome.sent == 1
    assert outcome.permanent_failures == 0
    assert outcome.transient_failures == 0
    assert len(fake.calls) == 1
    assert fake.calls[0]["text"] == "hola #clip #live"

    with bound_tenant(seeded["tenant_id"]):
        rows = await PublishJobsRepo(db).list_for_clip(seeded["clip_id"])
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "sent"
    assert row.external_id == "buf_update_1"
    assert row.attempts == 1
    assert row.last_error is None

    with bound_tenant(seeded["tenant_id"]):
        events = await EventsRepo(db).list_for_tenant(type="clip.published")
    assert any(e.payload.get("clip_id") == seeded["clip_id"] for e in events)


async def test_transient_then_success(
    db: Database, seeded: dict[str, str]
) -> None:
    """One 503, then a 200 - row ends up sent with attempts=2."""
    factory, fake = _factory(
        [
            BufferError("503", transient=True, status_code=503),
            {"updates": [{"id": "buf_update_2"}]},
        ]
    )
    outcome = await run_publish_jobs(
        seeded["tenant_id"], db, buffer_factory=factory, sleep=_no_sleep
    )
    assert outcome.sent == 1
    assert len(fake.calls) == 2

    with bound_tenant(seeded["tenant_id"]):
        rows = await PublishJobsRepo(db).list_for_clip(seeded["clip_id"])
    assert rows[0].status == "sent"
    assert rows[0].attempts == 2


async def test_fatal_marks_failed_immediately(
    db: Database, seeded: dict[str, str]
) -> None:
    factory, fake = _factory([BufferError("401", transient=False, status_code=401)])
    outcome = await run_publish_jobs(
        seeded["tenant_id"], db, buffer_factory=factory, sleep=_no_sleep
    )
    assert outcome.permanent_failures == 1
    assert len(fake.calls) == 1  # no retries on fatal

    with bound_tenant(seeded["tenant_id"]):
        rows = await PublishJobsRepo(db).list_for_clip(seeded["clip_id"])
    assert rows[0].status == "failed"
    assert "401" in (rows[0].last_error or "")

    with bound_tenant(seeded["tenant_id"]):
        events = await EventsRepo(db).list_for_tenant(type="publish_job.failed")
    assert events, "expected publish_job.failed event"


async def test_gives_up_after_max_attempts(
    db: Database, seeded: dict[str, str]
) -> None:
    """Three transient failures with max_attempts=3 -> failed."""
    factory, fake = _factory(
        [
            BufferError("503", transient=True, status_code=503),
            BufferError("503", transient=True, status_code=503),
            BufferError("503", transient=True, status_code=503),
        ]
    )
    outcome = await run_publish_jobs(
        seeded["tenant_id"],
        db,
        buffer_factory=factory,
        sleep=_no_sleep,
        max_attempts=3,
    )
    assert outcome.permanent_failures == 1
    assert len(fake.calls) == 3

    with bound_tenant(seeded["tenant_id"]):
        rows = await PublishJobsRepo(db).list_for_clip(seeded["clip_id"])
    assert rows[0].status == "failed"
    assert rows[0].attempts == 3


async def test_skips_account_without_token(
    db: Database, seeded: dict[str, str]
) -> None:
    """Strip the access token from the connected account -> permanent failure."""
    conn = await db.connect()
    await conn.execute(
        "UPDATE connected_accounts SET oauth_blob_json = NULL WHERE id = ?",
        (seeded["account_id"],),
    )
    await conn.commit()

    factory, _fake = _factory([])  # never called
    outcome = await run_publish_jobs(
        seeded["tenant_id"], db, buffer_factory=factory, sleep=_no_sleep
    )
    assert outcome.permanent_failures == 1

    with bound_tenant(seeded["tenant_id"]):
        rows = await PublishJobsRepo(db).list_for_clip(seeded["clip_id"])
    assert rows[0].status == "failed"
    assert "access_token" in (rows[0].last_error or "")


async def test_drain_isolates_per_tenant(
    db: Database, seeded: dict[str, str]
) -> None:
    """Adding a job for a different tenant - drain only touches the requested one."""
    from nexoclip.db import (
        CandidatesRepo,
        ClipsRepo,
        PersonasRepo,
        StreamsRepo,
        TenantsRepo,
        VariantsRepo,
    )
    from nexoclip.db.models import (
        CandidateRow,
        ClipRow,
        StreamRow,
        VariantRow,
    )

    other = await TenantsRepo(db).create(name="Other Co")
    with bound_tenant(other.id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_b",
                tenant_id=other.id,
                vod_url="x",
                platform="kick",
                title="t",
                channel="c",
                duration_s=10.0,
                source_video_path="/tmp/b.mp4",
                source_audio_path="/tmp/b.wav",
                status="ingested",
                created_at="2026-01-01T00:00:00+00:00",
            )
        )
        await CandidatesRepo(db).upsert_many(
            [
                CandidateRow(
                    id="cnd_b",
                    stream_id="str_b",
                    tenant_id=other.id,
                    ts=1.0,
                    score=0.5,
                    reason="voice",
                    evidence={},
                    created_at="2026-01-01T00:00:00+00:00",
                )
            ]
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id="clp_b",
                    stream_id="str_b",
                    tenant_id=other.id,
                    candidate_id="cnd_b",
                    start_s=0.0,
                    end_s=10.0,
                    duration_s=10.0,
                    width=1080,
                    height=1920,
                    path="/tmp/b.mp4",
                    status="approved",
                    created_at="2026-01-01T00:00:00+00:00",
                )
            ]
        )
        await PersonasRepo(db).create(
            persona_id="other",
            name="Other",
            primary_language="en",
            target_languages=["en"],
            voice_prompt="v",
        )
        await VariantsRepo(db).replace_for_clip_persona(
            "clp_b",
            "other",
            [
                VariantRow(
                    id="var_b",
                    clip_id="clp_b",
                    tenant_id=other.id,
                    persona_id="other",
                    language="en",
                    caption="hello",
                    title_card_text="",
                    hashtags=[],
                    model="claude-haiku-4-5-20251001",
                    created_at="2026-01-01T00:00:00+00:00",
                )
            ],
        )
        other_account = await ConnectedAccountsRepo(db).create(
            platform="buffer",
            external_id="buf_other",
            display_name="other",
            oauth_blob={"access_token": "tok_other"},
        )
        await PublishJobsRepo(db).enqueue(
            clip_id="clp_b",
            variant_id="var_b",
            account_id=other_account.id,
            platform="buffer",
        )

    factory, fake = _factory([{"updates": [{"id": "buf_update_seeded"}]}])
    outcome = await run_publish_jobs(
        seeded["tenant_id"], db, buffer_factory=factory, sleep=_no_sleep
    )
    assert outcome.sent == 1
    assert len(fake.calls) == 1  # only the seeded tenant's job ran

    # The other tenant's job is still pending.
    with bound_tenant(other.id):
        other_rows = await PublishJobsRepo(db).list_for_clip("clp_b")
    assert len(other_rows) == 1
    assert other_rows[0].status == "pending"
