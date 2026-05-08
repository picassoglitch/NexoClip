"""POST /streams/upload — multipart upload bypasses yt-dlp."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from nexoclip.api import create_app
from nexoclip.db import Database, StreamsRepo
from nexoclip.tenancy import bound_tenant

from .conftest import auth


def _stub_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real ffmpeg/ffprobe in CI — short-circuit both shell-outs and the
    PATH-availability check that gates the upload endpoints."""

    def fake_extract_audio(_video: Path, audio: Path) -> None:
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"\x00fakeaudio")

    def fake_ffprobe(_video: Path) -> float:
        return 17.5

    monkeypatch.setattr("nexoclip.ingest.service._extract_audio", fake_extract_audio)
    monkeypatch.setattr("nexoclip.ingest.service._ffprobe_duration", fake_ffprobe)
    # The handlers call `is_ffmpeg_available()` to gate the upload — stub
    # it in both modules where it's looked up so the fast-path returns True.
    monkeypatch.setattr("nexoclip.ingest.service.is_ffmpeg_available", lambda: True)
    monkeypatch.setattr("nexoclip.ingest.is_ffmpeg_available", lambda: True)


async def test_upload_creates_stream_and_kicks_off_pipeline(
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Multipart POST writes the file, runs ingest_uploaded, persists, schedules pipeline."""
    from nexoclip.api import PipelineKickoff

    _stub_ffmpeg(monkeypatch)
    monkeypatch.setattr(
        "nexoclip.settings.get_settings",
        lambda: type("S", (), {"default_output_dir": str(tmp_path / "out")})(),
    )

    captured: list[PipelineKickoff] = []

    async def fake_runner(kickoff: PipelineKickoff) -> None:
        captured.append(kickoff)

    app = create_app(db=db, pipeline_runner=fake_runner)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as c:
        files = {"file": ("recording.mp4", b"\x00fakevideo", "video/mp4")}
        r = await c.post(
            "/streams/upload",
            files=files,
            data={"persona_id": "aldo"},
            headers=auth(tenants["alice"]["token"]),
        )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["platform"] == "upload"
    assert body["title"] == "recording.mp4"
    assert body["duration_s"] == 17.5

    # Stream row landed under Alice's tenant.
    with bound_tenant(tenants["alice"]["id"]):
        row = await StreamsRepo(db).get(body["id"])
    assert row is not None
    assert row.platform == "upload"

    # Background pipeline task fired with the right kickoff.
    assert len(captured) == 1
    assert captured[0].tenant_id == tenants["alice"]["id"]
    assert captured[0].persona_id == "aldo"
    assert captured[0].stream.platform == "upload"


async def test_upload_rejects_empty_file(
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A zero-byte upload is a configuration mistake, not a stream."""
    _stub_ffmpeg(monkeypatch)
    monkeypatch.setattr(
        "nexoclip.settings.get_settings",
        lambda: type("S", (), {"default_output_dir": str(tmp_path / "out")})(),
    )

    app = create_app(db=db)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as c:
        files = {"file": ("empty.mp4", b"", "video/mp4")}
        r = await c.post(
            "/streams/upload",
            files=files,
            data={"persona_id": "aldo"},
            headers=auth(tenants["alice"]["token"]),
        )
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


async def test_dashboard_upload_redirects_to_stream_detail(
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The /dashboard/streams/upload form redirects to /dashboard/streams/<id>."""
    from nexoclip.api import PipelineKickoff

    _stub_ffmpeg(monkeypatch)
    monkeypatch.setattr(
        "nexoclip.settings.get_settings",
        lambda: type("S", (), {"default_output_dir": str(tmp_path / "out")})(),
    )

    async def fake_runner(_kickoff: PipelineKickoff) -> None:
        return None

    app = create_app(db=db, pipeline_runner=fake_runner)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://api.test"
    ) as c:
        # Sign in (sets the dashboard cookie).
        await c.post(
            "/dashboard/login", data={"token": tenants["alice"]["token"]}
        )
        files = {"file": ("clip.mp4", b"\x00fakevideo", "video/mp4")}
        r = await c.post(
            "/dashboard/streams/upload",
            files=files,
            data={"persona_id": "aldo"},
            follow_redirects=False,
        )
    assert r.status_code == 303
    assert r.headers["location"].startswith("/dashboard/streams/str_")


async def test_upload_503s_when_ffmpeg_missing(
    db: Database,
    tenants: dict[str, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When ffmpeg isn't on PATH, the upload endpoint refuses up front with a
    503 + install hint, so a multi-GB upload doesn't fail mid-pipeline."""
    monkeypatch.setattr(
        "nexoclip.settings.get_settings",
        lambda: type("S", (), {"default_output_dir": str(tmp_path / "out")})(),
    )
    monkeypatch.setattr("nexoclip.ingest.service.is_ffmpeg_available", lambda: False)
    monkeypatch.setattr("nexoclip.ingest.is_ffmpeg_available", lambda: False)

    app = create_app(db=db)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://api.test"
    ) as c:
        files = {"file": ("v.mp4", b"\x00fakevideo", "video/mp4")}
        r = await c.post(
            "/streams/upload",
            files=files,
            data={"persona_id": "aldo"},
            headers=auth(tenants["alice"]["token"]),
        )
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "ffmpeg" in detail
    assert "winget" in detail or "brew" in detail


async def test_upload_requires_full_scope(
    db: Database,
    tenants: dict[str, dict[str, str]],
) -> None:
    """No bearer = 401 (auth middleware rejects before form parsing)."""
    app = create_app(db=db)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as c:
        files = {"file": ("x.mp4", b"\x00x", "video/mp4")}
        r = await c.post(
            "/streams/upload",
            files=files,
            data={"persona_id": "aldo"},
        )
    assert r.status_code == 401
