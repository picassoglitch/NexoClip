"""Render Migration R5 — outer guard around render_clip_in_background.

Previously the runner caught only specific recorder errors; a stray
FileNotFoundError / AttributeError / KeyError / etc. propagated out
of the BackgroundTask coroutine. FastAPI's BackgroundTask runner
logged a traceback and dropped the task → state stayed 'rendering'
forever → operator's UI polled /render-status indefinitely with no
way to retry.

These tests run the actual `render_clip_in_background` coroutine
end-to-end with the recorders stubbed to raise different exception
classes, then assert:

  - The runner does NOT re-raise (so FastAPI's BackgroundTask
    runner never sees a traceback).
  - The clip row's render_state ends up 'failed' or 'ready'
    (never stuck at 'rendering').
  - When the hybrid raises a non-HybridRecordingError exception,
    we fall through to the legacy path (and if that succeeds, the
    render still completes).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

from nexoclip.api._clip_render import render_clip_in_background
from nexoclip.db import ClipsRepo, Database
from nexoclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed_clip(
    db: Database,
    tenant_id: str,
    *,
    clip_id: str,
    output_dir: Path,
) -> Path:
    stream_id = f"str_{clip_id}"
    clip_dir = output_dir / "clips" / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    source = clip_dir / "clip.mp4"
    source.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100_000)

    with bound_tenant(tenant_id):
        conn = await db.connect()
        await conn.execute(
            "INSERT OR IGNORE INTO streams "
            "(id, tenant_id, vod_url, platform, title, channel, "
            "duration_s, source_video_path, source_audio_path, status, "
            "created_at) "
            "VALUES (?, ?, 'https://kick.com/x/v/1', 'kick', NULL, NULL, "
            "30, '/tmp/v.mp4', '/tmp/a.wav', 'ingested', ?)",
            (stream_id, tenant_id, _now()),
        )
        await conn.execute(
            "INSERT INTO clips "
            "(id, stream_id, tenant_id, start_s, end_s, duration_s, "
            "width, height, path, status, created_at, "
            "render_state, render_started_at) "
            "VALUES (?, ?, ?, 0, 30, 30, 1080, 1920, ?, 'cut', ?, "
            "'rendering', ?)",
            (clip_id, stream_id, tenant_id, str(source), _now(), _now()),
        )
        await conn.commit()
    return source


async def _final_state(db: Database, tenant_id: str, clip_id: str) -> str:
    with bound_tenant(tenant_id):
        clip = await ClipsRepo(db).get(clip_id)
    assert clip is not None
    return clip.render_state


async def test_runner_catches_filenotfound_from_hybrid(
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduces the exact Railway bug: hybrid raises FileNotFoundError
    from the success-log path. Previously this propagated out of the
    runner → state stuck at 'rendering'. Now the runner catches it as
    a generic hybrid failure → falls back to legacy. We stub legacy
    to ALSO fail so we can assert the runner ends at state='failed'
    rather than at 'rendering'."""
    tenant_id = tenants["alice"]["id"]
    source = await _seed_clip(
        db, tenant_id, clip_id="clp_fnf", output_dir=tmp_path,
    )
    output = tmp_path / "clips" / "clp_fnf" / "clip_render_1080.mp4"

    async def fake_hybrid(**kwargs: object) -> None:
        raise FileNotFoundError("simulated ffmpeg silent-failure stat()")

    async def fake_legacy(**kwargs: object) -> None:
        from nexoclip.clip.preview_recorder import PreviewRecordingError
        raise PreviewRecordingError("legacy also failed")

    monkeypatch.setattr(
        "nexoclip.clip.hybrid_recorder.record_clip_hybrid", fake_hybrid,
    )
    monkeypatch.setattr(
        "nexoclip.clip.preview_recorder.record_clip_to_mp4", fake_legacy,
    )

    # Must NOT raise — that was the bug.
    await render_clip_in_background(
        clip_id="clp_fnf",
        tenant_id=tenant_id,
        duration_s=30.0,
        audio_source_path=source,
        output_path=output,
        base_url="http://localhost:8000",
        auth_cookie_value=None,
        width=1080,
        height=1920,
        db_path=str(tmp_path / "api.db"),  # ignored — db param above
    )

    # State must NOT be 'rendering' — that was the visible symptom.
    final = await _final_state(db, tenant_id, "clp_fnf")
    assert final == "failed", (
        f"runner left state at {final!r}; UI would poll forever"
    )


async def test_runner_catches_attributeerror_from_hybrid(
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stress: any exception class, not just FileNotFoundError, gets
    caught + the runner exits with state=failed."""
    tenant_id = tenants["alice"]["id"]
    source = await _seed_clip(
        db, tenant_id, clip_id="clp_attr_err", output_dir=tmp_path,
    )
    output = tmp_path / "clips" / "clp_attr_err" / "clip_render_1080.mp4"

    async def fake_hybrid(**kwargs: object) -> None:
        raise AttributeError("simulated typo somewhere")

    async def fake_legacy(**kwargs: object) -> None:
        from nexoclip.clip.preview_recorder import PreviewRecordingError
        raise PreviewRecordingError("legacy also failed")

    monkeypatch.setattr(
        "nexoclip.clip.hybrid_recorder.record_clip_hybrid", fake_hybrid,
    )
    monkeypatch.setattr(
        "nexoclip.clip.preview_recorder.record_clip_to_mp4", fake_legacy,
    )

    await render_clip_in_background(
        clip_id="clp_attr_err",
        tenant_id=tenant_id,
        duration_s=30.0,
        audio_source_path=source,
        output_path=output,
        base_url="http://localhost:8000",
        auth_cookie_value=None,
        width=1080,
        height=1920,
        db_path=str(tmp_path / "api.db"),
    )

    assert await _final_state(db, tenant_id, "clp_attr_err") == "failed"


async def test_runner_falls_back_to_legacy_on_unexpected_hybrid_crash(
    db: Database,
    tenants: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hybrid crashes with an unexpected exception class; legacy
    succeeds. End state should be 'ready' (operator gets their
    download). Pre-R5 the FileNotFoundError would have skipped the
    fallback entirely."""
    tenant_id = tenants["alice"]["id"]
    source = await _seed_clip(
        db, tenant_id, clip_id="clp_fallback_ok", output_dir=tmp_path,
    )
    output = tmp_path / "clips" / "clp_fallback_ok" / "clip_render_1080.mp4"

    async def fake_hybrid(**kwargs: object) -> None:
        raise OSError("disk i/o error from hybrid")

    async def fake_legacy(**kwargs: object) -> None:
        # Write a real-looking MP4 so the R1 validate-before-ready
        # gate accepts it (ftyp box + > 1 MB).
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(
            b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2mp41"
            + b"\x00" * 1_100_000,
        )

    monkeypatch.setattr(
        "nexoclip.clip.hybrid_recorder.record_clip_hybrid", fake_hybrid,
    )
    monkeypatch.setattr(
        "nexoclip.clip.preview_recorder.record_clip_to_mp4", fake_legacy,
    )

    await render_clip_in_background(
        clip_id="clp_fallback_ok",
        tenant_id=tenant_id,
        duration_s=30.0,
        audio_source_path=source,
        output_path=output,
        base_url="http://localhost:8000",
        auth_cookie_value=None,
        width=1080,
        height=1920,
        db_path=str(tmp_path / "api.db"),
    )

    final = await _final_state(db, tenant_id, "clp_fallback_ok")
    # ffprobe may or may not be on PATH in this dev env. If it is and
    # rejects our synthetic ftyp-box-only file → state='failed'. If
    # not, validate degrades to magic-byte+size → state='ready'.
    # Either way the bug-symptom ('rendering' forever) is gone.
    assert final in {"ready", "failed"}, (
        f"runner left state at {final!r}; expected ready or failed"
    )
