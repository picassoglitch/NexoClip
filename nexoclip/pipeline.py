"""End-to-end VOD-to-clips pipeline.

Wires the six Phase 0 steps in order:

    ingest → transcribe → detect → cut → variants(per clip) → manifest

Each step's service function is already idempotent on its own output file,
so the orchestrator just calls them in sequence; no extra resume bookkeeping
needed. `force=True` propagates down to all of them.

The final `manifest.json` written at `<output_dir>/<stream_id>/` summarizes the
full run — stream metadata, transcript stats, all candidates, every clip with
its variants, and a roll-up of LLM spend (read from `llm_calls.jsonl`).
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

from nexoclip.clip import Clip, cut_clips
from nexoclip.config import NexoClipConfig, load_config
from nexoclip.db import (
    CandidatesRepo,
    ClipsRepo,
    Database,
    PersonasRepo,
    StreamsRepo,
    TranscriptsRepo,
    VariantsRepo,
    db_session,
)
from nexoclip.db.adapters import (
    candidate_to_row,
    clip_to_row,
    stream_to_row,
    transcript_to_row,
    variant_to_row,
)
from nexoclip.detect import (
    Candidate,
    CandidateBatch,
    detect_candidates,
    detect_viral_moments,
    save_candidates,
)
from nexoclip.errors import DetectionError, NexoClipError, VariantError
from nexoclip.events import (
    CLIP_READY_FOR_REVIEW,
    STREAM_CREATED,
    STREAM_PROCESSED,
    emit,
)
from nexoclip.ingest import Stream, ingest_vod, load_chat_replay
from nexoclip.llm import LLMConfig, LLMRouter, Variant, load_llm_config
from nexoclip.llm.config import Quality
from nexoclip.logging import get_logger
from nexoclip.settings import Settings, get_settings
from nexoclip.transcribe import Transcript, transcribe
from nexoclip.variants import Persona, generate_variants, load_personas
from nexoclip.vision import analyze_video as _analyze_video
from nexoclip.vision import load_visual_signals

_MANIFEST_SCHEMA_VERSION = 1
_log = get_logger("nexoclip.pipeline")


@contextmanager
def _step(name: str, *, db: Database | None = None, **fields: Any) -> Iterator[None]:
    """Time + log one pipeline step. Re-raises typed errors with the bound context.

    Each step is wrapped to:
        - emit `step.<name>.start` / `step.<name>.done` events with `duration_s`
        - on `NexoClipError`, append the bound contextvars to the message so
          the user sees the stream_id when the CLI prints the error

    When `db` is provided, also writes `pipeline.step.start` / `.done` /
    `.failed` rows to the events table so the dashboard can poll and show
    live progress without scraping logs. The DB write is synchronous via
    sqlite3 to keep the context manager sync — the dashboard doesn't need
    millisecond freshness, just "this step is running" / "this step finished".
    """
    _log.info(f"step.{name}.start", **fields)
    t0 = time.perf_counter()
    stream_id = structlog.contextvars.get_contextvars().get("stream_id")
    _record_step_event(db, "pipeline.step.start", name, stream_id, fields)
    try:
        yield
    except NexoClipError as e:
        duration_s = time.perf_counter() - t0
        _log.error(
            f"step.{name}.failed",
            duration_s=duration_s,
            error_class=type(e).__name__,
            error_msg=str(e),
            **fields,
        )
        _record_step_event(
            db,
            "pipeline.step.failed",
            name,
            stream_id,
            {**fields, "duration_s": duration_s, "error": str(e)},
        )
        # Append every contextvar (stream_id, tenant_id, persona_id) into the
        # error message so the CLI / log readers see it without inspection.
        ctx = structlog.contextvars.get_contextvars()
        if ctx:
            ctx_str = " ".join(f"{k}={v}" for k, v in ctx.items())
            raise type(e)(f"{e} [{ctx_str}]") from e
        raise
    duration_s = time.perf_counter() - t0
    _log.info(f"step.{name}.done", duration_s=duration_s, **fields)
    _record_step_event(
        db,
        "pipeline.step.done",
        name,
        stream_id,
        {**fields, "duration_s": duration_s},
    )


def _record_step_event(
    db: Database | None,
    event_type: str,
    step_name: str,
    stream_id: str | None,
    fields: dict[str, Any],
) -> None:
    """Best-effort sync write to the events table.

    Uses raw sqlite3 because the surrounding `_step` is a sync context
    manager and the pipeline runs the same coro. We don't want a missing
    DB or transient disk error to bring the whole step down — the dashboard
    progress view degrades gracefully when an event row is missing.
    """
    if db is None:
        return
    tenant_id = structlog.contextvars.get_contextvars().get("tenant_id")
    if not tenant_id:
        return
    try:
        import sqlite3

        payload = {"step": step_name, "stream_id": stream_id, **fields}
        # Strip non-JSON-serializable values defensively.
        clean = {k: v for k, v in payload.items() if isinstance(v, str | int | float | bool | type(None) | list | dict)}
        with sqlite3.connect(db.path) as conn:
            conn.execute(
                "INSERT INTO events (id, tenant_id, type, payload_json, ts) VALUES (?, ?, ?, ?, ?)",
                (
                    f"evt_{step_name}_{int(time.time() * 1000)}",
                    tenant_id,
                    event_type,
                    json.dumps(clean),
                    _dt.datetime.now(_dt.UTC).isoformat(),
                ),
            )
            conn.commit()
    except Exception as e:  # noqa: BLE001 - intentional best-effort
        _log.debug("pipeline.step.event_write_failed", error=str(e))


class TranscriptSummary(BaseModel):
    """Compact summary so manifest.json doesn't carry the full transcript."""

    stream_id: str
    language: str
    duration_s: float
    model: str
    segment_count: int
    word_count: int

    @classmethod
    def from_transcript(cls, transcript: Transcript) -> TranscriptSummary:
        return cls(
            stream_id=transcript.stream_id,
            language=transcript.language,
            duration_s=transcript.duration_s,
            model=transcript.model,
            segment_count=len(transcript.segments),
            word_count=sum(len(s.words) for s in transcript.segments),
        )


class ClipEntry(BaseModel):
    """One clip with its persona-specific variant batch."""

    clip: Clip
    persona_id: str
    variants: list[Variant] = Field(default_factory=list)


class LLMSpend(BaseModel):
    """Cumulative roll-up of `llm_calls.jsonl` for this stream."""

    total_calls: int = 0
    total_cost_usd_micros: int = 0


class StreamManifest(BaseModel):
    """Full pipeline output, saved to `<stream_dir>/manifest.json`."""

    schema_version: int = _MANIFEST_SCHEMA_VERSION
    started_at: str
    completed_at: str
    tenant_id: str
    persona_id: str
    persona_name: str
    language: str
    n_variants_requested: int
    stream: Stream
    transcript: TranscriptSummary
    candidates: list[Candidate] = Field(default_factory=list)
    clip_entries: list[ClipEntry] = Field(default_factory=list)
    llm_spend: LLMSpend = Field(default_factory=LLMSpend)


@dataclass
class PipelineDeps:
    """Injectable dependencies — the CLI uses defaults, tests override."""

    config: NexoClipConfig | None = None
    llm_config: LLMConfig | None = None
    personas: dict[str, Persona] | None = None
    settings: Settings | None = None
    router_factory: Callable[[Path], LLMRouter] | None = None
    clock: Callable[[], _dt.datetime] = field(
        default_factory=lambda: lambda: _dt.datetime.now(_dt.UTC)
    )


async def process_vod(
    tenant_id: str,
    vod_url: str,
    output_dir: Path,
    *,
    persona_id: str,
    stream_id: str | None = None,
    language: str | None = None,
    n_variants: int = 5,
    quality: Quality | None = None,
    force: bool = False,
    db_path: str | None = None,
    chat_replay_source: Path | None = None,
    deps: PipelineDeps | None = None,
) -> StreamManifest:
    """Run the full Phase 0 pipeline against `vod_url`. Returns the manifest.

    Pass `stream_id` to resume a known stream; ingest will reuse the cached
    `stream.json` instead of minting a new ULID, and every downstream step
    finds its idempotency cache.

    Pass `db_path` to dual-write through the SQLite tenancy + repos. The
    tenant must already exist in the DB (use `nexoclip tenants add`).
    Without `db_path`, the pipeline runs filesystem-only — useful for
    tests and Phase 0-style invocations.

    Pass `chat_replay_source` (a path to a JSONL of `ChatMessage` rows)
    to feed the chat heat detector. Phase 1 doesn't fetch chat replay
    from platforms automatically — Phase 2 will add Kick / Twitch / YT
    fetchers behind the same import path.
    """
    deps = deps or PipelineDeps()
    config = deps.config or load_config()
    llm_config = deps.llm_config or load_llm_config()
    personas = deps.personas if deps.personas is not None else load_personas()
    settings = deps.settings or get_settings()

    if persona_id not in personas:
        known = ", ".join(sorted(personas)) or "(none)"
        raise VariantError(f"unknown persona {persona_id!r}; known: {known}")
    persona = personas[persona_id]

    started_at = deps.clock().isoformat()
    output_dir = Path(output_dir).resolve()

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(tenant_id=tenant_id, persona_id=persona.id)
    try:
        async with _maybe_db_session(tenant_id=tenant_id, db_path=db_path) as db:
            return await _run_pipeline(
                tenant_id=tenant_id,
                vod_url=vod_url,
                output_dir=output_dir,
                persona=persona,
                stream_id=stream_id,
                language=language,
                n_variants=n_variants,
                quality=quality,
                force=force,
                chat_replay_source=chat_replay_source,
                started_at=started_at,
                config=config,
                llm_config=llm_config,
                settings=settings,
                deps=deps,
                db=db,
            )
    finally:
        structlog.contextvars.clear_contextvars()


@asynccontextmanager
async def _maybe_db_session(
    *, tenant_id: str, db_path: str | None
) -> AsyncIterator[Database | None]:
    """Yield an open Database (with tenant bound) or None when no path is given."""
    if db_path is None:
        yield None
    else:
        async with db_session(tenant_id=tenant_id, db_path=db_path) as db:
            yield db


async def _run_pipeline(
    *,
    tenant_id: str,
    vod_url: str,
    output_dir: Path,
    persona: Persona,
    stream_id: str | None,
    language: str | None,
    n_variants: int,
    quality: Quality | None,
    force: bool,
    chat_replay_source: Path | None,
    started_at: str,
    config: NexoClipConfig,
    llm_config: LLMConfig,
    settings: Settings,
    deps: PipelineDeps,
    db: Database | None,
) -> StreamManifest:
    # Bind stream_id BEFORE the ingest step so its events carry stream_id
    # in the payload and the dashboard's progress card can find them. (For
    # the URL path where stream_id is None up-front, we re-bind below
    # using the id ingest_vod minted.)
    if stream_id:
        structlog.contextvars.bind_contextvars(stream_id=stream_id)

    # 1) ingest
    with _step("ingest", db=db, vod_url=vod_url):
        stream = await ingest_vod(
            tenant_id=tenant_id,
            vod_url=vod_url,
            output_dir=output_dir,
            stream_id=stream_id,
            force=force,
            chat_replay_source=chat_replay_source,
        )
        if db is not None:
            await StreamsRepo(db).upsert(stream_to_row(stream))
            await emit(
                db,
                STREAM_CREATED,
                {
                    "stream_id": stream.id,
                    "vod_url": stream.vod_url,
                    "platform": stream.platform,
                    "duration_s": stream.duration_s,
                },
            )

    structlog.contextvars.bind_contextvars(stream_id=stream.id)
    stream_dir = output_dir / stream.id
    call_log_path = stream_dir / "llm_calls.jsonl"

    # 2a) analyze video — local CV pipeline. Skip silently if:
    #   (a) the visual signals it produces aren't consumed by anything
    #       (visual detector disabled + no vision_rescore in the default
    #       pipeline) — pure wasted CPU per stream, and PySceneDetect on
    #       a multi-hour VOD on CPU can take longer than every other
    #       step combined.
    #   (b) the video file isn't decodable (test stubs, corrupted
    #       downloads — DetectionError) — log and continue.
    with _step("analyze_video", db=db):
        if not config.detection.visual.enabled:
            _log.info(
                "analyze_video.skipped",
                reason="visual detector disabled; nothing downstream consumes the output",
            )
        else:
            try:
                await _analyze_video(
                    tenant_id=tenant_id,
                    stream=stream,
                    output_dir=output_dir,
                    db=db,
                    force=force,
                )
            except DetectionError as e:
                _log.warning("analyze_video.skipped", reason=str(e))

    # 2) transcribe
    whisper_lang = language or "es"
    with _step("transcribe", db=db, model=settings.whisper_model, device=settings.whisper_device):
        transcript = await transcribe(
            tenant_id=tenant_id,
            stream=stream,
            model_size=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            language=whisper_lang,
            force=force,
        )
        if db is not None:
            await TranscriptsRepo(db).upsert(transcript_to_row(transcript))

    # 3) detect (also saves candidates.json). Build the router up-front so the
    # viral detector (LLM-based, runs inside this step when enabled) can use
    # it; the variants step at the bottom reuses the same router.
    router = (
        deps.router_factory(stream_dir)
        if deps.router_factory
        else LLMRouter(config=llm_config, call_log_path=call_log_path, db=db)
    )
    with _step("detect", db=db):
        chat_replay = load_chat_replay(stream_dir, stream_id=stream.id, tenant_id=tenant_id)
        visual_track = load_visual_signals(stream_dir)
        viral_cands: list[Candidate] = []
        if config.detection.viral.enabled:
            try:
                viral_cands = await detect_viral_moments(
                    tenant_id=tenant_id,
                    stream=stream,
                    transcript=transcript,
                    router=router,
                    config=config.detection.viral,
                )
            except Exception as e:  # noqa: BLE001 - belt-and-suspenders
                # detect_viral_moments already swallows expected errors;
                # this catches anything we missed so the pipeline keeps moving.
                _log.warning("viral.skipped", reason=str(e), stream_id=stream.id)
        candidates = detect_candidates(
            tenant_id=tenant_id,
            stream=stream,
            transcript=transcript,
            config=config.detection,
            chat_replay=chat_replay,
            visual_track=visual_track,
            viral_candidates=viral_cands,
        )
        save_candidates(
            stream_dir,
            CandidateBatch(stream_id=stream.id, tenant_id=tenant_id, candidates=candidates),
        )
        if db is not None:
            candidate_rows = [
                candidate_to_row(c, stream_id=stream.id, tenant_id=tenant_id) for c in candidates
            ]
            await CandidatesRepo(db).upsert_many(candidate_rows)
    _log.info("detect.candidates", count=len(candidates))

    # 4) cut clips
    with _step("cut", db=db, candidate_count=len(candidates)):
        clips = await cut_clips(
            tenant_id=tenant_id,
            stream=stream,
            candidates=candidates,
            output_dir=output_dir,
            config=config.clip,
            force=force,
        )
        if db is not None:
            await ClipsRepo(db).upsert_many([clip_to_row(c) for c in clips])

    # 5) variants per clip — reuse the router built above for detect+viral.
    if db is not None:
        # Persona must exist in DB so variants can FK to it.
        await PersonasRepo(db).upsert(
            persona_id=persona.id,
            name=persona.name,
            primary_language=persona.primary_language,
            target_languages=persona.target_languages,
            voice_prompt=persona.voice_prompt,
            routing_tags=persona.routing_tags,
        )
    clip_entries: list[ClipEntry] = []
    with _step("variants", db=db, clip_count=len(clips), n=n_variants):
        for clip in clips:
            variants = await generate_variants(
                tenant_id=tenant_id,
                clip=clip,
                persona=persona,
                router=router,
                n=n_variants,
                language=language,
                quality=quality,
                force=force,
            )
            clip_entries.append(ClipEntry(clip=clip, persona_id=persona.id, variants=variants))
            if db is not None:
                variant_rows = [
                    variant_to_row(
                        v,
                        clip_id=clip.id,
                        tenant_id=tenant_id,
                        persona_id=persona.id,
                    )
                    for v in variants
                ]
                await VariantsRepo(db).replace_for_clip_persona(clip.id, persona.id, variant_rows)
                await emit(
                    db,
                    CLIP_READY_FOR_REVIEW,
                    {
                        "clip_id": clip.id,
                        "stream_id": stream.id,
                        "persona_id": persona.id,
                        "variant_count": len(variants),
                    },
                )

    # 6) manifest
    manifest = StreamManifest(
        started_at=started_at,
        completed_at=deps.clock().isoformat(),
        tenant_id=tenant_id,
        persona_id=persona.id,
        persona_name=persona.name,
        language=language or persona.primary_language,
        n_variants_requested=n_variants,
        stream=stream,
        transcript=TranscriptSummary.from_transcript(transcript),
        candidates=candidates,
        clip_entries=clip_entries,
        llm_spend=_read_llm_spend(call_log_path),
    )
    manifest_path = stream_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    _log.info(
        "pipeline.done",
        clips=len(clip_entries),
        llm_calls=manifest.llm_spend.total_calls,
        cost_usd_micros=manifest.llm_spend.total_cost_usd_micros,
    )
    if db is not None:
        await emit(
            db,
            STREAM_PROCESSED,
            {
                "stream_id": stream.id,
                "clip_count": len(clip_entries),
                "llm_calls": manifest.llm_spend.total_calls,
                "cost_usd_micros": manifest.llm_spend.total_cost_usd_micros,
            },
        )
    return manifest


def _read_llm_spend(call_log_path: Path) -> LLMSpend:
    """Sum cost across every row in `llm_calls.jsonl` for this stream."""
    if not call_log_path.exists():
        return LLMSpend()
    total_calls = 0
    total_cost = 0
    for line in call_log_path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        total_calls += 1
        total_cost += int(row.get("cost_usd_micros", 0) or 0)
    return LLMSpend(total_calls=total_calls, total_cost_usd_micros=total_cost)


def load_manifest(stream_dir: Path) -> StreamManifest:
    """Read the manifest back from disk."""
    path = Path(stream_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    return StreamManifest.model_validate_json(path.read_text("utf-8"))
