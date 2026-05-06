"""NexoClip CLI entry point.

Phase 0 commands (see PHASE_0.md):
    nexoclip ingest <vod_url>
    nexoclip transcribe <stream_id>
    nexoclip detect <stream_id>
    nexoclip cut <stream_id>
    nexoclip variants <clip_id> --persona <persona_id>
    nexoclip process <vod_url>          # orchestrates all of the above

Phase 1 admin commands (see PHASE_1.md):
    nexoclip db init
    nexoclip tenants add <id> "<name>"
    nexoclip tenants list
    nexoclip tokens issue --tenant <id> [--scope full|read]
    nexoclip tokens list   --tenant <id>
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from nexoclip.db import Database
    from nexoclip.db.models import ApiTokenRow, Tenant

app = typer.Typer(
    name="nexoclip",
    help="VOD-to-clips pipeline.",
    no_args_is_help=True,
)

db_app = typer.Typer(name="db", help="Database admin commands.", no_args_is_help=True)
tenants_app = typer.Typer(name="tenants", help="Tenant management.", no_args_is_help=True)
tokens_app = typer.Typer(name="tokens", help="API token management.", no_args_is_help=True)
app.add_typer(db_app)
app.add_typer(tenants_app)
app.add_typer(tokens_app)


@app.callback()
def _root(
    log_level: str | None = typer.Option(
        None, "--log-level", help="DEBUG | INFO | WARNING | ERROR (overrides env)."
    ),
    log_format: str | None = typer.Option(
        None, "--log-format", help="`console` (default) or `json` for machine logs."
    ),
) -> None:
    """Configure logging once before any subcommand runs."""
    from nexoclip.logging import configure_logging
    from nexoclip.settings import get_settings

    settings = get_settings()
    configure_logging(
        level=log_level or settings.log_level,
        fmt=log_format or settings.log_format,
    )


@app.command()
def version() -> None:
    """Print the installed version."""
    from nexoclip import __version__

    typer.echo(__version__)


@app.command()
def ingest(
    vod_url: str = typer.Argument(..., help="VOD URL (Kick / Twitch / YouTube)"),
    output_dir: Path = typer.Option(
        Path("./out"), "--output-dir", "-o", help="Root directory for stream artifacts."
    ),
    tenant_id: str = typer.Option(
        "default", "--tenant-id", help="Tenant owning the stream (Phase 0: hardcoded)."
    ),
    stream_id: str | None = typer.Option(
        None, "--stream-id", help="Resume an existing stream (skips ID generation)."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-download and re-extract even if outputs exist."
    ),
    chat_replay: Path | None = typer.Option(
        None,
        "--chat-replay",
        help="Path to a JSONL of chat messages (Phase 1 doesn't fetch from platforms).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the Stream as JSON to stdout."),
) -> None:
    """Download a VOD and extract its audio. Idempotent."""
    from nexoclip.errors import IngestError
    from nexoclip.ingest import ingest_vod

    try:
        stream = asyncio.run(
            ingest_vod(
                tenant_id=tenant_id,
                vod_url=vod_url,
                output_dir=output_dir,
                stream_id=stream_id,
                force=force,
                chat_replay_source=chat_replay,
            )
        )
    except IngestError as e:
        typer.echo(f"ingest failed: {e}", err=True)
        raise typer.Exit(code=1) from e

    if json_output:
        typer.echo(stream.model_dump_json(indent=2))
        return

    typer.echo(f"stream_id:  {stream.id}")
    typer.echo(f"  platform: {stream.platform}")
    typer.echo(f"  title:    {stream.title or '(unknown)'}")
    typer.echo(f"  channel:  {stream.channel or '(unknown)'}")
    typer.echo(f"  duration: {stream.duration_s:.1f}s")
    typer.echo(f"  video:    {stream.source_video_path}")
    typer.echo(f"  audio:    {stream.source_audio_path}")


@app.command()
def transcribe(
    stream_id: str = typer.Argument(..., help="Stream ID produced by `nexoclip ingest`."),
    output_dir: Path = typer.Option(
        Path("./out"), "--output-dir", "-o", help="Root directory holding `<stream_id>/`."
    ),
    tenant_id: str | None = typer.Option(
        None, "--tenant-id", help="Override tenant; defaults to NEXOCLIP_DEFAULT_TENANT_ID."
    ),
    model_size: str | None = typer.Option(
        None, "--model", help="Whisper model size (overrides NEXOCLIP_WHISPER_MODEL)."
    ),
    device: str | None = typer.Option(
        None, "--device", help="`cuda` or `cpu` (overrides NEXOCLIP_WHISPER_DEVICE)."
    ),
    compute_type: str | None = typer.Option(
        None, "--compute-type", help="e.g. `float16`, `int8` (overrides env)."
    ),
    language: str = typer.Option("es", "--language", help="ISO 639-1 code, or `auto`."),
    force: bool = typer.Option(
        False, "--force", help="Re-transcribe even if `transcript.json` exists."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the full Transcript as JSON."),
) -> None:
    """Run Whisper on an ingested stream and write `transcript.json`."""
    from nexoclip.errors import IngestError, TranscriptionError
    from nexoclip.ingest import load_stream
    from nexoclip.settings import get_settings
    from nexoclip.transcribe import transcribe as run_transcribe

    settings = get_settings()
    stream_dir = Path(output_dir).resolve() / stream_id

    try:
        stream = load_stream(stream_dir)
        transcript = asyncio.run(
            run_transcribe(
                tenant_id=tenant_id or settings.default_tenant_id,
                stream=stream,
                model_size=model_size or settings.whisper_model,
                device=device or settings.whisper_device,
                compute_type=compute_type or settings.whisper_compute_type,
                language=None if language == "auto" else language,
                force=force,
            )
        )
    except (IngestError, TranscriptionError) as e:
        typer.echo(f"transcribe failed: {e}", err=True)
        raise typer.Exit(code=1) from e

    if json_output:
        typer.echo(transcript.model_dump_json(indent=2))
        return

    typer.echo(f"stream_id:  {transcript.stream_id}")
    typer.echo(f"  model:    {transcript.model}")
    typer.echo(f"  language: {transcript.language}")
    typer.echo(f"  duration: {transcript.duration_s:.1f}s")
    typer.echo(f"  segments: {len(transcript.segments)}")
    word_count = sum(len(s.words) for s in transcript.segments)
    typer.echo(f"  words:    {word_count}")


@app.command(name="analyze-video")
def analyze_video_cmd(
    stream_id: str = typer.Argument(..., help="Stream ID produced by `nexoclip ingest`."),
    output_dir: Path = typer.Option(
        Path("./out"), "--output-dir", "-o", help="Root directory holding `<stream_id>/`."
    ),
    tenant_id: str | None = typer.Option(
        None, "--tenant-id", help="Override tenant; defaults to NEXOCLIP_DEFAULT_TENANT_ID."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-run even when `visual_signals.json` exists."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the VisualSignalTrack as JSON."),
) -> None:
    """Run the local vision pipeline (scene cuts + motion + face/emotion)."""
    from nexoclip.errors import DetectionError, IngestError
    from nexoclip.ingest import load_stream
    from nexoclip.settings import get_settings
    from nexoclip.vision import analyze_video

    settings = get_settings()
    stream_dir = Path(output_dir).resolve() / stream_id
    effective_tenant = tenant_id or settings.default_tenant_id

    try:
        stream = load_stream(stream_dir)
        track = asyncio.run(
            analyze_video(
                tenant_id=effective_tenant,
                stream=stream,
                output_dir=Path(output_dir).resolve(),
                force=force,
            )
        )
    except (IngestError, DetectionError) as e:
        typer.echo(f"analyze-video failed: {e}", err=True)
        raise typer.Exit(code=1) from e

    if json_output:
        typer.echo(track.model_dump_json(indent=2))
        return

    cut_count = sum(1 for s in track.signals if s.scene_cut)
    smile_count = sum(1 for s in track.signals if s.face_emotion == "smile")
    motion_max = max(
        (s.motion_energy for s in track.signals if s.motion_energy is not None),
        default=0.0,
    )
    typer.echo(f"stream_id:  {track.stream_id}")
    typer.echo(f"  seconds:  {len(track.signals)}")
    typer.echo(f"  cuts:     {cut_count}")
    typer.echo(f"  smiles:   {smile_count}")
    typer.echo(f"  motion (max): {motion_max:.4f}")


@app.command()
def detect(
    stream_id: str = typer.Argument(..., help="Stream ID produced by `nexoclip ingest`."),
    output_dir: Path = typer.Option(
        Path("./out"), "--output-dir", "-o", help="Root directory holding `<stream_id>/`."
    ),
    tenant_id: str | None = typer.Option(
        None, "--tenant-id", help="Override tenant; defaults to NEXOCLIP_DEFAULT_TENANT_ID."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", help="Path to nexoclip.yaml (defaults to config/nexoclip.yaml)."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print candidates as JSON."),
) -> None:
    """Detect candidates (voice + chat) in an already-transcribed stream."""
    from nexoclip.config import load_config
    from nexoclip.detect import detect_candidates, save_candidates
    from nexoclip.detect.models import CandidateBatch
    from nexoclip.errors import DetectionError, IngestError, TranscriptionError
    from nexoclip.ingest import load_chat_replay, load_stream
    from nexoclip.settings import get_settings
    from nexoclip.transcribe import load_transcript

    settings = get_settings()
    stream_dir = Path(output_dir).resolve() / stream_id
    effective_tenant = tenant_id or settings.default_tenant_id

    try:
        stream = load_stream(stream_dir)
        transcript = load_transcript(stream_dir)
        config = load_config(config_path)
        chat_replay = load_chat_replay(stream_dir, stream_id=stream.id, tenant_id=effective_tenant)
        candidates = detect_candidates(
            tenant_id=effective_tenant,
            stream=stream,
            transcript=transcript,
            config=config.detection,
            chat_replay=chat_replay,
        )
    except (IngestError, TranscriptionError, DetectionError) as e:
        typer.echo(f"detect failed: {e}", err=True)
        raise typer.Exit(code=1) from e

    batch = CandidateBatch(stream_id=stream.id, tenant_id=effective_tenant, candidates=candidates)
    save_candidates(stream_dir, batch)

    if json_output:
        typer.echo(batch.model_dump_json(indent=2))
        return

    typer.echo(f"stream_id:  {batch.stream_id}")
    typer.echo(f"  candidates: {len(candidates)}")
    for c in candidates:
        phrase = c.evidence.get("phrase", "?")
        snippet = c.evidence.get("transcript_snippet", "")
        typer.echo(
            f"  [{c.timestamp:>7.1f}s]  score={c.score:.3f}  phrase={phrase!r}  snippet={snippet!r}"
        )


@app.command()
def cut(
    stream_id: str = typer.Argument(..., help="Stream ID produced by `nexoclip ingest`."),
    output_dir: Path = typer.Option(
        Path("./out"), "--output-dir", "-o", help="Root directory holding `<stream_id>/`."
    ),
    tenant_id: str | None = typer.Option(
        None, "--tenant-id", help="Override tenant; defaults to NEXOCLIP_DEFAULT_TENANT_ID."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", help="Path to nexoclip.yaml (defaults to config/nexoclip.yaml)."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-cut every clip even if `clips_manifest.json` exists."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the manifest as JSON."),
) -> None:
    """Cut + 9:16 reformat one clip per detected candidate."""
    from nexoclip.clip import cut_clips
    from nexoclip.clip.models import ClipManifest
    from nexoclip.config import load_config
    from nexoclip.detect import load_candidates
    from nexoclip.errors import ClipError, DetectionError, IngestError
    from nexoclip.ingest import load_stream
    from nexoclip.settings import get_settings

    settings = get_settings()
    stream_dir = Path(output_dir).resolve() / stream_id
    effective_tenant = tenant_id or settings.default_tenant_id

    try:
        stream = load_stream(stream_dir)
        batch = load_candidates(stream_dir)
        config = load_config(config_path)
        clips = asyncio.run(
            cut_clips(
                tenant_id=effective_tenant,
                stream=stream,
                candidates=batch.candidates,
                output_dir=Path(output_dir).resolve(),
                config=config.clip,
                force=force,
            )
        )
    except (IngestError, DetectionError, ClipError) as e:
        typer.echo(f"cut failed: {e}", err=True)
        raise typer.Exit(code=1) from e

    manifest = ClipManifest(stream_id=stream.id, tenant_id=effective_tenant, clips=clips)

    if json_output:
        typer.echo(manifest.model_dump_json(indent=2))
        return

    typer.echo(f"stream_id:  {manifest.stream_id}")
    typer.echo(f"  clips:    {len(clips)}")
    for c in clips:
        typer.echo(f"  [{c.start_s:>7.1f}s..{c.end_s:>7.1f}s]  {c.id}  -> {c.path}")


@app.command()
def variants(
    clip_id: str = typer.Argument(..., help="Clip ID produced by `nexoclip cut`."),
    persona: str = typer.Option(..., "--persona", help="Persona id from personas.yaml."),
    output_dir: Path = typer.Option(
        Path("./out"), "--output-dir", "-o", help="Root directory holding stream folders."
    ),
    tenant_id: str | None = typer.Option(
        None, "--tenant-id", help="Override tenant; defaults to NEXOCLIP_DEFAULT_TENANT_ID."
    ),
    n: int = typer.Option(5, "--n", min=1, help="How many variants to generate."),
    language: str | None = typer.Option(
        None, "--language", help="ISO 639-1 code; defaults to persona's primary language."
    ),
    quality: str | None = typer.Option(
        None,
        "--quality",
        help="Override default quality (`standard` or `premium`).",
    ),
    llm_config_path: Path | None = typer.Option(
        None, "--llm-config", help="Path to llm.yaml (defaults to config/llm.yaml)."
    ),
    personas_path: Path | None = typer.Option(None, "--personas", help="Path to personas.yaml."),
    force: bool = typer.Option(
        False, "--force", help="Re-generate even if `variants.json` already exists."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print variants as JSON."),
) -> None:
    """Generate N caption variants for a clip in a persona's voice."""
    from nexoclip.errors import LLMError, VariantError
    from nexoclip.llm import LLMRouter, load_llm_config
    from nexoclip.llm.config import Quality
    from nexoclip.settings import get_settings
    from nexoclip.variants import find_clip, generate_variants, get_persona

    settings = get_settings()
    effective_tenant = tenant_id or settings.default_tenant_id
    quality_arg: Quality | None = None
    if quality is not None:
        if quality not in ("standard", "premium"):
            typer.echo(f"--quality must be 'standard' or 'premium', got {quality!r}", err=True)
            raise typer.Exit(code=2)
        quality_arg = quality  # type: ignore[assignment]

    try:
        clip, clip_dir, stream_dir = find_clip(clip_id, Path(output_dir).resolve())
        persona_obj = get_persona(persona, path=personas_path)
        llm_config = load_llm_config(llm_config_path)
        router = LLMRouter(
            config=llm_config,
            call_log_path=stream_dir / "llm_calls.jsonl",
        )
        result = asyncio.run(
            generate_variants(
                tenant_id=effective_tenant,
                clip=clip,
                persona=persona_obj,
                router=router,
                n=n,
                language=language,
                quality=quality_arg,
                clip_dir=clip_dir,
                force=force,
            )
        )
    except (VariantError, LLMError) as e:
        typer.echo(f"variants failed: {e}", err=True)
        raise typer.Exit(code=1) from e

    if json_output:
        from nexoclip.variants.models import VariantsFile

        payload = VariantsFile(
            clip_id=clip.id,
            tenant_id=effective_tenant,
            persona_id=persona_obj.id,
            persona_name=persona_obj.name,
            language=language or persona_obj.primary_language,
            variants=result,
        )
        typer.echo(payload.model_dump_json(indent=2))
        return

    typer.echo(f"clip:    {clip.id}")
    typer.echo(f"persona: {persona_obj.id} ({persona_obj.name})")
    typer.echo(f"variants ({len(result)}):")
    for v in result:
        hashtags = " ".join(f"#{h}" for h in v.hashtags) if v.hashtags else ""
        title = f"  [{v.title_card_text}]" if v.title_card_text else ""
        typer.echo(f"  - {v.id} ({v.language}){title} {v.caption} {hashtags}".rstrip())


@app.command()
def process(
    vod_url: str = typer.Argument(..., help="VOD URL (Kick / Twitch / YouTube)"),
    persona: str = typer.Option(..., "--persona", help="Persona id from personas.yaml."),
    output_dir: Path = typer.Option(
        Path("./out"), "--output-dir", "-o", help="Root directory for stream artifacts."
    ),
    tenant_id: str | None = typer.Option(
        None, "--tenant-id", help="Override tenant; defaults to NEXOCLIP_DEFAULT_TENANT_ID."
    ),
    stream_id: str | None = typer.Option(
        None, "--stream-id", help="Resume an existing stream (skips ID generation)."
    ),
    language: str | None = typer.Option(
        None, "--language", help="ISO 639-1 code; defaults to persona's primary language."
    ),
    n: int = typer.Option(5, "--n", min=1, help="How many variants to generate per clip."),
    quality: str | None = typer.Option(
        None, "--quality", help="Override default quality (`standard` or `premium`)."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-run every step even when its output exists."
    ),
    no_db: bool = typer.Option(
        False, "--no-db", help="Skip dual-write to SQLite (filesystem only)."
    ),
    db_path: Path | None = typer.Option(
        None, "--db-path", help="Override NEXOCLIP_DB_PATH for this run."
    ),
    chat_replay: Path | None = typer.Option(
        None,
        "--chat-replay",
        help="Path to a JSONL of chat messages, fed to the chat heat detector.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the full manifest as JSON when complete."
    ),
) -> None:
    """Run the full ingest -> transcribe -> detect -> cut -> variants pipeline."""
    from nexoclip.errors import (
        ClipError,
        DetectionError,
        IngestError,
        LLMError,
        TenancyError,
        TranscriptionError,
        VariantError,
    )
    from nexoclip.llm.config import Quality
    from nexoclip.pipeline import process_vod
    from nexoclip.settings import get_settings

    settings = get_settings()
    effective_tenant = tenant_id or settings.default_tenant_id
    quality_arg: Quality | None = None
    if quality is not None:
        if quality not in ("standard", "premium"):
            typer.echo(f"--quality must be 'standard' or 'premium', got {quality!r}", err=True)
            raise typer.Exit(code=2)
        quality_arg = quality  # type: ignore[assignment]

    effective_db_path = None if no_db else (str(db_path) if db_path else settings.db_path)

    try:
        manifest = asyncio.run(
            process_vod(
                tenant_id=effective_tenant,
                vod_url=vod_url,
                output_dir=Path(output_dir).resolve(),
                persona_id=persona,
                stream_id=stream_id,
                language=language,
                n_variants=n,
                quality=quality_arg,
                force=force,
                db_path=effective_db_path,
                chat_replay_source=chat_replay,
            )
        )
    except (
        IngestError,
        TranscriptionError,
        DetectionError,
        ClipError,
        VariantError,
        LLMError,
        TenancyError,
    ) as e:
        typer.echo(f"process failed: {e}", err=True)
        raise typer.Exit(code=1) from e

    if json_output:
        typer.echo(manifest.model_dump_json(indent=2))
        return

    typer.echo(f"stream_id:  {manifest.stream.id}")
    typer.echo(f"  persona:  {manifest.persona_id} ({manifest.persona_name})")
    typer.echo(f"  language: {manifest.language}")
    typer.echo(f"  duration: {manifest.stream.duration_s:.1f}s")
    typer.echo(f"  candidates: {len(manifest.candidates)}")
    typer.echo(f"  clips:    {len(manifest.clip_entries)}")
    typer.echo(
        f"  llm:      {manifest.llm_spend.total_calls} calls, "
        f"${manifest.llm_spend.total_cost_usd_micros / 1_000_000:.4f}"
    )
    for entry in manifest.clip_entries:
        c = entry.clip
        typer.echo(
            f"  - {c.id}  [{c.start_s:>7.1f}s..{c.end_s:>7.1f}s]  {len(entry.variants)} variants"
        )


# ---------------------------------------------------------------------------
# Phase 1 admin commands: db init, tenants add/list, tokens issue/list.
# ---------------------------------------------------------------------------


def _open_db(db_path: str | Path | None) -> Database:
    from nexoclip.db import Database
    from nexoclip.settings import get_settings

    return Database(Path(db_path) if db_path else Path(get_settings().db_path))


@db_app.command("init")
def db_init_cmd(
    db_path: Path | None = typer.Option(
        None, "--db-path", help="Override NEXOCLIP_DB_PATH for this command."
    ),
) -> None:
    """Apply Phase 1 migrations against the configured SQLite file."""
    from nexoclip.db import apply_migrations

    async def _run() -> int:
        db = _open_db(db_path)
        try:
            return await apply_migrations(db)
        finally:
            await db.close()

    version = asyncio.run(_run())
    typer.echo(f"schema_version = {version}")


@tenants_app.command("add")
def tenants_add_cmd(
    tenant_id: str = typer.Argument(..., help="Stable tenant id, e.g. `aldo`."),
    name: str = typer.Argument(..., help="Display name."),
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """Create a new tenant row."""
    from nexoclip.db import TenantsRepo, apply_migrations

    async def _run() -> str:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            repo = TenantsRepo(db)
            t = await repo.create(tenant_id=tenant_id, name=name)
            return t.id
        finally:
            await db.close()

    created = asyncio.run(_run())
    typer.echo(f"created tenant: {created}")


@tenants_app.command("list")
def tenants_list_cmd(
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """List all tenants."""
    from nexoclip.db import TenantsRepo, apply_migrations

    async def _run() -> list[Tenant]:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            return await TenantsRepo(db).list_all()
        finally:
            await db.close()

    tenants = asyncio.run(_run())
    if not tenants:
        typer.echo("(no tenants)")
        return
    for t in tenants:
        typer.echo(f"  {t.id}  {t.name}  ({t.created_at})")


@tokens_app.command("issue")
def tokens_issue_cmd(
    tenant_id: str = typer.Option(..., "--tenant", help="Tenant the token authenticates."),
    scope: str = typer.Option("full", "--scope", help="`full` or `read`."),
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """Mint a new API token. Prints the raw token ONCE -- store it now."""
    from nexoclip.db import ApiTokensRepo, TenantsRepo, apply_migrations
    from nexoclip.tenancy import bound_tenant, mint_token

    if scope not in ("full", "read"):
        typer.echo(f"--scope must be 'full' or 'read', got {scope!r}", err=True)
        raise typer.Exit(code=2)

    async def _run() -> str:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            t = await TenantsRepo(db).get(tenant_id)
            if t is None:
                raise typer.Exit(code=1)
            raw, hashed = mint_token()
            with bound_tenant(tenant_id):
                await ApiTokensRepo(db).create(hash_=hashed, scope=scope)
            return raw
        finally:
            await db.close()

    raw = asyncio.run(_run())
    typer.echo(raw)
    typer.echo(
        "(store this token now -- it is not retrievable again; only its sha256 "
        "hash is persisted in the database)",
        err=True,
    )


@tokens_app.command("list")
def tokens_list_cmd(
    tenant_id: str = typer.Option(..., "--tenant"),
    db_path: Path | None = typer.Option(None, "--db-path"),
) -> None:
    """List tokens for a tenant (hashes + scopes only -- no raw tokens)."""
    from nexoclip.db import ApiTokensRepo, apply_migrations
    from nexoclip.tenancy import bound_tenant

    async def _run() -> list[ApiTokenRow]:
        db = _open_db(db_path)
        try:
            await apply_migrations(db)
            with bound_tenant(tenant_id):
                return await ApiTokensRepo(db).list_for_tenant()
        finally:
            await db.close()

    rows = asyncio.run(_run())
    if not rows:
        typer.echo("(no tokens)")
        return
    for r in rows:
        last = r.last_used_at or "(never)"
        typer.echo(f"  {r.id}  scope={r.scope}  hash={r.hash[:16]}…  last_used={last}")


if __name__ == "__main__":
    app()
