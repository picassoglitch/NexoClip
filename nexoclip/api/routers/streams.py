"""Stream + candidate + clip-listing endpoints.

`POST /streams` is the kick-off point - it ingests the VOD synchronously
(needed to know stream metadata + on-disk paths), persists the row, then
schedules the rest of `process_vod` as a FastAPI BackgroundTask. The
ingest step itself is fast (yt-dlp metadata fetch); the heavy work is
Whisper + vision + variants which fire after the response returns.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from nexoclip.db import (
    CandidatesRepo,
    ClipsRepo,
    Database,
    EventsRepo,
    StreamsRepo,
)
from nexoclip.errors import NexoClipError

from .._pipeline import PipelineKickoff
from ..deps import get_db, require_full_scope, tenant_binder
from ..schemas import (
    CandidateResponse,
    ClipResponse,
    StreamCreateRequest,
    StreamResponse,
)

router = APIRouter(prefix="/streams", tags=["streams"])


@router.post(
    "",
    response_model=StreamResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_full_scope)],
)
async def create_stream(
    payload: StreamCreateRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> StreamResponse:
    """Ingest a VOD and schedule the rest of the pipeline in the background.

    The ingest step (yt-dlp metadata + audio/video download) runs inline so
    the response can carry the resulting Stream id. Transcription, detection,
    cutting, and variant generation fire as a BackgroundTask after the
    response goes out.
    """
    # Imported here to avoid pulling yt-dlp into module-load time.
    from nexoclip.db.adapters import stream_to_row
    from nexoclip.ingest import ingest_vod
    from nexoclip.settings import get_settings

    output_dir = Path(get_settings().default_output_dir)
    try:
        stream = await ingest_vod(
            vod_url=payload.vod_url,
            tenant_id=tenant_id,
            output_dir=output_dir,
        )
    except NexoClipError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    row = await StreamsRepo(db).upsert(stream_to_row(stream))
    await EventsRepo(db).emit(type="stream.created", payload={"stream_id": row.id})

    runner = request.app.state.pipeline_runner
    kickoff = PipelineKickoff(
        tenant_id=tenant_id,
        stream=stream,
        persona_id=payload.persona_id,
        output_dir=output_dir,
        language=payload.language,
    )
    background_tasks.add_task(runner, kickoff)
    return StreamResponse.model_validate(row.model_dump())


@router.get("", response_model=list[StreamResponse])
async def list_streams(
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> list[StreamResponse]:
    rows = await StreamsRepo(db).list_for_tenant()
    return [StreamResponse.model_validate(r.model_dump()) for r in rows]


@router.get("/{stream_id}", response_model=StreamResponse)
async def get_stream(
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> StreamResponse:
    row = await StreamsRepo(db).get(stream_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="stream not found")
    return StreamResponse.model_validate(row.model_dump())


@router.get("/{stream_id}/candidates", response_model=list[CandidateResponse])
async def list_candidates(
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> list[CandidateResponse]:
    # Stream existence check ensures we 404 instead of returning [] for
    # other-tenant ids - tighter contract for the dashboard.
    if await StreamsRepo(db).get(stream_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="stream not found")
    rows = await CandidatesRepo(db).list_for_stream(stream_id)
    return [CandidateResponse.model_validate(r.model_dump()) for r in rows]


@router.get("/{stream_id}/clips", response_model=list[ClipResponse])
async def list_clips(
    stream_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> list[ClipResponse]:
    if await StreamsRepo(db).get(stream_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="stream not found")
    rows = await ClipsRepo(db).list_for_stream(stream_id)
    return [ClipResponse.model_validate(r.model_dump()) for r in rows]
