"""Clip detail + status update + publish enqueue."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from nexoclip.db import (
    ClipsRepo,
    ConnectedAccountsRepo,
    Database,
    EventsRepo,
    PublishJobsRepo,
    VariantsRepo,
)
from nexoclip.errors import TenancyError

from ..deps import get_db, require_full_scope, tenant_binder
from ..status_gate import require_active_tenant, require_top_tier
from ..schemas import (
    ClipResponse,
    ClipUpdateRequest,
    PublishJobResponse,
    PublishRequest,
)

router = APIRouter(prefix="/clips", tags=["clips"])

# clips.status moves through: cut -> ready_for_review -> approved | rejected
# -> (publish) -> published. The dashboard drives transitions; the worker
# (Task 11) writes `published` after a successful Buffer post.
_VALID_STATUS_TRANSITIONS: dict[str, set[str]] = {
    "cut": {"ready_for_review", "rejected"},
    "ready_for_review": {"approved", "rejected"},
    "approved": {"rejected", "published"},
    "rejected": {"ready_for_review"},
    "published": set(),
}


@router.get("/{clip_id}", response_model=ClipResponse)
async def get_clip(
    clip_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> ClipResponse:
    row = await ClipsRepo(db).get(clip_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="clip not found")
    return ClipResponse.model_validate(row.model_dump())


@router.patch(
    "/{clip_id}",
    response_model=ClipResponse,
    dependencies=[Depends(require_full_scope)],
)
async def update_clip(
    clip_id: str,
    payload: ClipUpdateRequest,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> ClipResponse:
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="clip not found")

    if payload.status is not None and payload.status != clip.status:
        allowed = _VALID_STATUS_TRANSITIONS.get(clip.status, set())
        if payload.status not in allowed:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"cannot transition clip from {clip.status!r} to {payload.status!r}",
            )
        # Direct UPDATE - the repo's bulk upserts are insert-only by design,
        # so a one-off status change goes via SQL here.
        conn = await db.connect()
        await conn.execute(
            "UPDATE clips SET status = ? WHERE id = ? AND tenant_id = ?",
            (payload.status, clip_id, tenant_id),
        )
        await conn.commit()
        await EventsRepo(db).emit(
            type=f"clip.{payload.status}",
            payload={"clip_id": clip_id, "from": clip.status, "to": payload.status},
        )

    refreshed = await ClipsRepo(db).get(clip_id)
    assert refreshed is not None
    return ClipResponse.model_validate(refreshed.model_dump())


@router.post(
    "/{clip_id}/publish",
    response_model=list[PublishJobResponse],
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[
        Depends(require_full_scope),
        Depends(require_active_tenant),
        Depends(require_top_tier),
    ],
)
async def publish_clip(
    clip_id: str,
    payload: PublishRequest,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> list[PublishJobResponse]:
    """Enqueue one publish_job per connected account for this clip + variant."""
    clip = await ClipsRepo(db).get(clip_id)
    if clip is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="clip not found")

    variants = await VariantsRepo(db).list_for_clip(clip_id)
    variant = next((v for v in variants if v.id == payload.variant_id), None)
    if variant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"variant {payload.variant_id!r} not found for this clip",
        )

    accounts = await ConnectedAccountsRepo(db).list_for_tenant()
    if payload.account_ids is not None:
        wanted = set(payload.account_ids)
        accounts = [a for a in accounts if a.id in wanted]
        missing = wanted - {a.id for a in accounts}
        if missing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown account ids: {sorted(missing)}",
            )
    if not accounts:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="no connected accounts to publish to",
        )

    jobs_repo = PublishJobsRepo(db)
    created: list[PublishJobResponse] = []
    try:
        for account in accounts:
            job = await jobs_repo.enqueue(
                clip_id=clip_id,
                variant_id=variant.id,
                account_id=account.id,
                platform=account.platform,
            )
            created.append(PublishJobResponse.model_validate(job.model_dump()))
    except TenancyError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e

    await EventsRepo(db).emit(
        type="clip.publish_requested",
        payload={"clip_id": clip_id, "variant_id": variant.id, "n_jobs": len(created)},
    )
    return created
