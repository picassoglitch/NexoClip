"""Webhook subscription CRUD via the REST API.

All routes scoped to the bound tenant. Secret is generated server-side
on create and returned in the response *once* (mirrors api_tokens).
Subsequent reads never expose the secret.
"""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from nexoclip.db import Database, WebhookSubscriptionsRepo

from ..deps import get_db, require_full_scope, tenant_binder

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


class WebhookCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=1)
    types: list[str] = Field(default_factory=list)


class WebhookCreateResponse(BaseModel):
    """Response on create — includes the secret. Other reads omit it."""

    model_config = ConfigDict(extra="ignore")
    id: str
    url: str
    types: list[str]
    secret: str  # only returned at create time
    status: str
    created_at: str


class WebhookResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    url: str
    types: list[str]
    status: str
    created_at: str
    last_dispatch_ts: str | None = None
    failure_count: int = 0


@router.post(
    "",
    response_model=WebhookCreateResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_full_scope)],
)
async def create_webhook(
    payload: WebhookCreateRequest,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> WebhookCreateResponse:
    """Register a subscriber URL. Secret is minted now and shown once."""
    secret = secrets.token_hex(32)
    sub = await WebhookSubscriptionsRepo(db).create(
        url=payload.url, types=payload.types, secret=secret
    )
    return WebhookCreateResponse(
        id=sub.id,
        url=sub.url,
        types=sub.types,
        secret=secret,
        status=sub.status,
        created_at=sub.created_at,
    )


@router.get("", response_model=list[WebhookResponse])
async def list_webhooks(
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> list[WebhookResponse]:
    rows = await WebhookSubscriptionsRepo(db).list_for_tenant()
    return [
        WebhookResponse(
            id=r.id,
            url=r.url,
            types=r.types,
            status=r.status,
            created_at=r.created_at,
            last_dispatch_ts=r.last_dispatch_ts,
            failure_count=r.failure_count,
        )
        for r in rows
    ]


@router.delete(
    "/{sub_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_full_scope)],
)
async def delete_webhook(
    sub_id: str,
    tenant_id: str = Depends(tenant_binder),
    db: Database = Depends(get_db),
) -> None:
    """Idempotent delete: 204 if the row was there, 404 if it wasn't."""
    if not await WebhookSubscriptionsRepo(db).delete(sub_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="webhook not found"
        )
