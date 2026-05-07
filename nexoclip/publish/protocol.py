"""Publisher protocol — what every native client (Buffer, TikTok, YT) implements.

The dispatcher in `service.py` looks up a factory by `connected_accounts.platform`
and asks it for a `Publisher` instance for the given OAuth blob. Phase 1's
`BufferClient` is now an instance of this protocol; Phase 2 adds TikTok + YT.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol

from nexoclip.db.models import ConnectedAccount, VariantRow


class PublishResult(Protocol):
    """Minimum shape an external publish response must expose."""

    @property
    def external_id(self) -> str: ...  # platform-side video id
    @property
    def external_url(self) -> str | None: ...  # shareable URL when known
    @property
    def metadata(self) -> dict[str, Any]: ...  # raw bits we want to keep


class PublisherError(Exception):
    """Common base so the dispatcher can split transient vs fatal."""

    transient: bool

    def __init__(self, message: str, *, transient: bool) -> None:
        super().__init__(message)
        self.transient = transient


class Publisher(Protocol):
    """Async-context-managed client that can publish one variant."""

    async def __aenter__(self) -> Publisher: ...

    async def __aexit__(self, *exc: object) -> None: ...

    def publish(
        self,
        *,
        clip_path: str,
        variant: VariantRow,
        account: ConnectedAccount,
    ) -> Awaitable[_Resp]: ...


class _Resp(Protocol):
    external_id: str
    external_url: str | None
    metadata: dict[str, Any]
