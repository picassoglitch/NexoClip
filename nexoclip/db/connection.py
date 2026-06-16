"""aiosqlite connection wrapper.

Phase 1 runs single-instance SQLite, so a single shared connection with
serialized async access is enough. Phase 3 swaps this out for a real
Aurora pool — same `Database.transaction()` API.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite


class Database:
    """One aiosqlite connection with WAL + foreign_keys + safe defaults."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> aiosqlite.Connection:
        """Lazily open the connection. Idempotent."""
        if self._conn is not None:
            return self._conn
        async with self._lock:
            if self._conn is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                conn = await aiosqlite.connect(self.path)
                conn.row_factory = aiosqlite.Row
                # Pragmas — safe defaults for a single-writer workload.
                await conn.execute("PRAGMA journal_mode = WAL")
                await conn.execute("PRAGMA foreign_keys = ON")
                await conn.execute("PRAGMA synchronous = NORMAL")
                # WAL removes reader/writer contention but NOT writer/writer:
                # only one write txn runs at a time, and any other connection
                # (background render task, pipeline workers) that hits the held
                # write lock raises `database is locked` IMMEDIATELY when the
                # busy_timeout is the default 0. Give contending writers up to
                # 5s to wait the lock out — SQLite retries internally, which
                # turns those bursty collisions into a short stall instead of a
                # 500. Must be set per-connection; connect() is the chokepoint.
                await conn.execute("PRAGMA busy_timeout = 5000")
                await conn.commit()
                self._conn = conn
        return self._conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Wrap a block in BEGIN / COMMIT, rolling back on exception."""
        conn = await self.connect()
        try:
            yield conn
        except BaseException:
            await conn.rollback()
            raise
        else:
            await conn.commit()
