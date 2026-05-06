"""Shared fixtures for DB tests.

Each test gets its own SQLite file under `tmp_path` so tests are
independent and can run in parallel later.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest_asyncio

from nexoclip.db import Database, apply_migrations


@pytest_asyncio.fixture
async def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest_asyncio.fixture
async def db(db_path: Path) -> AsyncIterator[Database]:
    d = Database(db_path)
    try:
        yield d
    finally:
        await d.close()


@pytest_asyncio.fixture
async def migrated_db(db: Database) -> Database:
    """A database with all migrations applied."""
    await apply_migrations(db)
    return db
