"""Forward-only migration runner.

Migrations live in `nexoclip/db/migrations/` as `NNN_<name>.sql`.
Schema version is tracked in the single-row `schema_version` table.
"""

from __future__ import annotations

import re
from pathlib import Path

import aiosqlite

from nexoclip.errors import NexoClipError

from .connection import Database

_MIGRATION_RE = re.compile(r"^(\d{3})_([a-z0-9_]+)\.sql$", re.IGNORECASE)


class MigrationError(NexoClipError):
    """Migration runner failure."""


async def _ensure_schema_version_table(conn: aiosqlite.Connection) -> int:
    """Return the current schema version, creating the table on first run."""
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            rowid   INTEGER PRIMARY KEY CHECK (rowid = 1),
            version INTEGER NOT NULL
        )
        """
    )
    await conn.commit()
    cur = await conn.execute("SELECT version FROM schema_version WHERE rowid = 1")
    row = await cur.fetchone()
    return int(row[0]) if row else 0


def _discover_migrations(directory: Path) -> list[tuple[int, str, Path]]:
    """Sorted (version, name, path) for every well-named .sql in `directory`."""
    out: list[tuple[int, str, Path]] = []
    for candidate in sorted(directory.glob("*.sql")):
        m = _MIGRATION_RE.match(candidate.name)
        if not m:
            raise MigrationError(
                f"migration filename does not match NNN_<name>.sql: {candidate.name}"
            )
        out.append((int(m.group(1)), m.group(2), candidate))
    out.sort()
    return out


async def apply_migrations(db: Database, *, migrations_dir: Path | None = None) -> int:
    """Apply any pending migrations. Returns the resulting schema version."""
    if migrations_dir is None:
        migrations_dir = Path(__file__).parent / "migrations"
    if not migrations_dir.exists():
        raise MigrationError(f"migrations dir not found: {migrations_dir}")

    available = _discover_migrations(migrations_dir)
    if not available:
        raise MigrationError(f"no migrations found in {migrations_dir}")

    versions = [v for v, _, _ in available]
    if versions != list(range(1, len(versions) + 1)):
        raise MigrationError(
            f"migration versions must be a contiguous 1..N sequence; got {versions}"
        )

    conn = await db.connect()
    current = await _ensure_schema_version_table(conn)
    target = available[-1][0]

    if current > target:
        raise MigrationError(
            f"schema_version={current} is newer than highest available migration={target}; "
            "downgrades are not supported"
        )

    for version, _name, path in available:
        if version <= current:
            continue
        sql = path.read_text(encoding="utf-8")
        try:
            # executescript implicitly commits any pending tx, then runs the
            # script as a single statement batch. We follow it with the
            # version bump so a partial failure leaves schema_version stale
            # (re-runnable migration files should be safe to retry).
            await conn.executescript(sql)
            await conn.execute(
                "INSERT OR REPLACE INTO schema_version (rowid, version) VALUES (1, ?)",
                (version,),
            )
            await conn.commit()
        except Exception as e:
            await conn.rollback()
            raise MigrationError(f"migration {path.name} failed: {e}") from e
        current = version

    return current


async def schema_version(db: Database) -> int:
    """Read the current schema version (0 if uninitialized)."""
    conn = await db.connect()
    return await _ensure_schema_version_table(conn)
