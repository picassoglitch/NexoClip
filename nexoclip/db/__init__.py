"""SQLite persistence layer.

Phase 1 lock-down: schema in `migrations/001_init.sql`, runner in
`migrations.py`, connection in `connection.py`. Repos live in `repos.py`.
The `db_session` helper is the canonical way to open the DB + bind a
tenant in one call.
"""

from .connection import Database
from .migrations import MigrationError, apply_migrations, schema_version
from .repos import (
    ApiTokensRepo,
    CandidatesRepo,
    ClipsRepo,
    EventsRepo,
    LLMCallsRepo,
    PersonasRepo,
    StreamsRepo,
    TenantsRepo,
    TranscriptsRepo,
    UsersRepo,
    VariantsRepo,
)
from .session import db_session

__all__ = [
    "ApiTokensRepo",
    "CandidatesRepo",
    "ClipsRepo",
    "Database",
    "EventsRepo",
    "LLMCallsRepo",
    "MigrationError",
    "PersonasRepo",
    "StreamsRepo",
    "TenantsRepo",
    "TranscriptsRepo",
    "UsersRepo",
    "VariantsRepo",
    "apply_migrations",
    "db_session",
    "schema_version",
]
