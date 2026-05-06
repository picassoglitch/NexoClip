"""SQLite persistence layer.

Phase 1 lock-down: schema in `migrations/001_init.sql`, runner in
`migrations.py`, connection in `connection.py`. Repos live in `repos.py`.
"""

from .connection import Database
from .migrations import MigrationError, apply_migrations, schema_version
from .repos import (
    ApiTokensRepo,
    EventsRepo,
    LLMCallsRepo,
    PersonasRepo,
    StreamsRepo,
    TenantsRepo,
    UsersRepo,
)

__all__ = [
    "ApiTokensRepo",
    "Database",
    "EventsRepo",
    "LLMCallsRepo",
    "MigrationError",
    "PersonasRepo",
    "StreamsRepo",
    "TenantsRepo",
    "UsersRepo",
    "apply_migrations",
    "schema_version",
]
