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
    BrandKitsRepo,
    CandidatesRepo,
    ClipsRepo,
    ConnectedAccountsRepo,
    DriveExportSettingsRepo,
    DriveWatchesRepo,
    EventsRepo,
    LiveStreamKeysRepo,
    LLMCallsRepo,
    PersonasRepo,
    PlatformSettingsRepo,
    PublishJobsRepo,
    PublishMetricsRepo,
    SpeakersRepo,
    StreamDestinationsRepo,
    StreamsRepo,
    TenantsRepo,
    TranscriptsRepo,
    UsersRepo,
    VariantsRepo,
    VisualSignalsRepo,
    VodSpeakersRepo,
    WebhookSecretsRepo,
    WebhookSubscriptionsRepo,
)
from .session import db_session

__all__ = [
    "ApiTokensRepo",
    "BrandKitsRepo",
    "CandidatesRepo",
    "ClipsRepo",
    "ConnectedAccountsRepo",
    "Database",
    "DriveExportSettingsRepo",
    "DriveWatchesRepo",
    "EventsRepo",
    "LiveStreamKeysRepo",
    "LLMCallsRepo",
    "MigrationError",
    "PersonasRepo",
    "PlatformSettingsRepo",
    "PublishJobsRepo",
    "PublishMetricsRepo",
    "SpeakersRepo",
    "StreamDestinationsRepo",
    "StreamsRepo",
    "TenantsRepo",
    "TranscriptsRepo",
    "UsersRepo",
    "VariantsRepo",
    "VisualSignalsRepo",
    "VodSpeakersRepo",
    "WebhookSecretsRepo",
    "WebhookSubscriptionsRepo",
    "apply_migrations",
    "db_session",
    "schema_version",
]
