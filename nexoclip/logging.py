"""structlog configuration shared by the CLI, FastAPI, and workers.

Per CLAUDE.md: structured fields, not f-string concatenation. Logs go to
stderr by default so a `--json` CLI command can stream the manifest to
stdout without log noise interleaving.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_CONFIGURED = False


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Idempotent process-wide logger setup. Call once at entry."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        stream=sys.stderr,
        format="%(message)s",
        force=True,
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if fmt.lower() == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Return a structlog logger; pass `__name__` from caller modules.

    Typed as `Any` because structlog's runtime logger class varies with the
    configured `wrapper_class`; pinning a specific class would lie to mypy.
    """
    return structlog.get_logger(name) if name else structlog.get_logger()


def reset_for_tests() -> None:
    """Allow tests to reconfigure log level by clearing the one-shot guard."""
    global _CONFIGURED
    _CONFIGURED = False
