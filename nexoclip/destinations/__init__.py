"""Multistream restream destinations — where the live relay fans out to."""

from .service import (
    PLATFORM_TEMPLATES,
    DestinationError,
    DestinationTarget,
    add_destination,
    list_destinations,
    resolve_targets,
    supported_platforms,
)

__all__ = [
    "PLATFORM_TEMPLATES",
    "DestinationError",
    "DestinationTarget",
    "add_destination",
    "list_destinations",
    "resolve_targets",
    "supported_platforms",
]
