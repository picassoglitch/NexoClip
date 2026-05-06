"""Lightweight fakes for `faster_whisper.WhisperModel` and friends.

The real WhisperModel constructor allocates a CUDA model — far too heavy
for unit tests. These fakes mimic just the duck-typed shape the service
actually consumes.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass
class FakeWord:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    words: list[FakeWord] | None


@dataclass
class FakeInfo:
    language: str
    duration: float


class FakeWhisperModel:
    """Records constructor + transcribe call args; replays canned segments."""

    constructor_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    transcribe_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    canned_segments: list[FakeSegment] = []
    canned_info: FakeInfo = FakeInfo(language="es", duration=0.0)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        FakeWhisperModel.constructor_calls.append((args, kwargs))

    def transcribe(
        self, *args: Any, **kwargs: Any
    ) -> tuple[Iterator[FakeSegment], FakeInfo]:
        FakeWhisperModel.transcribe_calls.append((args, kwargs))
        return iter(FakeWhisperModel.canned_segments), FakeWhisperModel.canned_info

    @classmethod
    def reset(cls) -> None:
        cls.constructor_calls = []
        cls.transcribe_calls = []
        cls.canned_segments = []
        cls.canned_info = FakeInfo(language="es", duration=0.0)
