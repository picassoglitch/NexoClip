"""Tests for `cut_window` (start/end/duration clamping)."""

from __future__ import annotations

import pytest

from nexoclip.clip import cut_window
from nexoclip.errors import ClipError


def test_normal_window() -> None:
    start, end, duration = cut_window(
        timestamp=120.0, pre_roll_s=30.0, post_roll_s=15.0, stream_duration_s=600.0
    )
    assert start == pytest.approx(90.0)
    assert end == pytest.approx(135.0)
    assert duration == pytest.approx(45.0)


def test_clamps_start_to_zero_for_early_anchor() -> None:
    start, end, duration = cut_window(
        timestamp=10.0, pre_roll_s=30.0, post_roll_s=15.0, stream_duration_s=600.0
    )
    assert start == 0.0
    assert end == pytest.approx(25.0)
    assert duration == pytest.approx(25.0)


def test_clamps_end_to_stream_length() -> None:
    start, end, duration = cut_window(
        timestamp=595.0, pre_roll_s=30.0, post_roll_s=15.0, stream_duration_s=600.0
    )
    assert start == pytest.approx(565.0)
    assert end == pytest.approx(600.0)
    assert duration == pytest.approx(35.0)


def test_anchor_after_stream_end_yields_zero_duration() -> None:
    start, end, duration = cut_window(
        timestamp=700.0, pre_roll_s=30.0, post_roll_s=15.0, stream_duration_s=600.0
    )
    assert duration == 0.0
    assert start == end


def test_zero_stream_duration_raises() -> None:
    with pytest.raises(ClipError):
        cut_window(timestamp=10.0, pre_roll_s=30.0, post_roll_s=15.0, stream_duration_s=0.0)
