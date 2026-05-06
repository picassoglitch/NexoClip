"""Tests for `find_clip` (clip lookup by id)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nexoclip.errors import VariantError
from nexoclip.variants import find_clip

from ._fixtures import make_clip


def test_find_clip_returns_clip_dirs(tmp_path: Path) -> None:
    clip = make_clip(tmp_path)
    result, clip_dir, stream_dir = find_clip(clip.id, tmp_path)
    assert result.id == clip.id
    assert clip_dir == tmp_path / clip.stream_id / "clips" / clip.id
    assert stream_dir == tmp_path / clip.stream_id


def test_find_clip_unknown_raises(tmp_path: Path) -> None:
    with pytest.raises(VariantError, match="clip not found"):
        find_clip("clp_nope", tmp_path)


def test_find_clip_collision_raises(tmp_path: Path) -> None:
    """Two streams accidentally containing the same clip id is a hard error."""
    make_clip(tmp_path)
    # Manually create a duplicate entry under a second stream.
    other = tmp_path / "str_OTHER" / "clips" / "clp_01TEST"
    other.mkdir(parents=True, exist_ok=True)
    (other / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(VariantError, match="collision"):
        find_clip("clp_01TEST", tmp_path)
