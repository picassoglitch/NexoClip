"""CLI smoke tests for `nexoclip variants`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nexoclip.cli import app
from nexoclip.llm import config as llm_config_module
from nexoclip.llm import router as router_module

from tests.llm._fakes import FakeProvider  # type: ignore[import]
from tests.llm._fixtures import make_llm_config  # type: ignore[import]
from ._fixtures import make_clip


def _seed_personas(tmp_path: Path) -> Path:
    path = tmp_path / "personas.yaml"
    path.write_text(
        """
personas:
  aldo_villanueva:
    name: "Aldo Villanueva"
    target_languages: [es, en]
    primary_language: es
    voice_prompt: "Direct entrepreneur voice."
""",
        encoding="utf-8",
    )
    return path


def _patch_router_to_use_fake(
    monkeypatch: pytest.MonkeyPatch, fake: FakeProvider
) -> None:
    """Force the CLI's router to use a fake Anthropic provider, no network."""
    config = make_llm_config(retry_attempts=1)

    def fake_load(_path: Path | None = None):
        return config

    def fake_factory(name: str, _config, _api_key: str):
        return fake if name == "anthropic" else None

    monkeypatch.setattr(llm_config_module, "load_llm_config", fake_load)
    # Anthropic API key needs to be present so the router builds the provider.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(router_module, "_default_provider_factory", fake_factory)


def _success_payload(n: int = 3) -> dict:
    return {
        "variants": [
            {
                "id": f"v_{i + 1}",
                "language": "es",
                "caption": f"Caption {i + 1}",
                "title_card_text": "",
                "hashtags": ["clip"],
            }
            for i in range(n)
        ]
    }


def test_variants_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["variants", "--help"])
    assert result.exit_code == 0
    assert "Persona id" in result.stdout


def test_variants_command_writes_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = make_clip(tmp_path)
    personas_path = _seed_personas(tmp_path)
    fake = FakeProvider("anthropic")
    fake.queue_success(_success_payload(n=3))
    _patch_router_to_use_fake(monkeypatch, fake)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "variants",
            clip.id,
            "--persona",
            "aldo_villanueva",
            "--output-dir",
            str(tmp_path),
            "--personas",
            str(personas_path),
            "--n",
            "3",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["clip_id"] == clip.id
    assert payload["persona_id"] == "aldo_villanueva"
    assert len(payload["variants"]) == 3

    assert (clip.path.parent / "variants.json").exists()
    # JSONL log row must have been written too (CLAUDE.md hard rule #6).
    assert (tmp_path / clip.stream_id / "llm_calls.jsonl").exists()


def test_variants_command_unknown_clip_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    personas_path = _seed_personas(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "variants",
            "clp_NOPE",
            "--persona",
            "aldo_villanueva",
            "--output-dir",
            str(tmp_path),
            "--personas",
            str(personas_path),
        ],
    )
    assert result.exit_code == 1
    assert "clip not found" in (result.stderr or result.output)


def test_variants_command_unknown_persona_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = make_clip(tmp_path)
    personas_path = _seed_personas(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "variants",
            clip.id,
            "--persona",
            "ghost",
            "--output-dir",
            str(tmp_path),
            "--personas",
            str(personas_path),
        ],
    )
    assert result.exit_code == 1
    assert "unknown persona" in (result.stderr or result.output)


def test_variants_command_invalid_quality_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = make_clip(tmp_path)
    personas_path = _seed_personas(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "variants",
            clip.id,
            "--persona",
            "aldo_villanueva",
            "--quality",
            "ultra",
            "--output-dir",
            str(tmp_path),
            "--personas",
            str(personas_path),
        ],
    )
    assert result.exit_code == 2
    assert "must be 'standard' or 'premium'" in (result.stderr or result.output)
