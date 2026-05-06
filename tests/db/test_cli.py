"""CLI smoke tests for the Phase 1 admin commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from nexoclip.cli import app


def test_db_init_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["db", "init", "--help"])
    assert result.exit_code == 0


def test_db_init_brings_schema_to_version_1(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "x.db"
    result = runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "schema_version = 1" in result.output


def test_tenants_add_and_list(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "x.db"
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    r1 = runner.invoke(
        app, ["tenants", "add", "aldo", "Aldo Villanueva", "--db-path", str(db_path)]
    )
    assert r1.exit_code == 0, r1.output
    assert "created tenant: aldo" in r1.output

    r2 = runner.invoke(app, ["tenants", "list", "--db-path", str(db_path)])
    assert r2.exit_code == 0
    assert "aldo" in r2.output


def test_tokens_issue_then_list(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "x.db"
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    runner.invoke(
        app, ["tenants", "add", "aldo", "Aldo", "--db-path", str(db_path)]
    )
    r1 = runner.invoke(
        app, ["tokens", "issue", "--tenant", "aldo", "--db-path", str(db_path)]
    )
    assert r1.exit_code == 0, r1.output
    raw_token = r1.output.strip().splitlines()[0]
    assert raw_token.startswith("tok_")

    r2 = runner.invoke(
        app, ["tokens", "list", "--tenant", "aldo", "--db-path", str(db_path)]
    )
    assert r2.exit_code == 0
    assert "scope=full" in r2.output
    # Raw token must NOT appear in the list output.
    assert raw_token not in r2.output


def test_tokens_issue_invalid_scope_exits_2(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "x.db"
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    runner.invoke(
        app, ["tenants", "add", "aldo", "Aldo", "--db-path", str(db_path)]
    )
    r = runner.invoke(
        app,
        [
            "tokens",
            "issue",
            "--tenant",
            "aldo",
            "--scope",
            "admin",
            "--db-path",
            str(db_path),
        ],
    )
    assert r.exit_code == 2
    assert "must be 'full' or 'read'" in (r.stderr or r.output)
