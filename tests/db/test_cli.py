"""CLI smoke tests for the Phase 1 admin commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from nexoclip.cli import app


def test_db_init_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["db", "init", "--help"])
    assert result.exit_code == 0


def test_db_init_brings_schema_to_current_version(tmp_path: Path) -> None:
    """`db init` runs every migration; current head is version 35
    (035_zernio_publish_snapshots)."""
    runner = CliRunner()
    db_path = tmp_path / "x.db"
    result = runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "schema_version = 35" in result.output


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


def test_tenants_set_budget_sets_dollars_micros(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "x.db"
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    runner.invoke(
        app, ["tenants", "add", "aldo", "Aldo", "--db-path", str(db_path)]
    )
    r = runner.invoke(
        app,
        [
            "tenants",
            "set-budget",
            "aldo",
            "--daily-usd",
            "5.00",
            "--publish-limit",
            "50",
            "--rescore-cap",
            "8",
            "--db-path",
            str(db_path),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "$5.00" in r.output
    assert "publish/d=50" in r.output
    assert "rescore_cap=8" in r.output

    # `tenants list` reflects the new values.
    r2 = runner.invoke(app, ["tenants", "list", "--db-path", str(db_path)])
    assert "$5.00" in r2.output


def test_tenants_set_budget_zero_clears_to_unlimited(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "x.db"
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    runner.invoke(
        app, ["tenants", "add", "aldo", "Aldo", "--db-path", str(db_path)]
    )
    runner.invoke(
        app,
        ["tenants", "set-budget", "aldo", "--daily-usd", "5.00", "--db-path", str(db_path)],
    )
    # Now clear it.
    r = runner.invoke(
        app,
        ["tenants", "set-budget", "aldo", "--daily-usd", "0", "--db-path", str(db_path)],
    )
    assert r.exit_code == 0, r.output
    assert "budget=unlimited" in r.output


def test_tenants_set_budget_no_args_exits_2(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "x.db"
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    runner.invoke(
        app, ["tenants", "add", "aldo", "Aldo", "--db-path", str(db_path)]
    )
    r = runner.invoke(
        app, ["tenants", "set-budget", "aldo", "--db-path", str(db_path)]
    )
    assert r.exit_code == 2
    assert "nothing to update" in (r.stderr or r.output)


def test_queue_list_unknown_tenant_exits_1(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "x.db"
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    result = runner.invoke(
        app,
        ["queue", "list", "--tenant", "ten_nope", "--db-path", str(db_path)],
    )
    assert result.exit_code == 1
    assert "unknown tenant" in (result.stderr or result.output)


def test_queue_list_renders_pending_and_recently_sent(tmp_path: Path) -> None:
    """Smoke test: enqueue + flip-to-sent, then `queue list` shows both sections."""
    import asyncio
    import datetime as _dt

    from nexoclip.db import (
        CandidatesRepo,
        ClipsRepo,
        ConnectedAccountsRepo,
        Database,
        PersonasRepo,
        PublishJobsRepo,
        StreamsRepo,
        TenantsRepo,
        VariantsRepo,
        apply_migrations,
    )
    from nexoclip.db.models import (
        CandidateRow,
        ClipRow,
        StreamRow,
        VariantRow,
    )
    from nexoclip.tenancy import bound_tenant

    db_path = tmp_path / "queue.db"

    async def _seed() -> None:
        db = Database(db_path)
        try:
            await apply_migrations(db)
            tenant = await TenantsRepo(db).create(tenant_id="aldo", name="Aldo")
            with bound_tenant(tenant.id):
                await StreamsRepo(db).upsert(
                    StreamRow(
                        id="str_q",
                        tenant_id=tenant.id,
                        vod_url="x",
                        platform="kick",
                        title=None,
                        channel=None,
                        duration_s=60.0,
                        source_video_path="/tmp/v",
                        source_audio_path="/tmp/a",
                        status="ingested",
                        created_at=_dt.datetime.now(_dt.UTC).isoformat(),
                    )
                )
                await CandidatesRepo(db).upsert_many(
                    [
                        CandidateRow(
                            id="cnd_q",
                            stream_id="str_q",
                            tenant_id=tenant.id,
                            ts=10.0,
                            score=0.5,
                            reason="voice",
                            evidence={},
                            created_at=_dt.datetime.now(_dt.UTC).isoformat(),
                        )
                    ]
                )
                await ClipsRepo(db).upsert_many(
                    [
                        ClipRow(
                            id="clp_q",
                            stream_id="str_q",
                            tenant_id=tenant.id,
                            candidate_id="cnd_q",
                            start_s=0.0,
                            end_s=10.0,
                            duration_s=10.0,
                            width=1080,
                            height=1920,
                            path="/tmp/c.mp4",
                            status="approved",
                            created_at=_dt.datetime.now(_dt.UTC).isoformat(),
                        )
                    ]
                )
                await PersonasRepo(db).create(
                    persona_id="p1",
                    name="P",
                    primary_language="es",
                    target_languages=["es"],
                    voice_prompt="v",
                )
                await VariantsRepo(db).replace_for_clip_persona(
                    "clp_q",
                    "p1",
                    [
                        VariantRow(
                            id="var_q",
                            clip_id="clp_q",
                            tenant_id=tenant.id,
                            persona_id="p1",
                            language="es",
                            caption="c",
                            title_card_text="",
                            hashtags=[],
                            model=None,
                            created_at=_dt.datetime.now(_dt.UTC).isoformat(),
                        )
                    ],
                )
                acct = await ConnectedAccountsRepo(db).create(
                    platform="tiktok", external_id="u"
                )
                pending = await PublishJobsRepo(db).enqueue(
                    clip_id="clp_q",
                    variant_id="var_q",
                    account_id=acct.id,
                    platform="tiktok",
                )
                sent = await PublishJobsRepo(db).enqueue(
                    clip_id="clp_q",
                    variant_id="var_q",
                    account_id=acct.id,
                    platform="tiktok",
                )
                conn = await db.connect()
                await conn.execute(
                    "UPDATE publish_jobs SET status='sent', external_id='ext_xyz' WHERE id=?",
                    (sent.id,),
                )
                await conn.commit()
                _ = pending
        finally:
            await db.close()

    asyncio.run(_seed())

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["queue", "list", "--tenant", "aldo", "--db-path", str(db_path)],
    )
    assert result.exit_code == 0, result.output
    assert "Pending (1)" in result.output
    assert "Recently sent (1)" in result.output
    assert "Failed (0)" in result.output
    assert "external=ext_xyz" in result.output


def test_mcp_serve_help_is_documented(tmp_path: Path) -> None:
    """`nexoclip mcp serve --help` shows usage so agents can discover the command."""
    runner = CliRunner()
    result = runner.invoke(app, ["mcp", "serve", "--help"])
    assert result.exit_code == 0
    assert "MCP server" in result.output or "stdio" in result.output


def test_mcp_serve_without_token_exits_1(tmp_path: Path) -> None:
    """No `--token` and no env var -> the server fails fast at boot."""
    import os

    runner = CliRunner()
    db_path = tmp_path / "x.db"
    runner.invoke(app, ["db", "init", "--db-path", str(db_path)])
    # Make sure the env var is empty for this invocation.
    env = {k: v for k, v in os.environ.items() if k != "NEXOCLIP_API_TOKEN"}
    env["NEXOCLIP_API_TOKEN"] = ""
    result = runner.invoke(
        app, ["mcp", "serve", "--db-path", str(db_path)], env=env
    )
    assert result.exit_code == 1
    assert "no API token" in (result.stderr or result.output) or "NEXOCLIP_API_TOKEN" in (
        result.stderr or result.output
    )


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
