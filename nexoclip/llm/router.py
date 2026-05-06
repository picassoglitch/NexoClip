"""LLMRouter — single point through which every LLM call flows.

Why a router (per CLAUDE.md hard rule #3):
    - Centralized cost tracking: every call writes a row to the call log
      (Phase 0: JSONL on disk; Phase 1+: `llm_calls` table).
    - Provider fallback: try `primary`, then each `fallback` in order.
    - Retries with exponential backoff on transient failures.
    - Structured output validation: callers get a typed Pydantic model back,
      never raw JSON.

Phase 0 only ships the Anthropic provider; OpenAI fallback is reserved.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from nexoclip.errors import LLMError

from .config import LLMConfig, ProviderConfig, Quality
from .provider import LLMProvider, ProviderResult, RetryableLLMError

if TYPE_CHECKING:
    from nexoclip.db import Database

T = TypeVar("T", bound=BaseModel)

ProviderFactory = Callable[[str, ProviderConfig, str], LLMProvider | None]


class CallLogRow(BaseModel):
    """One row appended to `llm_calls.jsonl` per `complete()` call."""

    ts: str
    tenant_id: str
    purpose: str
    provider: str
    model: str
    quality: Quality
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd_micros: int = 0
    status: str = "ok"
    error: str | None = None
    attempts: int = 1


def _default_provider_factory(
    name: str, config: ProviderConfig, api_key: str
) -> LLMProvider | None:
    """Construct providers we ship in Phase 0; return None for unimplemented ones."""
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=api_key, config=config)
    # OpenAI fallback is stubbed in Phase 0 — see PHASE_0.md.
    return None


class LLMRouter:
    """Routes typed LLM calls through `provider chain → retry → validate → log`."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        api_keys: dict[str, str] | None = None,
        call_log_path: Path | None = None,
        db: Database | None = None,
        provider_factory: ProviderFactory | None = None,
        clock: Callable[[], _dt.datetime] | None = None,
    ):
        self._config = config
        self._api_keys = api_keys if api_keys is not None else _read_api_keys(config)
        self._call_log_path = call_log_path
        self._db = db
        self._provider_factory = provider_factory or _default_provider_factory
        self._clock = clock or (lambda: _dt.datetime.now(_dt.UTC))
        self._providers: dict[str, LLMProvider | None] = {}

    async def complete(
        self,
        *,
        tenant_id: str,
        purpose: str,
        system: str,
        user: str,
        schema: type[T],
        quality: Quality | None = None,
    ) -> T:
        """Run one LLM completion with retries + provider fallback.

        Args:
            tenant_id: Owns the call (cost is attributed here).
            purpose: Routing key (e.g. `variant_generation`). Must exist in
                `config.routing`.
            system: System prompt (persona voice, instructions, etc.).
            user: User prompt — typically the clip context.
            schema: Pydantic model the response must validate against.
            quality: Override the routing rule's `default_quality`.
        """
        rule = self._config.routing.get(purpose)
        if rule is None:
            raise LLMError(f"unknown routing purpose: {purpose}")
        effective_quality: Quality = quality or rule.default_quality

        provider_chain = [rule.primary, *rule.fallbacks]
        last_error: Exception | None = None

        for provider_name in provider_chain:
            provider = self._get_provider(provider_name)
            if provider is None:
                last_error = LLMError(f"provider not available: {provider_name}")
                continue
            model = self._config.model_for(provider_name, effective_quality)

            try:
                validated, attempts, result = await self._call_with_retries(
                    provider=provider,
                    model=model,
                    tenant_id=tenant_id,
                    system=system,
                    user=user,
                    schema=schema,
                )
            except (LLMError, ValidationError) as e:
                await self._log(
                    tenant_id=tenant_id,
                    purpose=purpose,
                    provider=provider_name,
                    model=model,
                    quality=effective_quality,
                    status="error",
                    error=f"{type(e).__name__}: {e}",
                    attempts=getattr(e, "attempts", 1),
                )
                last_error = e
                # If there's another provider in the chain to try, emit
                # llm.fallback so dashboards can flag flaky primaries.
                if provider_name != provider_chain[-1]:
                    await self._emit_event(
                        tenant_id=tenant_id,
                        type_="llm.fallback",
                        payload={
                            "purpose": purpose,
                            "provider": provider_name,
                            "error": f"{type(e).__name__}: {e}",
                        },
                    )
                continue

            cost = self._compute_cost_micros(
                provider=provider_name,
                model=model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
            await self._log(
                tenant_id=tenant_id,
                purpose=purpose,
                provider=provider_name,
                model=model,
                quality=effective_quality,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd_micros=cost,
                attempts=attempts,
            )
            return validated

        await self._emit_event(
            tenant_id=tenant_id,
            type_="llm.exhausted",
            payload={
                "purpose": purpose,
                "providers_tried": provider_chain,
                "error": f"{type(last_error).__name__}: {last_error}"
                if last_error is not None
                else None,
            },
        )
        raise LLMError(
            f"all providers failed for purpose={purpose!r}: {last_error}"
        ) from last_error

    async def _call_with_retries(
        self,
        *,
        provider: LLMProvider,
        model: str,
        tenant_id: str,
        system: str,
        user: str,
        schema: type[T],
    ) -> tuple[T, int, ProviderResult]:
        """Drive a single provider through `RetryConfig.max_attempts`."""
        retry = self._config.retry
        last_err: Exception | None = None
        for attempt in range(1, retry.max_attempts + 1):
            try:
                result = await provider.complete(
                    tenant_id=tenant_id,
                    model=model,
                    system=system,
                    user=user,
                    schema=schema,
                )
                validated = schema.model_validate(result.output)
                return validated, attempt, result
            except RetryableLLMError as e:
                last_err = e
                if attempt < retry.max_attempts:
                    backoff = retry.initial_backoff_s * (retry.backoff_multiplier ** (attempt - 1))
                    await asyncio.sleep(backoff)
                continue
            except ValidationError as e:
                # Treat schema violations as retryable — the LLM may produce a
                # better-formed object on the next attempt.
                last_err = e
                if attempt < retry.max_attempts:
                    backoff = retry.initial_backoff_s * (retry.backoff_multiplier ** (attempt - 1))
                    await asyncio.sleep(backoff)
                continue

        # Annotate so the caller can record attempts in the failure log row.
        if last_err is not None:
            last_err.attempts = retry.max_attempts  # type: ignore[attr-defined]
            raise last_err
        raise LLMError("retry loop exited without success or error")

    def _get_provider(self, name: str) -> LLMProvider | None:
        if name not in self._providers:
            cfg = self._config.providers.get(name)
            api_key = self._api_keys.get(name, "")
            if cfg is None or not api_key:
                self._providers[name] = None
            else:
                self._providers[name] = self._provider_factory(name, cfg, api_key)
        return self._providers[name]

    def _compute_cost_micros(
        self, *, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> int:
        """`cost_usd_micros = round(input * input_pmtok + output * output_pmtok)`."""
        pricing = self._config.pricing_for(provider, model)
        cost = (
            input_tokens * pricing.input_per_mtok_usd + output_tokens * pricing.output_per_mtok_usd
        )
        return round(cost)

    async def _log(
        self,
        *,
        tenant_id: str,
        purpose: str,
        provider: str,
        model: str,
        quality: Quality,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd_micros: int = 0,
        status: str = "ok",
        error: str | None = None,
        attempts: int = 1,
    ) -> None:
        """Write one cost-tracking row to the JSONL breadcrumb and the DB.

        Both writes are best-effort — the LLM call already happened, so
        a write failure here must not propagate up. The JSONL is the
        Phase 0 carry-over; the DB row is the Phase 1 source of truth.
        """
        ts = self._clock().isoformat()
        row = CallLogRow(
            ts=ts,
            tenant_id=tenant_id,
            purpose=purpose,
            provider=provider,
            model=model,
            quality=quality,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd_micros=cost_usd_micros,
            status=status,
            error=error,
            attempts=attempts,
        )

        if self._call_log_path is not None:
            try:
                self._call_log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._call_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row.model_dump(), ensure_ascii=False))
                    f.write("\n")
            except OSError:
                pass

        if self._db is not None:
            from nexoclip.db import LLMCallsRepo
            from nexoclip.db.models import LLMCallRow
            from nexoclip.ids import new_id
            from nexoclip.tenancy import bound_tenant

            db_row = LLMCallRow(
                id=new_id("llm"),
                tenant_id=tenant_id,
                purpose=purpose,
                provider=provider,
                model=model,
                quality=quality,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd_micros=cost_usd_micros,
                status=status,
                error=error,
                attempts=attempts,
                ts=ts,
            )
            try:
                with bound_tenant(tenant_id):
                    await LLMCallsRepo(self._db).record(db_row)
            except Exception:
                # Best-effort — the LLM call already happened and the JSONL
                # has the row; a DB write failure must not propagate up.
                pass

    async def _emit_event(
        self,
        *,
        tenant_id: str,
        type_: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append an llm.* event row when a DB is wired in. Best-effort."""
        if self._db is None:
            return
        from nexoclip.events import emit
        from nexoclip.tenancy import bound_tenant

        try:
            with bound_tenant(tenant_id):
                await emit(self._db, type_, payload)
        except Exception:
            pass


def _read_api_keys(config: LLMConfig) -> dict[str, str]:
    """Pull each provider's API key out of the environment."""
    keys: dict[str, str] = {}
    for name, cfg in config.providers.items():
        keys[name] = os.environ.get(cfg.api_key_env, "")
    return keys
