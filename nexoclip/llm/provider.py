"""Provider abstraction shared by all LLM backends.

The router doesn't know about Anthropic vs OpenAI specifics — it just calls
`provider.complete(...)` and gets back a `ProviderResult`. New providers
implement the `LLMProvider` protocol and the router calls them via the
`provider_factory` injected into its constructor.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel

from nexoclip.errors import LLMError


class RetryableLLMError(LLMError):
    """A transient provider failure: 429, 5xx, network/timeout.

    The router retries up to `RetryConfig.max_attempts`. Non-retryable failures
    raise plain `LLMError` and skip straight to the next provider in the chain.
    """

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ProviderResult(BaseModel):
    """Raw structured output + token counts returned by a provider call."""

    output: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    model: str


class LLMProvider(Protocol):
    """Minimal surface every provider must expose to the router."""

    async def complete(
        self,
        *,
        tenant_id: str,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> ProviderResult: ...
