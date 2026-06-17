"""Pipeline auto-hook: every publish-worthy clip gets a viral title line
generated from its transcript, best-effort (never breaks the run)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nexoclip.pipeline import _auto_hook_for_clip
from nexoclip.variants.hooks import Hook, HookBatch


class _FakeRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kw: Any) -> HookBatch:
        self.calls.append(kw)
        assert kw["purpose"] == "hook_generation"
        return HookBatch(hooks=[Hook(text="Esto Cambió Toda La Partida")])


def _clip(snippet: str) -> Any:
    return SimpleNamespace(
        id="clp_x", candidate=SimpleNamespace(evidence={"transcript_snippet": snippet})
    )


_PERSONA = SimpleNamespace(voice_prompt="energético, directo", primary_language="es")


@pytest.mark.asyncio
async def test_generates_hook_from_transcript_snippet() -> None:
    router = _FakeRouter()
    hook = await _auto_hook_for_clip(
        clip=_clip("clipea esto, esa jugada fue una locura"),
        persona=_PERSONA, language="es", tenant_id="t1", router=router,
    )
    assert hook == "Esto Cambió Toda La Partida"
    # The clip's transcript snippet was passed through to the generator.
    assert "clipea esto" in router.calls[0]["user"]


@pytest.mark.asyncio
async def test_swallows_llm_failure_returns_empty() -> None:
    class _Boom:
        async def complete(self, **kw: Any) -> HookBatch:
            raise RuntimeError("llm down")

    hook = await _auto_hook_for_clip(
        clip=_clip(""), persona=_PERSONA, language=None,
        tenant_id="t1", router=_Boom(),
    )
    assert hook == ""  # best-effort: a hook hiccup never breaks the pipeline
