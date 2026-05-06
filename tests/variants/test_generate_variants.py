"""Tests for `generate_variants`."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nexoclip.errors import VariantError
from nexoclip.variants import generate_variants

from tests.llm._fakes import FakeProvider  # type: ignore[import]
from ._fixtures import make_clip, make_persona, make_router_with_fake


def _success_payload(n: int = 5) -> dict:
    return {
        "variants": [
            {
                "id": f"v_{i + 1}",
                "language": "es",
                "caption": f"Caption #{i + 1}",
                "title_card_text": "" if i % 2 == 0 else f"HOOK{i}",
                "hashtags": ["clip", f"v{i + 1}"],
            }
            for i in range(n)
        ]
    }


def test_generates_n_variants_and_writes_file(tmp_path: Path) -> None:
    clip = make_clip(tmp_path)
    persona = make_persona()
    fake = FakeProvider("anthropic")
    fake.queue_success(_success_payload(n=5))
    router = make_router_with_fake(fake)

    result = asyncio.run(
        generate_variants(
            tenant_id="default",
            clip=clip,
            persona=persona,
            router=router,
            n=5,
        )
    )

    assert len(result) == 5
    assert result[0].id == "v_1"
    assert result[0].language == "es"
    assert result[0].caption == "Caption #1"

    out_path = clip.path.parent / "variants.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text("utf-8"))
    assert payload["clip_id"] == clip.id
    assert payload["persona_id"] == persona.id
    assert payload["language"] == persona.primary_language
    assert len(payload["variants"]) == 5


def test_prompt_includes_persona_voice_and_clip_evidence(tmp_path: Path) -> None:
    clip = make_clip(tmp_path)
    persona = make_persona()
    fake = FakeProvider("anthropic")
    fake.queue_success(_success_payload(n=3))
    router = make_router_with_fake(fake)

    asyncio.run(
        generate_variants(
            tenant_id="default",
            clip=clip,
            persona=persona,
            router=router,
            n=3,
        )
    )
    call = fake.calls[0]
    assert "Direct, confrontational" in call["system"]
    assert "Write each variant in es" in call["system"]
    assert "exactly 3" in call["system"]
    # User prompt carries the trigger evidence.
    assert "clipéalo" in call["user"]
    assert "no manches eso clipéalo" in call["user"]
    assert call["schema"] == "VariantBatch"


def test_n_truncates_when_llm_returns_more(tmp_path: Path) -> None:
    clip = make_clip(tmp_path)
    persona = make_persona()
    fake = FakeProvider("anthropic")
    fake.queue_success(_success_payload(n=8))
    router = make_router_with_fake(fake)

    result = asyncio.run(
        generate_variants(
            tenant_id="default",
            clip=clip,
            persona=persona,
            router=router,
            n=3,
        )
    )
    assert len(result) == 3


def test_idempotent_when_cached_matches(tmp_path: Path) -> None:
    clip = make_clip(tmp_path)
    persona = make_persona()
    fake = FakeProvider("anthropic")
    fake.queue_success(_success_payload(n=5))
    router = make_router_with_fake(fake)

    first = asyncio.run(
        generate_variants(
            tenant_id="default", clip=clip, persona=persona, router=router, n=5
        )
    )
    second = asyncio.run(
        generate_variants(
            tenant_id="default", clip=clip, persona=persona, router=router, n=5
        )
    )
    assert second == first
    # Cache hit means no second LLM call.
    assert len(fake.calls) == 1


def test_force_regenerates(tmp_path: Path) -> None:
    clip = make_clip(tmp_path)
    persona = make_persona()
    fake = FakeProvider("anthropic")
    fake.queue_success(_success_payload(n=5))
    fake.queue_success(_success_payload(n=5))
    router = make_router_with_fake(fake)

    asyncio.run(
        generate_variants(
            tenant_id="default", clip=clip, persona=persona, router=router, n=5
        )
    )
    asyncio.run(
        generate_variants(
            tenant_id="default",
            clip=clip,
            persona=persona,
            router=router,
            n=5,
            force=True,
        )
    )
    assert len(fake.calls) == 2


def test_different_persona_ignores_cache(tmp_path: Path) -> None:
    clip = make_clip(tmp_path)
    fake = FakeProvider("anthropic")
    fake.queue_success(_success_payload(n=5))
    fake.queue_success(_success_payload(n=5))
    router = make_router_with_fake(fake)

    asyncio.run(
        generate_variants(
            tenant_id="default",
            clip=clip,
            persona=make_persona(persona_id="aldo_villanueva"),
            router=router,
            n=5,
        )
    )
    asyncio.run(
        generate_variants(
            tenant_id="default",
            clip=clip,
            persona=make_persona(persona_id="aara"),
            router=router,
            n=5,
        )
    )
    assert len(fake.calls) == 2


def test_tenant_mismatch_raises(tmp_path: Path) -> None:
    clip = make_clip(tmp_path)
    fake = FakeProvider("anthropic")
    router = make_router_with_fake(fake)
    with pytest.raises(VariantError, match="tenant mismatch"):
        asyncio.run(
            generate_variants(
                tenant_id="other",
                clip=clip,
                persona=make_persona(),
                router=router,
                n=3,
            )
        )


def test_zero_variants_from_llm_raises(tmp_path: Path) -> None:
    clip = make_clip(tmp_path)
    fake = FakeProvider("anthropic")
    fake.queue_success({"variants": []})
    router = make_router_with_fake(fake)
    with pytest.raises(VariantError, match="returned 0 variants"):
        asyncio.run(
            generate_variants(
                tenant_id="default",
                clip=clip,
                persona=make_persona(),
                router=router,
                n=3,
            )
        )


def test_language_override_propagates(tmp_path: Path) -> None:
    clip = make_clip(tmp_path)
    persona = make_persona(primary_language="es")
    fake = FakeProvider("anthropic")
    fake.queue_success(_success_payload(n=2))
    router = make_router_with_fake(fake)

    asyncio.run(
        generate_variants(
            tenant_id="default",
            clip=clip,
            persona=persona,
            router=router,
            n=2,
            language="en",
        )
    )
    assert "Write each variant in en" in fake.calls[0]["system"]


def test_invalid_n_raises(tmp_path: Path) -> None:
    clip = make_clip(tmp_path)
    fake = FakeProvider("anthropic")
    router = make_router_with_fake(fake)
    with pytest.raises(VariantError, match="n must be > 0"):
        asyncio.run(
            generate_variants(
                tenant_id="default",
                clip=clip,
                persona=make_persona(),
                router=router,
                n=0,
            )
        )
