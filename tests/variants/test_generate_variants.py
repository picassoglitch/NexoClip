"""Tests for `generate_variants`."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nexoclip.errors import VariantError
from nexoclip.llm import FrameCache
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


def _seed_vision_cache(clip, *, n_frames: int = 3) -> FrameCache:
    """Pre-populate the cache so `_gather_vision_frames` never decodes the placeholder mp4."""
    cache = FrameCache()
    duration = clip.duration_s
    for i in range(n_frames):
        offset = duration * (i + 1) / (n_frames + 1)
        source_ts = clip.start_s + offset
        cache.put(clip.stream_id, source_ts, f"jpeg-{i}".encode())
    return cache


def test_use_vision_routes_through_complete_multimodal(tmp_path: Path) -> None:
    """`use_vision=True` -> FakeProvider sees a multimodal call carrying 3 frames."""
    clip = make_clip(tmp_path)
    persona = make_persona()
    fake = FakeProvider("anthropic")
    fake.queue_success(_success_payload(n=3))
    router = make_router_with_fake(fake)
    cache = _seed_vision_cache(clip, n_frames=3)

    result = asyncio.run(
        generate_variants(
            tenant_id="default",
            clip=clip,
            persona=persona,
            router=router,
            n=3,
            use_vision=True,
            frame_cache=cache,
        )
    )
    assert len(result) == 3
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["kind"] == "multimodal"
    assert call["n_images"] == 3
    # Cache pre-population stuffed deterministic blobs at idx 0/1/2.
    assert call["images"][0].data == b"jpeg-0"
    assert call["images"][2].data == b"jpeg-2"


def test_use_vision_cache_hit_skips_redecoding(tmp_path: Path, monkeypatch) -> None:
    """A pre-warmed cache means we never call `sample_frames`."""
    clip = make_clip(tmp_path)
    persona = make_persona()
    fake = FakeProvider("anthropic")
    fake.queue_success(_success_payload(n=3))
    router = make_router_with_fake(fake)
    cache = _seed_vision_cache(clip, n_frames=3)

    sampler_calls: list[int] = []

    def explode(*_args, **_kwargs):
        sampler_calls.append(1)
        raise AssertionError("sample_frames should not be called on cache hit")

    monkeypatch.setattr("nexoclip.vision.frame_sampler.sample_frames", explode)

    asyncio.run(
        generate_variants(
            tenant_id="default",
            clip=clip,
            persona=persona,
            router=router,
            n=3,
            use_vision=True,
            frame_cache=cache,
        )
    )
    assert sampler_calls == []


def test_use_vision_cache_miss_invokes_sampler(tmp_path: Path, monkeypatch) -> None:
    """Empty cache -> sample_frames is called once per requested frame, results cached."""
    clip = make_clip(tmp_path)
    persona = make_persona()
    fake = FakeProvider("anthropic")
    fake.queue_success(_success_payload(n=3))
    router = make_router_with_fake(fake)
    cache = FrameCache()

    sampler_calls: list[float] = []

    def fake_sample_frames(video_path, ts, n=1, *, spread_s=None):
        sampler_calls.append(float(ts))
        return [f"decoded-{ts:.2f}".encode()]

    monkeypatch.setattr("nexoclip.vision.frame_sampler.sample_frames", fake_sample_frames)

    asyncio.run(
        generate_variants(
            tenant_id="default",
            clip=clip,
            persona=persona,
            router=router,
            n=3,
            use_vision=True,
            frame_cache=cache,
            n_vision_frames=3,
        )
    )
    # 3 frames sampled -> 3 sampler calls -> 3 cache entries seeded.
    assert len(sampler_calls) == 3
    assert len(cache) == 3
    # Frames flowed into the multimodal call.
    call = fake.calls[0]
    assert call["kind"] == "multimodal"
    assert call["n_images"] == 3
    assert call["images"][0].data.startswith(b"decoded-")


def test_use_vision_default_off_still_uses_text_path(tmp_path: Path) -> None:
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
    assert fake.calls[0]["kind"] == "text"


def test_use_vision_invalid_n_frames_raises(tmp_path: Path) -> None:
    clip = make_clip(tmp_path)
    persona = make_persona()
    fake = FakeProvider("anthropic")
    fake.queue_success(_success_payload(n=3))
    router = make_router_with_fake(fake)

    with pytest.raises(VariantError, match="n_vision_frames"):
        asyncio.run(
            generate_variants(
                tenant_id="default",
                clip=clip,
                persona=persona,
                router=router,
                n=3,
                use_vision=True,
                frame_cache=FrameCache(),
                n_vision_frames=0,
            )
        )
