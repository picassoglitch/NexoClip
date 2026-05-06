"""Tests for `detect_candidates` — voice + chat fusion."""

from __future__ import annotations

from nexoclip.config import ChatHeatConfig, DetectionConfig, VoiceDetectorConfig
from nexoclip.detect import detect_candidates
from nexoclip.ingest import ChatMessage, ChatReplay

from ._fixtures import make_stream, make_transcript


def _detection_config(
    *,
    voice_phrases: list[str] | None = None,
    chat_enabled: bool = True,
    merge_window_s: float = 30.0,
) -> DetectionConfig:
    return DetectionConfig(
        voice=VoiceDetectorConfig(
            enabled=True,
            weight=1.0,
            fuzzy_distance=2,
            phrases={"es": voice_phrases or ["clipéalo"]},
        ),
        chat_heat=ChatHeatConfig(
            enabled=chat_enabled,
            weight=0.7,
            baseline_window_s=10.0,
            spike_ratio=3.0,
            absolute_floor_msg_per_s=5.0,
        ),
        merge_window_s=merge_window_s,
    )


def _chat_with_spike_at(ts: int, *, baseline_seconds: int = 10) -> ChatReplay:
    """Build chat replay where the `baseline_seconds` immediately preceding
    `ts` are quiet (1 msg/sec) and `ts` itself is a 20-msg spike."""
    msgs = [
        ChatMessage(ts=float(s), user="u", text="filler")
        for s in range(max(0, ts - baseline_seconds), ts)
    ]
    msgs.extend(
        ChatMessage(ts=float(ts) + i * 0.001, user="fan", text="hype")
        for i in range(20)
    )
    return ChatReplay(stream_id="str_01TEST", tenant_id="default", messages=msgs)


def test_voice_only_when_no_chat_replay() -> None:
    transcript = make_transcript(words=[(10.0, 11.0, " clipéalo", 0.9)])
    candidates = detect_candidates(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=_detection_config(),
        chat_replay=None,
    )
    assert len(candidates) == 1
    assert candidates[0].reason == "voice"


def test_chat_only_when_no_voice_match() -> None:
    transcript = make_transcript(words=[(0.5, 1.0, " hola", 0.9)])
    chat = _chat_with_spike_at(15)
    candidates = detect_candidates(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=_detection_config(),
        chat_replay=chat,
    )
    assert len(candidates) == 1
    assert candidates[0].reason == "chat"
    assert candidates[0].timestamp == 15.0


def test_voice_and_chat_fuse_within_merge_window() -> None:
    """A voice trigger at t=12 and a chat spike at t=15, both within
    merge_window_s=30. They should collapse to one cluster with both
    pieces of evidence under `evidence['matches']`."""
    transcript = make_transcript(words=[(12.0, 13.0, " clipéalo", 0.9)])
    chat = _chat_with_spike_at(15)
    candidates = detect_candidates(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=_detection_config(merge_window_s=30.0),
        chat_replay=chat,
    )
    assert len(candidates) == 1
    c = candidates[0]
    assert "matches" in c.evidence
    assert len(c.evidence["matches"]) == 2
    reasons = {m.get("phrase") or m.get("ratio") for m in c.evidence["matches"]}
    # Either the phrase string or the ratio float — order-independent membership.
    assert any(r == "clipéalo" for r in reasons)


def test_voice_and_chat_far_apart_stay_separate() -> None:
    """Voice at t=12, chat spike at t=200 — well outside merge_window_s=30."""
    transcript = make_transcript(words=[(12.0, 13.0, " clipéalo", 0.9)])
    chat = _chat_with_spike_at(200)
    candidates = detect_candidates(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=_detection_config(merge_window_s=30.0),
        chat_replay=chat,
    )
    assert len(candidates) == 2
    reasons = sorted(c.reason for c in candidates)
    assert reasons == ["chat", "voice"]


def test_chat_disabled_still_returns_voice() -> None:
    """When chat_heat.enabled=False, fusion just returns voice candidates."""
    transcript = make_transcript(words=[(10.0, 11.0, " clipéalo", 0.9)])
    chat = _chat_with_spike_at(15)
    candidates = detect_candidates(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=_detection_config(chat_enabled=False),
        chat_replay=chat,
    )
    assert len(candidates) == 1
    assert candidates[0].reason == "voice"


def test_consecutive_chat_spike_seconds_collapse_to_one_candidate() -> None:
    """detect_chat_heat emits one candidate per spike second; fusion
    merges them via merge_window_s into a single cluster.

    The fourth spike second (sec 13) doesn't fire on its own — by then
    the prior spikes have lifted the rolling baseline above 5, so 15
    msg/sec is no longer 3x the baseline. The first three (secs 10, 11,
    12) all fire and get merged.
    """
    msgs = [ChatMessage(ts=float(s), user="u", text="f") for s in range(10)]
    for sec in (10, 11, 12, 13):
        msgs.extend(
            ChatMessage(ts=float(sec) + i * 0.001, user="fan", text="hype")
            for i in range(15)
        )
    chat = ChatReplay(stream_id="str_01TEST", tenant_id="default", messages=msgs)

    transcript = make_transcript(words=[(0.0, 0.5, " hola", 0.9)])
    candidates = detect_candidates(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=_detection_config(merge_window_s=30.0),
        chat_replay=chat,
    )
    assert len(candidates) == 1
    assert candidates[0].reason == "chat"
    assert "matches" in candidates[0].evidence
    assert len(candidates[0].evidence["matches"]) == 3
