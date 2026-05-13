"""Trigger detection — voice-phrase + chat-heat fan-in.

Detectors:
1. Voice triggers (existing Phase 0 logic, internal to this file): slide
   each configured phrase across the Whisper transcript and emit a hit
   when Levenshtein <= `fuzzy_distance`.
2. Chat heat (Phase 1, in `chat_heat.py`): rolling-baseline spike test
   on chat-replay msg/sec.

`detect_candidates(...)` is the public entry point that runs both detectors
and merges their output. Adjacent candidates within `merge_window_s` get
clustered; the highest-scoring per cluster wins, and per-signal evidence
unions under `evidence["matches"]` so reviewers can see which detectors
fired together.

Phase 1 ships voice + chat. Audio energy lands in Task 4; visual signals
in Tasks 5-6. All three plug into the same fusion path.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from nexoclip.config import DetectionConfig
from nexoclip.errors import DetectionError
from nexoclip.ingest import ChatReplay, Stream
from nexoclip.transcribe import Transcript
from nexoclip.vision import VisualSignalTrack

from .audio_energy import detect_audio_energy
from .chat_heat import detect_chat_heat
from .levenshtein import levenshtein
from .models import Candidate, CandidateBatch
from .viral import detect_viral_moments
from .visual_signals import detect_visual_candidates


@dataclass(frozen=True)
class _FlatWord:
    text: str
    ts: float
    end_ts: float
    prob: float


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Accents are preserved — `clipéalo` and `clipealo` should match via the
    fuzzy distance, not via aggressive normalization that loses signal.
    """
    text = unicodedata.normalize("NFC", text)
    text = _PUNCT_RE.sub(" ", text.lower())
    return _WS_RE.sub(" ", text).strip()


def _flatten(transcript: Transcript) -> list[_FlatWord]:
    flat: list[_FlatWord] = []
    for seg in transcript.segments:
        for w in seg.words:
            flat.append(_FlatWord(text=w.text, ts=w.ts, end_ts=w.end_ts, prob=w.prob))
    return flat


def detect_voice_triggers(
    tenant_id: str,
    stream: Stream,
    transcript: Transcript,
    config: DetectionConfig,
    *,
    diarization: object | None = None,
    extra_phrases_by_speaker: dict[str, object] | None = None,
) -> list[Candidate]:
    """Return merged voice-trigger candidates for `stream`.

    All inputs must agree on `tenant_id` (CLAUDE.md hard rule #1).

    `diarization` is optional. When supplied (a `nexoclip.diarize.Diarization`
    with `skipped=False`), each emitted candidate carries a `speaker_label`
    in evidence, and the cooldown is applied per (speaker, trigger_kind)
    instead of globally — so a co-host firing 'clipea esto' 1s after the
    host still produces a separate clip. The argument is typed as `object`
    here to avoid importing nexoclip.diarize at module top (would create a
    cycle: pipeline → detect → diarize → db, all already touched).

    `extra_phrases_by_speaker` lets each speaker contribute their brand-kit's
    custom trigger phrases on top of the tenant base list. Map shape:
        {speaker_label: CustomTriggerPhrases(forward=[...], retroactive=[...])}
    Typed as `dict[str, object]` here for the same reason as `diarization` —
    the nexoclip.db.models import would form a back-edge through the db
    package. The values must duck-type to `.forward` + `.retroactive`
    (list[str] each); the resolver in `nexoclip.branding.service` produces
    them. Extra-phrase hits are restricted to candidates whose attributed
    speaker matches the map key — a co-host's kit can't accidentally
    create candidates for words the host said.
    """
    if tenant_id != stream.tenant_id:
        raise DetectionError(f"tenant mismatch: caller={tenant_id!r}, stream={stream.tenant_id!r}")
    if tenant_id != transcript.tenant_id:
        raise DetectionError(
            f"tenant mismatch: caller={tenant_id!r}, transcript={transcript.tenant_id!r}"
        )
    if stream.id != transcript.stream_id:
        raise DetectionError(
            f"stream/transcript mismatch: stream={stream.id} transcript={transcript.stream_id}"
        )

    voice_cfg = config.voice
    if not voice_cfg.enabled:
        return []

    flat = _flatten(transcript)
    if not flat:
        return []

    raw: list[Candidate] = []
    # Forward triggers — clip extends forward from the timestamp.
    _scan_phrase_family(
        flat=flat,
        phrases_by_lang=voice_cfg.phrases,
        fuzzy_distance=voice_cfg.fuzzy_distance,
        weight=voice_cfg.weight,
        kind="forward",
        out=raw,
        diarization=diarization,
    )
    # Retroactive triggers — clip extends BACKWARD from the timestamp.
    _scan_phrase_family(
        flat=flat,
        phrases_by_lang=voice_cfg.retroactive_phrases,
        fuzzy_distance=voice_cfg.fuzzy_distance,
        weight=voice_cfg.weight,
        kind="retroactive",
        out=raw,
        retroactive_lookback_s=voice_cfg.retroactive_lookback_s,
        diarization=diarization,
    )
    # Per-kit additions — each speaker's brand kit contributes additional
    # forward/retroactive phrases. Each scan is restricted to candidates
    # whose attributed speaker matches the map key, so a co-host's kit
    # can't fire on words the host said. The detector cooldown step
    # below then dedupes anything that also matched a base phrase.
    if extra_phrases_by_speaker:
        for speaker_label, phrases in extra_phrases_by_speaker.items():
            kit_forward = list(getattr(phrases, "forward", []) or [])
            kit_retroactive = list(getattr(phrases, "retroactive", []) or [])
            if not kit_forward and not kit_retroactive:
                continue
            # Language is unknown for kit-added phrases; treat them as
            # "any language" by routing all of them through the 'es' bucket,
            # which is the only language the detector currently scans
            # word-for-word. Future i18n: kit phrases gain an explicit lang.
            if kit_forward:
                _scan_phrase_family(
                    flat=flat,
                    phrases_by_lang={"es": kit_forward},
                    fuzzy_distance=voice_cfg.fuzzy_distance,
                    weight=voice_cfg.weight,
                    kind="forward",
                    out=raw,
                    diarization=diarization,
                    restrict_to_speaker=speaker_label,
                )
            if kit_retroactive:
                _scan_phrase_family(
                    flat=flat,
                    phrases_by_lang={"es": kit_retroactive},
                    fuzzy_distance=voice_cfg.fuzzy_distance,
                    weight=voice_cfg.weight,
                    kind="retroactive",
                    out=raw,
                    retroactive_lookback_s=voice_cfg.retroactive_lookback_s,
                    diarization=diarization,
                    restrict_to_speaker=speaker_label,
                )
    # Per-speaker cooldown — keeps the first trigger of each
    # (speaker, kind) within `cooldown_s`. Falls back to a global
    # cooldown when no diarization was attached.
    pruned = _apply_per_speaker_cooldown(raw, cooldown_s=voice_cfg.cooldown_s)
    return _merge_candidates(pruned, window_s=config.merge_window_s)


def _scan_phrase_family(
    *,
    flat: list[_FlatWord],
    phrases_by_lang: dict[str, list[str]],
    fuzzy_distance: int,
    weight: float,
    kind: str,
    out: list[Candidate],
    retroactive_lookback_s: float | None = None,
    diarization: object | None = None,
    restrict_to_speaker: str | None = None,
) -> None:
    """Append candidates for one phrase family (forward OR retroactive) to `out`.

    The two families share the entire matching algorithm; only the
    `evidence['trigger_kind']` flag and the optional retroactive metadata
    differ. Cut step reads `trigger_kind` to decide the window direction.

    When `diarization` is provided and `.overlap_speaker(...)` returns a
    label, the candidate carries `evidence['speaker_label']` so the
    per-speaker cooldown + downstream brand-kit-by-speaker resolution
    can use it.

    `restrict_to_speaker` is the per-kit-phrases filter (slice C.2). When
    set, ONLY emissions whose attributed speaker_label matches the
    argument survive — a co-host's kit can't add candidates on the host's
    words.
    """
    for language, phrases in phrases_by_lang.items():
        for phrase in phrases:
            phrase_norm = _normalize(phrase)
            if not phrase_norm:
                continue
            phrase_tokens = phrase_norm.split()
            window_len = len(phrase_tokens)
            if window_len == 0 or window_len > len(flat):
                continue
            for i in range(len(flat) - window_len + 1):
                window = flat[i : i + window_len]
                joined = " ".join(_normalize(w.text) for w in window).strip()
                if not joined:
                    continue
                dist = levenshtein(phrase_norm, joined, max_dist=fuzzy_distance)
                if dist > fuzzy_distance:
                    continue
                confidence = sum(w.prob for w in window) / len(window)
                score = weight * confidence
                snippet = " ".join(w.text.strip() for w in window)
                evidence: dict[str, object] = {
                    "phrase": phrase,
                    "language": language,
                    "transcript_snippet": snippet,
                    "distance": dist,
                    "confidence": confidence,
                    "end_ts": window[-1].end_ts,
                    "trigger_kind": kind,
                }
                if kind == "retroactive" and retroactive_lookback_s is not None:
                    evidence["retroactive_lookback_s"] = retroactive_lookback_s
                # Attribute to a speaker if diarization is available and the
                # turn covers (or mostly overlaps) the phrase span.
                speaker_label = _attribute_speaker(
                    diarization, window[0].ts, window[-1].end_ts
                )
                if restrict_to_speaker is not None and speaker_label != restrict_to_speaker:
                    # Kit-added phrase fired on the wrong speaker — drop it.
                    continue
                if speaker_label is not None:
                    evidence["speaker_label"] = speaker_label
                out.append(
                    Candidate(
                        timestamp=window[0].ts,
                        score=score,
                        reason="voice",
                        evidence=evidence,
                    )
                )


def _attribute_speaker(
    diarization: object | None, ts: float, end_ts: float
) -> str | None:
    """Best-effort speaker attribution; returns None when diarization is absent
    or doesn't cover this phrase span.

    Typed loosely as `object` to keep the diarize package decoupled from
    detect at module-load time. Falls back to None if the dynamic call
    fails for any reason — never raises into the detector."""
    if diarization is None:
        return None
    overlap = getattr(diarization, "overlap_speaker", None)
    skipped = getattr(diarization, "skipped", True)
    if not callable(overlap) or skipped:
        return None
    try:
        label = overlap(ts, end_ts)
    except Exception:  # noqa: BLE001
        return None
    return label if isinstance(label, str) else None


def _apply_per_speaker_cooldown(
    candidates: list[Candidate], *, cooldown_s: float
) -> list[Candidate]:
    """Drop later candidates within `cooldown_s` of an earlier same-kind hit.

    Cooldown key is (speaker_label, trigger_kind). When speaker_label is
    absent (no diarization, or speaker unattributable on a given hit), all
    no-speaker candidates of the same kind share one key — equivalent to
    the global cooldown the spec describes when diarization is off.

    Stable order: candidates are walked in ascending-timestamp order, the
    first wins per cooldown bucket. Always preserves at least the first
    hit per (speaker, kind).
    """
    if cooldown_s <= 0.0 or not candidates:
        return list(candidates)
    sorted_c = sorted(candidates, key=lambda c: c.timestamp)
    kept: list[Candidate] = []
    last_ts_by_key: dict[tuple[str | None, str], float] = {}
    for c in sorted_c:
        ev = c.evidence or {}
        speaker = ev.get("speaker_label")
        speaker_key: str | None = speaker if isinstance(speaker, str) else None
        kind_raw = ev.get("trigger_kind", "forward")
        kind = str(kind_raw) if kind_raw is not None else "forward"
        key = (speaker_key, kind)
        last_ts = last_ts_by_key.get(key)
        if last_ts is not None and c.timestamp - last_ts < cooldown_s:
            continue
        kept.append(c)
        last_ts_by_key[key] = c.timestamp
    return kept


def _merge_candidates(candidates: list[Candidate], *, window_s: float) -> list[Candidate]:
    """Collapse temporally-close candidates: highest score wins, evidence unions."""
    if not candidates:
        return []
    if window_s <= 0:
        return sorted(candidates, key=lambda c: c.timestamp)

    sorted_c = sorted(candidates, key=lambda c: c.timestamp)
    clusters: list[list[Candidate]] = [[sorted_c[0]]]
    for c in sorted_c[1:]:
        if c.timestamp - clusters[-1][-1].timestamp <= window_s:
            clusters[-1].append(c)
        else:
            clusters.append([c])

    merged: list[Candidate] = []
    for cluster in clusters:
        winner = max(cluster, key=lambda c: c.score)
        if len(cluster) == 1:
            merged.append(winner)
            continue
        evidence = {
            **winner.evidence,
            "matches": [c.evidence for c in cluster],
            "merged_count": len(cluster),
        }
        merged.append(
            Candidate(
                timestamp=winner.timestamp,
                score=winner.score,
                reason=winner.reason,
                evidence=evidence,
            )
        )
    return merged


def detect_candidates(
    tenant_id: str,
    stream: Stream,
    transcript: Transcript,
    config: DetectionConfig,
    *,
    chat_replay: ChatReplay | None = None,
    visual_track: VisualSignalTrack | None = None,
    viral_candidates: list[Candidate] | None = None,
    diarization: object | None = None,
    extra_phrases_by_speaker: dict[str, object] | None = None,
) -> list[Candidate]:
    """Run every available detector and return the fused candidate stream.

    Detectors:
        * voice triggers — fuzzy phrase match on the transcript
        * chat heat — rolling-baseline spike on msg/sec (when chat_replay set)
        * audio energy — RMS spike on the source audio (when enabled)
        * visual signals — scene cuts / faces / motion (when visual_track set)
        * viral — pass-through; the caller runs `detect_viral_moments()`
          (async, needs an LLMRouter) and forwards the result here. This
          keeps the sync vs async boundary clean — `detect_candidates` stays
          sync and pure; the pipeline awaits the LLM call once and threads
          the candidates in.

    Each detector emits its own Candidates; this function concatenates them
    and runs `_merge_candidates` so hits from different signals within the
    same window collapse into one cluster with `evidence["matches"]`
    listing each source.
    """
    voice = detect_voice_triggers(
        tenant_id=tenant_id,
        stream=stream,
        transcript=transcript,
        config=config,
        diarization=diarization,
        extra_phrases_by_speaker=extra_phrases_by_speaker,
    )
    chat: list[Candidate] = []
    if chat_replay is not None:
        chat = detect_chat_heat(
            tenant_id=tenant_id,
            stream=stream,
            chat_replay=chat_replay,
            config=config.chat_heat,
        )
    audio: list[Candidate] = detect_audio_energy(
        tenant_id=tenant_id, stream=stream, config=config.audio_energy
    )
    visual: list[Candidate] = []
    if visual_track is not None:
        visual = detect_visual_candidates(
            tenant_id=tenant_id,
            stream=stream,
            track=visual_track,
            config=config.visual,
        )
    viral = viral_candidates or []
    if not chat and not audio and not visual and not viral:
        # Voice-only — already merged by detect_voice_triggers.
        return voice
    return _merge_candidates(
        voice + chat + audio + visual + viral, window_s=config.merge_window_s
    )


def save_candidates(stream_dir: Path, batch: CandidateBatch) -> Path:
    """Persist candidates to `<stream_dir>/candidates.json`."""
    out = Path(stream_dir) / "candidates.json"
    out.write_text(batch.model_dump_json(indent=2), encoding="utf-8")
    return out


def load_candidates(stream_dir: Path) -> CandidateBatch:
    """Read candidates back from disk."""
    path = Path(stream_dir) / "candidates.json"
    if not path.exists():
        raise DetectionError(f"candidates not found at {path}")
    return CandidateBatch.model_validate_json(path.read_text("utf-8"))
