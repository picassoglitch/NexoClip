"""Voice-trigger detection over a Whisper transcript.

Algorithm:
1. Flatten the transcript into `(text, ts, end_ts, prob)` tuples.
2. For each configured phrase, slide a window the size of the phrase token
   count across the flat list, computing Levenshtein on the joined+normalized
   strings. A hit is any window with edit distance <= `fuzzy_distance`.
3. Each hit becomes a `Candidate(timestamp, score, reason="voice", evidence)`.
4. Cluster candidates whose timestamps are within `merge_window_s` of the
   previous candidate; pick the highest-scoring as the cluster anchor and
   union evidence under `evidence["matches"]`.

Phase 0 has only this detector; chat/audio/visual fan-in arrives in Phase 1
and will share the same `Candidate` shape.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from nexoclip.config import DetectionConfig
from nexoclip.errors import DetectionError
from nexoclip.ingest import Stream
from nexoclip.transcribe import Transcript

from .levenshtein import levenshtein
from .models import Candidate, CandidateBatch


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
) -> list[Candidate]:
    """Return merged voice-trigger candidates for `stream`.

    All inputs must agree on `tenant_id` (CLAUDE.md hard rule #1).
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
    for language, phrases in voice_cfg.phrases.items():
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
                dist = levenshtein(phrase_norm, joined, max_dist=voice_cfg.fuzzy_distance)
                if dist > voice_cfg.fuzzy_distance:
                    continue
                confidence = sum(w.prob for w in window) / len(window)
                score = voice_cfg.weight * confidence
                snippet = " ".join(w.text.strip() for w in window)
                raw.append(
                    Candidate(
                        timestamp=window[0].ts,
                        score=score,
                        reason="voice",
                        evidence={
                            "phrase": phrase,
                            "language": language,
                            "transcript_snippet": snippet,
                            "distance": dist,
                            "confidence": confidence,
                            "end_ts": window[-1].end_ts,
                        },
                    )
                )

    return _merge_candidates(raw, window_s=config.merge_window_s)


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
