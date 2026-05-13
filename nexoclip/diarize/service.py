"""Async wrapper around the diarization subprocess worker.

Public entry: `diarize(...)` returns a `Diarization` (possibly empty +
flagged `skipped=True`). Never raises into the pipeline — a skipped
diarization is a valid outcome and downstream consumers tolerate
missing speaker labels.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import structlog

from nexoclip.config import DiarizationConfig
from nexoclip.ingest import Stream

from .models import Diarization, DiarizationSegment, SpeakerEmbedding

_log = structlog.get_logger(__name__)


def _diarization_path(stream: Stream) -> Path:
    return stream.source_audio_path.parent / "diarization.json"


def is_diarization_available() -> bool:
    """Cheap pre-flight check used by the dashboard's pipeline-status surface.

    True when both:
      * `HF_TOKEN` is set in the environment, AND
      * `pyannote.audio` can be imported.
    Either failure means diarization will be skipped; this lets the
    dashboard show 'speaker labels unavailable' without spawning a worker.
    """
    if not os.environ.get("HF_TOKEN", "").strip():
        return False
    try:
        import importlib.util

        return importlib.util.find_spec("pyannote.audio") is not None
    except Exception:  # noqa: BLE001
        return False


async def diarize(
    tenant_id: str,
    stream: Stream,
    *,
    config: DiarizationConfig | None = None,
    force: bool = False,
) -> Diarization:
    """Diarize `stream.source_audio_path`, returning per-speaker turns + embeddings.

    Idempotent: a cached `diarization.json` is reused unless `force=True`.

    Skippable: returns an empty Diarization with `skipped=True` and a
    one-line `skip_reason` when:
        * `config.enabled=False`
        * HF_TOKEN missing
        * pyannote.audio not importable
        * The subprocess exits non-zero (CUDA OOM, model load fail, etc.)
    """
    cfg = config or DiarizationConfig()
    out_path = _diarization_path(stream)

    if not cfg.enabled:
        return Diarization(
            stream_id=stream.id,
            tenant_id=tenant_id,
            skipped=True,
            skip_reason="diarization disabled in config",
        )

    if not force and out_path.exists():
        try:
            cached = Diarization.model_validate_json(out_path.read_text("utf-8"))
            # If the previous run was skipped for a transient reason and the
            # condition has cleared (e.g. user just added HF_TOKEN), force a
            # fresh attempt. We detect this only on the 'token missing' case
            # since it's the most common cause of a stale skip.
            if cached.skipped and (
                cached.skip_reason or ""
            ).lower().find("hf_token") >= 0 and os.environ.get("HF_TOKEN"):
                pass  # fall through to re-run
            else:
                return cached
        except Exception as e:  # noqa: BLE001
            _log.warning(
                "diarize.cache_invalid", path=str(out_path), reason=str(e)
            )

    if not stream.source_audio_path.exists():
        skipped = Diarization(
            stream_id=stream.id,
            tenant_id=tenant_id,
            skipped=True,
            skip_reason=f"audio file missing: {stream.source_audio_path}",
        )
        out_path.write_text(skipped.model_dump_json(indent=2), encoding="utf-8")
        return skipped

    if not is_diarization_available():
        # Decide which sub-reason applies so the dashboard can show a useful hint.
        if not os.environ.get("HF_TOKEN", "").strip():
            reason = (
                "HF_TOKEN not set. Add HF_TOKEN=hf_... to .env after accepting "
                "the pyannote license at hf.co/pyannote/speaker-diarization-3.1."
            )
        else:
            reason = (
                "pyannote.audio not installed. Run: "
                "pip install 'pyannote.audio>=3.1,<4'"
            )
        skipped = Diarization(
            stream_id=stream.id,
            tenant_id=tenant_id,
            skipped=True,
            skip_reason=reason,
        )
        out_path.write_text(skipped.model_dump_json(indent=2), encoding="utf-8")
        _log.info("diarize.skipped", reason=reason, stream_id=stream.id)
        return skipped

    result = await asyncio.to_thread(
        _run_worker,
        audio_path=stream.source_audio_path,
        stream_id=stream.id,
        tenant_id=tenant_id,
        model=cfg.model,
        device=cfg.device,
    )
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def _run_worker(
    *,
    audio_path: Path,
    stream_id: str,
    tenant_id: str,
    model: str,
    device: str,
) -> Diarization:
    """Spawn `nexoclip.diarize._worker` and parse its JSON output.

    A non-zero exit code is converted to a `skipped=True` Diarization so
    the pipeline can continue without speaker labels. The dashboard
    surfaces the stderr tail in the failed-step hint.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        out_path = Path(tmp.name)
    try:
        cmd = [
            sys.executable,
            "-m",
            "nexoclip.diarize._worker",
            "--audio",
            str(audio_path),
            "--stream-id",
            stream_id,
            "--tenant-id",
            tenant_id,
            "--model",
            model,
            "--device",
            device,
            "--out",
            str(out_path),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            tail = (result.stderr or "")[-1500:]
            _log.warning(
                "diarize.worker_failed",
                exit_code=result.returncode,
                stderr_tail=tail,
                stream_id=stream_id,
            )
            return Diarization(
                stream_id=stream_id,
                tenant_id=tenant_id,
                skipped=True,
                skip_reason=(
                    f"diarization worker exited {result.returncode}. "
                    f"Tail: {tail.strip().splitlines()[-1] if tail.strip() else 'no output'}"
                ),
            )
        if not out_path.exists():
            return Diarization(
                stream_id=stream_id,
                tenant_id=tenant_id,
                skipped=True,
                skip_reason=(
                    f"diarization worker exited 0 but produced no output at {out_path}"
                ),
            )
        raw = json.loads(out_path.read_text("utf-8"))
        return Diarization(
            stream_id=stream_id,
            tenant_id=tenant_id,
            segments=[
                DiarizationSegment.model_validate(s) for s in raw.get("segments", [])
            ],
            embeddings=[
                SpeakerEmbedding.model_validate(e)
                for e in raw.get("embeddings", [])
            ],
            skipped=False,
            skip_reason=None,
        )
    finally:
        try:
            out_path.unlink()
        except OSError:
            pass


def load_diarization(stream_dir: Path) -> Diarization | None:
    """Read the cached diarization from `<stream_dir>/source/diarization.json`."""
    path = Path(stream_dir) / "source" / "diarization.json"
    if not path.exists():
        return None
    try:
        return Diarization.model_validate_json(path.read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return None
