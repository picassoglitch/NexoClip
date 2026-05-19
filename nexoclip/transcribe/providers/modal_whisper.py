"""ModalWhisperProvider — GPU Whisper via Modal (slice O.44).

Deploy plan:
  1. `modal deploy infra/modal_whisper_app.py` — once on the operator's
     box. Modal prints the endpoint URL.
  2. Set in Railway env:
        NEXOCLIP_TRANSCRIBE_PROVIDER=modal
        NEXOCLIP_MODAL_ENDPOINT_URL=<the URL Modal printed>
        NEXOCLIP_MODAL_TOKEN=<long random string, same on both sides>
        NEXOCLIP_INTERNAL_SIGNING_SECRET=<long random string>
        NEXOCLIP_MODAL_MODEL=small  (optional; default small)
        NEXOCLIP_PUBLIC_URL=https://nexoclip-production.up.railway.app  (already
            set; Modal pulls audio from this base + signed query)
  3. On Modal side, create the secret the app expects:
        modal secret create nexoclip-modal-token MODAL_BEARER_TOKEN=<same>

Once both sides agree on the bearer token, NexoClip POSTs the audio
URL + signed token to Modal, Modal pulls + transcribes + returns
JSON, and the pipeline keeps going.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any

import httpx
import structlog

from nexoclip.errors import TranscriptionError
from nexoclip.transcribe.models import Segment, Transcript, Word

from .base import TranscribeRequest

_log = structlog.get_logger(__name__)


class ModalWhisperProvider:
    """Forwards transcribe to a Modal-hosted faster-whisper container.

    Behavior contract is identical to `LocalWhisperProvider`: returns
    a `Transcript` or raises `TranscriptionError`. The pipeline can
    swap between local and modal via NEXOCLIP_TRANSCRIBE_PROVIDER
    without changing anything else.
    """

    def __init__(
        self,
        *,
        endpoint_url: str,
        bearer_token: str,
        signing_secret: str,
        public_base_url: str,
        model: str = "small",
        request_timeout_s: float = 3600.0,
    ) -> None:
        self._endpoint_url = endpoint_url.rstrip("/")
        self._bearer_token = bearer_token
        self._signing_secret = signing_secret
        self._public_base_url = public_base_url.rstrip("/")
        self._model = model
        self._timeout_s = request_timeout_s

    @property
    def name(self) -> str:
        return f"modal-whisper-{self._model}"

    async def transcribe(self, req: TranscribeRequest) -> Transcript:
        if not self._endpoint_url or not self._bearer_token:
            raise TranscriptionError(
                "ModalWhisperProvider misconfigured: NEXOCLIP_MODAL_ENDPOINT_URL "
                "and NEXOCLIP_MODAL_TOKEN must be set."
            )
        if not self._signing_secret:
            raise TranscriptionError(
                "ModalWhisperProvider misconfigured: NEXOCLIP_INTERNAL_SIGNING_SECRET "
                "must be set so Modal can pull audio from this server."
            )
        if not self._public_base_url:
            raise TranscriptionError(
                "ModalWhisperProvider misconfigured: NEXOCLIP_PUBLIC_URL must "
                "be set to the externally-reachable base URL (e.g. "
                "https://nexoclip-production.up.railway.app) so Modal can "
                "fetch the audio."
            )

        audio_url = self._build_signed_audio_url(req.stream_id, req.tenant_id)
        body: dict[str, Any] = {
            "audio_url": audio_url,
            "language": req.language,
            "model": self._model,
            "stream_id": req.stream_id,
            # Bearer token also in the payload — Modal's web_endpoint
            # quirks make header auth awkward; keeping it in the body
            # is simpler + still over HTTPS so it's encrypted in flight.
            "auth_token": self._bearer_token,
        }
        _log.info(
            "modal_whisper.dispatch",
            stream_id=req.stream_id,
            tenant_id=req.tenant_id,
            model=self._model,
            audio_url_host=audio_url.split("/")[2] if "//" in audio_url else "?",
        )
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(self._endpoint_url, json=body)
                resp.raise_for_status()
                raw = resp.json()
        except httpx.HTTPStatusError as e:
            raise TranscriptionError(
                f"modal whisper returned {e.response.status_code}: "
                f"{e.response.text[:500]}"
            ) from e
        except httpx.HTTPError as e:
            raise TranscriptionError(
                f"modal whisper transport error: {e}"
            ) from e

        return _modal_response_to_transcript(raw, req=req)

    def _build_signed_audio_url(self, stream_id: str, tenant_id: str) -> str:
        """HMAC-signed URL valid for 30 min so Modal can pull the audio.

        Shape: /api/internal/audio/<stream_id>?tenant=<tid>&exp=<unix>&sig=<hmac>
        Server side verifies sig + exp before serving the file.
        """
        exp = int(time.time()) + 30 * 60  # 30 min window
        msg = f"{stream_id}|{tenant_id}|{exp}".encode()
        sig = hmac.new(
            self._signing_secret.encode(), msg, hashlib.sha256
        ).hexdigest()
        return (
            f"{self._public_base_url}/api/internal/audio/{stream_id}"
            f"?tenant={tenant_id}&exp={exp}&sig={sig}"
        )


def _modal_response_to_transcript(
    raw: dict[str, Any], *, req: TranscribeRequest
) -> Transcript:
    """Map the Modal app's JSON response into our Transcript model.

    The Modal app's `transcribe()` shape is documented in
    infra/modal_whisper_app.py — see the docstring there.
    """
    try:
        segments = []
        for s in raw.get("segments", []):
            words = [
                Word(
                    text=w.get("word", ""),
                    ts=float(w.get("start", 0.0)),
                    end_ts=float(w.get("end", 0.0)),
                    prob=max(0.0, min(1.0, float(w.get("prob", 0.0)))),
                )
                for w in s.get("words", [])
            ]
            segments.append(
                Segment(
                    text=s.get("text", "").strip(),
                    ts=float(s.get("start", 0.0)),
                    end_ts=float(s.get("end", 0.0)),
                    words=words,
                )
            )
        return Transcript(
            stream_id=req.stream_id,
            tenant_id=req.tenant_id,
            language=str(raw.get("language") or req.language or "es"),
            duration_s=float(raw.get("duration_s", 0.0)),
            model=f"modal-{raw.get('model', '?')}",
            segments=segments,
        )
    except Exception as e:  # noqa: BLE001
        raise TranscriptionError(
            f"modal whisper response unparseable: {e}; raw keys: "
            f"{sorted(raw.keys())[:8]}"
        ) from e
