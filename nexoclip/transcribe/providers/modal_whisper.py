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

import asyncio
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

# Modal's web-endpoint protocol for long-running functions: the initial
# POST returns 303 See Other with Location: ?__modal_function_call_id=fc-...,
# and that polling URL keeps redirecting (302 / 303) until the function
# completes (200 + body) or fails (4xx / 5xx). The previous version of
# this provider treated the first 303 as a hard error — that's the bug
# the operator hit on 06-03: `modal whisper returned 303: ` with empty
# body, because the body of a 303 is the redirect HTML, not the result.
_POLL_INTERVAL_S = 5.0
"""How long to wait between poll requests. Modal's Whisper-small on T4
takes ~30-90s for a typical VOD, so a 5s cadence is ~6-18 polls per run."""
_POLL_REDIRECT_STATUSES = (302, 303, 307, 308)
"""HTTP redirect codes Modal uses to mean 'still running, try the
Location URL again later'. We treat all four uniformly."""


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
                # Modal's async-function protocol — POST returns 303 with
                # a polling Location. Follow until the function completes
                # (200 + body) or the deadline expires. See module-level
                # `_POLL_*` constants for cadence rationale.
                resp = await _poll_until_terminal(
                    client=client,
                    initial=resp,
                    deadline_s=self._timeout_s,
                    stream_id=req.stream_id,
                )
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


async def _poll_until_terminal(
    *,
    client: httpx.AsyncClient,
    initial: httpx.Response,
    deadline_s: float,
    stream_id: str,
) -> httpx.Response:
    """Follow Modal's redirect-based polling protocol until the response
    is a terminal one (any non-redirect status).

    Modal's web-endpoint for a long-running function:
      POST  /  → 303 + Location: /?__modal_function_call_id=fc-...
      GET   /?__modal_function_call_id=fc-... → 302 / 303 while running,
                                                  200 once the function
                                                  returns its body.

    Behavior:
      * The initial response is the result of the POST. If it's already
        non-redirect (e.g. small/fast Modal app or short audio), return
        it immediately.
      * Otherwise GET the Location URL after `_POLL_INTERVAL_S` and keep
        looping until either a non-redirect response lands or `deadline_s`
        elapses.
      * On timeout we raise — caller's outer try/except surfaces it
        with the same TranscriptionError shape as a real HTTP failure.

    Same return contract as a direct request: `raise_for_status()` is the
    caller's responsibility.
    """
    if initial.status_code not in _POLL_REDIRECT_STATUSES:
        return initial

    deadline = time.monotonic() + max(0.0, deadline_s)
    current = initial
    polls = 0
    while current.status_code in _POLL_REDIRECT_STATUSES:
        location = current.headers.get("location")
        if not location:
            # Redirect without a Location — protocol broken; treat the
            # response as terminal so the caller's raise_for_status
            # surfaces the unexpected 30x as a clear error.
            _log.warning(
                "modal_whisper.poll.no_location",
                stream_id=stream_id, status=current.status_code,
            )
            return current
        # Resolve the (possibly relative) Location against the response's
        # own request URL so trailing-slash + query-string cases stay
        # correct on Modal's hostnames.
        next_url = httpx.URL(location)
        if not next_url.is_absolute_url:
            next_url = httpx.URL(location, base=str(current.request.url))

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TranscriptionError(
                f"modal whisper poll deadline reached after {polls} polls "
                f"(deadline={deadline_s:.0f}s)"
            )

        # Brief backoff before the next poll. Cap by remaining deadline so
        # the final poll fires at most just-before the deadline rather
        # than oversleeping past it.
        await asyncio.sleep(min(_POLL_INTERVAL_S, remaining))
        polls += 1
        if polls == 1 or polls % 12 == 0:
            # Log first poll + once per minute so prod logs show progress
            # without spamming on every sub-minute poll.
            _log.info(
                "modal_whisper.poll",
                stream_id=stream_id, poll_n=polls,
                next_url_host=str(next_url.host),
            )
        current = await client.get(str(next_url))
    return current


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
