# syntax=docker/dockerfile:1.7
#
# NexoClip production image — slim Debian + Python 3.11 + ffmpeg.
# CPU-only by design. Transcription + diarization run on AssemblyAI
# (Migration Tasks A1-A3) so there's no torch / faster-whisper /
# pyannote / CUDA in this image. The `diarize` and `local-whisper`
# optional extras stay available in pyproject.toml for users who
# want local-GPU inference; production deploys do NOT install them.
#
# For a GPU-enabled image: swap the base image to nvidia/cuda:12.4-
# runtime-ubuntu22.04 and add `pip install '.[local-whisper,diarize]'`
# below — the rest of the layout stays.

FROM python:3.11-slim-bookworm

# System packages:
#   ffmpeg          — cut + reformat clips, audio extraction
#   build-essential — some Python deps compile native extensions
#   ca-certificates — outbound HTTPS to Anthropic, Resend, AssemblyAI, etc.
#   curl + unzip    — fetch + unpack the deno runtime (next layer)
# Keep the layer minimal: rm apt lists after install.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
        ca-certificates \
        curl \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# JavaScript runtime for yt-dlp. Current yt-dlp needs a JS runtime to solve
# YouTube's player challenge (nsig/signature); WITHOUT one it falls back to
# the `android_vr` client, which YouTube gates behind "Sign in to confirm
# you're not a bot" — so every YouTube ingest 403s even with valid cookies
# (confirmed in prod: `[debug] JS runtimes: none` → `LOGIN_REQUIRED`). deno
# is the runtime yt-dlp enables by DEFAULT, so just having it on PATH fixes
# the extraction with no application-code change. Pinned via the `latest`
# release asset for linux x86_64 (Railway's arch).
RUN curl -fsSL \
        https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip \
        -o /tmp/deno.zip \
    && unzip -q /tmp/deno.zip -d /usr/local/bin \
    && rm /tmp/deno.zip \
    && chmod +x /usr/local/bin/deno \
    && /usr/local/bin/deno --version

WORKDIR /app

# Install Python deps. We COPY the package source rather than just
# pyproject.toml because `pip install .` needs the `nexoclip/` package to
# exist to compute metadata. Trade-off: changing any .py invalidates this
# layer. Acceptable for v1 deploys; optimize the cache split later if
# image-build time becomes a problem.
COPY pyproject.toml README.md ./
COPY nexoclip ./nexoclip
COPY run.py ./run.py
# Slice O.28 — ship the config/ dir so the LLM router actually finds
# its routing rules. Without this, load_llm_config() in the running
# container hits an empty `config/llm.yaml` lookup, returns defaults,
# and the pipeline crashes at the variants step with
# `LLMError: unknown routing purpose: variant_generation`.
COPY config ./config

# Migration Task A3 — CPU-only deps. The base install pulls only what
# the AssemblyAI-driven pipeline needs (~150 MB total deps including
# httpx, FastAPI, opencv-python, Pillow, yt-dlp). No torch, no CUDA.
# A separate `.[local-whisper,diarize]` install path stays available
# for self-hosted users who want the GPU-bound stack — see
# pyproject.toml for the extras' rationale.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir '.'

# Slice O.28 — Playwright + Chromium for the preview-recorder (slice
# O.20). Without these, `/clips/<id>/download` falls back to the
# ffmpeg burn which renders captions as plain libass without the
# CSS karaoke pop-color the operator sees in the editor preview.
# Installing chromium adds ~300 MB to the image but guarantees the
# downloaded MP4 is pixel-identical to the browser render.
RUN pip install --no-cache-dir playwright>=1.50 && \
    playwright install --with-deps chromium

# Ship the ops scripts (e.g. the SQLite→Postgres data cutover) so they can be
# run from a Railway shell. Placed after the dependency layers so editing a
# script doesn't invalidate the pip/playwright cache.
COPY scripts ./scripts

# All persistent state lives on /data:
#   * SQLite DB
#   * Output clips + frames (ffmpeg writes here)
#   * Whisper model cache (HuggingFace downloads — ~244MB for `small`)
# Without a persistent volume mounted at /data, every redeploy loses
# everything.
#
# Railway-specific note: we DON'T declare `VOLUME ["/data"]` here because
# Railway rejects anonymous Docker volumes — they have their own volume
# system that's configured per-service on the dashboard (Settings →
# Volumes → New Volume → mount path `/data`). Fly.io takes the same
# approach. If you ever switch to a platform that respects the `VOLUME`
# declaration (raw Docker, ECS, K8s), add it back.

# Sensible production defaults. Override any via Railway env-var dashboard.
#   NEXOCLIP_HOST=0.0.0.0                — bind to all interfaces (container)
#   NEXOCLIP_TRANSCRIBE_PROVIDER=assemblyai  — Migration Task A3 default;
#                                           pipeline.transcribe runs against
#                                           AssemblyAI's batch API. The
#                                           operator must set
#                                           NEXOCLIP_ASSEMBLYAI_API_KEY on
#                                           the Railway dashboard before the
#                                           first job runs.
#   NEXOCLIP_DIARIZATION_SOURCE          — leaves default ("pyannote" in
#                                           config) but pipeline auto-falls
#                                           through to skipped on the slim
#                                           image. Set to "transcribe" to use
#                                           AssemblyAI utterance speakers
#                                           directly.
#   PYTHONUNBUFFERED=1                   — see logs in real-time
ENV NEXOCLIP_DB_PATH=/data/nexoclip.db \
    NEXOCLIP_DEFAULT_OUTPUT_DIR=/data/out \
    NEXOCLIP_HOST=0.0.0.0 \
    NEXOCLIP_TRANSCRIBE_PROVIDER=assemblyai \
    PYTHONUNBUFFERED=1

# Documentation only — Railway dynamically assigns $PORT and our CMD
# wires it through to NEXOCLIP_PORT which run.py reads.
EXPOSE 8000

# Railway sets $PORT; run.py reads NEXOCLIP_PORT. Translate at boot.
# Use ${PORT:-8000} so the same image works locally (just `docker run -p
# 8000:8000`) without setting PORT explicitly.
CMD ["sh", "-c", "NEXOCLIP_PORT=${PORT:-8000} python run.py"]
