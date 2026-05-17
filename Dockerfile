# syntax=docker/dockerfile:1.7
#
# NexoClip production image — slim Debian + Python 3.11 + ffmpeg.
# Targets CPU-only deploys (Railway, Fly.io, Render). For GPU later,
# swap the base image to `nvidia/cuda:12.4-runtime-ubuntu22.04` and pip-install
# nvidia-cublas-cu12 / nvidia-cudnn-cu12 — the rest of the layout stays.
#
# IMPORTANT pinning: faster-whisper requires Python <= 3.12 per CLAUDE.md
# rule #1 of the run.py boot guard. Don't bump to 3.13+ without checking.

FROM python:3.11-slim-bookworm

# System packages:
#   ffmpeg          — cut + reformat clips, audio extraction
#   build-essential — some Python deps compile native extensions
#   ca-certificates — outbound HTTPS to Anthropic, Resend, Nexo AI, etc.
# Keep the layer minimal: rm apt lists after install.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

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

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Slice O.28 — Playwright + Chromium for the preview-recorder (slice
# O.20). Without these, `/clips/<id>/download` falls back to the
# ffmpeg burn which renders captions as plain libass without the
# CSS karaoke pop-color the operator sees in the editor preview.
# Installing chromium adds ~300 MB to the image but guarantees the
# downloaded MP4 is pixel-identical to the browser render.
RUN pip install --no-cache-dir playwright>=1.50 && \
    playwright install --with-deps chromium

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
#   NEXOCLIP_HOST=0.0.0.0       — bind to all interfaces (required in container)
#   NEXOCLIP_WHISPER_DEVICE=cpu — no GPU on slim image
#   NEXOCLIP_WHISPER_MODEL=base — 74 MB. We previously used `small` (244 MB)
#                                but it leaves no room on Railway's 500 MB free
#                                tier alongside stream artifacts. `base` is the
#                                quality floor for production Spanish/English
#                                streams; bump to `small` / `medium` once the
#                                volume is on a paid tier (1 GB / 5 GB).
#   HF_HOME                     — keep model cache on the persistent volume so
#                                cold-starts don't re-download
#   PYTHONUNBUFFERED=1          — see logs in real-time (no stdout buffering)
ENV NEXOCLIP_DB_PATH=/data/nexoclip.db \
    NEXOCLIP_DEFAULT_OUTPUT_DIR=/data/out \
    NEXOCLIP_HOST=0.0.0.0 \
    NEXOCLIP_WHISPER_DEVICE=cpu \
    NEXOCLIP_WHISPER_COMPUTE_TYPE=int8 \
    NEXOCLIP_WHISPER_MODEL=base \
    HF_HOME=/data/hf_cache \
    PYTHONUNBUFFERED=1

# Documentation only — Railway dynamically assigns $PORT and our CMD
# wires it through to NEXOCLIP_PORT which run.py reads.
EXPOSE 8000

# Railway sets $PORT; run.py reads NEXOCLIP_PORT. Translate at boot.
# Use ${PORT:-8000} so the same image works locally (just `docker run -p
# 8000:8000`) without setting PORT explicitly.
CMD ["sh", "-c", "NEXOCLIP_PORT=${PORT:-8000} python run.py"]
