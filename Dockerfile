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

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# All persistent state lives on /data — Railway / Fly volumes mount here.
#   * SQLite DB
#   * Output clips + frames (ffmpeg writes here)
#   * Whisper model cache (HuggingFace downloads — ~244MB for `small`)
# Without the volume, every redeploy loses everything. Volume mount is
# configured on the platform side; this VOLUME declaration documents the
# contract.
VOLUME ["/data"]

# Sensible production defaults. Override any via Railway env-var dashboard.
#   NEXOCLIP_HOST=0.0.0.0       — bind to all interfaces (required in container)
#   NEXOCLIP_WHISPER_DEVICE=cpu — no GPU on slim image
#   HF_HOME                     — keep model cache on the persistent volume
#   PYTHONUNBUFFERED=1          — see logs in real-time (no stdout buffering)
ENV NEXOCLIP_DB_PATH=/data/nexoclip.db \
    NEXOCLIP_DEFAULT_OUTPUT_DIR=/data/out \
    NEXOCLIP_HOST=0.0.0.0 \
    NEXOCLIP_WHISPER_DEVICE=cpu \
    NEXOCLIP_WHISPER_COMPUTE_TYPE=int8 \
    NEXOCLIP_WHISPER_MODEL=small \
    HF_HOME=/data/hf_cache \
    PYTHONUNBUFFERED=1

# Documentation only — Railway dynamically assigns $PORT and our CMD
# wires it through to NEXOCLIP_PORT which run.py reads.
EXPOSE 8000

# Railway sets $PORT; run.py reads NEXOCLIP_PORT. Translate at boot.
# Use ${PORT:-8000} so the same image works locally (just `docker run -p
# 8000:8000`) without setting PORT explicitly.
CMD ["sh", "-c", "NEXOCLIP_PORT=${PORT:-8000} python run.py"]
