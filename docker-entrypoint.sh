#!/bin/sh
# One image, two roles.
#
# ChalybClip splits into two Cloud Run services that run the SAME image:
#
#   api     — the FastAPI dashboard + the Chalyb hub contract
#             (POST /api/admin/tenants, GET /auth/sso). Default.
#   worker  — the pipeline worker (`nexoclip worker`). Speaks the kickoff/poll
#             HTTP contract the ModalJobDispatcher already talks, so the API
#             dispatches to it by pointing NEXOCLIP_MODAL_PIPELINE_ENDPOINT_URL
#             at this service's URL. No new dispatcher code needed.
#
# Set NEXOCLIP_ROLE=worker on the worker service. Anything else (or unset)
# serves the API, which keeps the existing Railway deploy working unchanged.
#
# Both roles bind $PORT — Cloud Run assigns it and requires the container to
# listen on it. run.py reads NEXOCLIP_PORT, and `nexoclip worker` takes --port.
set -eu

PORT="${PORT:-8000}"

case "${NEXOCLIP_ROLE:-api}" in
  worker)
    exec nexoclip worker --host 0.0.0.0 --port "$PORT"
    ;;
  api)
    NEXOCLIP_PORT="$PORT" exec python run.py
    ;;
  *)
    echo "docker-entrypoint: unknown NEXOCLIP_ROLE='${NEXOCLIP_ROLE}' (expected 'api' or 'worker')" >&2
    exit 64
    ;;
esac
